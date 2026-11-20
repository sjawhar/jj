//! Patched in this tree: the tests of gix-pack's reads through jj's chunked mapping, `chunked_mmap`. They live here,
//! in gix-pack, so that the test binary links one copy of the process-wide registry of chunks.

#![cfg(target_os = "linux")]

use std::any::Any;
#[cfg(not(feature = "wasm"))]
use std::cell::RefCell;
use std::fs;
use std::fs::File;
use std::hint::black_box;
use std::io::Write as _;
use std::ops::Range;
use std::os::unix::fs::MetadataExt as _;
use std::path::Path;
use std::path::PathBuf;
#[cfg(not(feature = "wasm"))]
use std::rc::Rc;
use std::sync::Mutex;
use std::sync::MutexGuard;
use std::sync::PoisonError;
use std::sync::atomic::AtomicBool;

use chunked_mmap::CHUNK_SIZE;
use chunked_mmap::testing;
use gix_hash::ObjectId;
use gix_hash::oid;
use gix_pack::FileData;

const SHA1: gix_hash::Kind = gix_hash::Kind::Sha1;

/// Serializes the tests of this binary, whose chunk size and budget are the whole process's.
static SETTINGS_LOCK: Mutex<()> = Mutex::new(());

/// The process's chunk size and budget for one test, restored on drop.
struct Settings {
    _lock: MutexGuard<'static, ()>,
    chunk_size: usize,
    limit: usize,
}

impl Drop for Settings {
    fn drop(&mut self) {
        testing::set_chunk_size(self.chunk_size);
        testing::set_limit(self.limit);
    }
}

fn settings(chunk_size: usize, limit: usize) -> Settings {
    let lock = SETTINGS_LOCK.lock().unwrap_or_else(PoisonError::into_inner);
    let previous = Settings {
        _lock: lock,
        chunk_size: testing::chunk_size(),
        limit: testing::limit(),
    };
    testing::set_chunk_size(chunk_size);
    testing::set_limit(limit);
    previous
}

/// A chunk for files of test size: 16 pages.
fn small_chunk() -> usize {
    16 * testing::page_size()
}

/// `len` bytes that differ for each `seed` and do not compress (SplitMix64).
fn random_bytes(seed: u64, len: usize) -> Vec<u8> {
    let mut state = seed;
    let mut bytes = Vec::with_capacity(len.next_multiple_of(8));
    while bytes.len() < len {
        state = state.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = state;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        bytes.extend_from_slice(&(z ^ (z >> 31)).to_le_bytes());
    }
    bytes.truncate(len);
    bytes
}

/// A blob of a test pack.
enum Blob {
    /// Stored whole.
    Whole(Vec<u8>),
    /// Stored as a delta of the earlier blob `base`: its content with `insert` inserted at `at`.
    Delta {
        base: usize,
        at: usize,
        insert: Vec<u8>,
    },
}

/// Blobs of random content and sizes, from `seed` on. Each starts with its seed, so no two are the same object.
fn random_blobs(
    seed: u64,
    count: usize,
    len: impl Fn(usize) -> usize,
) -> impl Iterator<Item = Blob> {
    (0..count).map(move |index| {
        let seed = seed + index as u64;
        Blob::Whole([&seed.to_le_bytes()[..], &random_bytes(seed, len(index))].concat())
    })
}

/// A chain of `depth` deltas, each of the one before, from the blob `base` of `blobs`.
fn push_delta_chain(blobs: &mut Vec<Blob>, mut base: usize, depth: usize, at: usize) {
    for link in 0..depth {
        blobs.push(Blob::Delta {
            base,
            at: at + link,
            insert: random_bytes(7_000 + link as u64, 300),
        });
        base = blobs.len() - 1;
    }
}

/// A pack and its index, written by gix-pack from blobs, with each blob's id and content and the blob it is a delta
/// of.
struct TestPack {
    data_path: PathBuf,
    index_path: PathBuf,
    blobs: Vec<(ObjectId, Vec<u8>)>,
    bases: Vec<Option<usize>>,
}

impl TestPack {
    fn data_len(&self) -> usize {
        file_len(&self.data_path)
    }
}

/// The pack stream of `blobs`, and each blob's content and base.
fn pack_stream(blobs: Vec<Blob>) -> (Vec<u8>, Vec<Vec<u8>>, Vec<Option<usize>>) {
    let mut pack = Vec::new();
    pack.extend_from_slice(b"PACK");
    pack.extend_from_slice(&2u32.to_be_bytes());
    pack.extend_from_slice(&u32::try_from(blobs.len()).unwrap().to_be_bytes());
    let mut offsets = Vec::new();
    let mut contents: Vec<Vec<u8>> = Vec::new();
    let mut bases = Vec::new();
    for blob in blobs {
        let offset = pack.len();
        offsets.push(offset);
        let (content, base) = match blob {
            Blob::Whole(content) => {
                write_entry_header(&mut pack, 3, content.len());
                pack.extend(deflate(&content));
                (content, None)
            }
            Blob::Delta { base, at, insert } => {
                let base_content = &contents[base];
                let delta = delta(base_content, at, &insert);
                write_entry_header(&mut pack, 6, delta.len());
                write_base_distance(&mut pack, offset - offsets[base]);
                pack.extend(deflate(&delta));
                (
                    [&base_content[..at], &insert, &base_content[at..]].concat(),
                    Some(base),
                )
            }
        };
        contents.push(content);
        bases.push(base);
    }
    let mut hasher = gix_hash::hasher(SHA1);
    hasher.update(&pack);
    pack.extend_from_slice(hasher.try_finalize().unwrap().as_bytes());
    (pack, contents, bases)
}

/// Writes a pack of `blobs` into `dir`, with the index that gix-pack's index writer makes for it, resolving entries
/// through memmap2's one mapping of the pack.
fn write_pack(dir: &Path, blobs: Vec<Blob>) -> TestPack {
    let (pack, contents, bases) = pack_stream(blobs);
    let checksum = ObjectId::from_bytes_or_panic(&pack[pack.len() - SHA1.len_in_bytes()..]);
    let data_path = dir.join(format!("pack-{checksum}.pack"));
    fs::write(&data_path, &pack).unwrap();
    let index_path = data_path.with_extension("idx");
    let mut entries = gix_pack::data::input::BytesToEntriesIter::new_from_header(
        pack.as_slice(),
        gix_pack::data::input::Mode::Verify,
        gix_pack::data::input::EntryDataMode::Crc32,
        SHA1,
    )
    .unwrap();
    let whole = Whole::open(&data_path);
    gix_pack::index::write_data_iter_to_stream(
        gix_pack::index::Version::V2,
        move || Ok((resolve_in_whole, whole)),
        &mut entries,
        None,
        &mut gix_features::progress::Discard,
        &mut File::create(&index_path).unwrap(),
        &AtomicBool::new(false),
        SHA1,
        None,
        gix_pack::data::Version::V2,
    )
    .unwrap();
    test_pack(data_path, index_path, contents, bases)
}

fn resolve_in_whole(range: gix_pack::data::EntryRange, whole: &Whole) -> Option<&[u8]> {
    whole.0.get(range.start as usize..range.end as usize)
}

/// Writes a pack of `blobs` into `dir` through gix-pack's bundle writer, whose resolver maps the pack it writes. The
/// `wasm` feature of gix-pack leaves the bundle writer out.
#[cfg(not(feature = "wasm"))]
fn write_pack_through_bundle_writer(dir: &Path, blobs: Vec<Blob>) -> TestPack {
    let (pack, contents, bases) = pack_stream(blobs);
    let outcome = gix_pack::Bundle::write_to_directory(
        &mut pack.as_slice(),
        Some(dir),
        &mut gix_features::progress::Discard,
        &AtomicBool::new(false),
        None::<gix_object::find::Never>,
        SHA1,
        gix_pack::bundle::write::Options::default(),
    )
    .unwrap();
    test_pack(
        outcome.data_path.unwrap(),
        outcome.index_path.unwrap(),
        contents,
        bases,
    )
}

fn test_pack(
    data_path: PathBuf,
    index_path: PathBuf,
    contents: Vec<Vec<u8>>,
    bases: Vec<Option<usize>>,
) -> TestPack {
    let blobs = contents
        .into_iter()
        .map(|content| {
            let id = gix_object::compute_hash(SHA1, gix_object::Kind::Blob, &content).unwrap();
            (id, content)
        })
        .collect();
    TestPack {
        data_path,
        index_path,
        blobs,
        bases,
    }
}

/// Writes a multi-pack-index of `indexes` into `dir`.
fn write_multi_index(dir: &Path, indexes: &[&Path]) -> PathBuf {
    let path = dir.join("multi-pack-index");
    let mut file = File::create(&path).unwrap();
    gix_pack::multi_index::write_from_index_paths(
        indexes.iter().map(|path| path.to_path_buf()).collect(),
        &mut file,
        &mut gix_features::progress::Discard,
        &AtomicBool::new(false),
        gix_pack::multi_index::write::Options { object_hash: SHA1 },
    )
    .unwrap();
    path
}

fn write_entry_header(pack: &mut Vec<u8>, kind: u8, size: usize) {
    let mut byte = (kind << 4) | (size & 0x0f) as u8;
    let mut rest = size >> 4;
    while rest != 0 {
        pack.push(byte | 0x80);
        byte = (rest & 0x7f) as u8;
        rest >>= 7;
    }
    pack.push(byte);
}

fn write_base_distance(pack: &mut Vec<u8>, mut distance: usize) {
    let mut bytes = vec![(distance & 0x7f) as u8];
    distance >>= 7;
    while distance != 0 {
        distance -= 1;
        bytes.push(0x80 | (distance & 0x7f) as u8);
        distance >>= 7;
    }
    bytes.reverse();
    pack.extend(bytes);
}

fn write_varint(out: &mut Vec<u8>, mut value: usize) {
    while value >= 0x80 {
        out.push((value & 0x7f) as u8 | 0x80);
        value >>= 7;
    }
    out.push(value as u8);
}

/// A delta that makes `base` with `insert` inserted at `at`.
fn delta(base: &[u8], at: usize, insert: &[u8]) -> Vec<u8> {
    fn copy(delta: &mut Vec<u8>, range: Range<usize>) {
        for start in range.clone().step_by(0xff_ffff) {
            let len = (range.end - start).min(0xff_ffff);
            delta.push(0x80 | 0x7f);
            delta.extend_from_slice(&u32::try_from(start).unwrap().to_le_bytes());
            delta.extend_from_slice(&u32::try_from(len).unwrap().to_le_bytes()[..3]);
        }
    }
    let mut delta = Vec::new();
    write_varint(&mut delta, base.len());
    write_varint(&mut delta, base.len() + insert.len());
    copy(&mut delta, 0..at);
    for piece in insert.chunks(0x7f) {
        delta.push(piece.len() as u8);
        delta.extend_from_slice(piece);
    }
    copy(&mut delta, at..base.len());
    delta
}

fn deflate(data: &[u8]) -> Vec<u8> {
    let mut out =
        gix_zlib::stream::deflate::Write::new(Vec::new(), gix_zlib::Compression::BEST_SPEED);
    out.write_all(data).unwrap();
    out.flush().unwrap();
    out.into_inner()
}

/// memmap2's one mapping of a whole file, as gix-pack maps files without the patch. Its reads mark nothing.
struct Whole(memmap2::Mmap);

impl Whole {
    fn open(path: &Path) -> Self {
        let file = File::open(path).unwrap();
        // SAFETY: nothing changes a test's file while it is mapped.
        Self(unsafe { memmap2::MmapOptions::new().map_copy_read_only(&file) }.unwrap())
    }
}

impl FileData for Whole {
    fn len(&self) -> usize {
        self.0.len()
    }

    fn slice(&self, range: Range<usize>) -> &[u8] {
        &self.0[range]
    }

    fn read_from<R>(&self, start: usize, read: impl FnOnce(&[u8]) -> (R, usize)) -> R {
        read(&self.0[start..]).0
    }
}

/// Decodes the blobs of one pack through gix-pack, from whichever mappings back its pack and index.
struct Decoder<T: FileData> {
    pack: gix_pack::data::File<T>,
    index: gix_pack::index::File<T>,
}

impl Decoder<gix_pack::MMap> {
    /// Opens `pack` as gix-pack opens packs: in chunks.
    fn chunked(pack: &TestPack) -> Self {
        Self {
            pack: gix_pack::data::File::at(&pack.data_path, SHA1).unwrap(),
            index: gix_pack::index::File::at(&pack.index_path, SHA1).unwrap(),
        }
    }
}

impl Decoder<Whole> {
    /// Opens `pack` with one memmap2 mapping of each file.
    fn whole(pack: &TestPack) -> Self {
        Self {
            pack: gix_pack::data::File::from_data(
                Whole::open(&pack.data_path),
                pack.data_path.clone(),
                SHA1,
            )
            .unwrap(),
            index: gix_pack::index::File::from_data(
                Whole::open(&pack.index_path),
                pack.index_path.clone(),
                SHA1,
            )
            .unwrap(),
        }
    }
}

impl<T: FileData> Decoder<T> {
    fn decode(&self, id: &oid) -> Vec<u8> {
        let index = self.index.lookup(id).expect("the blob is in the index");
        decode_at(&self.pack, self.index.pack_offset_at_index(index))
    }

    fn offset(&self, id: &oid) -> usize {
        let index = self.index.lookup(id).expect("the blob is in the index");
        usize::try_from(self.index.pack_offset_at_index(index)).unwrap()
    }
}

/// Decodes the blob at `offset` of `pack`.
fn decode_at<T: FileData>(pack: &gix_pack::data::File<T>, offset: u64) -> Vec<u8> {
    let entry = pack.entry(offset).unwrap();
    let mut out = Vec::new();
    let outcome = pack
        .decode_entry(
            entry,
            &mut out,
            &mut gix_zlib::Inflate::default(),
            &|_, _| None,
            &mut gix_pack::cache::Never,
        )
        .unwrap();
    assert_eq!(outcome.kind, gix_object::Kind::Blob);
    out
}

/// The byte range of each blob's entry in its pack: from its header to the next entry or the trailer.
fn entry_ranges(pack: &TestPack) -> Vec<Range<usize>> {
    let whole = Decoder::whole(pack);
    let starts: Vec<usize> = pack.blobs.iter().map(|(id, _)| whole.offset(id)).collect();
    let mut sorted = starts.clone();
    sorted.sort_unstable();
    sorted.push(pack.data_len() - SHA1.len_in_bytes());
    starts
        .iter()
        .map(|&start| start..sorted[sorted.partition_point(|&offset| offset <= start)])
        .collect()
}

fn file_len(path: &Path) -> usize {
    usize::try_from(fs::metadata(path).unwrap().len()).unwrap()
}

fn inode(path: &Path) -> u64 {
    fs::metadata(path).unwrap().ino()
}

/// The chunks that hold the bytes of `range`.
fn chunks_of(range: &Range<usize>, chunk: usize) -> Vec<usize> {
    if range.is_empty() {
        Vec::new()
    } else {
        (range.start / chunk..=(range.end - 1) / chunk).collect()
    }
}

fn marked(path: &Path) -> Vec<usize> {
    testing::marked_chunks(path).unwrap()
}

/// A mapping of the process, as /proc/self/maps or /proc/self/smaps lists it.
#[derive(Debug)]
struct Mapping {
    range: Range<usize>,
    perms: String,
    offset: usize,
    inode: u64,
    /// Resident KiB, as /proc/self/smaps gives it; 0 from /proc/self/maps.
    rss_kib: u64,
}

fn read_mappings(path: &str) -> Vec<Mapping> {
    let mut mappings: Vec<Mapping> = Vec::new();
    for line in fs::read_to_string(path).unwrap().lines() {
        if let Some(mapping) = parse_mapping(line) {
            mappings.push(mapping);
        } else if let Some(rss) = line.strip_prefix("Rss:") {
            let kib = rss.trim().strip_suffix("kB").unwrap().trim();
            mappings.last_mut().unwrap().rss_kib = kib.parse().unwrap();
        }
    }
    mappings
}

/// Parses a line `start-end perms offset dev inode [path]`.
fn parse_mapping(line: &str) -> Option<Mapping> {
    let mut fields = line.split_whitespace();
    let (start, end) = fields.next()?.split_once('-')?;
    let range = usize::from_str_radix(start, 16).ok()?..usize::from_str_radix(end, 16).ok()?;
    let perms = fields.next()?.to_owned();
    let offset = usize::from_str_radix(fields.next()?, 16).ok()?;
    let _device = fields.next()?;
    let inode = fields.next()?.parse().ok()?;
    Some(Mapping {
        range,
        perms,
        offset,
        inode,
        rss_kib: 0,
    })
}

/// The mappings of /proc/self/maps of the file at `path`, in address order.
fn maps_of(path: &Path) -> Vec<Mapping> {
    let inode = inode(path);
    read_mappings("/proc/self/maps")
        .into_iter()
        .filter(|mapping| mapping.inode == inode)
        .collect()
}

/// The mappings of /proc/self/smaps of the file at `path`, in address order, with their Rss.
fn smaps_of(path: &Path) -> Vec<Mapping> {
    let inode = inode(path);
    read_mappings("/proc/self/smaps")
        .into_iter()
        .filter(|mapping| mapping.inode == inode)
        .collect()
}

fn rss_kib(mappings: &[Mapping]) -> u64 {
    mappings.iter().map(|mapping| mapping.rss_kib).sum()
}

/// Fails unless `maps` are one `r--p` mapping for each chunk of a file of `len` bytes, at adjacent addresses, the
/// last at the file's remaining length.
fn assert_one_vma_per_chunk(what: &str, maps: &[Mapping], len: usize, chunk: usize) {
    assert_eq!(
        maps.len(),
        len.div_ceil(chunk),
        "{what} of {len} bytes: {maps:#?}"
    );
    for (index, map) in maps.iter().enumerate() {
        let chunk_len = chunk.min(len - index * chunk);
        assert_eq!(map.perms, "r--p", "{what}: {map:?}");
        assert_eq!(map.offset, index * chunk, "{what}: {map:?}");
        assert_eq!(
            map.range.len(),
            chunk_len.next_multiple_of(testing::page_size()),
            "{what}: {map:?}"
        );
        if index > 0 {
            assert_eq!(
                map.range.start,
                maps[index - 1].range.end,
                "{what}: {maps:#?}"
            );
        }
    }
}

#[test]
fn decodes_like_one_mapping() {
    let _settings = settings(CHUNK_SIZE, 512 << 20);
    let dir = tempfile::tempdir().unwrap();
    let mut blobs: Vec<Blob> = random_blobs(0, 30, |index| (2 << 20) + 4099 * index).collect();
    // An object larger than one chunk.
    blobs.push(Blob::Whole(random_bytes(100, CHUNK_SIZE + (4 << 20) + 1)));
    // Chains of deltas whose entries lie in other chunks than their bases.
    push_delta_chain(&mut blobs, 0, 6, 1000);
    push_delta_chain(&mut blobs, 30, 3, CHUNK_SIZE);
    blobs.extend(random_blobs(1000, 3000, |index| 1 + index % 200));
    let pack = write_pack(dir.path(), blobs);
    let len = pack.data_len();
    assert!(
        len > 64 << 20 && !len.is_multiple_of(CHUNK_SIZE),
        "the pack is {len} bytes"
    );

    let whole = Decoder::whole(&pack);
    // With 8 MiB chunks, and with small chunks that split the index too.
    for (chunk_size, limit) in [(CHUNK_SIZE, 512 << 20), (small_chunk(), 8 * small_chunk())] {
        testing::set_chunk_size(chunk_size);
        testing::set_limit(limit);
        let chunked = Decoder::chunked(&pack);
        assert!(maps_of(&pack.index_path).len() > 1 || chunk_size == CHUNK_SIZE);
        for (id, content) in &pack.blobs {
            let decoded = chunked.decode(id);
            assert!(
                decoded == whole.decode(id),
                "{id} decodes to other bytes than through one mapping, in chunks of {chunk_size}"
            );
            assert!(
                decoded == *content,
                "{id} decodes to other bytes than it holds"
            );
        }
    }
}

/// Blobs whose pack spans several small chunks, with deltas, and whose index and multi-pack-index span several too.
fn blobs_of_several_chunks(chunk: usize) -> Vec<Blob> {
    let mut blobs: Vec<Blob> = random_blobs(2000, 24, |index| chunk / 3 + 1234 + index).collect();
    push_delta_chain(&mut blobs, 0, 3, 100);
    push_delta_chain(&mut blobs, 5, 2, chunk / 4);
    blobs.extend(random_blobs(3000, 3 * chunk / 28, |index| 1 + index % 50));
    blobs
}

#[test]
fn maps_one_vma_per_chunk() {
    let chunk = small_chunk();
    let _settings = settings(chunk, 1 << 30);
    let dir = tempfile::tempdir().unwrap();

    let pack = write_pack(dir.path(), blobs_of_several_chunks(chunk));

    // The bundle writer maps the pack it writes, to resolve its entries.
    #[cfg(not(feature = "wasm"))]
    {
        let opened: Rc<RefCell<Vec<Vec<Mapping>>>> = Rc::default();
        testing::set_after_open_hook(Some(Box::new({
            let opened = Rc::clone(&opened);
            move |range: Range<usize>| {
                let maps = read_mappings("/proc/self/maps")
                    .into_iter()
                    .filter(|mapping| {
                        mapping.range.start >= range.start && mapping.range.start < range.end
                    })
                    .collect();
                opened.borrow_mut().push(maps);
            }
        })));
        let bundle_dir = tempfile::tempdir().unwrap();
        let written =
            write_pack_through_bundle_writer(bundle_dir.path(), blobs_of_several_chunks(chunk));
        testing::set_after_open_hook(None);
        let opened = opened.take();
        assert_eq!(opened.len(), 1, "{opened:#?}");
        assert_one_vma_per_chunk(
            "the bundle writer's pack",
            &opened[0],
            written.data_len(),
            chunk,
        );
    }

    let multi_index_path = write_multi_index(dir.path(), &[pack.index_path.as_path()]);
    let data = gix_pack::data::File::at(&pack.data_path, SHA1).unwrap();
    let index = gix_pack::index::File::at(&pack.index_path, SHA1).unwrap();
    let multi_index = gix_pack::multi_index::File::at(&multi_index_path, None).unwrap();
    for (id, content) in pack.blobs.iter().step_by(97) {
        let (_, offset) =
            multi_index.pack_id_and_pack_offset_at_index(multi_index.lookup(id).unwrap());
        assert_eq!(
            offset,
            index.pack_offset_at_index(index.lookup(id).unwrap())
        );
        assert!(decode_at(&data, offset) == *content);
    }
    for path in [&pack.data_path, &pack.index_path, &multi_index_path] {
        let len = file_len(path);
        assert!(len > chunk, "{} is only {len} bytes", path.display());
        assert_one_vma_per_chunk(&path.display().to_string(), &maps_of(path), len, chunk);
    }
}

#[test]
fn budget_holds_across_whole_file_reads() {
    let budget = 16 << 20;
    let _settings = settings(CHUNK_SIZE, 64 << 20);
    let dir = tempfile::tempdir().unwrap();
    let mut blobs: Vec<Blob> = random_blobs(4000, 9, |index| CHUNK_SIZE - 1000 * index).collect();
    // An object larger than one chunk.
    blobs.push(Blob::Whole(random_bytes(4100, CHUNK_SIZE + (4 << 20) + 1)));
    blobs.extend(random_blobs(5000, 100, |index| 1 + index));
    let pack = write_pack(dir.path(), blobs);
    let ranges = entry_ranges(&pack);
    let len = pack.data_len();
    let last_chunk = (len - 1) / CHUNK_SIZE;
    let decoder = Decoder::chunked(&pack);

    // The opens marked the headers they read.
    assert!(marked(&pack.data_path).contains(&0));
    assert!(marked(&pack.index_path).contains(&0));

    // An object larger than one chunk marks each chunk it spans.
    testing::clear_marks();
    let large = &pack.blobs[9];
    assert!(decoder.decode(&large.0) == large.1);
    let spanned = chunks_of(&ranges[9], CHUNK_SIZE);
    assert!(spanned.len() > 1, "{spanned:?}");
    let pack_marks = marked(&pack.data_path);
    assert!(
        spanned.iter().all(|chunk| pack_marks.contains(chunk)),
        "{spanned:?} in {pack_marks:?}"
    );

    // An index lookup and the trailer reads mark theirs.
    testing::clear_marks();
    decoder.index.lookup(large.0).unwrap();
    assert_eq!(marked(&pack.index_path), vec![0]);
    testing::clear_marks();
    assert_eq!(decoder.pack.checksum(), decoder.index.pack_checksum());
    black_box(decoder.index.index_checksum());
    assert_eq!(marked(&pack.data_path), vec![last_chunk]);
    assert_eq!(marked(&pack.index_path), vec![0]);

    // Reads spread over the whole file keep to the budget.
    testing::set_limit(budget);
    for (id, content) in &pack.blobs {
        assert!(decoder.decode(id) == *content);
        let marked_bytes = testing::marked_bytes();
        assert!(
            marked_bytes <= budget,
            "{marked_bytes} bytes are marked after reading {id}"
        );
    }
    // By now the first chunk was released, and the last, read last, is resident.
    let smaps = smaps_of(&pack.data_path);
    assert_eq!(
        smaps[0].rss_kib, 0,
        "the first chunk is still resident: {smaps:#?}"
    );
    assert!(
        smaps[last_chunk].rss_kib > 0,
        "the last chunk is not resident: {smaps:#?}"
    );

    // An unmarked registered chunk is released first: once every mark is forgotten, the next release drops the
    // last chunk, which no read marked since. The first chunks read back the same bytes.
    testing::clear_marks();
    for (id, content) in &pack.blobs[..3] {
        assert!(
            decoder.decode(id) == *content,
            "{id} reads back other bytes"
        );
        assert!(testing::marked_bytes() <= budget);
    }
    let smaps = smaps_of(&pack.data_path);
    assert_eq!(
        smaps[last_chunk].rss_kib, 0,
        "the unmarked last chunk was not released: {smaps:#?}"
    );
}

#[test]
fn one_budget_for_several_packs_and_a_midx() {
    let chunk = small_chunk();
    let budget = 4 * chunk;
    let _settings = settings(chunk, budget);
    let dir = tempfile::tempdir().unwrap();
    let large = write_pack(dir.path(), blobs_of_several_chunks(chunk));
    let small: Vec<TestPack> = [6000, 7000]
        .into_iter()
        .map(|seed| {
            write_pack(
                dir.path(),
                random_blobs(seed, 10, |index| 100 + index).collect(),
            )
        })
        .collect();
    for pack in &small {
        assert!(pack.data_len() <= chunk && file_len(&pack.index_path) <= chunk);
    }
    let packs: Vec<&TestPack> = std::iter::once(&large).chain(&small).collect();
    let index_paths: Vec<&Path> = packs.iter().map(|pack| pack.index_path.as_path()).collect();
    let multi_index_path = write_multi_index(dir.path(), &index_paths);
    assert!(file_len(&multi_index_path) > chunk);

    let multi_index = gix_pack::multi_index::File::at(&multi_index_path, None).unwrap();
    let data: Vec<_> = multi_index
        .index_names()
        .iter()
        .map(|name| {
            gix_pack::data::File::at(dir.path().join(name).with_extension("pack"), SHA1).unwrap()
        })
        .collect();
    let indexes: Vec<_> = packs
        .iter()
        .map(|pack| gix_pack::index::File::at(&pack.index_path, SHA1).unwrap())
        .collect();
    for (pack, index) in packs.iter().zip(&indexes) {
        for (id, content) in &pack.blobs {
            let (pack_id, offset) =
                multi_index.pack_id_and_pack_offset_at_index(multi_index.lookup(id).unwrap());
            assert!(decode_at(&data[pack_id as usize], offset) == *content);
            assert_eq!(
                index.pack_offset_at_index(index.lookup(id).unwrap()),
                offset
            );
            let marked_bytes = testing::marked_bytes();
            assert!(
                marked_bytes <= budget,
                "{marked_bytes} bytes are marked after reading {id}"
            );
        }
    }

    // A pack no larger than one chunk is one mapping, which counts as one chunk of its length.
    for (pack, index) in packs.iter().zip(&indexes).skip(1) {
        assert_eq!(maps_of(&pack.data_path).len(), 1);
        assert_eq!(maps_of(&pack.index_path).len(), 1);
        let id = &pack.blobs[0].0;
        let (pack_id, _) =
            multi_index.pack_id_and_pack_offset_at_index(multi_index.lookup(id).unwrap());
        testing::clear_marks();
        decode_at(
            &data[pack_id as usize],
            index.pack_offset_at_index(index.lookup(id).unwrap()),
        );
        assert_eq!(marked(&pack.data_path), vec![0]);
        assert_eq!(marked(&pack.index_path), vec![0]);
        assert_eq!(
            testing::marked_bytes(),
            pack.data_len() + file_len(&pack.index_path)
        );
    }
}

#[test]
fn stress_decode_release_drop() {
    let chunk = small_chunk();
    let budget = 3 * chunk;
    let _settings = settings(chunk, budget);
    let dir = tempfile::tempdir().unwrap();
    let mut blobs: Vec<Blob> = random_blobs(8000, 40, |index| chunk / 2 + 333 * index).collect();
    push_delta_chain(&mut blobs, 3, 4, 10);
    blobs.extend(random_blobs(9000, 500, |index| 1 + index));
    let pack = write_pack(dir.path(), blobs);
    let failed_before = testing::failed_releases();
    let shared = Decoder::chunked(&pack);
    std::thread::scope(|scope| {
        for thread in 0..8 {
            let (pack, shared) = (&pack, &shared);
            scope.spawn(move || {
                for round in 0..16 {
                    let own = Decoder::chunked(pack);
                    for (index, (id, content)) in pack.blobs.iter().enumerate() {
                        if (index + thread + round) % 5 != 0 {
                            continue;
                        }
                        if index % 2 == 0 {
                            assert!(own.decode(id) == *content, "{id} decodes to other bytes");
                        } else {
                            assert!(shared.decode(id) == *content, "{id} decodes to other bytes");
                        }
                        if index % 7 == 0 {
                            testing::release();
                        }
                    }
                    drop(own);
                }
            });
        }
    });
    assert_eq!(
        testing::failed_releases(),
        failed_before,
        "a release reached a range that was no longer mapped"
    );
    assert!(testing::marked_bytes() <= budget);
}

/// The files that the table of entry points reads, opened in chunks, and what the rows need to know of them.
struct Table {
    _dir: tempfile::TempDir,
    pack: TestPack,
    ranges: Vec<Range<usize>>,
    /// An index the test writes by hand, with a 64-bit offset for the entry `LARGE_ENTRY`.
    large_offsets_path: PathBuf,
    multi_index_path: PathBuf,
    data: gix_pack::data::File,
    index: gix_pack::index::File,
    large_offsets: gix_pack::index::File,
    multi_index: gix_pack::multi_index::File,
}

/// The entry of the hand-written index that has a 64-bit offset.
const LARGE_ENTRY: usize = 7;
/// The 64-bit offset of `LARGE_ENTRY`.
const LARGE_OFFSET: u64 = 5 << 32;

fn table() -> Table {
    let dir = tempfile::tempdir().unwrap();
    let chunk = testing::chunk_size();
    let pack = write_pack(dir.path(), blobs_of_several_chunks(chunk));
    let ranges = entry_ranges(&pack);
    let large_offsets_path = write_index_with_large_offset(dir.path(), 3 * chunk / 28);
    let multi_index_path = write_multi_index(
        dir.path(),
        &[pack.index_path.as_path(), large_offsets_path.as_path()],
    );
    Table {
        data: gix_pack::data::File::at(&pack.data_path, SHA1).unwrap(),
        index: gix_pack::index::File::at(&pack.index_path, SHA1).unwrap(),
        large_offsets: gix_pack::index::File::at(&large_offsets_path, SHA1).unwrap(),
        multi_index: gix_pack::multi_index::File::at(&multi_index_path, None).unwrap(),
        _dir: dir,
        pack,
        ranges,
        large_offsets_path,
        multi_index_path,
    }
}

/// Writes a version 2 index of `count` random ids, whose entry `LARGE_ENTRY` has the 64-bit offset `LARGE_OFFSET`.
/// No pack goes with it.
fn write_index_with_large_offset(dir: &Path, count: usize) -> PathBuf {
    let mut ids: Vec<Vec<u8>> = (0..count)
        .map(|index| random_bytes(20_000 + index as u64, 20))
        .collect();
    ids.sort();
    let mut index = Vec::new();
    index.extend_from_slice(b"\xfftOc");
    index.extend_from_slice(&2u32.to_be_bytes());
    for first_byte in 0..=255u8 {
        let below = ids.iter().filter(|id| id[0] <= first_byte).count();
        index.extend_from_slice(&u32::try_from(below).unwrap().to_be_bytes());
    }
    for id in &ids {
        index.extend_from_slice(id);
    }
    for position in 0..count {
        index.extend_from_slice(&u32::try_from(position).unwrap().to_be_bytes());
    }
    for position in 0..count {
        let offset = if position == LARGE_ENTRY {
            1 << 31
        } else {
            12 + 100 * u32::try_from(position).unwrap()
        };
        index.extend_from_slice(&offset.to_be_bytes());
    }
    index.extend_from_slice(&LARGE_OFFSET.to_be_bytes());
    index.extend_from_slice(&random_bytes(30_000, 40));
    let path = dir.join("pack-large-offsets.idx");
    fs::write(&path, index).unwrap();
    path
}

/// The layout of a version 2 pack index of `count` entries.
struct IndexLayout {
    count: usize,
    len: usize,
}

impl IndexLayout {
    const OIDS: usize = 8 + 256 * 4;

    fn of(path: &Path) -> Self {
        let len = file_len(path);
        let bytes = fs::read(path).unwrap();
        let count =
            u32::from_be_bytes(bytes[Self::OIDS - 4..Self::OIDS].try_into().unwrap()) as usize;
        Self { count, len }
    }

    fn oid(&self, index: usize) -> Range<usize> {
        Self::OIDS + 20 * index..Self::OIDS + 20 * (index + 1)
    }

    fn crc32(&self, index: usize) -> Range<usize> {
        let start = Self::OIDS + 20 * self.count + 4 * index;
        start..start + 4
    }

    fn offset32(&self, index: usize) -> Range<usize> {
        let start = Self::OIDS + 24 * self.count + 4 * index;
        start..start + 4
    }

    fn offset64(&self, index: usize) -> Range<usize> {
        let start = Self::OIDS + 28 * self.count + 8 * index;
        start..start + 8
    }

    fn tables(&self) -> Range<usize> {
        Self::OIDS..Self::OIDS + 28 * self.count
    }

    fn offset32_table(&self) -> Range<usize> {
        self.offset32(0).start..self.offset32(self.count).start
    }

    fn trailer(&self) -> Range<usize> {
        self.len - 20..self.len
    }
}

/// The table of contents of a multi-pack-index: each chunk's id and byte range, and the end of the table.
struct MultiIndexLayout {
    chunks: Vec<([u8; 4], Range<usize>)>,
    toc_end: usize,
    len: usize,
}

impl MultiIndexLayout {
    fn of(path: &Path) -> Self {
        let bytes = fs::read(path).unwrap();
        let num_chunks = usize::from(bytes[6]);
        let entry = |index: usize| {
            let start = 12 + 12 * index;
            let id: [u8; 4] = bytes[start..start + 4].try_into().unwrap();
            let offset = u64::from_be_bytes(bytes[start + 4..start + 12].try_into().unwrap());
            (id, usize::try_from(offset).unwrap())
        };
        let chunks = (0..num_chunks)
            .map(|index| (entry(index).0, entry(index).1..entry(index + 1).1))
            .collect();
        Self {
            chunks,
            toc_end: 12 + 12 * (num_chunks + 1),
            len: bytes.len(),
        }
    }

    fn chunk(&self, id: &[u8; 4]) -> Range<usize> {
        self.chunks
            .iter()
            .find(|(chunk_id, _)| chunk_id == id)
            .unwrap()
            .1
            .clone()
    }

    fn oid(&self, index: usize) -> Range<usize> {
        let start = self.chunk(b"OIDL").start + 20 * index;
        start..start + 20
    }

    fn offsets(&self, index: usize) -> Range<usize> {
        let start = self.chunk(b"OOFF").start + 8 * index;
        start..start + 8
    }

    fn trailer(&self) -> Range<usize> {
        self.len - 20..self.len
    }
}

/// The range of ids whose first byte is `id`'s, in an id table that starts at `table`, read from the fan-out table
/// at `fan` of the file at `path`.
fn bucket(path: &Path, fan: usize, table: usize, id: &oid) -> Range<usize> {
    let bytes = fs::read(path).unwrap();
    let fan_at = |byte: usize| {
        u32::from_be_bytes(
            bytes[fan + 4 * byte..fan + 4 * byte + 4]
                .try_into()
                .unwrap(),
        ) as usize
    };
    let first_byte = usize::from(id.first_byte());
    let below = if first_byte == 0 {
        0
    } else {
        fan_at(first_byte - 1)
    };
    table + 20 * below..table + 20 * fan_at(first_byte)
}

type Call = Box<dyn Fn(&Table) -> Box<dyn Any>>;

/// One public entry point of gix-pack that reads a file: the bytes it surely consumes, and the bytes it may consume.
struct Row {
    name: &'static str,
    call: Call,
    consumed: Vec<(PathBuf, Range<usize>)>,
    within: Vec<(PathBuf, Range<usize>)>,
}

fn row(
    name: &'static str,
    call: impl Fn(&Table) -> Box<dyn Any> + 'static,
    consumed: Vec<(PathBuf, Range<usize>)>,
) -> Row {
    Row {
        name,
        call: Box::new(call),
        within: consumed.clone(),
        consumed,
    }
}

fn kept<T: 'static>(value: T) -> Box<dyn Any> {
    Box::new(value)
}

/// The rows of R8's inventory: each public entry point of gix-pack that reaches a read of a pack, an index or a
/// multi-pack-index.
fn rows(table: &Table) -> Vec<Row> {
    let pack_path = table.pack.data_path.clone();
    let index_path = table.pack.index_path.clone();
    let large_path = table.large_offsets_path.clone();
    let midx_path = table.multi_index_path.clone();
    let pack_len = table.pack.data_len();
    let index = IndexLayout::of(&index_path);
    let large = IndexLayout::of(&large_path);
    let midx = MultiIndexLayout::of(&midx_path);
    let whole = Decoder::whole(&table.pack);

    // A blob in the middle of the pack, the tip of the longest delta chain, and the chain's entries.
    let middle = 12;
    let middle_id = table.pack.blobs[middle].0;
    let middle_start = table.ranges[middle].start;
    let middle_entry = whole.pack.entry(middle_start as u64).unwrap();
    let middle_header = middle_start..middle_start + middle_entry.header_size();
    let tip = 24 + 2;
    let mut chain = vec![tip];
    while let Some(base) = table.pack.bases[*chain.last().unwrap()] {
        chain.push(base);
    }
    assert_eq!(chain.len(), 4, "the chain is {chain:?}");
    let chain_entries: Vec<_> = chain
        .iter()
        .map(|&blob| (pack_path.clone(), table.ranges[blob].clone()))
        .collect();
    let chain_headers: Vec<_> = chain
        .iter()
        .map(|&blob| {
            let start = table.ranges[blob].start;
            let header = whole.pack.entry(start as u64).unwrap().header_size();
            (pack_path.clone(), start..start + header)
        })
        .collect();
    let middle_index = table.index.lookup(middle_id).unwrap() as usize;
    let midx_index = table.multi_index.lookup(middle_id).unwrap() as usize;
    let large_midx_index = (0..table.multi_index.num_objects())
        .find(|&entry| table.multi_index.pack_id_and_pack_offset_at_index(entry).1 == LARGE_OFFSET)
        .unwrap() as usize;
    let large_loff = midx.chunk(b"LOFF");
    let midx_fan = midx.chunk(b"OIDF").start;
    let midx_oids = midx.chunk(b"OIDL").start;
    let prefix = gix_hash::Prefix::new(&middle_id, 12).unwrap();

    let pack_trailer = pack_len - 20..pack_len;
    let with_within = |mut row: Row, within: Vec<(PathBuf, Range<usize>)>| {
        row.within = within;
        row
    };
    vec![
        // Opening.
        row(
            "data::File::at",
            |table| kept(gix_pack::data::File::at(&table.pack.data_path, SHA1).unwrap()),
            vec![(pack_path.clone(), 0..12)],
        ),
        row(
            "index::File::at",
            |table| kept(gix_pack::index::File::at(&table.pack.index_path, SHA1).unwrap()),
            vec![
                (index_path.clone(), 0..IndexLayout::OIDS),
                (index_path.clone(), index.offset32_table()),
            ],
        ),
        row(
            "multi_index::File::at",
            |table| kept(gix_pack::multi_index::File::at(&table.multi_index_path, None).unwrap()),
            vec![
                (midx_path.clone(), 0..midx.toc_end),
                (midx_path.clone(), midx.chunk(b"PNAM")),
                (midx_path.clone(), midx.chunk(b"OIDF")),
            ],
        ),
        // data::File.
        row(
            "data::File::entry",
            move |table| kept(table.data.entry(middle_start as u64).unwrap()),
            vec![(pack_path.clone(), middle_header)],
        ),
        row(
            "data::File::decode_entry",
            move |table| {
                let offset = table.ranges[tip].start as u64;
                kept(decode_at(&table.data, offset))
            },
            chain_entries.clone(),
        ),
        with_within(
            row(
                "data::File::decode_header",
                move |table| {
                    let entry = table.data.entry(table.ranges[tip].start as u64).unwrap();
                    let outcome = table
                        .data
                        .decode_header(entry, &mut gix_zlib::Inflate::default(), &|_| None)
                        .unwrap();
                    kept(outcome)
                },
                chain_headers,
            ),
            chain_entries.clone(),
        ),
        row(
            "data::File::decompress_entry",
            {
                let entry = middle_entry.clone();
                move |table| {
                    let mut out = vec![0; usize::try_from(entry.decompressed_size).unwrap()];
                    kept(
                        table
                            .data
                            .decompress_entry(&entry, &mut gix_zlib::Inflate::default(), &mut out)
                            .unwrap(),
                    )
                }
            },
            vec![(
                pack_path.clone(),
                usize::try_from(middle_entry.data_offset).unwrap()..table.ranges[middle].end,
            )],
        ),
        row(
            "data::File::entry_slice",
            {
                let range = table.ranges[middle].clone();
                move |table| {
                    kept(
                        table
                            .data
                            .entry_slice(range.start as u64..range.end as u64)
                            .unwrap()
                            .len(),
                    )
                }
            },
            vec![(pack_path.clone(), table.ranges[middle].clone())],
        ),
        row(
            "data::File::entry_crc32",
            {
                let range = table.ranges[middle].clone();
                move |table| kept(table.data.entry_crc32(range.start as u64, range.len()))
            },
            vec![(pack_path.clone(), table.ranges[middle].clone())],
        ),
        row(
            "data::File::checksum",
            |table| kept(table.data.checksum()),
            vec![(pack_path.clone(), pack_trailer.clone())],
        ),
        with_within(
            row(
                "data::File::verify_checksum",
                |table| {
                    kept(
                        table
                            .data
                            .verify_checksum(
                                &mut gix_features::progress::Discard,
                                &AtomicBool::new(false),
                            )
                            .unwrap(),
                    )
                },
                vec![(pack_path.clone(), pack_trailer)],
            ),
            vec![(pack_path.clone(), 0..pack_len)],
        ),
        // index::File.
        row(
            "index::File::iter",
            |table| kept(table.index.iter().count()),
            vec![(index_path.clone(), index.tables())],
        ),
        with_within(
            row(
                "index::File::lookup",
                move |table| kept(table.index.lookup(middle_id)),
                vec![(index_path.clone(), index.oid(middle_index))],
            ),
            vec![(
                index_path.clone(),
                bucket(&index_path, 8, IndexLayout::OIDS, &middle_id),
            )],
        ),
        with_within(
            row(
                "index::File::lookup_prefix",
                move |table| kept(table.index.lookup_prefix(prefix, None)),
                vec![(index_path.clone(), index.oid(middle_index))],
            ),
            vec![(
                index_path.clone(),
                bucket(&index_path, 8, IndexLayout::OIDS, &middle_id),
            )],
        ),
        row(
            "index::File::oid_at_index",
            move |table| kept(table.index.oid_at_index(middle_index as u32).to_owned()),
            vec![(index_path.clone(), index.oid(middle_index))],
        ),
        row(
            "index::File::pack_offset_at_index",
            move |table| kept(table.index.pack_offset_at_index(middle_index as u32)),
            vec![(index_path.clone(), index.offset32(middle_index))],
        ),
        row(
            "index::File::pack_offset_at_index with a 64-bit offset",
            |table| {
                let offset = table.large_offsets.pack_offset_at_index(LARGE_ENTRY as u32);
                assert_eq!(offset, LARGE_OFFSET);
                kept(offset)
            },
            vec![
                (large_path.clone(), large.offset32(LARGE_ENTRY)),
                (large_path.clone(), large.offset64(0)),
            ],
        ),
        row(
            "index::File::crc32_at_index",
            move |table| kept(table.index.crc32_at_index(middle_index as u32)),
            vec![(index_path.clone(), index.crc32(middle_index))],
        ),
        with_within(
            row(
                "index::File::sorted_offsets",
                |table| kept(table.large_offsets.sorted_offsets()),
                vec![
                    (large_path.clone(), large.offset32_table()),
                    (large_path.clone(), large.offset64(0)),
                ],
            ),
            vec![
                (large_path.clone(), large.offset32_table()),
                (large_path.clone(), large.offset64(0)),
            ],
        ),
        row(
            "index::File::index_checksum",
            |table| kept(table.index.index_checksum()),
            vec![(index_path.clone(), index.trailer())],
        ),
        row(
            "index::File::pack_checksum",
            |table| kept(table.index.pack_checksum()),
            vec![(index_path.clone(), index.len - 40..index.len - 20)],
        ),
        with_within(
            row(
                "index::File::verify_checksum",
                |table| {
                    kept(
                        table
                            .index
                            .verify_checksum(
                                &mut gix_features::progress::Discard,
                                &AtomicBool::new(false),
                            )
                            .unwrap(),
                    )
                },
                vec![(index_path.clone(), index.trailer())],
            ),
            vec![(index_path.clone(), 0..index.len)],
        ),
        // multi_index::File.
        with_within(
            row(
                "multi_index::File::lookup",
                move |table| kept(table.multi_index.lookup(middle_id)),
                vec![(midx_path.clone(), midx.oid(midx_index))],
            ),
            vec![(
                midx_path.clone(),
                bucket(&midx_path, midx_fan, midx_oids, &middle_id),
            )],
        ),
        with_within(
            row(
                "multi_index::File::lookup_prefix",
                move |table| kept(table.multi_index.lookup_prefix(prefix, None)),
                vec![(midx_path.clone(), midx.oid(midx_index))],
            ),
            vec![(
                midx_path.clone(),
                bucket(&midx_path, midx_fan, midx_oids, &middle_id),
            )],
        ),
        row(
            "multi_index::File::iter",
            |table| kept(table.multi_index.iter().count()),
            vec![
                (midx_path.clone(), midx.chunk(b"OIDL")),
                (midx_path.clone(), midx.chunk(b"OOFF")),
                (midx_path.clone(), large_loff.clone()),
            ],
        ),
        row(
            "multi_index::File::pack_id_and_pack_offset_at_index",
            move |table| {
                kept(
                    table
                        .multi_index
                        .pack_id_and_pack_offset_at_index(midx_index as u32),
                )
            },
            vec![(midx_path.clone(), midx.offsets(midx_index))],
        ),
        row(
            "multi_index::File::pack_id_and_pack_offset_at_index with a 64-bit offset",
            move |table| {
                kept(
                    table
                        .multi_index
                        .pack_id_and_pack_offset_at_index(large_midx_index as u32),
                )
            },
            vec![
                (midx_path.clone(), midx.offsets(large_midx_index)),
                (midx_path.clone(), large_loff),
            ],
        ),
        row(
            "multi_index::File::checksum",
            |table| kept(table.multi_index.checksum()),
            vec![(midx_path.clone(), midx.trailer())],
        ),
        with_within(
            row(
                "multi_index::File::verify_checksum",
                |table| {
                    kept(
                        table
                            .multi_index
                            .verify_checksum(
                                &mut gix_features::progress::Discard,
                                &AtomicBool::new(false),
                            )
                            .unwrap(),
                    )
                },
                vec![(midx_path.clone(), midx.trailer())],
            ),
            vec![(midx_path, 0..midx.len)],
        ),
    ]
}

/// Runs each row with the registry's marks cleared before it, and `check`s the marks it left, by file, while what
/// it opened is still open. Then runs the bundle writer, whose resolver's mapping is dropped within the call, and
/// checks the marks it had when it was dropped.
fn for_each_row(check: impl Fn(&str, &Row, &dyn Fn(&Path) -> Vec<usize>)) {
    let _settings = settings(small_chunk(), 1 << 30);
    let table = table();
    for row in rows(&table) {
        testing::clear_marks();
        let opened = (row.call)(&table);
        check(row.name, &row, &|path| marked(path));
        drop(opened);
    }
    #[cfg(not(feature = "wasm"))]
    check_bundle_writer(&check);
}

/// The bundle writer's resolver, which reads every entry of the pack it writes to index it.
#[cfg(not(feature = "wasm"))]
fn check_bundle_writer(check: &impl Fn(&str, &Row, &dyn Fn(&Path) -> Vec<usize>)) {
    let dropped: Rc<RefCell<Vec<(usize, Vec<usize>)>>> = Rc::default();
    testing::set_before_close_hook(Some(Box::new({
        let dropped = Rc::clone(&dropped);
        move |len: usize, marked: Vec<usize>| dropped.borrow_mut().push((len, marked))
    })));
    let dir = tempfile::tempdir().unwrap();
    testing::clear_marks();
    let written =
        write_pack_through_bundle_writer(dir.path(), blobs_of_several_chunks(small_chunk()));
    testing::set_before_close_hook(None);
    let pack_len = written.data_len();
    let marks = dropped
        .take()
        .into_iter()
        .find(|(len, _)| *len == pack_len)
        .expect("the bundle writer mapped the pack it wrote")
        .1;
    let entries = vec![(written.data_path.clone(), 12..pack_len - 20)];
    let resolver = Row {
        name: "Bundle::write_to_directory, through resolve_entry and new_pack_file_resolver",
        call: Box::new(|_: &Table| kept(())),
        consumed: entries.clone(),
        within: entries,
    };
    let marks_of_written = |path: &Path| {
        if path == written.data_path.as_path() {
            marks.clone()
        } else {
            Vec::new()
        }
    };
    check(resolver.name, &resolver, &marks_of_written);
}

#[test]
fn every_gix_pack_read_marks_its_chunks() {
    let chunk = small_chunk();
    for_each_row(|name, row, marks| {
        for (path, range) in &row.consumed {
            let marked = marks(path);
            let unmarked: Vec<usize> = chunks_of(range, chunk)
                .into_iter()
                .filter(|chunk| !marked.contains(chunk))
                .collect();
            assert!(
                unmarked.is_empty(),
                "{name} consumed bytes {range:?} of {} but left chunks {unmarked:?} unmarked; it \
                 marked {marked:?}",
                path.display()
            );
        }
    });
}

#[test]
fn no_read_marks_far_beyond_what_it_consumed() {
    let chunk = small_chunk();
    for_each_row(|name, row, marks| {
        let mut paths: Vec<&PathBuf> = row.within.iter().map(|(path, _)| path).collect();
        paths.dedup();
        for path in paths {
            let allowed: Vec<usize> = row
                .within
                .iter()
                .filter(|(within_path, _)| within_path == path)
                .flat_map(|(_, range)| chunks_of(range, chunk))
                .flat_map(|chunk| [chunk.saturating_sub(1), chunk, chunk + 1])
                .collect();
            let far: Vec<usize> = marks(path)
                .into_iter()
                .filter(|chunk| !allowed.contains(chunk))
                .collect();
            assert!(
                far.is_empty(),
                "{name} marked chunks {far:?} of {}, far beyond the bytes it may consume, {:?}",
                path.display(),
                row.within
            );
        }
    });
}

#[test]
#[ignore = "runs under gVisor, with JJ_PACK_MMAP_GVISOR=1, against a pack on a disk volume"]
fn gvisor_counts_each_chunk_alone() {
    assert_eq!(
        std::env::var("JJ_PACK_MMAP_GVISOR").as_deref(),
        Ok("1"),
        "this test runs under gVisor only"
    );
    let budget = 16 << 20;
    let _settings = settings(CHUNK_SIZE, budget);
    let dir = tempfile::tempdir().unwrap();
    let mut blobs: Vec<Blob> = random_blobs(10_000, 9, |index| CHUNK_SIZE - 512 * index).collect();
    blobs.push(Blob::Whole(b"small".to_vec()));
    let pack = write_pack(dir.path(), blobs);
    let path = &pack.data_path;
    let len = pack.data_len();
    assert!(len > 64 << 20, "the pack is {len} bytes");
    let chunk_kib = (CHUNK_SIZE >> 10) as u64;
    let len_kib = (len >> 10) as u64;
    let rss = || rss_kib(&smaps_of(path));
    let at = 3 * CHUNK_SIZE + 100;

    let chunked = chunked_mmap::Mmap::open(path).unwrap();
    println!(
        "gvisor: the pack of {len} bytes is {} mappings",
        maps_of(path).len()
    );
    let start = rss();
    black_box(chunked.slice(at..at + 1)[0]);
    let one_read = rss() - start;
    println!(
        "gvisor: chunked: a one-byte read grew the pack's Rss by {one_read} KiB, a chunk is \
         {chunk_kib} KiB"
    );
    for offset in (0..len).step_by(1 << 20) {
        black_box(chunked.slice(offset..offset + 1)[0]);
    }
    let spread = rss();
    println!(
        "gvisor: chunked: after reads spread over the whole file, Rss is {spread} KiB, the budget \
         {} KiB",
        budget >> 10
    );
    testing::set_limit(CHUNK_SIZE);
    testing::release();
    let released = spread - rss();
    println!("gvisor: chunked: a release took {released} KiB off");
    drop(chunked);

    let whole = Whole::open(path);
    let start = rss();
    black_box(whole.0[at]);
    let whole_read = rss() - start;
    println!(
        "gvisor: memmap2's whole mapping: a one-byte read grew the pack's Rss by {whole_read} \
         KiB, the file is {len_kib} KiB"
    );

    assert_eq!(maps_of(path).len(), 1, "memmap2 maps the file once");
    assert!(
        one_read.abs_diff(chunk_kib) <= 1024,
        "a one-byte read grew Rss by {one_read} KiB"
    );
    assert!(
        spread <= (budget >> 10) as u64,
        "Rss is {spread} KiB after reads over the whole file"
    );
    assert!(
        released.abs_diff(chunk_kib) <= 1024,
        "a release took {released} KiB off"
    );
    assert!(
        whole_read >= len_kib * 9 / 10,
        "memmap2's one-byte read grew Rss by {whole_read} KiB only"
    );
}
