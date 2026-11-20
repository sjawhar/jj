// Copyright 2026 The Jujutsu Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//! The process-wide account of the chunks that reads marked, and their
//! release.
//!
//! Each chunk keeps the epoch of its latest mark, or 0 while no read has
//! marked it since it was mapped or last released. A chunk becomes marked or
//! released only under the registry's lock, which also guards the count of
//! marked bytes; a read of a chunk that is marked already only moves its epoch
//! forward, without the lock. The epoch advances each time a chunk becomes
//! marked, so the chunk with the smallest epoch is the least recently marked,
//! ties aside.

use std::ffi::OsStr;
use std::io;
use std::ops::Range;
use std::ptr::NonNull;
use std::slice;
use std::sync::Arc;
use std::sync::LazyLock;
use std::sync::Mutex;
use std::sync::MutexGuard;
use std::sync::PoisonError;
use std::sync::atomic::AtomicU64;
use std::sync::atomic::AtomicUsize;
use std::sync::atomic::Ordering::Relaxed;

/// The environment variable that sets the budget.
const LIMIT_VARIABLE: &str = "JJ_PACK_MAPPED_LIMIT";

/// The budget when `JJ_PACK_MAPPED_LIMIT` is unset.
pub(crate) const DEFAULT_LIMIT: usize = 512 << 20;

static REGISTRY: Mutex<Registry> = Mutex::new(Registry {
    mappings: Vec::new(),
    marked_bytes: 0,
});

/// The epoch that a mark records; it advances each time a chunk becomes
/// marked.
static EPOCH: AtomicU64 = AtomicU64::new(1);

/// The budget that `JJ_PACK_MAPPED_LIMIT` gives, read at first use.
static LIMIT_FROM_ENV: LazyLock<Result<usize, String>> =
    LazyLock::new(|| parse_limit(std::env::var_os(LIMIT_VARIABLE).as_deref()));

/// A budget that a test set, or `usize::MAX` for none.
static LIMIT_FOR_TESTS: AtomicUsize = AtomicUsize::new(usize::MAX);

/// How many `madvise` calls of a release failed, which only a release of a
/// range that is no longer mapped can do.
static FAILED_RELEASES: AtomicU64 = AtomicU64::new(0);

/// One file's mapping, as the registry sees it.
pub(crate) struct Mapping {
    pub(crate) ptr: NonNull<u8>,
    pub(crate) len: usize,
    chunk_size: usize,
    /// The device and inode of the file, for tests to find the mapping by.
    file: (u64, u64),
    /// For each chunk, the epoch of its latest mark, or 0.
    marks: Box<[AtomicU64]>,
}

// SAFETY: the mapping is read-only memory that stays mapped while the mapping
// lives; the registry only releases its chunks, which read back the same bytes.
unsafe impl Send for Mapping {}
// SAFETY: as above; the marks are atomics.
unsafe impl Sync for Mapping {}

impl Mapping {
    pub(crate) fn bytes(&self) -> &[u8] {
        // SAFETY: the mapping covers `len` readable bytes from `ptr` while it
        // lives, and nothing writes to a private read-only mapping. A released
        // chunk reads back the same bytes from the file.
        unsafe { slice::from_raw_parts(self.ptr.as_ptr(), self.len) }
    }

    pub(crate) fn mark(&self, range: Range<usize>) {
        let end = range.end.min(self.len);
        if range.start >= end {
            return;
        }
        for chunk in range.start / self.chunk_size..=(end - 1) / self.chunk_size {
            let mark = &self.marks[chunk];
            let epoch = EPOCH.load(Relaxed);
            let mut current = mark.load(Relaxed);
            // A chunk that is marked already only takes the current epoch, and
            // keeps a newer one that another read gave it meanwhile. One that is
            // not marked takes the lock to become marked, unless another read
            // marks it first.
            while current < epoch {
                if current == 0 {
                    lock().mark_unmarked(self, chunk);
                    break;
                }
                match mark.compare_exchange_weak(current, epoch, Relaxed, Relaxed) {
                    Ok(_) => break,
                    Err(actual) => current = actual,
                }
            }
        }
    }

    fn chunk_range(&self, chunks: Range<usize>) -> (*mut u8, usize) {
        let start = chunks.start * self.chunk_size;
        let end = (chunks.end * self.chunk_size).min(self.len);
        (self.ptr.as_ptr().wrapping_add(start), end - start)
    }

    fn chunk_len(&self, chunk: usize) -> usize {
        self.chunk_range(chunk..chunk + 1).1
    }

    /// Releases every chunk of this mapping that no read has marked since it
    /// was mapped or last released, one `madvise` per run of such chunks.
    fn release_unmarked(&self) {
        let mut run_start = None;
        for chunk in 0..=self.marks.len() {
            let unmarked = self
                .marks
                .get(chunk)
                .is_some_and(|mark| mark.load(Relaxed) == 0);
            match (unmarked, run_start) {
                (true, None) => run_start = Some(chunk),
                (false, Some(start)) => {
                    release(self.chunk_range(start..chunk));
                    run_start = None;
                }
                _ => {}
            }
        }
    }
}

struct Registry {
    mappings: Vec<Arc<Mapping>>,
    /// The sum of the lengths of the chunks whose mark is not 0.
    marked_bytes: usize,
}

impl Registry {
    fn mark_unmarked(&mut self, mapping: &Mapping, chunk: usize) {
        let epoch = EPOCH.fetch_add(1, Relaxed) + 1;
        if mapping.marks[chunk].swap(epoch, Relaxed) == 0 {
            self.marked_bytes += mapping.chunk_len(chunk);
            if self.marked_bytes > limit() {
                self.release();
            }
        }
    }

    /// Releases every chunk that no read has marked since it was mapped or
    /// last released, then the least recently marked chunks until the marked
    /// bytes are within the budget.
    fn release(&mut self) {
        for mapping in &self.mappings {
            mapping.release_unmarked();
        }
        let limit = limit();
        while self.marked_bytes > limit {
            let Some((mapping, chunk, epoch)) = self.least_recently_marked() else {
                break;
            };
            let mapping = &self.mappings[mapping];
            // A read that marks the chunk meanwhile keeps it.
            if mapping.marks[chunk]
                .compare_exchange(epoch, 0, Relaxed, Relaxed)
                .is_ok()
            {
                let range = mapping.chunk_range(chunk..chunk + 1);
                release(range);
                self.marked_bytes -= range.1;
            }
        }
    }

    /// The mapping's index, the chunk and the epoch of the least recently
    /// marked chunk.
    fn least_recently_marked(&self) -> Option<(usize, usize, u64)> {
        self.mappings
            .iter()
            .enumerate()
            .flat_map(|(index, mapping)| {
                mapping
                    .marks
                    .iter()
                    .enumerate()
                    .map(move |(chunk, mark)| (index, chunk, mark.load(Relaxed)))
            })
            .filter(|&(_, _, epoch)| epoch != 0)
            .min_by_key(|&(_, _, epoch)| epoch)
    }
}

fn lock() -> MutexGuard<'static, Registry> {
    REGISTRY.lock().unwrap_or_else(PoisonError::into_inner)
}

/// Drops the resident pages of `len` bytes from `ptr`, inside a registered
/// mapping. The next read faults them back in from the file.
fn release((ptr, len): (*mut u8, usize)) {
    // SAFETY: the range is part of a registered mapping, which stays mapped
    // while the caller holds the registry's lock. Its pages are clean pages of
    // a private read-only file mapping, so dropping them changes no byte that
    // a read sees.
    if unsafe { libc::madvise(ptr.cast(), len, libc::MADV_DONTNEED) } != 0 {
        FAILED_RELEASES.fetch_add(1, Relaxed);
    }
}

/// Adds the mapping of `len` bytes at `ptr` of `file`, in chunks of
/// `chunk_size`, to the registry.
pub(crate) fn register(
    ptr: NonNull<u8>,
    len: usize,
    chunk_size: usize,
    file: (u64, u64),
) -> Arc<Mapping> {
    let mapping = Arc::new(Mapping {
        ptr,
        len,
        chunk_size,
        file,
        marks: (0..len.div_ceil(chunk_size))
            .map(|_| AtomicU64::new(0))
            .collect(),
    });
    lock().mappings.push(Arc::clone(&mapping));
    mapping
}

/// Removes `mapping` from the registry, under the lock that every release
/// takes, so no release reaches it afterwards.
pub(crate) fn unregister(mapping: &Arc<Mapping>) {
    let mut registry = lock();
    if let Some(index) = registry
        .mappings
        .iter()
        .position(|registered| Arc::ptr_eq(registered, mapping))
    {
        registry.mappings.swap_remove(index);
    }
    for (chunk, mark) in mapping.marks.iter().enumerate() {
        if mark.swap(0, Relaxed) != 0 {
            registry.marked_bytes -= mapping.chunk_len(chunk);
        }
    }
}

/// Fails when `JJ_PACK_MAPPED_LIMIT` holds a value that is not a budget.
pub(crate) fn check_limit() -> io::Result<()> {
    match &*LIMIT_FROM_ENV {
        Ok(_) => Ok(()),
        Err(message) => Err(io::Error::new(io::ErrorKind::InvalidInput, message.clone())),
    }
}

/// The budget of marked bytes.
pub(crate) fn limit() -> usize {
    match LIMIT_FOR_TESTS.load(Relaxed) {
        usize::MAX => *LIMIT_FROM_ENV.as_ref().unwrap_or(&DEFAULT_LIMIT),
        limit => limit,
    }
}

pub(crate) fn set_limit_for_tests(bytes: usize) {
    LIMIT_FOR_TESTS.store(bytes, Relaxed);
}

/// Reads the budget from the value of `JJ_PACK_MAPPED_LIMIT`, if it is set: a
/// number of bytes, with an optional suffix `m` (MiB) or `g` (GiB).
pub(crate) fn parse_limit(value: Option<&OsStr>) -> Result<usize, String> {
    let Some(value) = value else {
        return Ok(DEFAULT_LIMIT);
    };
    let refuse = || {
        format!(
            "{LIMIT_VARIABLE}={value:?} is not a number of bytes, with an optional suffix m or g"
        )
    };
    let value_str = value.to_str().ok_or_else(refuse)?;
    let (digits, unit) = if let Some(digits) = value_str.strip_suffix('m') {
        (digits, 1 << 20)
    } else if let Some(digits) = value_str.strip_suffix('g') {
        (digits, 1 << 30)
    } else {
        (value_str, 1)
    };
    if digits.is_empty() || !digits.bytes().all(|byte| byte.is_ascii_digit()) {
        return Err(refuse());
    }
    digits
        .parse::<usize>()
        .ok()
        .and_then(|number| number.checked_mul(unit))
        .ok_or_else(refuse)
}

pub(crate) fn marked_bytes() -> usize {
    lock().marked_bytes
}

pub(crate) fn release_now() {
    lock().release();
}

/// Forgets every mark without releasing a chunk, as if no read had marked
/// one.
pub(crate) fn clear_marks() {
    let mut registry = lock();
    for mapping in &registry.mappings {
        for mark in &mapping.marks {
            mark.store(0, Relaxed);
        }
    }
    registry.marked_bytes = 0;
}

/// The marked chunks of the mappings of `file`, by index, in order.
pub(crate) fn marked_chunks(file: (u64, u64)) -> Vec<usize> {
    let registry = lock();
    let mut chunks: Vec<usize> = registry
        .mappings
        .iter()
        .filter(|mapping| mapping.file == file)
        .flat_map(|mapping| mapping.marked_chunks())
        .collect();
    chunks.sort_unstable();
    chunks.dedup();
    chunks
}

impl Mapping {
    /// The marked chunks of this mapping, by index, in order.
    pub(crate) fn marked_chunks(&self) -> Vec<usize> {
        self.marks
            .iter()
            .enumerate()
            .filter(|(_, mark)| mark.load(Relaxed) != 0)
            .map(|(chunk, _)| chunk)
            .collect()
    }
}

pub(crate) fn failed_releases() -> u64 {
    FAILED_RELEASES.load(Relaxed)
}
