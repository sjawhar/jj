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

//! The blocks of VB and M: the reader, which finds the blocks that the writer
//! wrote in a file (see the documentation of [`super`]), and the finder,
//! which finds the blocks of VB that text from the base of a conflict holds
//! whole. Both read lines in the form of marker lines with the same scans.

use std::cmp::Reverse;
use std::collections::HashMap;
use std::ops::Range;

use bstr::ByteSlice as _;
use indexmap::IndexMap;
use itertools::Itertools as _;

use super::CLOSE;
use super::OPEN;
use super::SEPARATOR;
use super::block_hasher;
use super::hasher_with_after;
use super::parse_names;
use super::salted_hex;
use super::split_hunk_level;
use crate::files::FileMergeHunkLevel;

/// A block of VB or M: the offsets of its opening line and of the end of its
/// closing line, the hunk level that its opening line records and the names
/// of the terms that it carries, its sides, each with its newlines, and
/// their offsets, and its marker lines.
pub(super) struct Block<'a> {
    pub(super) start: usize,
    pub(super) end: usize,
    pub(super) hunk_level: FileMergeHunkLevel,
    pub(super) names: &'a [u8],
    pub(super) sides: Vec<&'a [u8]>,
    pub(super) side_ranges: Vec<Range<usize>>,
    /// Each marker line, with its newline, in order: its range, and its kind.
    pub(super) markers: Vec<(Range<usize>, Marker)>,
}

/// The kind of a marker line of a block.
#[derive(Clone, Copy, Debug)]
pub(super) enum Marker {
    Open,
    Separator,
    Close,
}

impl Marker {
    /// The kind of marker line that `line` is in the form of, if any.
    pub(super) fn of_form(line: &[u8]) -> Option<Self> {
        if parse_opening_line(line).is_some() {
            Some(Self::Open)
        } else if marker_hex(line, SEPARATOR).is_some() {
            Some(Self::Separator)
        } else if marker_hex(line, CLOSE).is_some() {
            Some(Self::Close)
        } else {
            None
        }
    }
}

/// Whether `line`, a line of text that came from the base of a conflict that
/// the person kept, is in the form of a marker line of a virtual merge base.
pub(super) fn is_marker_form(line: &[u8]) -> bool {
    Marker::of_form(line).is_some()
}

/// The most bytes that [`blocks`] hashes for opening marker lines whose blocks
/// do not verify, as a multiple of the length of the content.
const MAX_UNVERIFIED_HASHING: usize = 4;

/// The blocks of `content`, in order, as the writer wrote them. A line in the
/// form of an opening marker line can open a block where the next closing
/// marker line of its hex follows, with no other opening marker line of that
/// hex before it, and where it names terms. The writer's blocks all carry
/// one salt, above the salt of every such line of V's text (2^64-1 where one
/// has that salt), and the same level and names, so those lines are tried in
/// groups of one salt, level and names, from the highest salt down, and at
/// one salt in the order of each group's first line. The blocks are those
/// of the first group that verifies: each line of it opens a block whose hex
/// is the hash of the block with the group's salt, where the block's ordinal
/// counts the blocks before it, the text before it starts where the previous
/// block ends (see [`verified_blocks`]). Every other line is text, whatever
/// its form.
///
/// `None` where no group verifies, as for a file that is neither VB nor M;
/// and where the groups that do not verify would take more than
/// [`MAX_UNVERIFIED_HASHING`] times the content's length to hash, so that
/// reading any content takes time in proportion to its length. A file that
/// the writer wrote takes none of that, as its own group is tried first.
pub(super) fn blocks(content: &[u8]) -> Option<Vec<Block<'_>>> {
    let lines = lines_at(content, 0);
    let mut groups: IndexMap<(u64, &[u8]), Vec<Candidate<'_>>> = IndexMap::new();
    for candidate in candidates(&lines) {
        let key = (candidate.opening.salt, candidate.opening.tail);
        groups.entry(key).or_default().push(candidate);
    }
    let mut groups = groups
        .into_values()
        .filter(|group| parse_names(group[0].opening.names).is_some())
        .collect_vec();
    groups.sort_by_key(|group| Reverse(group[0].opening.salt));
    let mut budget = content.len().saturating_mul(MAX_UNVERIFIED_HASHING);
    for group in &groups {
        match verified_blocks(content, &lines, group) {
            Ok(blocks) => return Some(blocks),
            Err(hashed) => budget = budget.checked_sub(hashed)?,
        }
    }
    None
}

/// The lines of `bytes`, each with its newline and with its offset, where
/// `bytes` starts at the offset `start`.
fn lines_at(bytes: &[u8], start: usize) -> Vec<(usize, &[u8])> {
    bytes
        .lines_with_terminator()
        .scan(start, |offset, line| {
            let line_start = *offset;
            *offset += line.len();
            Some((line_start, line))
        })
        .collect()
}

/// A line in the form of an opening marker line whose block closes: what the
/// line holds, and the lines of the block, from it to its closing line.
struct Candidate<'a> {
    opening: OpeningLine<'a>,
    lines: Range<usize>,
}

/// Each of `lines` in the form of an opening marker line whose block closes,
/// in order, with its closing line: the next closing marker line of its hex,
/// unless another opening marker line of that hex comes first.
fn candidates<'a>(lines: &[(usize, &'a [u8])]) -> Vec<Candidate<'a>> {
    let mut candidates = vec![];
    let mut next_close: HashMap<&[u8], usize> = HashMap::new();
    for (index, &(_, line)) in lines.iter().enumerate().rev() {
        if let Some(hex) = marker_hex(line, CLOSE) {
            next_close.insert(hex, index);
        } else if let Some(opening) = parse_opening_line(line)
            && let Some(close) = next_close.remove(opening.hex)
        {
            candidates.push(Candidate {
                opening,
                lines: index..close + 1,
            });
        }
    }
    candidates.reverse();
    candidates
}

/// The lines among `lines`, the lines of a block after its opening line up to
/// its closing line, each with its offset, that end a side of the block, whose
/// hex is `hex`: each line that is [`SEPARATOR`] and `hex`, and the closing
/// line, each with its kind.
fn side_ends<'a>(
    lines: &[(usize, &'a [u8])],
    hex: &[u8],
) -> impl Iterator<Item = (usize, &'a [u8], Marker)> {
    lines.iter().filter_map(move |&(start, line)| {
        let marker = if marker_hex(line, SEPARATOR) == Some(hex) {
            Marker::Separator
        } else if marker_hex(line, CLOSE) == Some(hex) {
            Marker::Close
        } else {
            return None;
        };
        Some((start, line, marker))
    })
}

/// The blocks that `group`, lines of `content` in the form of an opening
/// marker line with one salt, level and names, in order, open, where each
/// block verifies: it starts after the block before it, and its hex is the
/// hash of the block, where the text before it starts where that block ends
/// (at the start of the content for the first), and the text after it is the
/// rest of the content for the last block and nothing for another. Each side
/// ends at a line that is [`SEPARATOR`] and the block's hex, or at the
/// closing line. `lines` are the lines of `content`, each with its offset.
/// Where a block does not verify, the error is how many bytes the check
/// read.
fn verified_blocks<'a>(
    content: &'a [u8],
    lines: &[(usize, &'a [u8])],
    group: &[Candidate<'a>],
) -> Result<Vec<Block<'a>>, usize> {
    let mut blocks = vec![];
    let mut text_start = 0;
    let mut read = 0;
    for (position, candidate) in group.iter().enumerate() {
        let OpeningLine {
            hex,
            salt,
            tail,
            hunk_level,
            names,
        } = candidate.opening;
        let (start, line) = lines[candidate.lines.start];
        if start < text_start {
            return Err(read);
        }
        let mut sides = vec![];
        let mut side_ranges = vec![];
        let mut side_start = start + line.len();
        let mut markers = vec![(start..side_start, Marker::Open)];
        for (line_start, line, marker) in side_ends(&lines[candidate.lines.clone()][1..], hex) {
            sides.push(&content[side_start..line_start]);
            side_ranges.push(side_start..line_start);
            side_start = line_start + line.len();
            markers.push((line_start..side_start, marker));
        }
        let end = side_start;
        let after: &[u8] = if position + 1 == group.len() {
            &content[end..]
        } else {
            &[]
        };
        read += end - text_start + after.len();
        let hasher = block_hasher(blocks.len(), tail, &content[text_start..start], &sides);
        if salted_hex(&hasher_with_after(hasher, after), salt).as_bytes() != hex {
            return Err(read);
        }
        blocks.push(Block {
            start,
            end,
            hunk_level,
            names,
            sides,
            side_ranges,
            markers,
        });
        text_start = end;
    }
    Ok(blocks)
}

/// What a line in the form of an opening marker line holds: its hex, its
/// salt, and what follows the salt, which is the hunk level that it records
/// and the names of the terms (see [`split_hunk_level`]).
#[derive(Clone, Copy)]
pub(super) struct OpeningLine<'a> {
    hex: &'a [u8],
    pub(super) salt: u64,
    tail: &'a [u8],
    hunk_level: FileMergeHunkLevel,
    names: &'a [u8],
}

/// What `line` holds if it is in the form of an opening marker line:
/// [`OPEN`], 32 lowercase hex digits, a salt in decimal, the hunk level and
/// the names as [`super::opening_tail`] writes them, and a newline. The level
/// is read here, once, so that no reader takes it for a name.
pub(super) fn parse_opening_line(line: &[u8]) -> Option<OpeningLine<'_>> {
    let rest = line.strip_prefix(OPEN.as_bytes())?.strip_suffix(b"\n")?;
    let (hex, rest) = rest.split_once_str(" ")?;
    let (salt, tail) = rest.split_once_str(" ")?;
    let salt_text = salt.to_str().ok()?;
    let salt: u64 = salt_text.parse().ok()?;
    let canonical = salt.to_string() == salt_text;
    let (hunk_level, names) = split_hunk_level(tail);
    (is_hex(hex) && canonical && !tail.is_empty()).then_some(OpeningLine {
        hex,
        salt,
        tail,
        hunk_level,
        names,
    })
}

/// The hex of `line` if `line` is `prefix`, 32 lowercase hex digits and a
/// newline.
fn marker_hex<'a>(line: &'a [u8], prefix: &str) -> Option<&'a [u8]> {
    let hex = line.strip_prefix(prefix.as_bytes())?.strip_suffix(b"\n")?;
    is_hex(hex).then_some(hex)
}

/// Whether `hex` is 32 lowercase hex digits.
fn is_hex(hex: &[u8]) -> bool {
    hex.len() == 32
        && hex
            .iter()
            .all(|&byte| matches!(byte, b'0'..=b'9' | b'a'..=b'f'))
}

/// The blocks of VB, looked up for a snapshot, which reads them many times:
/// by their marker lines, by the offsets of those, and by their sides. A
/// snapshot builds it once for each VB that it reads, which reads each
/// block's sides once (see [`Self::held`]).
pub(super) struct BlockFinder<'a> {
    /// VB.
    pub(super) vb: &'a [u8],
    /// VB's blocks.
    pub(super) blocks: &'a [Block<'a>],
    /// The length of the sides of the block whose sides are the longest, all
    /// of them together.
    longest: usize,
    /// The block of each marker line of VB's blocks, by the line, with its
    /// newline.
    by_line: HashMap<&'a [u8], usize>,
    /// The kind of each marker line of VB's blocks, by its offset.
    by_offset: HashMap<usize, Marker>,
    /// The blocks that hold each list of sides, in order.
    by_sides: HashMap<&'a [&'a [u8]], Vec<usize>>,
}

/// The most bytes that [`BlockFinder::held`] reads as sides for the lines of
/// a base in the form of an opening marker line, as a multiple of the length
/// of that base.
const MAX_COMPARED: usize = 4;

impl<'a> BlockFinder<'a> {
    /// The finder of `blocks`, the blocks of `vb`.
    pub(super) fn new(vb: &'a [u8], blocks: &'a [Block<'a>]) -> Self {
        let mut by_line = HashMap::new();
        let mut by_offset = HashMap::new();
        let mut by_sides: HashMap<&'a [&'a [u8]], Vec<usize>> = HashMap::new();
        for (index, block) in blocks.iter().enumerate() {
            for (range, marker) in &block.markers {
                by_line.entry(&vb[range.clone()]).or_insert(index);
                by_offset.insert(range.start, *marker);
            }
            by_sides
                .entry(block.sides.as_slice())
                .or_default()
                .push(index);
        }
        let longest = blocks
            .iter()
            .map(|block| block.sides.iter().map(|side| side.len()).sum::<usize>())
            .max()
            .unwrap_or(0);
        count_read(blocks.len());
        Self {
            vb,
            blocks,
            longest,
            by_line,
            by_offset,
            by_sides,
        }
    }

    /// The block whose marker line `line`, with its newline, is, if any.
    pub(super) fn block_of(&self, line: &[u8]) -> Option<usize> {
        self.by_line.get(line).copied()
    }

    /// Whether `line`, with its newline, is a marker line of VB's blocks.
    pub(super) fn is_marker_line(&self, line: &[u8]) -> bool {
        self.by_line.contains_key(line)
    }

    /// The kind of the marker line of VB's blocks that starts at `offset` of
    /// VB, if one does.
    pub(super) fn marker_at(&self, offset: usize) -> Option<Marker> {
        self.by_offset.get(&offset).copied()
    }

    /// The blocks of VB that `range` of `bytes`, text that came from a base,
    /// holds whole: the range of each, and the block's index. A block is held
    /// where a line in the form of an opening marker line, the lines up to
    /// the next closing marker line of its hex, and the lines between them
    /// that separate sides at that hex, hold the sides of a block of VB: the
    /// block with that opening line, else the next one with those sides after
    /// the block held before, else the first one.
    ///
    /// A line in the form of an opening marker line opens no block where the
    /// lines up to its closing line are longer than the sides of every block
    /// of VB with their marker lines, or where they would take the bytes read
    /// as sides for `range` past [`MAX_COMPARED`] times its length. So the
    /// bytes read and looked up as sides are bounded by a multiple of the
    /// length of `range`, and each lookup looks the sides up once in the map
    /// of sides and compares them with those of at most one block (see
    /// [`Self::block_holding`]), whatever the number of VB's blocks.
    pub(super) fn held(&self, bytes: &[u8], range: Range<usize>) -> Vec<(Range<usize>, usize)> {
        let lines = lines_at(&bytes[range.clone()], range.start);
        let mut budget = range.len().saturating_mul(MAX_COMPARED);
        let mut held = vec![];
        let mut next_block = 0;
        let mut next_line = 0;
        for candidate in candidates(&lines) {
            let Range { start: open, end } = candidate.lines;
            if open < next_line {
                continue;
            }
            let hex = candidate.opening.hex;
            let (start, line) = lines[open];
            let (close_start, close_line) = lines[end - 1];
            let first_side = start + line.len();
            // The sides are no longer than the block's lines between its
            // opening line and its closing line.
            let length = close_start - first_side;
            if length > self.longest + (end - 1 - open) * (SEPARATOR.len() + hex.len() + 1) {
                continue;
            }
            let Some(left) = budget.checked_sub(length) else {
                continue;
            };
            budget = left;
            let mut sides = vec![];
            let mut side_start = first_side;
            for (line_start, end_line, _) in side_ends(&lines[open + 1..end], hex) {
                sides.push(&bytes[side_start..line_start]);
                side_start = line_start + end_line.len();
            }
            if let Some(block) = self.block_holding(line, &sides, next_block) {
                held.push((start..close_start + close_line.len(), block));
                next_block = block + 1;
                next_line = end;
            }
        }
        held
    }

    /// The block of VB that holds `sides`, the sides that a text holds after
    /// `line`, a line in the form of an opening marker line: the block whose
    /// opening line `line` is, if it holds them, else the first block from
    /// `next_block` on that holds them, else the first one that does. A
    /// lookup compares `sides` with those of at most one block, the block of
    /// `line`, by [`Self::holds`]; the map of sides gives only blocks that
    /// have them.
    fn block_holding(&self, line: &[u8], sides: &[&[u8]], next_block: usize) -> Option<usize> {
        let own = self
            .block_of(line)
            .filter(|&block| self.holds(block, sides));
        own.or_else(|| {
            let holding = self.blocks_with(sides);
            let next = holding.partition_point(|&block| block < next_block);
            holding.get(next).or(holding.first()).copied()
        })
    }

    /// Whether the block at `block` of VB has the sides `sides`: the one
    /// comparison of a block's sides with a text's that a lookup makes,
    /// which the tests that bound a finder's reads count.
    fn holds(&self, block: usize, sides: &[&[u8]]) -> bool {
        count_read(1);
        self.blocks[block].sides == sides
    }

    /// The blocks of VB whose sides are `sides`, in order.
    fn blocks_with<'s>(&'s self, sides: &[&'s [u8]]) -> &'s [usize] {
        let by_sides: &'s HashMap<&'s [&'s [u8]], Vec<usize>> = &self.by_sides;
        by_sides.get(sides).map(Vec::as_slice).unwrap_or_default()
    }
}

#[cfg(test)]
thread_local! {
    /// The blocks whose sides the finders on this thread read.
    static READ: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

/// Counts `count` more blocks whose sides a finder reads, for the tests that
/// bound that work.
#[cfg(test)]
fn count_read(count: usize) {
    READ.set(READ.get() + count);
}

#[cfg(not(test))]
fn count_read(_count: usize) {}

/// The blocks whose sides the finders on this thread read so far.
#[cfg(test)]
pub(super) fn blocks_read() -> usize {
    READ.get()
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use super::super::Draft;
    use super::super::VirtualBaseHunk;
    use super::super::opening_tail;
    use super::super::recorded_hunk_level;
    use super::super::tests::quoted_blocks;
    use super::super::tests::sides;
    use super::super::virtual_base_content;
    use super::*;

    /// A line in the form of an opening marker line, whose block never
    /// closes.
    const OPENING_FORM: &str =
        "<<<<<<< jj virtual merge base 0123456789abcdef0123456789abcdef 0 quoted\n";

    /// The names of three terms, as an opening marker line holds them.
    const NAMES: &str = "file:01 file:02 file:03";

    /// VB that names [`NAMES`], made of `hunks`.
    fn vb(hunks: Vec<VirtualBaseHunk>) -> Vec<u8> {
        virtual_base_content(hunks, NAMES, &HashSet::new()).expect("a salt is free")
    }

    /// A block with the sides `x` and `y`.
    fn block() -> VirtualBaseHunk {
        VirtualBaseHunk::Block(vec![b"x\n".to_vec(), b"y\n".to_vec()])
    }

    #[test]
    fn an_opening_line_whose_block_never_closes_is_text() {
        let text = format!("1\n{OPENING_FORM}3\n");
        let content = vb(vec![
            VirtualBaseHunk::Resolved(text.into_bytes()),
            block(),
            VirtualBaseHunk::Resolved(b"after\n".to_vec()),
        ]);
        let blocks = blocks(&content).expect("VB reads back");
        assert_eq!(sides(&blocks), [[b"x\n" as &[u8], b"y\n"]]);
    }

    #[test]
    fn a_quoted_block_whose_hash_does_not_check_out_is_text() {
        // VB's own text quotes another VB whole, closing line and all, after
        // a line of text before VB's own block, and after it.
        let quoted = vb(vec![block()]);
        let before = [b"1\n".as_slice(), &quoted].concat();
        let content = vb(vec![
            VirtualBaseHunk::Resolved(before),
            block(),
            VirtualBaseHunk::Resolved(quoted),
        ]);
        let blocks = blocks(&content).expect("VB reads back");
        assert_eq!(sides(&blocks), [[b"x\n" as &[u8], b"y\n"]]);
    }

    #[test]
    fn vb_of_text_that_starts_with_a_copy_of_a_vb_reads_back() {
        // As the first text of a file, a byte copy of a VB that ends with its
        // block verifies by itself: the writer's salt is above the copy's, so
        // the reader tries the writer's block first.
        let quoted = vb(vec![VirtualBaseHunk::Block(vec![
            b"p\n".to_vec(),
            b"q\n".to_vec(),
        ])]);
        assert_eq!(
            blocks(&quoted).map(|blocks| sides(&blocks)),
            Some(vec![vec![b"p\n" as &[u8], b"q\n"]])
        );
        let content = vb(vec![VirtualBaseHunk::Resolved(quoted), block()]);
        let blocks = blocks(&content).expect("VB reads back");
        assert_eq!(sides(&blocks), [[b"x\n" as &[u8], b"y\n"]]);
    }

    #[test]
    fn vb_of_text_that_quotes_many_blocks_reads_back() {
        // The reader tries the writer's blocks before the quoted ones, however
        // many there are, so reading them back hashes none of the others.
        let content = vb(vec![
            VirtualBaseHunk::Resolved(quoted_blocks(200, 7)),
            block(),
            VirtualBaseHunk::Resolved(b"after\n".to_vec()),
        ]);
        let blocks = blocks(&content).expect("VB reads back");
        assert_eq!(sides(&blocks), [[b"x\n" as &[u8], b"y\n"]]);
    }

    #[test]
    fn checks_of_blocks_that_do_not_verify_are_bounded() {
        // A file that no merge wrote: its own block is at salt 0, below the
        // salt of the blocks that its text quotes, so the reader tries those
        // first, each a hash of about the whole file. One is within the
        // bound; eight are not, and the file is then not VB, although its
        // block verifies.
        let at_salt_0 = |quoted: usize| {
            let hunks = vec![VirtualBaseHunk::Resolved(quoted_blocks(quoted, 1)), block()];
            Draft::new(hunks)
                .write(NAMES, 0, &HashSet::new())
                .expect("salt 0 is free")
        };
        assert_eq!(
            blocks(&at_salt_0(1)).map(|blocks| sides(&blocks)),
            Some(vec![vec![b"x\n" as &[u8], b"y\n"]])
        );
        assert!(blocks(&at_salt_0(8)).is_none());
    }

    #[test]
    fn vb_with_any_byte_changed_is_not_vb() {
        let content = vb(vec![
            block(),
            VirtualBaseHunk::Resolved(b"between\n".to_vec()),
            block(),
            VirtualBaseHunk::Resolved(b"after\n".to_vec()),
        ]);
        assert_eq!(blocks(&content).map(|blocks| blocks.len()), Some(2));
        let changed = content.replace("after", "AFTER");
        assert!(blocks(&changed).is_none());
        let changed = content.replace("between", "BETWEEN");
        assert!(blocks(&changed).is_none());
    }

    #[test]
    fn the_hunk_level_reads_back() {
        let tail = opening_tail(FileMergeHunkLevel::Word, NAMES);
        assert_eq!(tail, format!("word {NAMES}"));
        let by_word = virtual_base_content(vec![block()], &tail, &HashSet::new()).unwrap();
        let by_line = vb(vec![block()]);
        let levels = [
            (&by_word, FileMergeHunkLevel::Word),
            (&by_line, FileMergeHunkLevel::Line),
        ];
        for (content, hunk_level) in levels {
            let blocks = blocks(content).expect("VB reads back");
            assert_eq!(recorded_hunk_level(&blocks), hunk_level);
            assert_eq!(blocks[0].names, NAMES.as_bytes());
        }
    }

    #[test]
    fn each_block_records_its_marker_lines() {
        let content = vb(vec![
            VirtualBaseHunk::Resolved(b"1\n".to_vec()),
            VirtualBaseHunk::Block(vec![b"x\n".to_vec(), b"o\n".to_vec(), b"y\n".to_vec()]),
        ]);
        let blocks = blocks(&content).expect("VB reads back");
        let [block] = &blocks[..] else {
            panic!("expected one block");
        };
        let markers = block
            .markers
            .iter()
            .map(|(range, marker)| (Marker::of_form(&content[range.clone()]), *marker))
            .collect_vec();
        assert_eq!(
            format!("{markers:?}"),
            "[(Some(Open), Open), (Some(Separator), Separator), (Some(Separator), Separator), \
             (Some(Close), Close)]"
        );
        assert_eq!(block.markers[0].0.start, block.start);
        assert_eq!(block.markers[3].0.end, block.end);
    }
}
