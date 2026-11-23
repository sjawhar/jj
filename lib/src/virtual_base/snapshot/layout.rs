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

//! VB laid out over the hunks of the merge of V's terms, and the changes to
//! VB made to those terms (see the documentation of [`super`]).

use std::collections::HashSet;
use std::iter::zip;
use std::ops::Range;

use bstr::BString;
use bstr::ByteSlice as _;
use itertools::Itertools as _;

use super::super::Block;
use super::edits::Edit;
use super::edits::Text;
use super::edits::held_blocks;
use super::edits::is_marker_form;
use crate::files;
use crate::files::RangedMergeHunk;
use crate::merge::Merge;
use crate::tree_merge::MergeOptions;

/// VB laid out over the hunks of the merge of V's terms: each hunk takes a
/// range of VB, its merged text where it resolves and a block where it
/// conflicts.
pub(super) struct Layout<'a> {
    vb: &'a [u8],
    blocks: &'a [Block<'a>],
    /// The texts of V's terms.
    contents: &'a Merge<BString>,
    /// The hunks of the merge of V's terms, before the ones that resolve are
    /// joined.
    hunks: Vec<RangedMergeHunk>,
    /// The range of VB that each hunk takes, and the index of its block where
    /// it conflicts.
    segments: Vec<Segment>,
    /// The index of the hunk of each block.
    block_hunks: Vec<usize>,
    /// The marker lines of VB's blocks.
    marker_lines: HashSet<&'a [u8]>,
}

/// The range of VB that a hunk takes, and the index of its block where it
/// conflicts.
struct Segment {
    range: Range<usize>,
    block: Option<usize>,
}

/// How a change that is made where it falls is made.
enum InPlace {
    /// To every term: it falls in text that every term holds, or between
    /// two hunks.
    EveryTerm,
    /// To the term whose side of the block of the hunk `hunk` it falls in.
    Side { hunk: usize, side: usize },
}

/// A change to the text of one term of V: the range that it replaces, the
/// text that takes its place, and the range of VB that the change came from,
/// which orders the changes at one point of the term.
struct TermEdit {
    range: Range<usize>,
    text: Vec<u8>,
    from: Range<usize>,
}

impl<'a> Layout<'a> {
    /// `vb`, whose blocks are `blocks`, laid out over the hunks of the merge
    /// of V's terms, whose texts are `contents`, merged with `options`.
    /// `None` where `vb` does not hold each hunk's merged text, or each
    /// conflicting hunk's sides in a block, in order: `vb` is then not the VB
    /// of `contents` at those options.
    pub(super) fn new(
        vb: &'a [u8],
        blocks: &'a [Block<'a>],
        contents: &'a Merge<BString>,
        options: &MergeOptions,
    ) -> Option<Self> {
        let hunks = files::split_ranged_hunks(contents, options);
        let mut segments = Vec::with_capacity(hunks.len());
        let mut block_hunks = Vec::with_capacity(blocks.len());
        let mut cursor = 0;
        for (index, hunk) in hunks.iter().enumerate() {
            let (block, end) = match &hunk.resolved {
                None => {
                    let block = blocks.get(block_hunks.len())?;
                    let adds = zip(contents.adds(), hunk.ranges.adds());
                    let sides_hold_adds = block.start == cursor
                        && block.sides.len() == contents.adds().len()
                        && zip(&block.sides, adds).all(|(side, (add, range))| {
                            side_holds(side, &add.as_slice()[range.clone()])
                        });
                    if !sides_hold_adds {
                        return None;
                    }
                    block_hunks.push(index);
                    (Some(block_hunks.len() - 1), block.end)
                }
                Some(text) => {
                    if vb.get(cursor..cursor + text.len())? != text.as_slice() {
                        return None;
                    }
                    (None, cursor + text.len())
                }
            };
            segments.push(Segment {
                range: cursor..end,
                block,
            });
            cursor = end;
        }
        if cursor != vb.len() || block_hunks.len() != blocks.len() {
            return None;
        }
        let marker_lines = blocks
            .iter()
            .flat_map(|block| {
                let mut at = block.start;
                let sides = block.side_ranges.iter();
                let mut lines = sides
                    .map(|side| {
                        let line = &vb[at..side.start];
                        at = side.end;
                        line
                    })
                    .collect_vec();
                lines.push(&vb[at..block.end]);
                lines
            })
            .collect();
        Some(Self {
            vb,
            blocks,
            contents,
            hunks,
            segments,
            block_hunks,
            marker_lines,
        })
    }

    /// The texts of V's terms with `edits`, changes to VB in order, none
    /// overlapping another, made as the documentation of [`super`] says.
    pub(super) fn edited(&self, edits: &[Edit]) -> Merge<Vec<u8>> {
        // The changes to each term, in the order of the merge's terms.
        let mut term_edits: Vec<Vec<TermEdit>> = self.contents.iter().map(|_| vec![]).collect_vec();
        // The changes that are not made where they fall: each grows to the
        // whole hunks that it touches, with every change that reaches into
        // them, and the text that is left for those hunks goes into every
        // term.
        let mut rewritten = vec![false; edits.len()];
        for index in 0..edits.len() {
            if rewritten[index] || self.in_place(&edits[index].range).is_some() {
                continue;
            }
            let (mut first, mut last) = (index, index);
            let mut span = self.whole_hunks(&edits[index].range);
            loop {
                let before = first.checked_sub(1).filter(|&before| {
                    !rewritten[before] && reaches_into(&edits[before].range, &span)
                });
                let after = Some(last + 1)
                    .filter(|&after| after < edits.len())
                    .filter(|&after| reaches_into(&edits[after].range, &span));
                let Some(next) = before.or(after) else {
                    break;
                };
                (first, last) = (first.min(next), last.max(next));
                let grown = self.whole_hunks(&edits[next].range);
                span = span.start.min(grown.start)..span.end.max(grown.end);
            }
            rewritten[first..=last].fill(true);
            // What is left of the hunks: VB's own lines around the changes,
            // without its marker lines, and the changes' text.
            let mut text = Text::default();
            let mut at = span.start;
            for edit in &edits[first..=last] {
                text.push(&self.unmarked(at..edit.range.start), false);
                text.append(&edit.text);
                at = edit.range.end;
            }
            text.push(&self.unmarked(at..span.end), false);
            self.write_to_every_term(&span, &text, &mut term_edits);
        }
        for (edit, _) in zip(edits, &rewritten).filter(|(_, rewritten)| !**rewritten) {
            match self.in_place(&edit.range) {
                Some(InPlace::Side { hunk, side }) => {
                    term_edits[2 * side].push(self.side_edit(hunk, side, edit));
                }
                _ => self.write_to_every_term(&edit.range, &edit.text, &mut term_edits),
            }
        }
        let new_contents = zip(self.contents.iter(), term_edits)
            .map(|(content, mut edits)| {
                edits.sort_by_key(|edit| (edit.range.start, edit.from.start, edit.from.end));
                let mut new_content = vec![];
                let mut at = 0;
                for edit in edits {
                    if edit.range.start < at {
                        // Two changes never take the same text of a term.
                        debug_assert!(false, "overlapping changes to a term");
                        continue;
                    }
                    new_content.extend_from_slice(&content[at..edit.range.start]);
                    new_content.extend_from_slice(&edit.text);
                    at = edit.range.end;
                }
                new_content.extend_from_slice(&content[at..]);
                new_content
            })
            .collect_vec();
        Merge::from_vec(new_contents)
    }

    /// The hunks that a change to `range` of VB touches where V's terms do
    /// not all hold the same text: the ones that hold a part of a change that
    /// replaces text, or, for an insertion, the one that it is inside, not
    /// at the edge of.
    fn touched(&self, range: &Range<usize>) -> impl Iterator<Item = usize> {
        // A hunk holds a part of the change where it ends after the change's
        // start and starts before its end, which, for an insertion, is a
        // hunk that the insertion is inside.
        let (start, end) = (range.start, range.end);
        let first = self
            .segments
            .partition_point(|segment| segment.range.end <= start);
        zip(first.., &self.segments[first..])
            .take_while(move |(_, segment)| segment.range.start < end)
            .map(|(index, _)| index)
            .filter(|&index| !self.hunks[index].matching)
    }

    /// How a change to `range` of VB is made where it falls, if it is:
    /// in text that every term holds or between two hunks, to every term;
    /// inside one side of a block, to that side's term.
    fn in_place(&self, range: &Range<usize>) -> Option<InPlace> {
        let mut touched = self.touched(range);
        let Some(hunk) = touched.next() else {
            return Some(InPlace::EveryTerm);
        };
        if touched.next().is_some() {
            return None;
        }
        let block = &self.blocks[self.segments[hunk].block?];
        let side = block
            .side_ranges
            .iter()
            .position(|side| side.start <= range.start && range.end <= side.end)?;
        Some(InPlace::Side { hunk, side })
    }

    /// `range` of VB, grown to the whole of each hunk that a change to it
    /// touches where V's terms do not all hold the same text.
    fn whole_hunks(&self, range: &Range<usize>) -> Range<usize> {
        self.touched(range).fold(range.clone(), |span, hunk| {
            let hunk = &self.segments[hunk].range;
            span.start.min(hunk.start)..span.end.max(hunk.end)
        })
    }

    /// The range of the term at `term` that `range` of VB stands for, where
    /// each end of `range` is between two hunks, or inside a hunk whose terms
    /// all hold the same text. The range starts in the hunk that starts at or
    /// before its start, and ends in the hunk that ends at or after its end,
    /// so that it takes none of a term's text in a hunk that takes none of
    /// VB. An insertion is at the start of the hunk that it is before.
    fn term_range(&self, range: &Range<usize>, term: usize) -> Range<usize> {
        let offset = |hunk: usize, point: usize| match self.segments.get(hunk) {
            Some(segment) => {
                let range = &self.hunks[hunk].ranges.as_slice()[term];
                (range.start + point - segment.range.start).min(range.end)
            }
            None => self.contents.as_slice()[term].len(),
        };
        let first = self
            .segments
            .partition_point(|segment| segment.range.end <= range.start);
        let start = offset(first, range.start);
        if range.is_empty() {
            return start..start;
        }
        let last = self
            .segments
            .partition_point(|segment| segment.range.start < range.end);
        start..offset(last - 1, range.end)
    }

    /// Makes the change of `range` of VB to `text` to every term: each term's
    /// own text there gives way to `text` as [`Self::term_text`] gives it for
    /// the term. Text put at the end of a term that ends without a newline
    /// comes after one, as jj's snapshot reads text that the person put after
    /// such a term in a conflict.
    fn write_to_every_term(
        &self,
        range: &Range<usize>,
        text: &Text,
        term_edits: &mut [Vec<TermEdit>],
    ) {
        for (term, edits) in term_edits.iter_mut().enumerate() {
            let term_range = self.term_range(range, term);
            let mut text = self.term_text(text, term);
            if term_range.is_empty() && text.is_empty() {
                continue;
            }
            let content = &self.contents.as_slice()[term];
            if term_range.is_empty()
                && term_range.start == content.len()
                && !text.is_empty()
                && content.last().is_some_and(|&byte| byte != b'\n')
            {
                text.insert(0, b'\n');
            }
            edits.push(TermEdit {
                range: term_range,
                text,
                from: range.clone(),
            });
        }
    }

    /// `range` of VB without the marker lines of VB's blocks.
    fn unmarked(&self, range: Range<usize>) -> Vec<u8> {
        self.vb[range]
            .lines_with_terminator()
            .filter(|line| !self.marker_lines.contains(line))
            .collect_vec()
            .concat()
    }

    /// `text` as it goes into the term at `term`. In the ranges of it that
    /// came from the base of a conflict that the person kept, a whole block
    /// of VB (see [`held_blocks`]) stands for the term's own text at the
    /// block's hunk, and any other line in the form of a marker line is left
    /// out. The rest is the person's text, whatever its lines look like.
    fn term_text(&self, text: &Text, term: usize) -> Vec<u8> {
        let unmarked = |bytes: &[u8], result: &mut Vec<u8>| {
            for line in bytes.lines_with_terminator() {
                if !is_marker_form(line) {
                    result.extend_from_slice(line);
                }
            }
        };
        let mut result = vec![];
        let mut at = 0;
        for base in &text.of_base {
            result.extend_from_slice(&text.bytes[at..base.start]);
            at = base.start;
            for (copy, block) in held_blocks(self.vb, self.blocks, &text.bytes, base.clone()) {
                unmarked(&text.bytes[at..copy.start], &mut result);
                let ranges = &self.hunks[self.block_hunks[block]].ranges;
                let own = &self.contents.as_slice()[term][ranges.as_slice()[term].clone()];
                result.extend_from_slice(own);
                if own.last().is_some_and(|&byte| byte != b'\n') {
                    result.push(b'\n');
                }
                at = copy.end;
            }
            unmarked(&text.bytes[at..base.end], &mut result);
            at = base.end;
        }
        result.extend_from_slice(&text.bytes[at..]);
        result
    }

    /// `edit`, a change to VB inside the side at `side` of the block of the
    /// hunk at `hunk`, as a change to the term of V whose side it is.
    fn side_edit(&self, hunk: usize, side: usize, edit: &Edit) -> TermEdit {
        let term = 2 * side;
        let term_range = &self.hunks[hunk].ranges.as_slice()[term];
        let block = self.segments[hunk].block.map(|block| &self.blocks[block]);
        let side_range = block.map_or(edit.range.clone(), |block| block.side_ranges[side].clone());
        let (start, end) = (edit.range.start, edit.range.end);
        let (from, to) = (start - side_range.start, end - side_range.start);
        let mut text = self.term_text(&edit.text, term);
        let range = if to <= term_range.len() {
            from..to
        } else {
            // The term's last line has no final newline, which VB added: the
            // changed lines keep that.
            if text.last() == Some(&b'\n') {
                text.pop();
            }
            if from == side_range.len() {
                text.insert(0, b'\n');
                term_range.len()..term_range.len()
            } else {
                from..term_range.len()
            }
        };
        TermEdit {
            range: term_range.start + range.start..term_range.start + range.end,
            text,
            from: edit.range.clone(),
        }
    }
}

/// Whether a change to `range` of VB reaches into `span`, a range of whole
/// hunks: it replaces text of it, or inserts inside it, not at its edge.
fn reaches_into(range: &Range<usize>, span: &Range<usize>) -> bool {
    if range.is_empty() {
        span.start < range.start && range.start < span.end
    } else {
        range.start < span.end && span.start < range.end
    }
}

/// Whether `side`, a side of a block of VB, holds `text`, a term's text there:
/// the same text, or, where the text's last line has no final newline, the
/// text with the newline that VB ends each side with.
fn side_holds(side: &[u8], text: &[u8]) -> bool {
    side == text
        || (!text.is_empty() && !text.ends_with(b"\n") && side.strip_suffix(b"\n") == Some(text))
}
