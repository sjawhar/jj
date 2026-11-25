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

//! The changes that a snapshot makes to a virtual merge base: what the person
//! left in place of each range of it that they changed (see the
//! documentation of [`super`]).

use std::iter;
use std::iter::zip;
use std::ops::Range;

use bstr::BString;
use bstr::ByteSlice as _;
use itertools::Itertools as _;

use super::super::blocks::Block;
use super::super::blocks::BlockFinder;
use super::super::blocks::Marker;
use super::super::blocks::parse_opening_line;
use crate::diff::ContentDiff;
use crate::diff::DiffHunkKind;
use crate::files;
use crate::merge::Merge;
use crate::tree_merge::MergeOptions;

/// Text that a snapshot gives a virtual merge base, with the ranges of it
/// that came from the base of a conflict that the person kept, where a line
/// in the form of a marker line is VB's; the rest is text that the person
/// left outside conflicts, which is text whatever its lines look like.
#[derive(Clone, Debug, Default)]
pub(super) struct Text {
    pub(super) bytes: Vec<u8>,
    /// The ranges of `bytes` that came from a base, in order, none next to
    /// another.
    pub(super) of_base: Vec<Range<usize>>,
}

impl Text {
    /// `bytes` as text that the person left outside conflicts.
    pub(super) fn literal(bytes: &[u8]) -> Self {
        Self {
            bytes: bytes.to_vec(),
            of_base: vec![],
        }
    }

    /// The new text of the base at `index` among the removes of the terms
    /// that `hunks` give, as jj's snapshot reads it: the text of each hunk
    /// that resolves, and each conflict's base. `None` where a conflict has
    /// no such base.
    pub(super) fn of_base(hunks: &[Merge<BString>], index: usize) -> Option<Self> {
        let mut text = Self::default();
        for hunk in hunks {
            match hunk.as_resolved() {
                Some(resolved) => text.push(resolved, false),
                None => text.push(hunk.get_remove(index)?, true),
            }
        }
        Some(text)
    }

    /// Appends `bytes`, which came from a base if `of_base` is set.
    pub(super) fn push(&mut self, bytes: &[u8], of_base: bool) {
        let start = self.bytes.len();
        self.bytes.extend_from_slice(bytes);
        if !of_base || bytes.is_empty() {
            return;
        }
        match self.of_base.last_mut() {
            Some(last) if last.end == start => last.end = self.bytes.len(),
            _ => self.of_base.push(start..self.bytes.len()),
        }
    }

    /// Appends `other`.
    pub(super) fn append(&mut self, other: &Self) {
        let mut at = 0;
        for range in &other.of_base {
            self.push(&other.bytes[at..range.start], false);
            self.push(&other.bytes[range.clone()], true);
            at = range.end;
        }
        self.push(&other.bytes[at..], false);
    }

    /// `range` of this text.
    fn slice(&self, range: Range<usize>) -> Self {
        let of_base = self
            .of_base
            .iter()
            .map(|base| base.start.max(range.start)..base.end.min(range.end))
            .filter(|base| base.start < base.end)
            .map(|base| base.start - range.start..base.end - range.start)
            .collect();
        Self {
            bytes: self.bytes[range].to_vec(),
            of_base,
        }
    }

    /// Whether the byte at `offset` came from a base.
    fn is_of_base(&self, offset: usize) -> bool {
        let next = self.of_base.partition_point(|base| base.end <= offset);
        self.of_base
            .get(next)
            .is_some_and(|base| base.start <= offset)
    }

    /// The lines of this text, where each line that came from a base and is
    /// in the form of a marker line is the line that stands for its kind
    /// and, where it is told, for its block of VB, whose blocks `finder`
    /// finds: the block whose marker line that very line is, or the block
    /// that this text holds whole around it (see [`BlockFinder::held`]).
    fn lines(&self, finder: &BlockFinder<'_>) -> Lines {
        let held = self
            .of_base
            .iter()
            .flat_map(|base| finder.held(&self.bytes, base.clone()))
            .collect_vec();
        let mut next = 0;
        let mut lines = Lines::default();
        let mut at = 0;
        for line in self.bytes.lines_with_terminator() {
            while held.get(next).is_some_and(|(range, _)| range.end <= at) {
                next += 1;
            }
            match Marker::of_form(line).filter(|_| self.is_of_base(at)) {
                Some(marker) => {
                    let around = held
                        .get(next)
                        .filter(|(range, _)| range.start <= at)
                        .map(|(_, block)| *block);
                    let block = finder.block_of(line).or(around);
                    lines.push(at, &marker.token_of(block));
                }
                None => lines.push(at, line),
            }
            at += line.len();
        }
        lines
    }

    /// Whether `range` of this text is lines of the base of a conflict that
    /// the person kept, or a point in or at the edge of one, and holds no
    /// line in the form of an opening marker line, so no whole block.
    fn is_inside_base(&self, range: &Range<usize>) -> bool {
        let next = self.of_base.partition_point(|base| base.end < range.start);
        let inside = self
            .of_base
            .get(next)
            .is_some_and(|base| base.start <= range.start && range.end <= base.end);
        inside
            && !self.bytes[range.clone()]
                .lines_with_terminator()
                .any(|line| parse_opening_line(line).is_some())
    }
}

/// The changes from `old`, the text of a term of V, to `new`, by a line diff:
/// each is text that takes the place of a range of `old`, whatever its lines
/// look like.
pub(super) fn text_edits(old: &[u8], new: &[u8]) -> Vec<Edit> {
    line_changes(old, new)
        .into_iter()
        .map(|(range, new_range)| Edit {
            range,
            text: Text::literal(&new[new_range]),
        })
        .collect()
}

/// A change to a virtual merge base: the range of it that the change
/// replaces, and the text that takes its place.
#[derive(Clone, Debug)]
pub(super) struct Edit {
    pub(super) range: Range<usize>,
    pub(super) text: Text,
}

/// How a line diff reads the marker lines of VB.
impl Marker {
    /// The line that stands for every marker line of this kind where a line
    /// diff lines VB up with its new text: no line of a text file.
    fn token(self) -> &'static [u8] {
        match self {
            Self::Open => b"\0opening marker line\n",
            Self::Separator => b"\0separator marker line\n",
            Self::Close => b"\0closing marker line\n",
        }
    }

    /// The line that stands for the marker line of this kind of the block at
    /// `block` of VB, or for every marker line of this kind where no block
    /// is told, where a line diff lines up the base that a file showed with
    /// its new text: a conflict that the person deleted or moved is then a
    /// change of its own block's lines.
    fn token_of(self, block: Option<usize>) -> Vec<u8> {
        match block {
            Some(block) => {
                let name = format!("\0block {block}: ");
                [name.as_bytes(), &self.token()[1..]].concat()
            }
            None => self.token().to_vec(),
        }
    }
}

/// The base of a conflict, VB, as the file that jj materialized for the
/// conflict showed it, with the changes that the conflict's sides make to VB
/// where the conflict's own merge resolved a hunk inside one of its blocks.
pub(super) struct ShownBase {
    /// VB but for the hunks that the conflict's own merge resolved to its
    /// sides' text, which show that text.
    pub(super) text: Text,
    /// At each hunk that the merge resolved inside a side of a block of VB,
    /// the sides' changes there made to VB's text, in order (see
    /// [`Self::new`]).
    pub(super) side_edits: Vec<Edit>,
}

impl ShownBase {
    /// VB, whose blocks `finder` finds, the base at `index` among the
    /// removes of `contents`, the contents of a conflict's terms, as their
    /// merge with `options` shows it.
    ///
    /// At a hunk that the merge resolves to `S`, jj's snapshot writes `S`
    /// into every side of the conflict, so each side's change there is its
    /// own text `P_i` to `S`. Where the hunk lies inside one side of a block
    /// of VB, holding none of its marker lines, those changes are made to
    /// VB's text `B` there: the text that takes its place is `B` plus, for
    /// each side, `S` less `P_i`, where those merge. That is `S` where a
    /// side holds `B` and the other `S`, and `B` where every side holds
    /// `S`, which made no change there. VB's marker lines, and the lines of
    /// a side of a block that the merge did not take, are in no such hunk.
    pub(super) fn new(
        finder: &BlockFinder<'_>,
        contents: &Merge<BString>,
        index: usize,
        options: &MergeOptions,
    ) -> Self {
        let vb = finder.vb;
        let mut text = Text::default();
        let mut side_edits = vec![];
        for hunk in files::split_ranged_hunks(contents, options) {
            let Some(range) = hunk.ranges.get_remove(index).cloned() else {
                continue;
            };
            let Some(merged) = &hunk.resolved else {
                text.push(&vb[range], true);
                continue;
            };
            text.push(merged, false);
            if hunk.matching || !is_inside_a_side(finder.blocks, &range) {
                continue;
            }
            let own = &vb[range.clone()];
            let sides = zip(contents.adds(), hunk.ranges.adds())
                .map(|(content, range)| &content.as_slice()[range.clone()]);
            let merged = merged.as_slice();
            let changes = iter::once(own).chain(iter::repeat_n(merged, contents.num_sides()));
            let changed = files::try_merge(&Merge::from_removes_adds(sides, changes), options);
            if let Some(changed) = changed
                && changed.as_slice() != own
            {
                side_edits.push(Edit {
                    range,
                    text: Text::literal(&changed),
                });
            }
        }
        Self { text, side_edits }
    }
}

/// Whether `range` of VB, whose blocks are `blocks`, lies inside one side of
/// a block, holding none of the block's marker lines.
fn is_inside_a_side(blocks: &[Block<'_>], range: &Range<usize>) -> bool {
    let next = blocks.partition_point(|block| block.end <= range.start);
    blocks.get(next).is_some_and(|block| {
        block
            .side_ranges
            .iter()
            .any(|side| side.start <= range.start && range.end <= side.end)
    })
}

/// The changes from VB, VB or M whose blocks `finder` finds, to `new`, the
/// text that a snapshot gives it, in order, none overlapping another, where
/// `seen` is the base that the file that jj materialized showed.
///
/// The text of `seen` is VB but for the hunks that the conflict's own merge
/// resolved to its sides' text, which the file holds in place of VB's.
/// Outside every block of VB, that text is what the file showed for the
/// base, and it goes into V's terms as the person left it, as jj's snapshot
/// writes the text of a hunk that resolves into every term, a plain base
/// among them. Where such hunks cut a block, VB keeps the block: its marker
/// lines and the side that the merge did not take stay, it takes each
/// side's change inside a side of the block (see [`ShownBase::new`]), and a
/// person's edit inside its sides goes into the terms of the block's sides,
/// not into V's base term (see [`onto_virtual_base`]).
///
/// The working copy can hold the marker lines of an earlier VB of the same
/// conflict, which an earlier snapshot replaced, so a block is told by its
/// sides, not by its hex: text that came from the base of a conflict that the
/// person kept holds a block of VB where it holds, between marker lines of
/// one hex, the block's sides (see [`BlockFinder::held`]). The blocks that
/// `new` holds in VB's order line up with themselves, and a line diff of the
/// text between them gives the changes. In that diff a block that `new` holds
/// out of that order is one line, so a conflict that the person moved is
/// deleted in one place and put, whole, in another; and each other line of a
/// base in the form of a marker line is equal to VB's marker line of that
/// kind, so a change inside a block's side is a change of those lines alone.
pub(super) fn base_edits(finder: &BlockFinder<'_>, seen: &ShownBase, new: &Text) -> Vec<Edit> {
    let vb = finder.vb;
    let new = &onto_virtual_base(finder, seen, new);
    let held = new
        .of_base
        .iter()
        .flat_map(|base| finder.held(&new.bytes, base.clone()))
        .collect_vec();
    let mut edits = vec![];
    let (mut vb_at, mut new_at, mut next) = (0, 0, 0);
    for anchor in increasing(&held) {
        let (copy, block) = &held[anchor];
        let block = &finder.blocks[*block];
        let gap = Gap {
            vb: vb_at..block.start,
            new: new_at..copy.start,
            moved: &held[next..anchor],
        };
        gap_edits(finder, new, &gap, &mut edits);
        (vb_at, new_at, next) = (block.end, copy.end, anchor + 1);
    }
    let gap = Gap {
        vb: vb_at..vb.len(),
        new: new_at..new.bytes.len(),
        moved: &held[next..],
    };
    gap_edits(finder, new, &gap, &mut edits);
    edits
}

/// `new`, the text that the person left for a base, as a change of VB, whose
/// blocks `finder` finds, where `seen` is that base as the file that jj
/// materialized showed it.
///
/// A line diff of the text that the file showed against `new` gives the
/// person's changes, and one of it against VB the hunks that the conflict's
/// own merge resolved to its sides' text. The result is the text that the
/// file showed with the person's changes. A hunk that the merge resolved
/// outside every block of VB keeps the text that the file showed, or the
/// person's, so it goes into V's terms. Where such hunks cut a block, the
/// lines that the file showed for the block give way to VB's own lines of
/// it, with the sides' changes inside its sides (see [`ShownBase::new`]),
/// and with the person's changes of those lines made among them where they
/// can be (see [`resolved_hunks`], [`untouched`] and [`lifted`]).
fn onto_virtual_base(finder: &BlockFinder<'_>, seen: &ShownBase, new: &Text) -> Text {
    let ShownBase {
        text: seen,
        side_edits,
    } = seen;
    let vb = finder.vb;
    let seen_lines = seen.lines(finder);
    // VB's own lines, with each marker line of its blocks as the line that
    // stands for it.
    let mut own_lines = Lines::default();
    let mut at = 0;
    for line in vb.lines_with_terminator() {
        match finder.marker_at(at) {
            Some(marker) => own_lines.push(at, &marker.token_of(finder.block_of(line))),
            None => own_lines.push(at, line),
        }
        at += line.len();
    }
    let new_lines = new.lines(finder);
    // The person's changes, but for a change of the lines of the base of a
    // conflict that they kept: those lines are VB's, which V's terms give,
    // and they stay as they were. A conflict that the person moved, copied
    // or deleted takes a whole block with it, and is a change.
    let changes = line_changes(&seen_lines.bytes, &new_lines.bytes)
        .into_iter()
        .filter_map(|(range, new_range)| {
            let start = seen_lines.original(range.start, seen.bytes.len());
            let end = seen_lines.original(range.end, seen.bytes.len());
            let new_start = new_lines.original(new_range.start, new.bytes.len());
            let new_end = new_lines.original(new_range.end, new.bytes.len());
            let of_base =
                seen.is_inside_base(&(start..end)) && new.is_inside_base(&(new_start..new_end));
            (!of_base).then(|| (range, new.slice(new_start..new_end)))
        })
        .collect_vec();
    let own_changes = line_changes(&seen_lines.bytes, &own_lines.bytes);
    let resolved = resolved_hunks(finder, &own_lines, &own_changes);
    let (kept, lifted) = untouched(
        finder,
        &seen_lines,
        &own_lines,
        &own_changes,
        &changes,
        resolved,
        side_edits,
    );
    // Each range of the lines of `seen` to replace, with the text that takes
    // its place.
    let mut replacements = zip(changes, lifted)
        .filter(|(_, lifted)| !lifted)
        .map(|(change, _)| change)
        .collect_vec();
    replacements.extend(kept);
    replacements.sort_by_key(|(range, _)| (range.start, range.end));
    let mut text = Text::default();
    let mut at = 0;
    for (range, replacement) in &replacements {
        let start = seen_lines.original(range.start, seen.bytes.len());
        text.append(&seen.slice(at..start));
        text.append(replacement);
        at = seen_lines.original(range.end, seen.bytes.len());
    }
    text.append(&seen.slice(at..seen.bytes.len()));
    text
}

/// The hunks that the conflict's own merge resolved to its sides' text where
/// they cut a block of VB, whose blocks `finder` finds: `own_changes`, the
/// changes from the lines of the base that the file showed to VB's own lines,
/// `own_lines`, that take or reach into a block, those of one block together.
/// A hunk that the merge resolved outside every block is none of these: the
/// text that the file showed there stays.
fn resolved_hunks(
    finder: &BlockFinder<'_>,
    own_lines: &Lines,
    own_changes: &[(Range<usize>, Range<usize>)],
) -> Vec<Resolved> {
    let (vb, blocks) = (finder.vb, finder.blocks);
    let mut resolved: Vec<Resolved> = vec![];
    for (range, own_range) in own_changes.iter().cloned() {
        let start = own_lines.original(own_range.start, vb.len());
        let end = own_lines.original(own_range.end, vb.len());
        let next = blocks.partition_point(|block| block.end <= start);
        let block = blocks.get(next).and_then(|block| {
            let inside = if start == end {
                block.start < start
            } else {
                block.start < end
            };
            inside.then_some(next)
        });
        let Some(block) = block else {
            continue;
        };
        match resolved.last_mut() {
            Some(last) if last.block == block => {
                last.span.end = range.end;
                last.own.end = own_range.end;
            }
            _ => resolved.push(Resolved {
                block,
                span: range,
                own: own_range,
            }),
        }
    }
    resolved
}

/// What takes the place of the lines that the file showed for each block of
/// `resolved`, blocks of VB, whose blocks `finder` finds, that hunks that the
/// conflict's own merge resolved cut, and, for each of `changes`, the
/// person's changes of the lines of the base that the file showed,
/// `seen_lines`, whether it is made there.
///
/// VB keeps each such block: its own lines of the block and of the hunks
/// take the place of the lines that the file showed, with each of
/// `side_edits`, the sides' changes to VB, that falls inside them. Where the
/// file showed some of the block's marker lines in a conflict's base, each
/// change of the person's to those lines is made among them (see
/// [`lifted`]). So an edit at the end or the start of a side, as one next to
/// the conflict at the block's tail is, goes into the terms of the block's
/// sides, A's and B's, and one between two lines of a side into that side's
/// term; none goes into V's base term, o's, and the block stays whole. Where
/// the merge resolved the whole block, or a change takes a marker line or
/// text of a base (a resolution of the conflict at the block's tail), the
/// lines stay as the person left them, and the change goes into every term
/// (case 3 of [`super`]). `own_changes` are the changes from the lines that
/// the file showed to VB's own lines, `own_lines`.
fn untouched(
    finder: &BlockFinder<'_>,
    seen_lines: &Lines,
    own_lines: &Lines,
    own_changes: &[(Range<usize>, Range<usize>)],
    changes: &[(Range<usize>, Text)],
    resolved: Vec<Resolved>,
    side_edits: &[Edit],
) -> (Vec<(Range<usize>, Text)>, Vec<bool>) {
    let vb = finder.vb;
    let mut kept = vec![];
    let mut made = vec![false; changes.len()];
    // The end of VB's own lines of the block before, which a hunk that cuts
    // it can reach past.
    let mut done = 0;
    for Resolved { block, span, own } in resolved {
        let block = &finder.blocks[block];
        // The block's own lines, and the lines that the file showed for them,
        // without a change that is next to the block but outside it.
        let block_start = offset_among(own_lines, block.start).max(done);
        let block_end = offset_among(own_lines, block.end);
        let (own_start, start) = if own.start <= block_start {
            (own.start, span.start)
        } else {
            (block_start, lined_up(own_changes, block_start, true))
        };
        let (own_end, end) = if block_end <= own.end {
            (own.end, span.end)
        } else {
            (block_end, lined_up(own_changes, block_end, false))
        };
        done = own_end;
        let range = own_lines.original(own_start, vb.len())..own_lines.original(own_end, vb.len());
        let own = with_side_edits(finder, range, side_edits);
        let first = changes.partition_point(|(change, _)| change.end <= start);
        let touching = changes[first..]
            .iter()
            .take_while(|(change, _)| change.start < end)
            .count();
        let touching = first..first + touching;
        if touching.is_empty() {
            let mut text = Text::default();
            for line in &own {
                text.append(&line.text);
            }
            kept.push((start..end, text));
            continue;
        }
        let inside = changes[touching.clone()]
            .iter()
            .map(|(change, text)| {
                (start <= change.start && change.end <= end)
                    .then(|| (change.start - start..change.end - start, text))
            })
            .collect::<Option<Vec<_>>>();
        let shown = &seen_lines.bytes[start..end];
        if let Some(text) = inside.and_then(|inside| lifted(block, &own, shown, &inside)) {
            kept.push((start..end, text));
            made[touching].fill(true);
        }
    }
    (kept, made)
}

/// A line of VB, or of a side's change to VB at a hunk that the conflict's
/// own merge resolved inside a block: the line that a line diff reads in its
/// place, its text, where a marker line of VB's blocks came from a base, and
/// the offset in VB of such a marker line.
struct OwnLine {
    token: Vec<u8>,
    text: Text,
    marker_at: Option<usize>,
}

/// The lines of `range` of VB, whose blocks `finder` finds, with each of
/// `side_edits`, changes to VB in order, that falls inside it made.
fn with_side_edits(
    finder: &BlockFinder<'_>,
    range: Range<usize>,
    side_edits: &[Edit],
) -> Vec<OwnLine> {
    let push_own = |lines: &mut Vec<OwnLine>, range: Range<usize>| {
        let mut at = range.start;
        for line in finder.vb[range].lines_with_terminator() {
            let marker = finder.marker_at(at);
            let token = marker.map_or(line.to_vec(), |marker| {
                marker.token_of(finder.block_of(line))
            });
            let mut text = Text::default();
            text.push(line, marker.is_some());
            lines.push(OwnLine {
                token,
                text,
                marker_at: marker.map(|_| at),
            });
            at += line.len();
        }
    };
    let mut lines = vec![];
    let mut at = range.start;
    let first = side_edits.partition_point(|edit| edit.range.start < range.start);
    for edit in side_edits[first..]
        .iter()
        .take_while(|edit| edit.range.end <= range.end)
    {
        push_own(&mut lines, at..edit.range.start);
        for line in edit.text.bytes.lines_with_terminator() {
            lines.push(OwnLine {
                token: line.to_vec(),
                text: Text::literal(line),
                marker_at: None,
            });
        }
        at = edit.range.end;
    }
    push_own(&mut lines, at..range.end);
    lines
}

/// Where a change of the person's goes among VB's own lines of a block.
enum Lift {
    /// In place of the lines that it replaces.
    InPlace,
    /// At the start of every side of the block, where it starts at the start
    /// of a side.
    Before,
    /// At the end of every side of the block, where it ends at the end of a
    /// side.
    After,
}

/// `own`, VB's own lines of `block` and around it with the sides' changes to
/// VB (see [`with_side_edits`]), with `changes` made among them, where the
/// file that jj materialized showed `shown` for them, and the person made
/// `changes`, each a range of `shown` and the text that takes its place.
///
/// A line diff of `shown` against `own` lines them up: the lines that the
/// conflict's own merge left out of the block, such as its marker lines and
/// a side that it did not take, are `own`'s alone. Each change that takes
/// only lines that both hold, and puts only the person's text, is made where
/// it falls among `own`. Inside a side of the block it is made there, so it
/// goes into that side's term of V. A change that ends at the end of a side,
/// as an edit next to a conflict at the block's tail does, or starts at its
/// start, takes that side's lines that it replaces there, and its text goes
/// at the end, or the start, of every side of the block: into the term of
/// each side, just where the person put it in the side that the file showed.
/// The block's marker lines stay where they are, so the conflict's own merge
/// of the block, a conflict at its tail among it, stays as it was; V's base
/// term keeps its own text there. This is only where the file showed some of
/// the block's marker lines in the base of a conflict, which the merge left
/// there: where it resolved the whole block, the person's text for it goes
/// into every term as they left it (merge-cases' VB5d). `None` there, and
/// where a change is not of that kind.
fn lifted(
    block: &Block<'_>,
    own: &[OwnLine],
    shown: &[u8],
    changes: &[(Range<usize>, &Text)],
) -> Option<Text> {
    let tokens = own
        .iter()
        .map(|line| line.token.as_slice())
        .collect_vec()
        .concat();
    let mut starts = Vec::with_capacity(own.len());
    let mut at = 0;
    for line in own {
        starts.push(at);
        at += line.token.len();
    }
    // The block's marker lines among `own`: its opening line first, its
    // closing line last.
    let markers = own
        .iter()
        .positions(|line| {
            line.marker_at
                .is_some_and(|at| block.markers.iter().any(|(range, _)| range.start == at))
        })
        .collect_vec();
    let (&open, &close) = (markers.first()?, markers.last()?);
    let shows_a_marker = shown
        .lines_with_terminator()
        .any(|line| markers.iter().any(|&marker| own[marker].token == line));
    if !shows_a_marker {
        return None;
    }
    let restored = line_changes(shown, &tokens);
    // The offset among `tokens` that `offset` among `shown` lines up with:
    // lines of `own`'s alone there that open the block come before it, any
    // other such lines after it.
    let line_up = |offset: usize| -> Option<usize> {
        let mut last = (0, 0);
        for (shown_range, own_range) in &restored {
            if shown_range.end < offset || (shown_range.end == offset && !shown_range.is_empty()) {
                last = (shown_range.end, own_range.end);
                continue;
            }
            if shown_range.start < offset {
                return None;
            }
            if shown_range.is_empty() && shown_range.start == offset {
                let opens = tokens[own_range.clone()].starts_with(&own[open].token);
                return Some(if opens {
                    own_range.end
                } else {
                    own_range.start
                });
            }
            break;
        }
        Some(last.1 + (offset - last.0))
    };
    let line_at = |offset: usize| match starts.binary_search(&offset) {
        Ok(index) => Some(index),
        Err(index) => (index == own.len() && offset == tokens.len()).then_some(index),
    };
    let (mut before, mut after, mut cuts) = (vec![], vec![], vec![]);
    for (range, text) in changes {
        let crosses = restored.iter().any(|(shown_range, _)| {
            if shown_range.is_empty() {
                range.start < shown_range.start && shown_range.start < range.end
            } else {
                shown_range.start < range.end && range.start < shown_range.end
            }
        });
        if crosses || !text.of_base.is_empty() {
            return None;
        }
        let start = line_at(line_up(range.start)?)?;
        let end = if range.is_empty() {
            start
        } else {
            line_at(line_up(range.end)?)?
        };
        let lines = own.get(start..end)?;
        if lines.iter().any(|line| line.marker_at.is_some()) {
            return None;
        }
        let inside = open < start && end <= close;
        let place = if inside && own[end].marker_at.is_some() {
            Lift::After
        } else if inside && own[start - 1].marker_at.is_some() {
            Lift::Before
        } else {
            Lift::InPlace
        };
        match place {
            Lift::InPlace => cuts.push((start..end, Some(*text))),
            Lift::Before => {
                before.push(*text);
                cuts.push((start..end, None));
            }
            Lift::After => {
                after.push(*text);
                cuts.push((start..end, None));
            }
        }
    }
    let mut result = Text::default();
    let mut cuts = cuts.into_iter().peekable();
    let mut index = 0;
    loop {
        let mut skip_to = None;
        while let Some((range, text)) = cuts.next_if(|(range, _)| range.start == index) {
            if let Some(text) = text {
                result.append(text);
            }
            if !range.is_empty() {
                skip_to = Some(range.end);
                break;
            }
        }
        if let Some(end) = skip_to {
            index = end;
            continue;
        }
        let Some(line) = own.get(index) else {
            break;
        };
        let is_marker = markers.binary_search(&index).is_ok();
        if is_marker && index != open {
            for text in &after {
                result.append(text);
            }
        }
        result.append(&line.text);
        if is_marker && index != close {
            for text in &before {
                result.append(text);
            }
        }
        index += 1;
    }
    Some(result)
}

/// The offset among `lines` of `original`, the start of a line of the text
/// that they are the lines of, or its end.
fn offset_among(lines: &Lines, original: usize) -> usize {
    let line = lines.starts.partition_point(|(_, start)| *start < original);
    lines
        .starts
        .get(line)
        .map_or(lines.bytes.len(), |(offset, _)| *offset)
}

/// The offset among the lines that a file showed for a base that `offset`,
/// one among VB's own lines, lines up with, where `changes` are the changes
/// from the first lines to the second, in order: the start of the lines
/// that the file showed in place of the ones at `offset`, or their end if
/// `end` is set.
fn lined_up(changes: &[(Range<usize>, Range<usize>)], offset: usize, end: bool) -> usize {
    let next = changes.partition_point(|(_, own)| own.end < offset);
    match changes.get(next) {
        Some((shown, own)) if own.start <= offset => {
            if end {
                if offset > own.start || own.is_empty() {
                    shown.end
                } else {
                    shown.start
                }
            } else if offset < own.end || own.is_empty() {
                shown.start
            } else {
                // The change ends here, and the next one can start here.
                match changes.get(next + 1) {
                    Some((next_shown, next_own)) if next_own.start == offset => next_shown.start,
                    _ => shown.end,
                }
            }
        }
        // No change takes these lines: they are as far past the change
        // before them in the one as in the other.
        _ => match next.checked_sub(1) {
            Some(last) => changes[last].0.end + (offset - changes[last].1.end),
            None => offset,
        },
    }
}

/// Hunks that a conflict's own merge resolved to its sides' text in one block
/// of VB, as changes of the lines of the base that the file showed to VB's
/// own lines.
struct Resolved {
    /// The index of the block.
    block: usize,
    /// The lines of the base that the file showed that the hunks span.
    span: Range<usize>,
    /// VB's own lines that they span.
    own: Range<usize>,
}

/// The lines of `range` of VB, whose blocks `finder` finds, where each marker
/// line of its blocks is the line that stands for its kind.
fn marked_lines(finder: &BlockFinder<'_>, range: Range<usize>) -> Lines {
    let mut lines = Lines::default();
    let mut at = range.start;
    for line in finder.vb[range].lines_with_terminator() {
        match finder.marker_at(at) {
            Some(marker) => lines.push(at, marker.token()),
            None => lines.push(at, line),
        }
        at += line.len();
    }
    lines
}

/// The indices of the most of `held`, blocks that the new text holds, each
/// with its block's index, that are in the order of their blocks.
fn increasing(held: &[(Range<usize>, usize)]) -> Vec<usize> {
    // The last block of the best run of each length so far, and for each
    // block the one before it in its run.
    let mut lasts: Vec<usize> = vec![];
    let mut previous = vec![None; held.len()];
    for (index, (_, block)) in held.iter().enumerate() {
        let length = lasts.partition_point(|&last| held[last].1 < *block);
        previous[index] = length.checked_sub(1).map(|shorter| lasts[shorter]);
        if length == lasts.len() {
            lasts.push(index);
        } else {
            lasts[length] = index;
        }
    }
    let mut run = vec![];
    let mut next = lasts.last().copied();
    while let Some(index) = next {
        run.push(index);
        next = previous[index];
    }
    run.reverse();
    run
}

/// The text between two blocks that line up, in VB and in its new text, and
/// the blocks that the new text holds there, out of VB's order.
struct Gap<'a> {
    vb: Range<usize>,
    new: Range<usize>,
    moved: &'a [(Range<usize>, usize)],
}

/// Text as a line diff reads it: its lines, some replaced with a line that
/// stands for them, and where each line starts, here and in the text itself,
/// with both ends last.
#[derive(Default)]
struct Lines {
    bytes: Vec<u8>,
    starts: Vec<(usize, usize)>,
}

impl Lines {
    /// Appends `line`, which stands for the text at `start`.
    fn push(&mut self, start: usize, line: &[u8]) {
        self.starts.push((self.bytes.len(), start));
        self.bytes.extend_from_slice(line);
    }

    /// The offset in the text itself of `offset` here, the start of a line or
    /// the end, where the text ends at `end`.
    fn original(&self, offset: usize, end: usize) -> usize {
        let line = self.starts.partition_point(|(start, _)| *start < offset);
        self.starts.get(line).map_or(end, |(_, original)| *original)
    }
}

/// Appends to `edits` the changes from VB, whose blocks `finder` finds, to
/// its new text, `new`, in `gap`, by a line diff.
fn gap_edits(finder: &BlockFinder<'_>, new: &Text, gap: &Gap<'_>, edits: &mut Vec<Edit>) {
    let vb = finder.vb;
    if vb[gap.vb.clone()] == new.bytes[gap.new.clone()] {
        return;
    }
    let old = marked_lines(finder, gap.vb.clone());
    let mut changed = Lines::default();
    let mut moved = gap.moved.iter().peekable();
    let mut at = gap.new.start;
    while at < gap.new.end {
        if let Some((copy, block)) = moved.next_if(|(copy, _)| copy.start == at) {
            changed.push(at, format!("\0block {block}\n").as_bytes());
            at = copy.end;
            continue;
        }
        let rest = &new.bytes[at..gap.new.end];
        let line = &rest[..rest.find_byte(b'\n').map_or(rest.len(), |end| end + 1)];
        match Marker::of_form(line).filter(|_| new.is_of_base(at)) {
            Some(marker) => changed.push(at, marker.token()),
            None => changed.push(at, line),
        }
        at += line.len();
    }
    for (old_range, new_range) in line_changes(&old.bytes, &changed.bytes) {
        let range =
            old.original(old_range.start, gap.vb.end)..old.original(old_range.end, gap.vb.end);
        let new_range = changed.original(new_range.start, gap.new.end)
            ..changed.original(new_range.end, gap.new.end);
        let text = new.slice(new_range);
        if vb[range.clone()] != text.bytes {
            edits.push(Edit { range, text });
        }
    }
}

/// The changes from `old` to `new` by a line diff: each range of `old` that
/// changed, with the range of `new` that takes its place.
fn line_changes(old: &[u8], new: &[u8]) -> Vec<(Range<usize>, Range<usize>)> {
    if old == new {
        return vec![];
    }
    ContentDiff::by_line([old, new])
        .hunk_ranges()
        .filter(|hunk| matches!(hunk.kind, DiffHunkKind::Different))
        .map(|hunk| (hunk.ranges[0].clone(), hunk.ranges[1].clone()))
        .collect()
}

/// The changes that the person made to each side of a conflict: the sides'
/// old texts, their new ones, and, for each side, the ranges that a line diff
/// of the two gives as changed, in the old text and in the new one.
struct SideChanges<'a> {
    old: Vec<&'a BString>,
    new: Vec<Vec<u8>>,
    changes: Vec<Vec<(Range<usize>, Range<usize>)>>,
}

impl<'a> SideChanges<'a> {
    /// The changes from the sides of `old_contents`, the contents of a
    /// conflict's terms, to the sides that `new_hunks`, the hunks that the
    /// person left, give, as jj's snapshot reads them.
    fn new(old_contents: &'a Merge<BString>, new_hunks: &[Merge<BString>]) -> Self {
        let old = old_contents.adds().collect_vec();
        let mut new = vec![vec![]; old.len()];
        for hunk in new_hunks {
            match hunk.as_resolved() {
                Some(resolved) => {
                    for side in &mut new {
                        side.extend_from_slice(resolved);
                    }
                }
                None => {
                    for (side, text) in zip(&mut new, hunk.adds()) {
                        side.extend_from_slice(text);
                    }
                }
            }
        }
        let changes = zip(&old, &new)
            .map(|(old, new)| side_changes(old, new))
            .collect();
        Self { old, new, changes }
    }

    /// Takes out of the changes the text that every side gained at its very
    /// start, or at its very end if `at_start` is not set, if it is the same
    /// text in all, and returns it.
    fn take_shared_insertion(&mut self, at_start: bool) -> Option<Vec<u8>> {
        let texts = zip(&self.changes, zip(&self.old, &self.new))
            .map(|(changes, (old, new))| {
                let (range, changed) = if at_start {
                    changes.first()?
                } else {
                    changes.last()?
                };
                let point = if at_start { 0 } else { old.len() };
                (*range == (point..point)).then(|| &new[changed.clone()])
            })
            .collect::<Option<Vec<_>>>()?;
        if !texts.iter().all_equal() {
            return None;
        }
        let text = texts.first()?.to_vec();
        for changes in &mut self.changes {
            if at_start {
                changes.remove(0);
            } else {
                changes.pop();
            }
        }
        Some(text)
    }
}

/// The changes from `old`, a side's text, to `new`, its text after a
/// snapshot, by a line diff. Where `old` ends without a newline, jj shows it
/// with one, so text that the person put after it comes after that newline:
/// `new` then starts with `old` and a newline, and that text is an insertion
/// at the end of `old`, which goes into a term after a newline where the term
/// has none (see `edited_m_terms` and the layout's `write_to_every_term`).
/// An empty side is shown as a newline where another side of its conflict
/// ends without one, and the text after that newline is the insertion too.
fn side_changes(old: &[u8], new: &[u8]) -> Vec<(Range<usize>, Range<usize>)> {
    if !old.ends_with(b"\n")
        && new.len() > old.len() + 1
        && new.starts_with(old)
        && new[old.len()] == b'\n'
    {
        return vec![(old.len()..old.len(), old.len() + 1..new.len())];
    }
    line_changes(old, new)
}

/// The changes to a base of `len` bytes that shows none of V's text, M, that
/// the hunks `new_hunks`, which the person left, stand for, where
/// `old_contents` are the contents of the conflict's terms: the text that
/// every side of the conflict gained at its very start goes before the whole
/// of the base, and the text that every side gained at its very end after
/// it.
pub(super) fn edge_insertions(
    old_contents: &Merge<BString>,
    new_hunks: &[Merge<BString>],
    len: usize,
) -> Vec<Edit> {
    let mut sides = SideChanges::new(old_contents, new_hunks);
    let leading = sides.take_shared_insertion(true).map(|text| (0, text));
    let trailing = sides.take_shared_insertion(false).map(|text| (len, text));
    leading
        .into_iter()
        .chain(trailing)
        .map(|(point, text)| Edit {
            range: point..point,
            text: Text::literal(&text),
        })
        .collect()
}

/// The changes to `reference`, the VB of V, that the hunks `new_hunks`,
/// which the person left, stand for, where the conflict's base, at `index`
/// among the removes of its terms' contents `old_contents`, shows none of
/// V's text: the changes that the person made to every side of the conflict
/// alike, in text that every side shares with `reference`.
///
/// With `reference` as the base, the conflict's terms merge, with `options`,
/// into hunks; the person's changes to each side are a line diff of the
/// side's old text against its new one. In a hunk where every term holds the
/// same text, a change that every side has, at the same lines and to the same
/// text, is a change to `reference` there. In a hunk that resolves otherwise,
/// where the sides' own text took the place of `reference`'s, the text that
/// every side holds after its changes takes that place, if it is the same in
/// all. Text that every side gained at its very start, or at its very end,
/// goes before or after the whole of `reference`. A change that only some
/// sides have, or that reaches out of the hunk, is a change to those sides
/// alone.
pub(super) fn shared_text_edits(
    reference: &[u8],
    old_contents: &Merge<BString>,
    index: usize,
    new_hunks: &[Merge<BString>],
    options: &MergeOptions,
) -> Vec<Edit> {
    let mut sides = SideChanges::new(old_contents, new_hunks);
    let leading = sides.take_shared_insertion(true);
    let trailing = sides.take_shared_insertion(false);
    let mut edits = vec![];
    edits.extend(leading.map(|text| Edit {
        range: 0..0,
        text: Text::literal(&text),
    }));
    let removes = old_contents.removes().enumerate().map(|(remove, content)| {
        if remove == index {
            reference
        } else {
            content.as_slice()
        }
    });
    let adds = sides.old.iter().map(|content| content.as_slice());
    let inputs = Merge::from_removes_adds(removes, adds);
    // The next change of each side that no hunk has looked at.
    let mut next = vec![0; sides.old.len()];
    for hunk in files::split_ranged_hunks(&inputs, options) {
        let Some(base_range) = hunk.ranges.get_remove(index) else {
            continue;
        };
        // Each side's changes inside the hunk: the range of each from the
        // start of the side's text there, and its text. An insertion between
        // two hunks is in the one that resolves, or the first of two that
        // do.
        let resolves = hunk.resolved.is_some();
        let mut side_changes = vec![];
        for (side, range) in hunk.ranges.adds().enumerate() {
            let mut inside = vec![];
            while let Some((old, new)) = sides.changes[side].get(next[side]) {
                let within = range.start <= old.start
                    && old.end <= range.end
                    && (old.start < range.end || (old.is_empty() && resolves));
                if old.start < range.start {
                    // It starts in a hunk before this one, and reaches here.
                    next[side] += 1;
                } else if within {
                    next[side] += 1;
                    inside.push((
                        old.start - range.start..old.end - range.start,
                        &sides.new[side][new.clone()],
                    ));
                } else {
                    break;
                }
            }
            side_changes.push(inside);
        }
        if !resolves {
            continue;
        }
        if hunk.matching {
            let (first, others) = side_changes.split_first().expect("a side");
            let shared = first
                .iter()
                .filter(|change| others.iter().all(|other| other.contains(change)));
            edits.extend(shared.map(|(range, text)| Edit {
                range: base_range.start + range.start..base_range.start + range.end,
                text: Text::literal(text),
            }));
            continue;
        }
        if side_changes.iter().any(|changes| changes.is_empty()) {
            continue;
        }
        let new_texts = zip(hunk.ranges.adds(), zip(&sides.old, &side_changes))
            .map(|(range, (old, changes))| {
                let old = &old.as_slice()[range.clone()];
                let mut text = vec![];
                let mut at = 0;
                for (range, new) in changes {
                    text.extend_from_slice(&old[at..range.start]);
                    text.extend_from_slice(new);
                    at = range.end;
                }
                text.extend_from_slice(&old[at..]);
                text
            })
            .collect_vec();
        if new_texts.iter().all_equal() {
            edits.push(Edit {
                range: base_range.clone(),
                text: Text::literal(&new_texts[0]),
            });
        }
    }
    edits.extend(trailing.map(|text| Edit {
        range: reference.len()..reference.len(),
        text: Text::literal(&text),
    }));
    edits
}
