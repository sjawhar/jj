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

//! The virtual merge base: the file that stands for V, the unresolved merge of
//! the merge bases of two sides that both changed it, as the base of their
//! merge (<https://github.com/jj-vcs/jj/issues/6369>; see
//! [`crate::merge_rule`]).
//!
//! Where V is a conflict of text files, the file is VB: jj's line merge of V's
//! terms, with V's sides between marker lines where they do not merge, as git
//! builds a virtual merge base. Otherwise it is M: one block whose sides each
//! name a term of V. VB and M are only ever bases of conflicts: where a merge
//! leaves one as a side, [`MergedTree::merge`] replaces it with the terms of V
//! that it names.
//!
//! A block is an opening marker line, `<<<<<<< jj virtual merge base <hex>
//! <salt> <term>...`, its sides, a line `======= <hex>` between two sides, and
//! a closing marker line `>>>>>>> <hex>`. The terms name V's terms after
//! `simplify()`, adds and removes alternating, and every block names the same
//! ones. `<hex>` is 32 hex digits of a hash over a fixed tag, the block's
//! ordinal among the file's blocks, the terms, the text between the previous
//! block (or the start of the file) and this one, the block's sides, the text
//! after the block if it is the last one, and the salt. So the hashes cover
//! every byte of the file: a file that anyone changed is no longer VB or M. The
//! writer takes the lowest salt at which no marker line is a line of either
//! side of the merge, and the reader hashes each block once, with the salt
//! that the block carries.
//!
//! [`MergedTree::merge`]: crate::merged_tree::MergedTree::merge

use std::collections::HashSet;
use std::fmt::Write as _;
use std::iter;
use std::iter::Peekable;
use std::iter::zip;
use std::mem;
use std::ops::Range;

use blake2::Blake2b512;
use bstr::BString;
use bstr::ByteSlice as _;
use digest::Digest;
use futures::AsyncReadExt as _;
use futures::try_join;
use itertools::Itertools as _;

use crate::backend::BackendError;
use crate::backend::BackendResult;
use crate::backend::CommitId;
use crate::backend::CopyId;
use crate::backend::FileId;
use crate::backend::MergedTreeValue;
use crate::backend::MergedTreeValueExt as _;
use crate::backend::SymlinkId;
use crate::backend::TreeId;
use crate::backend::TreeValue;
use crate::diff::ContentDiff;
use crate::diff::DiffHunkKind;
use crate::diff::DiffHunkRange;
use crate::files;
use crate::files::MergeResult;
use crate::files::is_binary_content;
use crate::hex_util;
use crate::merge::Merge;
use crate::merge::SameChange;
use crate::object_id::ObjectId as _;
use crate::repo_path::RepoPath;
use crate::store::Store;

/// The tag that the hash of every block starts with.
const TAG: &str = "jj virtual merge base";
/// The start of the opening marker line of a block.
const OPEN: &str = "<<<<<<< jj virtual merge base ";
/// The start of the marker line between two sides of a block.
const SEPARATOR: &str = "======= ";
/// The start of the closing marker line of a block.
const CLOSE: &str = ">>>>>>> ";

/// Writes to the store the virtual base against which two parents that both
/// changed `base`, V, an unresolved merge at `path`, are merged, and returns it
/// as a file value.
///
/// Where `base` is a conflict of text files (each add a regular file whose
/// first 8,000 bytes hold no NUL byte, each remove such a file or absent, read
/// as an empty file), the file is VB, jj's line merge of `base`'s terms: each
/// hunk that merges keeps its merged text, and each hunk that does not is a
/// block that holds `base`'s sides at that hunk, in `base`'s order. Otherwise
/// the file is M, one block whose sides each name a term of `base`, in
/// `base`'s order. Every block names `base`'s terms, and its marker lines are
/// at the lowest salt at which no line of any term of either parent equals one
/// of them (see the module's documentation). The file is executable where the
/// executable bits of `base` merge to `true`, and keeps the copy id of `base`
/// where its copy ids merge, else the placeholder. The result does not depend
/// on the order of the parents.
pub async fn virtual_merge_base(
    store: &Store,
    path: &RepoPath,
    base: &MergedTreeValue,
    parent1: &MergedTreeValue,
    parent2: &MergedTreeValue,
) -> BackendResult<TreeValue> {
    let base = base.simplify();
    let base_contents = text_file_contents(store, path, &base);
    let contents1 = file_term_contents(store, path, parent1);
    let contents2 = file_term_contents(store, path, parent2);
    let (base_contents, contents1, contents2) = try_join!(base_contents, contents1, contents2)?;
    let parent_lines = marker_like_lines(
        contents1
            .iter()
            .chain(contents2.iter())
            .map(|content| content.as_slice()),
    );
    write_virtual_base(store, path, &base, base_contents, &parent_lines).await
}

/// The terms of V that `content` names, if `content` is VB or M as
/// [`virtual_merge_base`] writes it: it holds at least one block, every block
/// names the same terms, and the hex of each block is the hash of the block, as
/// the writer hashes it, with the salt that the block carries. Text that no
/// merge wrote is neither VB nor M, nor is VB or M with any byte changed.
pub fn virtual_base_terms(content: &[u8]) -> Option<MergedTreeValue> {
    let blocks = blocks(content)?;
    let [first, ..] = &blocks[..] else {
        return None;
    };
    let names = first.terms;
    if blocks.iter().any(|block| block.terms != names) {
        return None;
    }
    let terms = str::from_utf8(names)
        .ok()?
        .split(' ')
        .map(parse_term_name)
        .collect::<Option<Vec<_>>>()?;
    if terms.len().is_multiple_of(2) {
        return None;
    }
    let last = blocks.len() - 1;
    let mut text_start = 0;
    for (ordinal, block) in blocks.iter().enumerate() {
        let before = &content[text_start..block.start];
        let after: &[u8] = if ordinal == last {
            &content[block.end..]
        } else {
            &[]
        };
        let hasher = block_hasher(ordinal, names, before, &block.sides, after);
        if salted_hex(&hasher, block.salt).as_bytes() != block.hex {
            return None;
        }
        text_start = block.end;
    }
    Some(Merge::from_vec(terms))
}

/// The terms of V that `value` stands for, if `value` is VB or M: a file whose
/// content names them (see [`virtual_base_terms`]), each of which the store
/// holds.
pub(crate) async fn stored_virtual_base_terms(
    store: &Store,
    path: &RepoPath,
    value: &TreeValue,
) -> BackendResult<Option<MergedTreeValue>> {
    let TreeValue::File { id, .. } = value else {
        return Ok(None);
    };
    let content = read_file(store, path, id).await?;
    match virtual_base_terms(&content) {
        Some(terms) if holds_terms(store, path, &terms).await? => Ok(Some(terms)),
        _ => Ok(None),
    }
}

/// The file that takes the place of the base at `index` of `old_contents`,
/// the contents of the terms of a conflict at `path` whose base `old_id` is
/// VB or M, once a parse of the conflict's materialized markers (a snapshot of
/// the working copy, or a merge tool's output) gave the hunks `new_hunks`;
/// `None` where that base is not VB or M.
///
/// The person changed the file that jj materialized, not the terms. Each
/// change to a hunk that resolves, a line diff of the hunk as jj materialized
/// it against the new hunk, is made to VB at the VB lines that the changed
/// lines are; VB keeps every other line, its marker lines included. An
/// insertion next to VB lines that the materialized hunk does not hold goes
/// outside the block they belong to, and so does a hunk that resolves where
/// none was. A change to VB's own lines in a hunk that conflicts is taken as
/// it is. The changes are then made to V's terms by region (see
/// [`edited_terms`]), and the file is the VB or M of the edited terms, or
/// their text where they resolve. Where the new hunks do not match the old
/// ones, or no change can be made, the base stays `old_id`: a snapshot never
/// writes a plain file that holds VB's marker lines.
pub(crate) async fn rewrite_virtual_base(
    store: &Store,
    path: &RepoPath,
    old_id: &FileId,
    old_contents: &Merge<BString>,
    index: usize,
    new_hunks: &[Merge<BString>],
    contents: impl IntoIterator<Item = &[u8]>,
) -> BackendResult<Option<FileId>> {
    let old = old_contents.as_slice()[index].as_slice();
    let Some(terms) = virtual_base_terms(old) else {
        return Ok(None);
    };
    if !holds_terms(store, path, &terms).await? {
        return Ok(None);
    }
    let same_change = store.merge_options().same_change;
    let Some(edits) = output_edits(old_contents, index, new_hunks, same_change) else {
        return Ok(Some(old_id.clone()));
    };
    let new_contents = match text_file_contents(store, path, &terms).await? {
        Some(term_contents) => edited_terms(old, &term_contents, &edits, same_change)
            .map(|new_contents| (term_contents, new_contents)),
        None if terms
            .iter()
            .flatten()
            .all(|term| matches!(term, TreeValue::File { .. })) =>
        {
            let term_contents = file_term_contents(store, path, &terms).await?;
            let new_contents = edited_m_terms(old, &term_contents, &edits);
            Some((term_contents, new_contents))
        }
        None => None,
    };
    let Some((term_contents, new_contents)) = new_contents else {
        return Ok(Some(old_id.clone()));
    };
    let mut new_terms = vec![];
    let old_terms = zip(terms.iter(), term_contents.iter());
    for ((term, content), new_content) in zip(old_terms, new_contents) {
        if new_content.as_slice() == content.as_slice() {
            new_terms.push(term.clone());
            continue;
        }
        let id = store.write_file(path, &mut new_content.as_slice()).await?;
        let (executable, copy_id) = match term {
            Some(TreeValue::File {
                executable,
                copy_id,
                ..
            }) => (*executable, copy_id.clone()),
            _ => (false, CopyId::placeholder()),
        };
        new_terms.push(Some(TreeValue::File {
            id,
            executable,
            copy_id,
        }));
    }
    let base = Merge::from_vec(new_terms).simplify();
    if base == terms {
        return Ok(Some(old_id.clone()));
    }
    if let Some(resolved) = base.as_resolved() {
        return Ok(match resolved {
            Some(TreeValue::File { id, .. }) => Some(id.clone()),
            _ => Some(old_id.clone()),
        });
    }
    let base_contents = text_file_contents(store, path, &base).await?;
    let parent_lines = marker_like_lines(contents);
    match write_virtual_base(store, path, &base, base_contents, &parent_lines).await? {
        TreeValue::File { id, .. } => Ok(Some(id)),
        _ => unreachable!("a virtual base is a file"),
    }
}

/// A change to a text: the range it replaces, and the text that takes its
/// place.
#[derive(Clone, Debug)]
struct Edit {
    range: Range<usize>,
    text: Vec<u8>,
}

/// The changes to the base at `index` of `old_contents`, VB or M, that the
/// person's changes to the hunks of `old_contents`, as jj materializes them,
/// stand for, where `new_hunks` are the hunks the person left (see
/// [`rewrite_virtual_base`]); `None` where the new hunks do not match the old
/// ones.
fn output_edits(
    old_contents: &Merge<BString>,
    index: usize,
    new_hunks: &[Merge<BString>],
    same_change: SameChange,
) -> Option<Vec<Edit>> {
    let vb = old_contents.as_slice()[index].as_slice();
    let blocks = blocks(vb)?;
    let mut new_hunks = new_hunks.iter().peekable();
    let mut edits = vec![];
    // A hunk that resolves where none was is an insertion where the previous
    // hunk ends.
    let mut end = 0;
    for hunk in coalesced(line_hunks(old_contents, same_change)) {
        let region = hunk.ranges[index].clone();
        if hunk.kind == HunkKind::Conflict {
            insert_resolved(&mut new_hunks, region.start, &mut edits);
            let new_hunk = new_hunks.next()?;
            if new_hunk.as_slice().len() != old_contents.as_slice().len() {
                return None;
            }
            let changes = line_changes(&vb[region.clone()], &new_hunk.as_slice()[index]);
            edits.extend(changes.into_iter().map(|(range, text)| Edit {
                range: region.start + range.start..region.start + range.end,
                text,
            }));
        } else {
            let empty = BString::default();
            let new_text = match new_hunks.next_if(|hunk| hunk.is_resolved()) {
                Some(new_hunk) => new_hunk.as_resolved().unwrap_or(&empty),
                None => &empty,
            };
            let anchors = ContentDiff::by_line([hunk.text.as_slice(), &vb[region.clone()]])
                .hunk_ranges()
                .collect_vec();
            for (range, text) in line_changes(&hunk.text, new_text) {
                let target = if range.is_empty() {
                    let point = insertion_point(&anchors, range.start, vb, region.start, &blocks);
                    Some(point..point)
                } else {
                    matched_range(&anchors, &range, region.start)
                };
                if let Some(range) = target {
                    edits.push(Edit { range, text });
                }
            }
        }
        end = region.end;
    }
    insert_resolved(&mut new_hunks, end, &mut edits);
    new_hunks.next().is_none().then_some(edits)
}

/// Adds to `edits` an insertion at `at` of each of the next hunks of
/// `new_hunks` that resolve.
fn insert_resolved<'a>(
    new_hunks: &mut Peekable<impl Iterator<Item = &'a Merge<BString>>>,
    at: usize,
    edits: &mut Vec<Edit>,
) {
    while let Some(new_hunk) = new_hunks.next_if(|hunk| hunk.is_resolved()) {
        let text = new_hunk.as_resolved().map_or(vec![], |text| text.to_vec());
        edits.push(Edit {
            range: at..at,
            text,
        });
    }
}

/// The changes from `old` to `new` by a line diff: each range of `old` that
/// changed, with the text of `new` that takes its place.
fn line_changes(old: &[u8], new: &[u8]) -> Vec<(Range<usize>, Vec<u8>)> {
    ContentDiff::by_line([old, new])
        .hunk_ranges()
        .filter(|hunk| matches!(hunk.kind, DiffHunkKind::Different))
        .map(|hunk| (hunk.ranges[0].clone(), new[hunk.ranges[1].clone()].to_vec()))
        .collect()
}

/// The range of the target whose lines are those of `range` of the source,
/// by `anchors`, a line diff of the source against the target, shifted by
/// `offset`; `None` where the target holds none of them, or where target
/// lines that the source does not hold lie between them.
fn matched_range(
    anchors: &[DiffHunkRange],
    range: &Range<usize>,
    offset: usize,
) -> Option<Range<usize>> {
    let mut matched: Option<Range<usize>> = None;
    for anchor in anchors {
        if !matches!(anchor.kind, DiffHunkKind::Matching) {
            continue;
        }
        let (source, target) = (&anchor.ranges[0], &anchor.ranges[1]);
        let (start, end) = (range.start.max(source.start), range.end.min(source.end));
        if start >= end {
            continue;
        }
        let piece = offset + target.start + start - source.start
            ..offset + target.start + end - source.start;
        matched = match matched {
            None => Some(piece),
            Some(previous) if previous.end == piece.start => Some(previous.start..piece.end),
            Some(_) => return None,
        };
    }
    matched
}

/// Where VB takes an insertion at `position` of a materialized hunk, by
/// `anchors`, a line diff of the hunk against VB's region at `offset`: next to
/// the VB lines that are the lines around it, and where VB lines that the hunk
/// does not hold lie there, at the first point among them that is outside
/// every block of `blocks`, if any.
fn insertion_point(
    anchors: &[DiffHunkRange],
    position: usize,
    vb: &[u8],
    offset: usize,
    blocks: &[Block<'_>],
) -> usize {
    let mut gap = offset..offset;
    for anchor in anchors {
        let source = &anchor.ranges[0];
        let target = offset + anchor.ranges[1].start..offset + anchor.ranges[1].end;
        if matches!(anchor.kind, DiffHunkKind::Different) {
            if source.start <= position && position <= source.end {
                gap = target;
                break;
            }
        } else if source.start < position && position < source.end {
            return target.start + position - source.start;
        } else if source.start == position {
            gap = target.start..target.start;
            break;
        } else if source.end == position {
            gap = target.end..target.end;
        }
    }
    let line_ends = vb[gap.clone()]
        .lines_with_terminator()
        .scan(gap.start, |end, line| {
            *end += line.len();
            Some(*end)
        });
    iter::once(gap.start)
        .chain(line_ends)
        .find(|&point| outside_blocks(blocks, point))
        .unwrap_or(gap.start)
}

/// Whether `point` is outside every block of `blocks`, or at its edge.
fn outside_blocks(blocks: &[Block<'_>], point: usize) -> bool {
    blocks
        .iter()
        .all(|block| point <= block.start || block.end <= point)
}

/// Whether a line of `text` is a marker line of a virtual base.
fn holds_marker_line(text: &[u8]) -> bool {
    text.lines_with_terminator().any(|line| {
        parse_opening_line(line).is_some()
            || marker_hex(line, SEPARATOR).is_some()
            || marker_hex(line, CLOSE).is_some()
    })
}

/// A line hunk of a merge: how its terms merge, the text they merge to unless
/// they conflict, and the range that each term takes in it, in the order of
/// [`Merge::as_slice`].
struct LineHunk {
    kind: HunkKind,
    text: Vec<u8>,
    ranges: Vec<Range<usize>>,
}

/// How the terms of a [`LineHunk`] merge.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum HunkKind {
    /// Every term holds the same text.
    Same,
    /// The terms differ, and merge trivially.
    Merged,
    /// The terms differ, and do not merge.
    Conflict,
}

/// The line hunks of the merge of `contents`, as jj's line merge splits it.
fn line_hunks(contents: &Merge<BString>, same_change: SameChange) -> Vec<LineHunk> {
    let num_removes = contents.removes().len();
    let term = |input: usize| {
        if input < num_removes {
            2 * input + 1
        } else {
            2 * (input - num_removes)
        }
    };
    let diff = ContentDiff::by_line(contents.removes().chain(contents.adds()));
    diff.hunk_ranges()
        .map(|hunk| {
            let mut ranges = vec![Range::default(); contents.as_slice().len()];
            for (input, range) in hunk.ranges.iter().enumerate() {
                ranges[term(input)] = range.clone();
            }
            let texts = Merge::from_vec(
                zip(contents.as_slice(), &ranges)
                    .map(|(content, range)| &content.as_slice()[range.clone()])
                    .collect_vec(),
            );
            let (kind, text) = match (hunk.kind, texts.resolve_trivial(same_change)) {
                (DiffHunkKind::Matching, _) => (HunkKind::Same, texts.first().to_vec()),
                (_, Some(text)) => (HunkKind::Merged, text.to_vec()),
                (_, None) => (HunkKind::Conflict, vec![]),
            };
            LineHunk { kind, text, ranges }
        })
        .collect()
}

/// `hunks` with each run of hunks that do not conflict made one, and the runs
/// that merge to no text left out, as [`files::merge_hunks`] gives them.
fn coalesced(hunks: Vec<LineHunk>) -> Vec<LineHunk> {
    let mut result: Vec<LineHunk> = vec![];
    for hunk in hunks {
        let resolves = hunk.kind != HunkKind::Conflict;
        match result.last_mut() {
            Some(last) if resolves && last.kind == HunkKind::Merged => {
                last.text.extend(hunk.text);
                for (range, next) in zip(&mut last.ranges, hunk.ranges) {
                    range.end = next.end;
                }
            }
            _ if resolves => result.push(LineHunk {
                kind: HunkKind::Merged,
                ..hunk
            }),
            _ => result.push(hunk),
        }
    }
    result.retain(|hunk| hunk.kind == HunkKind::Conflict || !hunk.text.is_empty());
    result
}

/// A line hunk of the merge of V's terms, with the range of VB that it takes,
/// and its block where it conflicts.
struct Segment<'a> {
    range: Range<usize>,
    hunk: &'a LineHunk,
    block: Option<&'a Block<'a>>,
}

/// Changes to VB that rewrite one range of it; where `expand` is set, the
/// range is the whole of the hunks `segments`, which take the changed text in
/// each term.
struct Group<'a> {
    span: Range<usize>,
    segments: Range<usize>,
    expand: bool,
    edits: Vec<&'a Edit>,
}

/// The texts of V's terms, `contents`, with `edits`, changes to `vb`, their
/// VB, made by region. A change to lines where V's terms are the same is made
/// to each term. A change that touches lines where V's terms differ but merge
/// writes the changed text of the hunks it touches into each term, in place
/// of the term's own text there. A change inside a side of a block is made to
/// the term of V whose side it is. An insertion at a block's edge is outside
/// the block, and goes into each term. A change that touches a marker line,
/// crosses a block's edge or two sides, or whose text holds a marker line, is
/// not made. `None` where `vb` is not the VB of `contents`.
fn edited_terms(
    vb: &[u8],
    contents: &Merge<BString>,
    edits: &[Edit],
    same_change: SameChange,
) -> Option<Vec<Vec<u8>>> {
    let blocks = blocks(vb)?;
    let hunks = line_hunks(contents, same_change);
    let mut segments = vec![];
    let mut next_blocks = blocks.iter();
    let mut cursor = 0;
    for hunk in &hunks {
        let block = if hunk.kind == HunkKind::Conflict {
            let block = next_blocks.next()?;
            let adds = zip(contents.adds(), hunk.ranges.iter().step_by(2));
            let sides_hold_adds = block.start == cursor
                && block.sides.len() == contents.adds().len()
                && zip(&block.sides, adds)
                    .all(|(side, (add, range))| *side == &add.as_slice()[range.clone()]);
            if !sides_hold_adds {
                return None;
            }
            Some(block)
        } else {
            if vb.get(cursor..cursor + hunk.text.len())? != hunk.text.as_slice() {
                return None;
            }
            None
        };
        let end = block.map_or(cursor + hunk.text.len(), |block| block.end);
        segments.push(Segment {
            range: cursor..end,
            hunk,
            block,
        });
        cursor = end;
    }
    if cursor != vb.len() || next_blocks.next().is_some() {
        return None;
    }

    let num_terms = contents.as_slice().len();
    let mut term_edits: Vec<Vec<Edit>> = vec![vec![]; num_terms];
    let mut groups: Vec<Group<'_>> = vec![];
    let mut edits = edits.iter().collect_vec();
    edits.sort_by_key(|edit| (edit.range.start, edit.range.end));
    for edit in edits {
        if holds_marker_line(&edit.text) {
            continue;
        }
        let (start, end) = (edit.range.start, edit.range.end);
        let touched = segments
            .iter()
            .enumerate()
            .filter(|(_, segment)| {
                let range = &segment.range;
                if start < end {
                    range.start < end && start < range.end
                } else if segment.block.is_some() {
                    range.start < start && start < range.end
                } else {
                    range.start <= start && start <= range.end
                }
            })
            .map(|(index, _)| index)
            .collect_vec();
        if let Some(&index) = touched
            .iter()
            .find(|&&index| segments[index].block.is_some())
        {
            // A change inside a block goes to the term whose side it is in.
            let segment = &segments[index];
            let Some(block) = segment.block.filter(|_| touched.len() == 1) else {
                continue;
            };
            let Some(side) = block
                .side_ranges
                .iter()
                .position(|side| side.start <= start && end <= side.end)
            else {
                continue;
            };
            let term = 2 * side;
            let shift = segment.hunk.ranges[term].start;
            let side_start = block.side_ranges[side].start;
            term_edits[term].push(Edit {
                range: shift + start - side_start..shift + end - side_start,
                text: edit.text.clone(),
            });
            continue;
        }
        let expand = touched.iter().any(|&index| {
            let segment = &segments[index];
            segment.hunk.kind == HunkKind::Merged
                && (start < end || (segment.range.start < start && start < segment.range.end))
        });
        let (first, last) = (
            touched.first().copied().unwrap_or_default(),
            touched.last().copied().unwrap_or_default(),
        );
        let span = if expand {
            segments[first].range.start..segments[last].range.end
        } else {
            edit.range.clone()
        };
        match groups.last_mut() {
            Some(group) if span.start < group.span.end => {
                group.span = group.span.start.min(span.start)..group.span.end.max(span.end);
                group.segments = group.segments.start.min(first)..group.segments.end.max(last + 1);
                group.expand |= expand;
                group.edits.push(edit);
            }
            _ => groups.push(Group {
                span,
                segments: first..last + 1,
                expand,
                edits: vec![edit],
            }),
        }
    }

    for group in &groups {
        let (first, last) = (
            &segments[group.segments.start],
            &segments[group.segments.end - 1],
        );
        let span = if group.expand {
            first.range.start..last.range.end
        } else {
            group.span.clone()
        };
        let mut text = vec![];
        let mut at = span.start;
        for edit in &group.edits {
            if edit.range.start < at {
                return None;
            }
            text.extend_from_slice(&vb[at..edit.range.start]);
            text.extend_from_slice(&edit.text);
            at = edit.range.end;
        }
        text.extend_from_slice(&vb[at..span.end]);
        for (term, edits) in term_edits.iter_mut().enumerate() {
            let range = if group.expand {
                first.hunk.ranges[term].start..last.hunk.ranges[term].end
            } else {
                term_offset(&segments, span.start, term)..term_offset(&segments, span.end, term)
            };
            edits.push(Edit {
                range,
                text: text.clone(),
            });
        }
    }

    let mut new_contents = vec![];
    for (content, mut edits) in zip(contents.iter(), term_edits) {
        let content: &[u8] = content;
        edits.sort_by_key(|edit| (edit.range.start, edit.range.end));
        let mut text = vec![];
        let mut at = 0;
        for edit in edits {
            if edit.range.start < at {
                return None;
            }
            text.extend_from_slice(&content[at..edit.range.start]);
            text.extend(edit.text);
            at = edit.range.end;
        }
        text.extend_from_slice(&content[at..]);
        new_contents.push(text);
    }
    Some(new_contents)
}

/// The offset in the term at `term` that `point` of VB stands for, where
/// `point` is inside a hunk whose terms are the same, or at a hunk's edge.
fn term_offset(segments: &[Segment<'_>], point: usize, term: usize) -> usize {
    for segment in segments {
        let (range, term_range) = (&segment.range, &segment.hunk.ranges[term]);
        if range.start <= point && point <= range.end {
            if segment.hunk.kind == HunkKind::Same {
                return term_range.start + point - range.start;
            } else if point == range.start {
                return term_range.start;
            } else if point == range.end {
                return term_range.end;
            }
        }
    }
    segments
        .last()
        .map_or(0, |segment| segment.hunk.ranges[term].end)
}

/// The texts of V's terms, `contents`, with `edits`, changes to M, `m`, made:
/// an insertion before or after M's block goes into each term at its start or
/// its end; no other change is made, as M holds none of V's text.
fn edited_m_terms(m: &[u8], contents: &Merge<BString>, edits: &[Edit]) -> Vec<Vec<u8>> {
    let (mut before, mut after) = (vec![], vec![]);
    for edit in edits {
        if !edit.range.is_empty() || holds_marker_line(&edit.text) {
            continue;
        }
        if edit.range.start == 0 {
            before.extend_from_slice(&edit.text);
        } else if edit.range.start == m.len() {
            after.extend_from_slice(&edit.text);
        }
    }
    contents
        .iter()
        .map(|content| [before.as_slice(), content.as_slice(), after.as_slice()].concat())
        .collect()
}

/// Writes to the store the VB or M of `base`, V after `simplify()`, whose
/// contents are `base_contents` if it is a conflict of text files, with marker
/// lines that are none of `parent_lines`.
async fn write_virtual_base(
    store: &Store,
    path: &RepoPath,
    base: &MergedTreeValue,
    base_contents: Option<Merge<BString>>,
    parent_lines: &HashSet<&[u8]>,
) -> BackendResult<TreeValue> {
    let hunks = match base_contents {
        Some(base_contents) => text_hunks(&base_contents, store),
        None => {
            let names = base
                .iter()
                .map(|term| term_name(term.as_ref()).into_bytes());
            vec![VirtualBaseHunk::Block(names.collect())]
        }
    };
    let content = virtual_base_content(hunks, base, parent_lines);
    let id = store.write_file(path, &mut content.as_slice()).await?;
    // The executable bit and the copy id depend on `base` alone, not on the
    // order of the parents.
    let executable = base
        .to_executable_merge()
        .and_then(|bits| bits.resolve_trivial(SameChange::Accept).copied().flatten())
        .unwrap_or(false);
    let copy_id = base
        .to_copy_id_merge()
        .and_then(|ids| ids.resolve_trivial(SameChange::Accept).cloned().flatten())
        .unwrap_or_else(CopyId::placeholder);
    Ok(TreeValue::File {
        id,
        executable,
        copy_id,
    })
}

/// The lines of `contents` that start like a marker line: only those can
/// equal one.
fn marker_like_lines<'a>(contents: impl IntoIterator<Item = &'a [u8]>) -> HashSet<&'a [u8]> {
    contents
        .into_iter()
        .flat_map(|content| content.lines())
        .filter(|line| {
            [OPEN, SEPARATOR, CLOSE]
                .iter()
                .any(|prefix| line.starts_with(prefix.as_bytes()))
        })
        .collect()
}

/// A hunk of VB or M: text that V's terms merge to, or the sides of a block.
enum VirtualBaseHunk {
    Resolved(Vec<u8>),
    Block(Vec<Vec<u8>>),
}

/// The hunks of the VB of `contents`, the terms of a conflict of text files,
/// merged with the store's merge options.
fn text_hunks(contents: &Merge<BString>, store: &Store) -> Vec<VirtualBaseHunk> {
    match files::merge_hunks(contents, store.merge_options()) {
        MergeResult::Resolved(content) => vec![VirtualBaseHunk::Resolved(content.into())],
        MergeResult::Conflict(hunks) => hunks
            .into_iter()
            .map(|hunk| match hunk.into_resolved() {
                Ok(content) => VirtualBaseHunk::Resolved(content.into()),
                Err(hunk) => {
                    VirtualBaseHunk::Block(hunk.adds().map(|side| side.to_vec()).collect())
                }
            })
            .collect(),
    }
}

/// The content of VB or M of `terms`, V's terms, made of `hunks`: the text of
/// each resolved hunk, and each block's sides, each ending with a newline
/// unless it is empty, between marker lines at the lowest salt at which none
/// is one of `parent_lines`.
fn virtual_base_content(
    hunks: Vec<VirtualBaseHunk>,
    terms: &MergedTreeValue,
    parent_lines: &HashSet<&[u8]>,
) -> Vec<u8> {
    let names = terms.iter().map(|term| term_name(term.as_ref())).join(" ");
    // Each block with the text before it, and the text after the last block.
    let mut blocks: Vec<(Vec<u8>, Vec<Vec<u8>>)> = vec![];
    let mut text = vec![];
    for hunk in hunks {
        match hunk {
            VirtualBaseHunk::Resolved(content) => text.extend(content),
            VirtualBaseHunk::Block(sides) => {
                let mut before = mem::take(&mut text);
                end_line(&mut before);
                let sides = sides
                    .into_iter()
                    .map(|mut side| {
                        end_line(&mut side);
                        side
                    })
                    .collect();
                blocks.push((before, sides));
            }
        }
    }
    let after = text;
    let last = blocks.len().saturating_sub(1);
    let hashers = blocks
        .iter()
        .enumerate()
        .map(|(ordinal, (before, sides))| {
            let after: &[u8] = if ordinal == last { &after } else { &[] };
            block_hasher(ordinal, names.as_bytes(), before, sides, after)
        })
        .collect_vec();
    // Each salt gives other opening lines, so the search ends after at most one
    // salt more than there are `parent_lines`.
    let markers = (0..)
        .map(|salt| {
            hashers
                .iter()
                .map(|hasher| marker_lines(&salted_hex(hasher, salt), salt, &names))
                .collect_vec()
        })
        .find(|markers| {
            markers
                .iter()
                .flatten()
                .all(|line| !parent_lines.contains(line.as_bytes()))
        })
        .expect("some salt gives marker lines that no parent holds");
    let mut content = vec![];
    for ((before, sides), [open, separator, close]) in zip(blocks, markers) {
        content.extend(before);
        push_line(&mut content, &open);
        for (index, side) in sides.into_iter().enumerate() {
            if index > 0 {
                push_line(&mut content, &separator);
            }
            content.extend(side);
        }
        push_line(&mut content, &close);
    }
    content.extend(after);
    content
}

/// Appends a newline to `text` unless it is empty or ends with one.
fn end_line(text: &mut Vec<u8>) {
    if text.last().is_some_and(|&byte| byte != b'\n') {
        text.push(b'\n');
    }
}

/// Appends `line` and a newline to `content`.
fn push_line(content: &mut Vec<u8>, line: &str) {
    content.extend_from_slice(line.as_bytes());
    content.push(b'\n');
}

/// The opening, separator and closing marker lines of a block whose hex is
/// `hex`, at `salt`, that names the terms `names`, without their newlines.
fn marker_lines(hex: &str, salt: u64, names: &str) -> [String; 3] {
    [
        format!("{OPEN}{hex} {salt} {names}"),
        format!("{SEPARATOR}{hex}"),
        format!("{CLOSE}{hex}"),
    ]
}

/// The hash of a block before its salt: the tag, the block's ordinal, the
/// names of the terms, the text before the block, its sides, and the text
/// after it, each with its length first.
fn block_hasher(
    ordinal: usize,
    names: &[u8],
    before: &[u8],
    sides: &[impl AsRef<[u8]>],
    after: &[u8],
) -> Blake2b512 {
    let mut hasher = Blake2b512::new();
    hash_bytes(&mut hasher, TAG.as_bytes());
    Digest::update(&mut hasher, (ordinal as u64).to_le_bytes());
    hash_bytes(&mut hasher, names);
    hash_bytes(&mut hasher, before);
    Digest::update(&mut hasher, (sides.len() as u64).to_le_bytes());
    for side in sides {
        hash_bytes(&mut hasher, side.as_ref());
    }
    hash_bytes(&mut hasher, after);
    hasher
}

/// Hashes `bytes` into `hasher`, with their length first.
fn hash_bytes(hasher: &mut Blake2b512, bytes: &[u8]) {
    Digest::update(hasher, (bytes.len() as u64).to_le_bytes());
    Digest::update(hasher, bytes);
}

/// 32 hex digits of the hash of `hasher` with `salt` hashed last.
fn salted_hex(hasher: &Blake2b512, salt: u64) -> String {
    let mut hasher = hasher.clone();
    Digest::update(&mut hasher, salt.to_le_bytes());
    hex_util::encode_hex(&hasher.finalize()[..16])
}

/// The name of a term of V in an opening marker line and in a side of M:
/// `absent`; `file:<id>`, then `:x` if the file is executable and
/// `:copy-<id>` if its copy id is not the placeholder; `symlink:<id>`;
/// `tree:<id>`; or `submodule:<id>`.
fn term_name(term: Option<&TreeValue>) -> String {
    match term {
        None => "absent".to_owned(),
        Some(TreeValue::File {
            id,
            executable,
            copy_id,
        }) => {
            let mut name = format!("file:{}", id.hex());
            if *executable {
                name.push_str(":x");
            }
            if !copy_id.as_bytes().is_empty() {
                write!(name, ":copy-{}", copy_id.hex()).unwrap();
            }
            name
        }
        Some(TreeValue::Symlink(id)) => format!("symlink:{}", id.hex()),
        Some(TreeValue::Tree(id)) => format!("tree:{}", id.hex()),
        Some(TreeValue::GitSubmodule(id)) => format!("submodule:{}", id.hex()),
    }
}

/// The term that `name` names, if [`term_name`] gives `name` for it.
fn parse_term_name(name: &str) -> Option<Option<TreeValue>> {
    let term = if name == "absent" {
        None
    } else {
        let mut parts = name.split(':');
        let kind = parts.next()?;
        let hex = parts.next()?;
        Some(match kind {
            "file" => {
                let id = FileId::try_from_hex(hex)?;
                let mut executable = false;
                let mut copy_id = CopyId::placeholder();
                for part in parts {
                    if part == "x" {
                        executable = true;
                    } else {
                        copy_id = CopyId::try_from_hex(part.strip_prefix("copy-")?)?;
                    }
                }
                TreeValue::File {
                    id,
                    executable,
                    copy_id,
                }
            }
            "symlink" => TreeValue::Symlink(SymlinkId::try_from_hex(hex)?),
            "tree" => TreeValue::Tree(TreeId::try_from_hex(hex)?),
            "submodule" => TreeValue::GitSubmodule(CommitId::try_from_hex(hex)?),
            _ => return None,
        })
    };
    // Each term has one name, so that the hashes cover the terms themselves.
    (term_name(term.as_ref()) == name).then_some(term)
}

/// A block of VB or M: the offsets of its opening line and of the end of its
/// closing line, the hex, the salt and the names of the terms its opening line
/// carries, and its sides, each with its newlines, and their offsets.
struct Block<'a> {
    start: usize,
    end: usize,
    hex: &'a [u8],
    salt: u64,
    terms: &'a [u8],
    sides: Vec<&'a [u8]>,
    side_ranges: Vec<Range<usize>>,
}

/// The blocks of `content`, in order, or `None` if a block does not close. A
/// block opens at an opening marker line, and each of its sides ends at the
/// next line that is [`SEPARATOR`] or [`CLOSE`] and the same hex.
fn blocks(content: &[u8]) -> Option<Vec<Block<'_>>> {
    let mut blocks = vec![];
    // The block being read, and the offset at which its current side starts.
    let mut current: Option<(Block<'_>, usize)> = None;
    let mut offset = 0;
    for line in content.lines_with_terminator() {
        let line_start = offset;
        offset += line.len();
        match current.take() {
            None => {
                if let Some((hex, salt, terms)) = parse_opening_line(line) {
                    let block = Block {
                        start: line_start,
                        end: offset,
                        hex,
                        salt,
                        terms,
                        sides: vec![],
                        side_ranges: vec![],
                    };
                    current = Some((block, offset));
                }
            }
            Some((mut block, side_start)) => {
                let is_separator = marker_hex(line, SEPARATOR) == Some(block.hex);
                let is_close = marker_hex(line, CLOSE) == Some(block.hex);
                if is_separator || is_close {
                    block.sides.push(&content[side_start..line_start]);
                    block.side_ranges.push(side_start..line_start);
                }
                if is_close {
                    block.end = offset;
                    blocks.push(block);
                } else if is_separator {
                    current = Some((block, offset));
                } else {
                    current = Some((block, side_start));
                }
            }
        }
    }
    current.is_none().then_some(blocks)
}

/// The hex, the salt and the names of the terms of `line` if it is an opening
/// marker line: [`OPEN`], 32 lowercase hex digits, a salt in decimal, the
/// names, and a newline.
fn parse_opening_line(line: &[u8]) -> Option<(&[u8], u64, &[u8])> {
    let rest = line.strip_prefix(OPEN.as_bytes())?.strip_suffix(b"\n")?;
    let (hex, rest) = rest.split_once_str(" ")?;
    let (salt, terms) = rest.split_once_str(" ")?;
    let salt_text = salt.to_str().ok()?;
    let salt: u64 = salt_text.parse().ok()?;
    let canonical = salt.to_string() == salt_text;
    (is_hex(hex) && canonical && !terms.is_empty()).then_some((hex, salt, terms))
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

/// Whether `terms`, the terms of V that a VB or M names, can be V's: the store
/// holds each of them, and no side of V is itself VB or M, which no merge
/// writes as a side. A submodule's commit is not an object of the store, so it
/// counts as held. A chain of files that each name the one before as a side
/// is thus refused at its first link, so that expanding it takes one step.
async fn holds_terms(
    store: &Store,
    path: &RepoPath,
    terms: &MergedTreeValue,
) -> BackendResult<bool> {
    for (index, term) in terms.iter().enumerate() {
        let read = match term {
            None => continue,
            Some(TreeValue::File { id, .. }) if index.is_multiple_of(2) => {
                match read_file(store, path, id).await {
                    Ok(content) if virtual_base_terms(&content).is_some() => return Ok(false),
                    result => result.map(drop),
                }
            }
            Some(TreeValue::File { id, .. }) => store.read_file(path, id).await.map(drop),
            Some(TreeValue::Symlink(id)) => store.read_symlink(path, id).await.map(drop),
            Some(TreeValue::Tree(id)) => store.backend().read_tree(path, id).await.map(drop),
            Some(TreeValue::GitSubmodule(_)) => Ok(()),
        };
        match read {
            Ok(()) => {}
            Err(BackendError::ObjectNotFound { .. }) => return Ok(false),
            Err(err) => return Err(err),
        }
    }
    Ok(true)
}

/// The contents of the terms of `base` if `base` is a conflict of text files:
/// each add is a regular file whose first 8,000 bytes hold no NUL byte, which
/// is git's test for a binary file, and each remove is such a file or absent,
/// which reads as an empty file.
async fn text_file_contents(
    store: &Store,
    path: &RepoPath,
    base: &MergedTreeValue,
) -> BackendResult<Option<Merge<BString>>> {
    let Some(file_ids) = base.to_file_merge() else {
        return Ok(None);
    };
    if file_ids.adds().any(Option::is_none) {
        return Ok(None);
    }
    let contents = file_contents(store, path, &file_ids).await?;
    let is_text = contents.iter().all(|content| !is_binary_content(content));
    Ok(is_text.then_some(contents))
}

/// The contents of the file terms of `value`, and an empty content for each
/// other term.
async fn file_term_contents(
    store: &Store,
    path: &RepoPath,
    value: &MergedTreeValue,
) -> BackendResult<Merge<BString>> {
    let file_ids = value.map(|term| match term {
        Some(TreeValue::File { id, .. }) => Some(id.clone()),
        _ => None,
    });
    file_contents(store, path, &file_ids).await
}

/// The contents of the files `file_ids`, and an empty content for each absent
/// term.
async fn file_contents(
    store: &Store,
    path: &RepoPath,
    file_ids: &Merge<Option<FileId>>,
) -> BackendResult<Merge<BString>> {
    file_ids
        .try_map_async(async |id| match id {
            Some(id) => Ok(BString::new(read_file(store, path, id).await?)),
            None => Ok(BString::default()),
        })
        .await
}

/// The content of the file `id` at `path`.
async fn read_file(store: &Store, path: &RepoPath, id: &FileId) -> BackendResult<Vec<u8>> {
    let mut content = vec![];
    store
        .read_file(path, id)
        .await?
        .read_to_end(&mut content)
        .await
        .map_err(|err| BackendError::ReadFile {
            path: path.to_owned(),
            id: id.clone(),
            source: err.into(),
        })?;
    Ok(content)
}
