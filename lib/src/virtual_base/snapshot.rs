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

//! A snapshot of a conflict whose base is VB or M: the person's changes to the
//! file that jj materialized, made to the terms of V that the base stands for
//! (see [`rewrite_virtual_base`]).

use std::iter;
use std::iter::zip;
use std::ops::Range;

use bstr::BString;
use bstr::ByteSlice as _;
use itertools::Itertools as _;

use super::Block;
use super::NamedTerms;
use super::blocks;
use super::file_term_contents;
use super::holds_terms;
use super::marker_like_lines;
use super::named_terms;
use super::recorded_hunk_level;
use super::simplified_with_labels;
use super::text_file_contents;
use super::write_virtual_base;
use crate::backend::BackendResult;
use crate::backend::CopyId;
use crate::backend::FileId;
use crate::backend::MergedTreeValue;
use crate::backend::TreeValue;
use crate::conflict_labels::ConflictLabels;
use crate::diff::ContentDiff;
use crate::diff::DiffHunkKind;
use crate::diff::DiffHunkRange;
use crate::files;
use crate::files::RangedMergeHunk;
use crate::merge::Merge;
use crate::repo_path::RepoPath;
use crate::store::Store;
use crate::tree_merge::MergeOptions;

/// The file that takes the place of the base at `index` among the removes of
/// `old_contents`, the contents of the terms of a conflict at `path` whose
/// base `old_id` is VB or M, and whose hunks, as jj materializes them at the
/// store's hunk level, are `old_hunks`, once a parse of the conflict's
/// materialized markers (a snapshot of the working copy, or a merge tool's
/// output) gave the hunks `new_hunks`; `None` where that base is not VB or M.
///
/// The person changed the file that jj materialized, not the terms. A line
/// diff of the old hunks against the new ones, each conflict one line, lines
/// them up (see [`output_edits`]). The merge that wrote VB split V's terms,
/// and the file it materialized, at its own hunk level, which VB records, so
/// the old hunks are split at that level, whatever `merge.hunk-level` is at
/// the snapshot. VB's structure is told by position, by the blocks of the old
/// base, never by a line's form: a line the person writes is text, whatever
/// it looks like. A conflict that the person replaced with text, or deleted,
/// and a change to a hunk that resolves and covers whole blocks of VB,
/// replace the region of VB that the hunks they touch take with the new text
/// there, which then goes into every term of V there, as jj's snapshot
/// writes a resolved hunk into every term. Each other change to a hunk that
/// resolves is made to VB at the VB lines that the changed lines are; VB
/// keeps every other line, its marker lines included.
/// An insertion next to VB lines that the materialized hunk does not hold
/// goes outside the block they belong to. In a conflict the person kept, a
/// change to VB's
/// lines is taken as it is where a side of the conflict holds each line it
/// changes; a change to lines that only VB holds, which jj shows only in the
/// conflict's base, is not made. The changes are then made to V's terms by
/// region, laid out at the hunk level that VB records (see [`edited_terms`]);
/// for M, an insertion before or after its block goes into each file term,
/// binary ones too, an absent one read as empty, and a term that is not a file
/// is left as it is (see [`edited_m_terms`]). The file is the VB or M of the
/// edited terms, or their text where they resolve. Where the new hunks cannot
/// be lined up with the old ones, or no change can be made, the base stays
/// `old_id`: a snapshot never writes a plain file that holds VB's or M's
/// marker lines.
pub(crate) async fn rewrite_virtual_base(
    store: &Store,
    path: &RepoPath,
    old_id: &FileId,
    old_contents: &Merge<BString>,
    old_hunks: &[RangedMergeHunk],
    index: usize,
    new_hunks: &[Merge<BString>],
) -> BackendResult<Option<FileId>> {
    let Some(old) = old_contents.get_remove(index) else {
        return Ok(None);
    };
    let old = old.as_slice();
    let Some(blocks) = blocks(old) else {
        return Ok(None);
    };
    let Some(NamedTerms { terms, labels }) = named_terms(&blocks) else {
        return Ok(None);
    };
    if !holds_terms(store, path, &terms).await? {
        return Ok(None);
    }
    // The merge that wrote VB split V's terms into hunks, and the file it
    // materialized for the conflict, at its own hunk level, which VB records:
    // the snapshot lays VB's regions out at that level, whatever the store's
    // is now, and lines the file up with the hunks at that level. A checkout
    // under another `merge.hunk-level` writes the file at the store's level
    // instead, so the hunks at that level are tried where the recorded ones
    // do not line up with it.
    let options = MergeOptions {
        hunk_level: recorded_hunk_level(&blocks),
        same_change: store.merge_options().same_change,
    };
    let recorded_hunks = (options.hunk_level != store.merge_options().hunk_level)
        .then(|| files::ranged_merge_hunks(old_contents, &options));
    let mut hunk_lists = recorded_hunks.as_deref().into_iter().chain([old_hunks]);
    let edits =
        hunk_lists.find_map(|hunks| output_edits(old_contents, hunks, index, &blocks, new_hunks));
    let Some(edits) = edits else {
        return Ok(Some(old_id.clone()));
    };
    let new_contents = match text_file_contents(store, path, &terms).await? {
        Some(term_contents) => edited_terms(old, &blocks, &term_contents, &edits, &options)
            .map(|new_contents| (term_contents, new_contents)),
        None => {
            let term_contents = file_term_contents(store, path, &terms).await?;
            let new_contents = edited_m_terms(old, &terms, &term_contents, &edits);
            Some((term_contents, new_contents))
        }
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
    let (base, labels) = simplified_with_labels(
        &Merge::from_vec(new_terms),
        &ConflictLabels::from_merge(labels),
    );
    if base == terms {
        return Ok(Some(old_id.clone()));
    }
    if let Some(resolved) = base.as_resolved() {
        // Every term that an edit changes is a file, and VB's terms are all
        // files or absent, so where they resolve it is to a file; any other
        // value keeps the old base.
        return Ok(match resolved {
            Some(TreeValue::File { id, .. }) => Some(id.clone()),
            _ => Some(old_id.clone()),
        });
    }
    // The base is the VB of the edited terms where they are a conflict of text
    // files, M otherwise, whatever the old base was: where an edit makes every
    // term of M's V text (an absent term written as the person's text), it is
    // the VB of the edited terms. Its marker lines are none of the lines that
    // the person left.
    let base_contents = text_file_contents(store, path, &base).await?;
    let new_lines = new_hunks
        .iter()
        .flat_map(|hunk| hunk.iter())
        .map(|text| text.as_slice());
    let parent_lines = marker_like_lines(new_lines);
    let id = write_virtual_base(store, path, &base, &labels, base_contents, &parent_lines).await?;
    Ok(Some(id))
}

/// A change to a text: the range it replaces, and the text that takes its
/// place.
#[derive(Clone, Debug)]
struct Edit {
    range: Range<usize>,
    text: Vec<u8>,
}

/// `span` of `text` with `edits`, changes to it in order, made; `None` where
/// two of them overlap.
fn apply_edits<'a>(
    text: &[u8],
    span: Range<usize>,
    edits: impl IntoIterator<Item = &'a Edit>,
) -> Option<Vec<u8>> {
    let mut result = vec![];
    let mut at = span.start;
    for edit in edits {
        if edit.range.start < at {
            return None;
        }
        result.extend_from_slice(&text[at..edit.range.start]);
        result.extend_from_slice(&edit.text);
        at = edit.range.end;
    }
    result.extend_from_slice(&text[at..span.end]);
    Some(result)
}

/// A token of the file that jj materialized for a conflict, as
/// [`output_edits`] lines it up with the file the person left: a line of a
/// hunk that resolves, with its range in the hunk's text, or a hunk that
/// conflicts, whole.
#[derive(Clone, Debug)]
enum Token {
    Line { hunk: usize, range: Range<usize> },
    Conflict { hunk: usize },
}

impl Token {
    /// The index of the hunk that the token is in.
    fn hunk(&self) -> usize {
        match self {
            Self::Line { hunk, .. } | Self::Conflict { hunk } => *hunk,
        }
    }
}

/// The tokens of a conflict whose hunks resolve to `hunks`, `None` for a hunk
/// that conflicts: each line of a hunk that resolves, and each hunk that
/// conflicts. With them, a text with one line for each token, for a line diff
/// to line them up: a line marked as one, with or without its final newline,
/// and every conflict the same line; and the offsets in that text at which
/// the tokens start, with its length last.
fn tokens<'a>(
    hunks: impl IntoIterator<Item = Option<&'a [u8]>>,
) -> (Vec<Token>, Vec<u8>, Vec<usize>) {
    let mut tokens = vec![];
    let mut text = vec![];
    let mut offsets = vec![];
    for (hunk, resolved) in hunks.into_iter().enumerate() {
        let Some(resolved) = resolved else {
            offsets.push(text.len());
            text.extend_from_slice(b"C\n");
            tokens.push(Token::Conflict { hunk });
            continue;
        };
        let mut start = 0;
        for line in resolved.lines_with_terminator() {
            offsets.push(text.len());
            if line.ends_with(b"\n") {
                text.push(b'L');
                text.extend_from_slice(line);
            } else {
                text.push(b'N');
                text.extend_from_slice(line);
                text.push(b'\n');
            }
            tokens.push(Token::Line {
                hunk,
                range: start..start + line.len(),
            });
            start += line.len();
        }
    }
    offsets.push(text.len());
    (tokens, text, offsets)
}

/// A hunk of the line diff of the old tokens against the new ones: the range
/// of old tokens and of new tokens it takes, and whether they are the same.
struct TokenHunk {
    old: Range<usize>,
    new: Range<usize>,
    matching: bool,
}

impl TokenHunk {
    /// Whether this hunk, a change, reaches into `span`, a range of old
    /// tokens: it changes a token of it, or inserts at a point in it or at
    /// its edge.
    fn reaches(&self, span: &Range<usize>) -> bool {
        if self.old.is_empty() {
            span.start <= self.old.start && self.old.start <= span.end
        } else {
            self.old.start < span.end && span.start < self.old.end
        }
    }
}

/// The changes to the base at `index` among the removes of `old_contents`, VB
/// or M, whose blocks are `blocks`, that the person's changes to the hunks
/// `old_hunks` of `old_contents`, as jj materialized them, stand for, where
/// `new_hunks` are the hunks the person left (see [`rewrite_virtual_base`]).
/// A line diff of the old hunks against the new ones, each line of a hunk
/// that resolves one token and each conflict one, lines them up: a conflict
/// the person kept lines up with a conflict, and one the person replaced or
/// deleted lines up with the text left in its place. A change that touches a
/// conflict, or a hunk that resolves over whole blocks, rewrites the old hunks
/// it touches whole, with every other change that reaches into them. `None`
/// where the new hunks hold a conflict that lines up with none of the old
/// ones.
fn output_edits(
    old_contents: &Merge<BString>,
    old_hunks: &[RangedMergeHunk],
    index: usize,
    blocks: &[Block<'_>],
    new_hunks: &[Merge<BString>],
) -> Option<Vec<Edit>> {
    let vb = old_contents.get_remove(index)?.as_slice();
    let regions = old_hunks
        .iter()
        .map(|hunk| hunk.ranges.get_remove(index).cloned())
        .collect::<Option<Vec<_>>>()?;
    let old_texts = old_hunks
        .iter()
        .map(|hunk| hunk.resolved.as_ref().map(|text| text.as_slice()))
        .collect_vec();
    let new_texts = new_hunks
        .iter()
        .map(|hunk| hunk.as_resolved().map(|text| text.as_slice()))
        .collect_vec();
    let (old_tokens, old_lines, old_offsets) = tokens(old_texts.iter().copied());
    let (new_tokens, new_lines, new_offsets) = tokens(new_texts.iter().copied());
    let token_index = |offsets: &[usize], offset: usize| offsets.binary_search(&offset).ok();
    let token_hunks = ContentDiff::by_line([old_lines.as_slice(), new_lines.as_slice()])
        .hunk_ranges()
        .map(|hunk| {
            let (old, new) = (&hunk.ranges[0], &hunk.ranges[1]);
            Some(TokenHunk {
                old: token_index(&old_offsets, old.start)?..token_index(&old_offsets, old.end)?,
                new: token_index(&new_offsets, new.start)?..token_index(&new_offsets, new.end)?,
                matching: matches!(hunk.kind, DiffHunkKind::Matching),
            })
        })
        .collect::<Option<Vec<_>>>()?;
    // Where each boundary between old tokens is among the new tokens, before
    // and after any insertion there.
    let mut before_insertion = vec![None; old_tokens.len() + 1];
    let mut after_insertion = vec![None; old_tokens.len() + 1];
    for hunk in &token_hunks {
        if hunk.matching || !hunk.old.is_empty() {
            let old = hunk.old.clone();
            let len = if hunk.matching { old.len() } else { 0 };
            for offset in 0..=len {
                before_insertion[old.start + offset].get_or_insert(hunk.new.start + offset);
                after_insertion[old.start + offset] = Some(hunk.new.start + offset);
            }
            before_insertion[old.end].get_or_insert(hunk.new.end);
            after_insertion[old.end] = Some(hunk.new.end);
        } else {
            before_insertion[hunk.old.start].get_or_insert(hunk.new.start);
            after_insertion[hunk.old.start] = Some(hunk.new.end);
        }
    }
    let new_text = |range: Range<usize>| -> Option<Vec<u8>> {
        new_tokens[range]
            .iter()
            .map(|token| match token {
                Token::Line { hunk, range } => Some(&new_texts[*hunk]?[range.clone()]),
                Token::Conflict { .. } => None,
            })
            .collect::<Option<Vec<_>>>()
            .map(|lines| lines.concat())
    };
    let hunk_tokens = (0..old_hunks.len())
        .map(|hunk| {
            let start = old_tokens.partition_point(|token| token.hunk() < hunk);
            let end = old_tokens.partition_point(|token| token.hunk() <= hunk);
            start..end
        })
        .collect_vec();
    let rewrites_whole = |token: &Token| match token {
        Token::Conflict { .. } => true,
        Token::Line { hunk, .. } => covers_whole_blocks(blocks, &regions[*hunk]),
    };
    let mut edits = vec![];
    // The changes that rewrite whole old hunks, each with every change that
    // reaches into the hunks it rewrites.
    let mut in_span = vec![false; token_hunks.len()];
    for (position, hunk) in token_hunks.iter().enumerate() {
        let rewrites = old_tokens[hunk.old.clone()].iter().any(rewrites_whole);
        if hunk.matching || in_span[position] || !rewrites {
            continue;
        }
        let mut span = hunk.old.clone();
        loop {
            let first = old_tokens[span.start].hunk();
            let last = old_tokens[span.end - 1].hunk();
            let mut next = hunk_tokens[first].start..hunk_tokens[last].end;
            for (other_position, other) in token_hunks.iter().enumerate() {
                if !other.matching && other.reaches(&next) {
                    next = next.start.min(other.old.start)..next.end.max(other.old.end);
                    in_span[other_position] = true;
                }
            }
            if next == span {
                break;
            }
            span = next;
        }
        let new_range = before_insertion[span.start]?..after_insertion[span.end]?;
        let first = old_tokens[span.start].hunk();
        let last = old_tokens[span.end - 1].hunk();
        edits.push(Edit {
            range: regions[first].start..regions[last].end,
            text: new_text(new_range)?,
        });
    }
    let mut anchors: Vec<Option<Vec<DiffHunkRange>>> = vec![None; old_hunks.len()];
    let mut anchors_of = |hunk: usize| -> Vec<DiffHunkRange> {
        anchors[hunk]
            .get_or_insert_with(|| {
                let text = old_texts[hunk].unwrap_or_default();
                ContentDiff::by_line([text, &vb[regions[hunk].clone()]])
                    .hunk_ranges()
                    .collect_vec()
            })
            .clone()
    };
    for (position, hunk) in token_hunks.iter().enumerate() {
        if in_span[position] {
            continue;
        }
        if hunk.matching {
            for (old, new) in zip(hunk.old.clone(), hunk.new.clone()) {
                let (Token::Conflict { hunk: old }, Token::Conflict { hunk: new }) =
                    (&old_tokens[old], &new_tokens[new])
                else {
                    continue;
                };
                let new_hunk = &new_hunks[*new];
                let old_hunk = &old_hunks[*old];
                let region = &regions[*old];
                edits.extend(conflict_edits(
                    vb,
                    old_contents,
                    old_hunk,
                    region,
                    index,
                    new_hunk,
                )?);
            }
            continue;
        }
        let text = new_text(hunk.new.clone())?;
        if hunk.old.is_empty() {
            let at = hunk.old.start;
            let before = at.checked_sub(1).map(|token| &old_tokens[token]);
            let point = match (before, old_tokens.get(at)) {
                (Some(Token::Line { hunk, range }), _) => {
                    let anchors = anchors_of(*hunk);
                    insertion_point(&anchors, range.end, vb, regions[*hunk].start, blocks)
                }
                (_, Some(Token::Line { hunk, range })) => {
                    let anchors = anchors_of(*hunk);
                    insertion_point(&anchors, range.start, vb, regions[*hunk].start, blocks)
                }
                (Some(Token::Conflict { hunk }), _) => regions[*hunk].end,
                (_, Some(Token::Conflict { hunk })) => regions[*hunk].start,
                (None, None) => 0,
            };
            edits.push(Edit {
                range: point..point,
                text,
            });
            continue;
        }
        // The changed lines are of one hunk that resolves, as a change that
        // reaches two would touch the conflict between them.
        let (
            Token::Line {
                hunk: old_hunk,
                range: first,
            },
            Token::Line { range: last, .. },
        ) = (&old_tokens[hunk.old.start], &old_tokens[hunk.old.end - 1])
        else {
            return None;
        };
        let anchors = anchors_of(*old_hunk);
        let lines = first.start..last.end;
        if let Some(range) = matched_range(&anchors, &lines, regions[*old_hunk].start) {
            edits.push(Edit { range, text });
        }
    }
    Some(edits)
}

/// The changes to VB, the base at `index` among the removes of
/// `old_contents`, in `hunk`, a conflict that the person kept, whose region of
/// VB is `region`, that the person's changes to its base in `new_hunk` stand
/// for: a change to lines that only VB holds, its marker lines and lines of a
/// block's side that no side of the conflict holds, which jj shows only in
/// the conflict's base, is not made, and VB keeps them. `None` where
/// `new_hunk` has another number of sides.
fn conflict_edits(
    vb: &[u8],
    old_contents: &Merge<BString>,
    hunk: &RangedMergeHunk,
    region: &Range<usize>,
    index: usize,
    new_hunk: &Merge<BString>,
) -> Option<Vec<Edit>> {
    if new_hunk.num_sides() != old_contents.num_sides() {
        return None;
    }
    let old_text = &vb[region.clone()];
    let held = held_lines(old_text, old_contents, hunk);
    let changes = line_changes(old_text, new_hunk.get_remove(index)?);
    let edits = changes
        .into_iter()
        .filter(|(range, _)| all_held(old_text, &held, range))
        .map(|(range, text)| Edit {
            range: region.start + range.start..region.start + range.end,
            text,
        })
        .collect();
    Some(edits)
}

/// Whether `region` of VB, whose blocks are `blocks`, holds whole blocks and
/// cuts none.
fn covers_whole_blocks(blocks: &[Block<'_>], region: &Range<usize>) -> bool {
    let mut touched = blocks
        .iter()
        .filter(|block| block.start < region.end && region.start < block.end)
        .peekable();
    touched.peek().is_some()
        && touched.all(|block| region.start <= block.start && block.end <= region.end)
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

/// A hunk of the merge of V's terms, with the range of VB that it takes,
/// and its block where it conflicts.
struct Segment<'a> {
    range: Range<usize>,
    hunk: &'a RangedMergeHunk,
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
/// VB, whose blocks are `blocks`, made by region. A change to lines where V's
/// terms are the same is made to each term. A change that touches lines where
/// V's terms differ but merge writes the changed text of the hunks it touches
/// into each term, in place of the term's own text there, and so does a change
/// that covers whole blocks. Another change inside a side of a block is made
/// to the term of V whose side it is. A term's text whose last line has no
/// final newline is a block side all the same, as jj's line split takes that
/// line as a line, though VB ends the side with a newline. An insertion at a
/// block's edge is outside the block, and goes into each term. A change that
/// touches a marker line without covering its block, or crosses a block's
/// edge or two sides, is not made. The text of a change is text, marker-shaped
/// lines included. `None` where `vb` is not the VB of `contents`.
fn edited_terms(
    vb: &[u8],
    blocks: &[Block<'_>],
    contents: &Merge<BString>,
    edits: &[Edit],
    options: &MergeOptions,
) -> Option<Merge<Vec<u8>>> {
    let hunks = files::split_ranged_hunks(contents, options);
    let segments = segments(vb, blocks, contents, &hunks)?;
    let (groups, side_edits) = grouped_edits(&segments, edits, contents.adds().len());
    let mut term_edits =
        Merge::from_removes_adds(iter::repeat_n(vec![], contents.removes().len()), side_edits);
    for group in &groups {
        let (first, last) = (
            &segments[group.segments.start],
            &segments[group.segments.end - 1],
        );
        let (span, ranges) = if group.expand {
            let ranges = first.hunk.ranges.clone().zip(last.hunk.ranges.clone());
            (
                first.range.start..last.range.end,
                ranges.into_map(|(first, last)| first.start..last.end),
            )
        } else {
            let starts = term_offsets(&segments, group.span.start)?;
            let ends = term_offsets(&segments, group.span.end)?;
            (
                group.span.clone(),
                starts.zip(ends).into_map(|(start, end)| start..end),
            )
        };
        let text = apply_edits(vb, span, group.edits.iter().copied())?;
        for (edits, range) in zip(term_edits.iter_mut(), ranges) {
            edits.push(Edit {
                range,
                text: text.clone(),
            });
        }
    }
    let new_contents = zip(contents.iter(), term_edits)
        .map(|(content, mut edits)| {
            edits.sort_by_key(|edit| (edit.range.start, edit.range.end));
            apply_edits(content, 0..content.len(), &edits)
        })
        .collect::<Option<Vec<_>>>()?;
    Some(Merge::from_vec(new_contents))
}

/// VB, `vb`, whose blocks are `blocks`, laid out over `hunks`, the hunks of
/// the merge of V's terms, `contents`: each hunk with the range of VB that it
/// takes, and its block where it conflicts. `None` where VB does not hold
/// each hunk's text, or each conflicting hunk's sides in a block, in order.
fn segments<'a>(
    vb: &[u8],
    blocks: &'a [Block<'a>],
    contents: &Merge<BString>,
    hunks: &'a [RangedMergeHunk],
) -> Option<Vec<Segment<'a>>> {
    let mut segments = vec![];
    let mut next_blocks = blocks.iter();
    let mut cursor = 0;
    for hunk in hunks {
        let (block, end) = match &hunk.resolved {
            None => {
                let block = next_blocks.next()?;
                let adds = zip(contents.adds(), hunk.ranges.adds());
                let sides_hold_adds = block.start == cursor
                    && block.sides.len() == contents.adds().len()
                    && zip(&block.sides, adds).all(|(side, (add, range))| {
                        side_holds(side, &add.as_slice()[range.clone()])
                    });
                if !sides_hold_adds {
                    return None;
                }
                (Some(block), block.end)
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
            hunk,
            block,
        });
        cursor = end;
    }
    (cursor == vb.len() && next_blocks.next().is_none()).then_some(segments)
}

/// `edits`, changes to VB laid out as `segments`, grouped: the groups of
/// changes that go into every term over one range of VB, and, for each of the
/// `num_sides` sides of the blocks, the changes inside that side, at their
/// place in the term of V whose side it is. A change that touches a block
/// without covering it, and is not inside one side of it, is left out.
fn grouped_edits<'a>(
    segments: &[Segment<'_>],
    edits: &'a [Edit],
    num_sides: usize,
) -> (Vec<Group<'a>>, Vec<Vec<Edit>>) {
    let mut side_edits: Vec<Vec<Edit>> = vec![vec![]; num_sides];
    let mut groups: Vec<Group<'_>> = vec![];
    let mut edits = edits.iter().collect_vec();
    edits.sort_by_key(|edit| (edit.range.start, edit.range.end));
    for edit in edits {
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
        // A change that covers whole blocks, a hunk the person resolved over
        // them, goes into every term below; any other change that touches a
        // block goes to the term whose side it is in, and is left out where it
        // is not inside one side (it then reaches past the block, or across a
        // separator).
        let covers = |range: &Range<usize>| start <= range.start && range.end <= end;
        if let Some(&index) = touched
            .iter()
            .find(|&&index| segments[index].block.is_some() && !covers(&segments[index].range))
        {
            if let Some((side, edit)) = side_edit(&segments[index], edit) {
                side_edits[side].push(edit);
            }
            continue;
        }
        let expand = touched.iter().any(|&index| {
            let segment = &segments[index];
            let inside = segment.range.start < start && start < segment.range.end;
            let merged = segment.hunk.resolved.is_some() && !segment.hunk.matching;
            segment.block.is_some() || (merged && (start < end || inside))
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
    (groups, side_edits)
}

/// `edit`, a change to VB inside `segment`'s block, as a change to the term of
/// V whose side it is in, with the side's index; `None` where it is not
/// inside one side.
fn side_edit(segment: &Segment<'_>, edit: &Edit) -> Option<(usize, Edit)> {
    let block = segment.block?;
    let (start, end) = (edit.range.start, edit.range.end);
    let side = block
        .side_ranges
        .iter()
        .position(|side| side.start <= start && end <= side.end)?;
    let term_range = segment.hunk.ranges.get_add(side)?;
    let side_range = &block.side_ranges[side];
    let (from, to) = (start - side_range.start, end - side_range.start);
    let (range, text) = if to <= term_range.len() {
        (from..to, edit.text.clone())
    } else {
        // The term's last line has no final newline, which VB added: the
        // changed lines keep that.
        let mut text = edit.text.clone();
        if text.last() == Some(&b'\n') {
            text.pop();
        }
        if from == side_range.len() {
            text.insert(0, b'\n');
            (term_range.len()..term_range.len(), text)
        } else {
            (from..term_range.len(), text)
        }
    };
    let edit = Edit {
        range: term_range.start + range.start..term_range.start + range.end,
        text,
    };
    Some((side, edit))
}

/// Whether `side`, a side of a block of VB, holds `text`, a term's text there:
/// the same text, or, where the text's last line has no final newline, the
/// text with the newline that VB ends each side with.
fn side_holds(side: &[u8], text: &[u8]) -> bool {
    side == text
        || (!text.is_empty() && !text.ends_with(b"\n") && side.strip_suffix(b"\n") == Some(text))
}

/// The offset in each term that `point` of VB stands for, where `point` is
/// inside a hunk whose terms are the same, or at a hunk's edge.
fn term_offsets(segments: &[Segment<'_>], point: usize) -> Option<Merge<usize>> {
    let found = segments.iter().find(|segment| {
        let range = &segment.range;
        range.start <= point
            && point <= range.end
            && (segment.hunk.matching || point == range.start || point == range.end)
    });
    let offsets = match found {
        Some(segment) if segment.hunk.matching => {
            let shift = point - segment.range.start;
            segment.hunk.ranges.map(|range| range.start + shift)
        }
        Some(segment) if point == segment.range.start => {
            segment.hunk.ranges.map(|range| range.start)
        }
        Some(segment) => segment.hunk.ranges.map(|range| range.end),
        None => segments.last()?.hunk.ranges.map(|range| range.end),
    };
    Some(offsets)
}

/// The contents of V's terms, `terms` with the contents `contents`, with
/// `edits`, changes to M, `m`, made: an insertion before or after M's block
/// goes into each file term, at its start or its end, as jj's line merge,
/// which splits any content at newline bytes, makes it into a binary file
/// too, an absent term read as empty; a term that is not a file (a symlink, a
/// directory or a submodule) is left as it is, and no other change is made,
/// as M holds none of V's text. The text of an insertion is text, marker-shaped
/// lines included.
fn edited_m_terms(
    m: &[u8],
    terms: &MergedTreeValue,
    contents: &Merge<BString>,
    edits: &[Edit],
) -> Merge<Vec<u8>> {
    let (mut before, mut after) = (vec![], vec![]);
    for edit in edits.iter().filter(|edit| edit.range.is_empty()) {
        if edit.range.start == 0 {
            before.extend_from_slice(&edit.text);
        } else if edit.range.start == m.len() {
            after.extend_from_slice(&edit.text);
        }
    }
    let new_contents = zip(terms.iter(), contents.iter())
        .map(|(term, content)| {
            if matches!(term, None | Some(TreeValue::File { .. })) {
                [before.as_slice(), content.as_slice(), after.as_slice()].concat()
            } else {
                content.as_slice().to_vec()
            }
        })
        .collect_vec();
    Merge::from_vec(new_contents)
}

/// The ranges of `text`, the text of a base of `contents` in `hunk`, whose
/// lines a side of `contents` holds there too, by a line diff of each.
fn held_lines(text: &[u8], contents: &Merge<BString>, hunk: &RangedMergeHunk) -> Vec<Range<usize>> {
    zip(contents.adds(), hunk.ranges.adds())
        .flat_map(|(content, range)| {
            ContentDiff::by_line([text, &content.as_slice()[range.clone()]])
                .hunk_ranges()
                .filter(|diff_hunk| matches!(diff_hunk.kind, DiffHunkKind::Matching))
                .map(|diff_hunk| diff_hunk.ranges[0].clone())
                .collect_vec()
        })
        .collect()
}

/// Whether each line of `range` of `text` is in one of `held`.
fn all_held(text: &[u8], held: &[Range<usize>], range: &Range<usize>) -> bool {
    let mut start = range.start;
    text[range.clone()].lines_with_terminator().all(|line| {
        let end = start + line.len();
        let is_held = held
            .iter()
            .any(|held| held.start <= start && end <= held.end);
        start = end;
        is_held
    })
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use super::super::text_hunks;
    use super::super::virtual_base_content;
    use super::*;
    use crate::backend::SymlinkId;
    use crate::files::FileMergeHunkLevel;
    use crate::merge::SameChange;

    /// A line shaped like a separator line of VB.
    const MARKER_SHAPED: &str = "======= 0123456789abcdef0123456789abcdef\n";

    fn options() -> MergeOptions {
        MergeOptions {
            hunk_level: FileMergeHunkLevel::Line,
            same_change: SameChange::Accept,
        }
    }

    /// V's terms: `x 2 3 4 5 6`, `o 2 3 4 5 6` and `y 2 3 4 5 6`, one word a
    /// line, so that they conflict at line 1 only.
    fn terms() -> Merge<BString> {
        let text = |first: &str| BString::from(format!("{first}\n2\n3\n4\n5\n6\n"));
        Merge::from_vec(vec![text("x"), text("o"), text("y")])
    }

    /// The VB of `terms`.
    fn vb_of(terms: &Merge<BString>) -> Vec<u8> {
        virtual_base_content(text_hunks(terms, &options()), "x o y", &HashSet::new())
    }

    /// The range of the first line of `vb` that is `line`, newline included.
    fn line_range(vb: &[u8], line: &str) -> Range<usize> {
        let start = vb.find(format!("\n{line}")).unwrap() + 1;
        start..start + line.len()
    }

    /// A change of `vb` from the line `from` to the line `to`, the first of
    /// each, to `text`.
    fn edit(vb: &[u8], from: &str, to: &str, text: &str) -> Edit {
        Edit {
            range: line_range(vb, from).start..line_range(vb, to).end,
            text: text.as_bytes().to_vec(),
        }
    }

    /// `terms` with `edit`, a change to their VB, made by region.
    fn edited(terms: &Merge<BString>, edit: Edit) -> Merge<Vec<u8>> {
        let vb = vb_of(terms);
        let blocks = blocks(&vb).unwrap();
        edited_terms(&vb, &blocks, terms, &[edit], &options()).unwrap()
    }

    #[test]
    fn a_marker_shaped_line_is_text() {
        let terms = terms();
        let vb = vb_of(&terms);
        let edited = edited(&terms, edit(&vb, "3\n", "3\n", MARKER_SHAPED));
        for (old, new) in zip(terms.iter(), edited.iter()) {
            assert_eq!(new, &old.replace("3\n", MARKER_SHAPED));
        }
    }

    #[test]
    fn a_change_that_reaches_past_a_block_is_not_made() {
        let terms = terms();
        let vb = vb_of(&terms);
        let edited = edited(&terms, edit(&vb, "y\n", "2\n", "Y\nTWO\n"));
        assert_eq!(edited, terms.map(|content| content.to_vec()));
    }

    #[test]
    fn a_change_across_two_sides_is_not_made() {
        let terms = terms();
        let vb = vb_of(&terms);
        let edited = edited(&terms, edit(&vb, "x\n", "y\n", "z\n"));
        assert_eq!(edited, terms.map(|content| content.to_vec()));
    }

    #[test]
    fn an_insertion_into_m_is_text_and_leaves_a_term_that_is_not_a_file() {
        let file = |id: u8| {
            Some(TreeValue::File {
                id: FileId::new(vec![id]),
                executable: false,
                copy_id: CopyId::placeholder(),
            })
        };
        let symlink = Some(TreeValue::Symlink(SymlinkId::new(vec![9])));
        let terms = Merge::from_vec(vec![file(1), symlink, file(2)]);
        let contents = Merge::from_vec(vec![
            BString::from("x\n"),
            BString::default(),
            BString::from("y\n"),
        ]);
        let m = b"<<<<<<< m\n";
        let insertion = Edit {
            range: m.len()..m.len(),
            text: MARKER_SHAPED.as_bytes().to_vec(),
        };
        let edited = edited_m_terms(m, &terms, &contents, &[insertion]);
        let expected = [
            format!("x\n{MARKER_SHAPED}").into_bytes(),
            vec![],
            format!("y\n{MARKER_SHAPED}").into_bytes(),
        ];
        assert_eq!(edited.as_slice(), expected);
    }
}
