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
//!
//! # The rule
//!
//! jj's own snapshot reads the file the person left as a new text for each
//! term of the conflict: the text of a hunk that they left resolved goes into
//! every term, and each conflict that they kept gives each term its own text.
//! The file that jj materialized reads the same way, as the base that the
//! person saw: VB, but for the hunks that the conflict's own merge resolved
//! to its sides' text. Outside every block of VB, such a hunk is text that
//! the file showed, which jj's snapshot writes into every term, a plain base
//! among them; it goes into V's terms too, so that a later rewrite that puts
//! V's terms in VB's place keeps it (a change that a parent made there).
//! Where such hunks cut a block, VB keeps its marker lines and the side of
//! the block that the merge did not take (merge-cases' VB6), which no side
//! of the conflict changes; but jj's snapshot writes each such hunk's text
//! into every side, so each side changes its own text there to that text,
//! and where the hunk lies inside one side of the block, that change goes
//! into V's terms as well (see [`edits::ShownBase`]), so that a rewrite that
//! takes the snapshot's sides away gives back what the parents' merge held
//! there (a parent's own line in a block that the merge cuts). A change of
//! the person's to the lines that the file showed for such a block is made
//! among VB's own lines of it, so that V keeps the block: where it ends at
//! the end of a side of the block, or starts at its start, as an edit next
//! to the conflict at the block's tail does, its text goes at the end, or
//! the start, of every side of the block, into each side's term of V, and
//! the lines that it replaces go from the side that holds them. V's base
//! term keeps its own text there, so the conflict's own merge of the block,
//! the conflict at its tail with it, stays as it was. Elsewhere the change
//! is made inside the side (see `onto_virtual_base` in [`edits`]). The rest
//! is the person's changes to the base: a conflict that they resolved,
//! deleted, moved or copied, and a line that they changed. The snapshot
//! makes each one, each hunk that the merge resolved outside every block,
//! and each side's change inside a block, to V's terms, by where it falls in
//! VB:
//!
//! 1. in text that every term of V holds, or at the edge of a hunk of V's
//!    terms' merge, outside every block: the change is made to every term, at
//!    that place;
//! 2. inside one side of a block, with the block's marker lines as they were:
//!    the change is made to the term of V whose side it is;
//! 3. anywhere else (it touches a marker line, reaches past a block's edge or
//!    across two sides, or touches a hunk where V's terms differ but merge):
//!    the change grows to the whole of the hunks that it touches, with every
//!    other change that reaches into them, and the text that the person left
//!    for those hunks takes the place of each term's own text there, in every
//!    term.
//!
//! 3 is the fallback: no change is left out, and the text that the person
//! left is on every term. What a change writes is the person's text, whatever
//! its lines look like. The lines that came from the base of a conflict that
//! the person kept are VB's, though: there, a whole block stands for each
//! term's own text at the block's hunk (a conflict that was moved or copied
//! takes V's conflict with it), and any other line in the form of a marker
//! line is left out, as are VB's own marker lines around a change, so no
//! term of V ever gets a marker line of VB.
//!
//! VB's blocks tell where its hunks are. The working copy can hold the
//! marker lines of an earlier VB of the conflict, which an earlier snapshot
//! replaced, so in the base of a conflict that the person kept a block is
//! told by its sides, not by its hex. The changes are found by a line diff
//! of VB against its new text, between the blocks that the new text still
//! holds whole and in order, which line up with themselves (see
//! [`edits::base_edits`]). VB is laid out over the hunks of V's terms' merge
//! at the hunk level that VB records (see [`layout::Layout`]), or at the
//! other level, for a VB that records none. The file that jj materialized is
//! read at that level too, the level of the merge that wrote both. A file
//! that a later checkout wrote at the other level shows a hunk that resolves
//! there in place of a conflict: that text reads as the person's, and goes
//! into V's terms as jj's own snapshot writes the text of a hunk that
//! resolves into every term.
//!
//! M holds none of V's text, so the person's changes to V's text are not in
//! the base's new text. Where V is a conflict of text files, the snapshot
//! builds V's VB, which the merge would have written, and takes, as the
//! changes to it, the changes that the person made to every side of the
//! conflict alike, in text that every side shares with that VB (see
//! [`edits::shared_text_edits`]); it then makes them by the same three cases.
//! The same goes for a file that names V's terms but is not laid out as their
//! VB. Where V's terms are all files but not all text, the snapshot does the
//! same with the VB that a line merge of them would give. For any other M (a
//! term is absent, or is not a file), only the text that every side of the
//! conflict gained at its very start or at its very end has a place in V's
//! terms (see [`edited_m_terms`]).
//!
//! The lines of the base of a conflict that the person kept are VB's, which
//! V's terms give: a line that the person added, changed or removed there is
//! not made to V's terms, and the base stays as it was. A conflict that they
//! moved, copied or deleted takes its whole block, and is a change.
//!
//! A term of V can itself be a VB: the merge bases of V's sides were merged
//! against one. A change to such a term is made to the terms that it names,
//! by the same cases, and the term becomes their VB. Its text is never
//! changed, so it never becomes a plain file that holds marker lines. A term
//! that is M, or a VB that is not laid out over its terms, stays as it is.
//!
//! A file whose last conflict's end marker has no newline ends every term of
//! that conflict one newline short (see `parse_conflict`): the person, or a
//! merge tool, removed the file's final newline, and each term of V loses its
//! own. No VB or M names a V one of whose sides is itself VB or M (see
//! [`holds_terms`]): where the person's changes would make one such, as M
//! pasted into an absent term does, the base stays as it was.

mod edits;
mod layout;

use std::collections::HashSet;
use std::iter::zip;

use bstr::BString;
use futures::future::BoxFuture;
use itertools::Itertools as _;

use self::edits::Edit;
use self::edits::ShownBase;
use self::edits::Text;
use self::edits::base_edits;
use self::edits::edge_insertions;
use self::edits::shared_text_edits;
use self::edits::text_edits;
use self::layout::Layout;
use super::NamedTerms;
use super::blocks::BlockFinder;
use super::blocks::blocks;
use super::file_term_contents;
use super::holds_terms;
use super::marker_like_lines;
use super::named_terms;
use super::recorded_hunk_level;
use super::simplified_with_labels;
use super::text_file_contents;
use super::virtual_base_file;
use super::write_virtual_base;
use crate::backend::BackendResult;
use crate::backend::CopyId;
use crate::backend::FileId;
use crate::backend::MergedTreeValue;
use crate::backend::TreeValue;
use crate::conflict_labels::ConflictLabels;
use crate::files::FileMergeHunkLevel;
use crate::merge::Merge;
use crate::repo_path::RepoPath;
use crate::store::Store;
use crate::tree_merge::MergeOptions;

/// The file that takes the place of the base at `index` among the removes of
/// `old_contents`, the contents of the terms of a conflict at `path` whose
/// base `old_id` is VB or M, once a parse of the conflict's materialized
/// markers (a snapshot of the working copy, or a merge tool's output) gave
/// the hunks `new_hunks`; `None` where that base is not VB or M.
///
/// The person's changes are made to the terms of V that the base names, as
/// the module's documentation says, and the file is the VB or M of the
/// edited terms, or their text where they resolve to a file. The base stays
/// `old_id` where the changes change no term, where the edited terms resolve
/// to a value that is not a file, and where the base names the terms of a
/// conflict of text files but neither it nor the VB that a merge would write
/// for them is laid out over them. A snapshot never writes a plain file that
/// holds VB's or M's marker lines.
pub(crate) async fn rewrite_virtual_base(
    store: &Store,
    path: &RepoPath,
    old_id: &FileId,
    old_contents: &Merge<BString>,
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
    let (term_contents, is_text) = match text_file_contents(store, path, &terms).await? {
        Some(contents) => (contents, true),
        None => (file_term_contents(store, path, &terms).await?, false),
    };
    let finder = BlockFinder::new(old, &blocks);
    let base = VirtualBase {
        finder: &finder,
        terms: &terms,
        labels: &labels,
    };
    let options = store.merge_options();
    let edited = edited_at_final_newline(
        old,
        old_contents,
        index,
        new_hunks,
        &term_contents,
        |new_hunks| {
            let Some(new_base) = Text::of_base(new_hunks, index) else {
                return Edited::Unread;
            };
            let snapshot = Snapshot {
                old_contents,
                index,
                new_hunks,
                new_base: &new_base,
            };
            if is_text {
                edited_text_terms(&base, &term_contents, &snapshot, options)
                    .map_or(Edited::Kept, Edited::Terms)
            } else {
                Edited::Terms(edited_file_terms(&base, &term_contents, &snapshot, options))
            }
        },
    );
    let new_contents = match edited {
        Edited::Unread => return Ok(None),
        Edited::Kept => return Ok(Some(old_id.clone())),
        Edited::Terms(new_contents) => new_contents,
    };
    // The marker lines of a base that is written are none of the lines that
    // the person left.
    let new_lines = new_hunks
        .iter()
        .flat_map(|hunk| hunk.iter())
        .map(|text| text.as_slice());
    let parent_lines = marker_like_lines(new_lines);
    let named = NamedTerms { terms, labels };
    let id = write_edited_terms(
        store,
        path,
        old_id,
        &named,
        &term_contents,
        new_contents,
        &parent_lines,
    )
    .await?;
    Ok(Some(id))
}

/// What a snapshot makes of the terms of V.
enum Edited {
    /// The hunks that the person left give the base no text: jj's snapshot
    /// writes it as it writes a plain base.
    Unread,
    /// The base stays as it was.
    Kept,
    /// The new contents of V's terms.
    Terms(Merge<Vec<u8>>),
}

/// What `edit` makes of the terms of V, whose contents are `term_contents`,
/// given `new_hunks`, the hunks that a parse of the markers of a conflict
/// whose terms' contents were `old_contents` gave, where the base at `index`
/// among their removes, `old`, is VB or M.
///
/// A file whose last conflict ends with an end marker that has no newline
/// ends every term of that conflict one newline short (see
/// `parse_conflict`): the person removed the file's final newline, which is
/// a change of every term of V alike. `edit` then finds the changes with
/// those newlines back (see [`with_final_newlines`]), and each term of V that
/// ended with a newline loses it after. As jj's parse takes a newline only
/// from a term that has text there, a term of V that was absent or empty
/// keeps whatever text it gained.
fn edited_at_final_newline(
    old: &[u8],
    old_contents: &Merge<BString>,
    index: usize,
    new_hunks: &[Merge<BString>],
    term_contents: &Merge<BString>,
    edit: impl FnOnce(&[Merge<BString>]) -> Edited,
) -> Edited {
    let drops_final_newline = old.ends_with(b"\n")
        && new_hunks.last().is_some_and(|hunk| {
            !hunk.is_resolved()
                && hunk
                    .get_remove(index)
                    .is_some_and(|base| !base.ends_with(b"\n"))
        });
    if !drops_final_newline {
        return edit(new_hunks);
    }
    match edit(&with_final_newlines(old_contents, new_hunks)) {
        Edited::Terms(new_contents) => {
            let popped = zip(term_contents.iter(), new_contents)
                .map(|(old, mut content)| {
                    if old.ends_with(b"\n") {
                        content.pop_if(|byte| *byte == b'\n');
                    }
                    content
                })
                .collect_vec();
            Edited::Terms(Merge::from_vec(popped))
        }
        unchanged => unchanged,
    }
}

/// `new_hunks`, the hunks of a file whose last conflict's end marker has no
/// newline, with the newline that each term of that conflict lost back:
/// each term that is not empty and does not end with a newline, where the
/// term of `old_contents` at its place, the conflict's terms before, did.
fn with_final_newlines(
    old_contents: &Merge<BString>,
    new_hunks: &[Merge<BString>],
) -> Vec<Merge<BString>> {
    let mut hunks = new_hunks.to_vec();
    if let Some(last) = hunks.last_mut() {
        *last = Merge::from_vec(
            zip(last.iter(), old_contents.iter())
                .map(|(term, old)| {
                    let mut term = term.clone();
                    if !term.is_empty() && !term.ends_with(b"\n") && old.ends_with(b"\n") {
                        term.push(b'\n');
                    }
                    term
                })
                .collect_vec(),
        );
    }
    hunks
}

/// The file that stands for `named`, the terms of V that the file `old_id`
/// names, whose contents are `term_contents`, once changes give them the
/// contents `new_contents`: `old_id` where no term changes, or where the
/// edited terms resolve to a value that is not a file; their file where they
/// resolve to one; else their VB where they are a conflict of text files,
/// and M otherwise, with marker lines that are none of `parent_lines`.
///
/// A term that changes and is itself VB or M is not written as text (see
/// [`rewrite_inner_virtual_base`]).
async fn write_edited_terms(
    store: &Store,
    path: &RepoPath,
    old_id: &FileId,
    named: &NamedTerms,
    term_contents: &Merge<BString>,
    new_contents: Merge<Vec<u8>>,
    parent_lines: &HashSet<&[u8]>,
) -> BackendResult<FileId> {
    let NamedTerms { terms, labels } = named;
    let mut new_terms = vec![];
    let old_terms = zip(terms.iter(), term_contents.iter());
    for ((term, content), new_content) in zip(old_terms, new_contents) {
        if new_content.as_slice() == content.as_slice() {
            new_terms.push(term.clone());
            continue;
        }
        let (term_id, executable, copy_id) = match term {
            Some(TreeValue::File {
                id,
                executable,
                copy_id,
            }) => (Some(id), *executable, copy_id.clone()),
            _ => (None, false, CopyId::placeholder()),
        };
        let inner = match term_id {
            Some(term_id) => {
                let old = content.as_slice();
                rewrite_inner_virtual_base(store, path, term_id, old, &new_content, parent_lines)
                    .await?
            }
            None => None,
        };
        let id = match inner {
            Some(id) => id,
            None => store.write_file(path, &mut new_content.as_slice()).await?,
        };
        new_terms.push(Some(TreeValue::File {
            id,
            executable,
            copy_id,
        }));
    }
    let (base, labels) = simplified_with_labels(
        &Merge::from_vec(new_terms),
        &ConflictLabels::from_merge(labels.clone()),
    );
    if base == *terms {
        return Ok(old_id.clone());
    }
    if let Some(resolved) = base.as_resolved() {
        // Every term that an edit changes is a file, and VB's terms are all
        // files or absent, so where they resolve it is to a file; any other
        // value keeps the old base.
        return Ok(match resolved {
            Some(TreeValue::File { id, .. }) => id.clone(),
            _ => old_id.clone(),
        });
    }
    // No VB or M can name the edited terms where a side among them is itself
    // VB or M, such as text that the person pasted from a base into an absent
    // term: none would read back (see [`holds_terms`]). The base then stays
    // the one that the file showed.
    if !holds_terms(store, path, &base).await? {
        return Ok(old_id.clone());
    }
    // The base is the VB of the edited terms where they are a conflict of text
    // files, M otherwise, whatever the old base was: where an edit makes every
    // term of M's V text (an absent term written as the person's text), it is
    // the VB of the edited terms.
    let base_contents = text_file_contents(store, path, &base).await?;
    write_virtual_base(store, path, &base, &labels, base_contents, parent_lines).await
}

/// The file that takes the place of `old_id`, a term of V whose text `old` a
/// change makes `new`, where that term is itself VB or M: the change is made
/// to the terms that it names, as the module's documentation says, and the
/// file is what stands for those (see [`write_edited_terms`]), so that the
/// term is never a plain file that holds the marker lines of a virtual merge
/// base. `None` where the term is not VB or M; `old_id` where it is M, or a
/// VB that is not laid out over the terms that it names.
fn rewrite_inner_virtual_base<'a>(
    store: &'a Store,
    path: &'a RepoPath,
    old_id: &'a FileId,
    old: &'a [u8],
    new: &'a [u8],
    parent_lines: &'a HashSet<&'a [u8]>,
) -> BoxFuture<'a, BackendResult<Option<FileId>>> {
    Box::pin(async move {
        let Some(blocks) = blocks(old) else {
            return Ok(None);
        };
        let Some(named) = named_terms(&blocks) else {
            return Ok(None);
        };
        if !holds_terms(store, path, &named.terms).await? {
            return Ok(None);
        }
        let Some(term_contents) = text_file_contents(store, path, &named.terms).await? else {
            return Ok(Some(old_id.clone()));
        };
        let new_contents = {
            let options = store.merge_options();
            let finder = BlockFinder::new(old, &blocks);
            let Some((layout, _)) = laid_out(&finder, &term_contents, options) else {
                return Ok(Some(old_id.clone()));
            };
            layout.edited(&text_edits(old, new))
        };
        let id = write_edited_terms(
            store,
            path,
            old_id,
            &named,
            &term_contents,
            new_contents,
            parent_lines,
        )
        .await?;
        Ok(Some(id))
    })
}

/// VB or M: its blocks, found in it by `finder`, and the terms of V that it
/// names, with their labels.
struct VirtualBase<'a> {
    finder: &'a BlockFinder<'a>,
    terms: &'a MergedTreeValue,
    labels: &'a Merge<String>,
}

/// What a snapshot read: the contents of the conflict's terms before it, the
/// index of the base among their removes, the hunks that the person left, and
/// the base's new text, which those hunks give.
struct Snapshot<'a> {
    old_contents: &'a Merge<BString>,
    index: usize,
    new_hunks: &'a [Merge<BString>],
    new_base: &'a Text,
}

/// VB, whose blocks `finder` finds, laid out over the hunks of the merge of
/// `term_contents`, the texts of the terms that it names, with the merge
/// options of that layout, where the store's are `options`.
///
/// The merge that wrote VB split V's terms into hunks at the hunk level that
/// VB records, so VB is laid out over them at that level, whatever
/// `merge.hunk-level` is now; a VB that an earlier build wrote records no
/// level where the merge split by word, so the other level is tried where the
/// recorded one does not lay VB out.
fn laid_out<'a>(
    finder: &'a BlockFinder<'a>,
    term_contents: &'a Merge<BString>,
    options: &MergeOptions,
) -> Option<(Layout<'a>, MergeOptions)> {
    let recorded = recorded_hunk_level(finder.blocks);
    let other = match recorded {
        FileMergeHunkLevel::Line => FileMergeHunkLevel::Word,
        FileMergeHunkLevel::Word => FileMergeHunkLevel::Line,
    };
    [recorded, other].into_iter().find_map(|hunk_level| {
        let options = MergeOptions {
            hunk_level,
            same_change: options.same_change,
        };
        let layout = Layout::new(finder, term_contents, &options)?;
        Some((layout, options))
    })
}

/// The texts of V's terms, `term_contents`, a conflict of text files, with
/// the changes of `snapshot` to `base`, which names them, made (see the
/// module's documentation). The store's merge options are `options`.
///
/// Where `base` is laid out over V's terms (see [`laid_out`]), the
/// conflict's terms, merged at the level of that layout, give the base as the
/// file that jj materialized showed it, and the person's changes are changes
/// of that. `None` where `base` is M, or another file that is not laid out
/// over V's terms, and the VB that a merge would write for them is not
/// either.
fn edited_text_terms(
    base: &VirtualBase<'_>,
    term_contents: &Merge<BString>,
    snapshot: &Snapshot<'_>,
    options: &MergeOptions,
) -> Option<Merge<Vec<u8>>> {
    let Some((layout, options)) = laid_out(base.finder, term_contents, options) else {
        return edited_by_reference(base, term_contents, snapshot, options);
    };
    let seen = ShownBase::new(base.finder, snapshot.old_contents, snapshot.index, &options);
    let edits = base_edits(base.finder, &seen, snapshot.new_base);
    Some(layout.edited(&edits))
}

/// The contents of V's terms, `term_contents`, files that are not all text,
/// with the changes of `snapshot` to `base`, which names them, made. The
/// store's merge options are `options`.
///
/// Files that are not all text merge by line as text does, so where every
/// term is a file, the VB that such a merge would give them tells where the
/// changes that every side of the conflict has fall in them (see
/// [`edited_by_reference`]). Otherwise, or where that VB is not laid out
/// over them, only the text that every side gained at its very start or at
/// its very end goes into them (see [`edited_m_terms`]).
fn edited_file_terms(
    base: &VirtualBase<'_>,
    term_contents: &Merge<BString>,
    snapshot: &Snapshot<'_>,
    options: &MergeOptions,
) -> Merge<Vec<u8>> {
    let all_files = base
        .terms
        .iter()
        .all(|term| matches!(term, Some(TreeValue::File { .. })));
    let by_reference = all_files
        .then(|| edited_by_reference(base, term_contents, snapshot, options))
        .flatten();
    by_reference.unwrap_or_else(|| {
        let old = base.finder.vb;
        let edits = edge_insertions(snapshot.old_contents, snapshot.new_hunks, old.len());
        edited_m_terms(old, base.terms, term_contents, &edits)
    })
}

/// The contents of V's terms, `term_contents`, with the changes of
/// `snapshot` that every side of the conflict has alike made, where `base`,
/// which names them, shows none of their text: the VB that a merge of them
/// with `options` would write tells where those changes fall (see
/// [`shared_text_edits`]). `None` where that VB is not laid out over them.
fn edited_by_reference(
    base: &VirtualBase<'_>,
    term_contents: &Merge<BString>,
    snapshot: &Snapshot<'_>,
    options: &MergeOptions,
) -> Option<Merge<Vec<u8>>> {
    let reference = virtual_base_file(
        base.terms,
        base.labels,
        Some(term_contents.clone()),
        options,
        &HashSet::new(),
    )?;
    let reference_blocks = blocks(&reference)?;
    let finder = BlockFinder::new(&reference, &reference_blocks);
    let layout = Layout::new(&finder, term_contents, options)?;
    let edits = shared_text_edits(
        &reference,
        snapshot.old_contents,
        snapshot.index,
        snapshot.new_hunks,
        options,
    );
    Some(layout.edited(&edits))
}

/// The contents of V's terms, `terms` with the contents `contents`, with
/// `edits`, changes to M, `m`, made, where V is not a conflict of text files:
/// an insertion before or after M's block (see [`edge_insertions`]) goes into
/// each file term, at its start or its end, as jj's line merge, which splits
/// any content at newline bytes, makes it into a binary file too, an absent
/// term read as empty; a term that is not a file (a symlink, a directory or a
/// submodule) is left as it is, and no other change is made, as M holds none
/// of V's text. The text of an insertion is text, marker-shaped lines
/// included.
fn edited_m_terms(
    m: &[u8],
    terms: &MergedTreeValue,
    contents: &Merge<BString>,
    edits: &[Edit],
) -> Merge<Vec<u8>> {
    let (mut before, mut after) = (vec![], vec![]);
    for edit in edits.iter().filter(|edit| edit.range.is_empty()) {
        if edit.range.start == 0 {
            before.extend_from_slice(&edit.text.bytes);
        } else if edit.range.start == m.len() {
            after.extend_from_slice(&edit.text.bytes);
        }
    }
    let new_contents = zip(terms.iter(), contents.iter())
        .map(|(term, content)| {
            if !matches!(term, None | Some(TreeValue::File { .. })) {
                return content.as_slice().to_vec();
            }
            // Text put after a term that ends without a newline comes after
            // one, as jj's snapshot reads it after such a side.
            let newline: &[u8] = if !after.is_empty() && content.last().is_some_and(|&b| b != b'\n')
            {
                b"\n"
            } else {
                b""
            };
            [
                before.as_slice(),
                content.as_slice(),
                newline,
                after.as_slice(),
            ]
            .concat()
        })
        .collect_vec();
    Merge::from_vec(new_contents)
}

#[cfg(test)]
mod tests {
    use super::super::blocks::blocks_read;
    use super::super::text_hunks;
    use super::super::virtual_base_content;
    use super::*;
    use crate::backend::SymlinkId;
    use crate::merge::SameChange;

    /// A line shaped like a separator line of VB.
    const MARKER_SHAPED: &str = "======= 0123456789abcdef0123456789abcdef\n";

    /// The names of three terms, as an opening marker line holds them.
    const NAMES: &str = "file:01 file:02 file:03";

    fn options() -> MergeOptions {
        MergeOptions {
            hunk_level: FileMergeHunkLevel::Line,
            same_change: SameChange::Accept,
        }
    }

    /// V's terms: `first 2 3 4 5 last`, one word a line, for `x`, `o` and `y`
    /// as `first` and `a6`, `6` and `b6` as `last`, so that they conflict at
    /// lines 1 and 6.
    fn terms() -> Merge<BString> {
        let text =
            |first: &str, last: &str| BString::from(format!("{first}\n2\n3\n4\n5\n{last}\n"));
        Merge::from_vec(vec![text("x", "a6"), text("o", "6"), text("y", "b6")])
    }

    /// The VB of `terms`.
    fn vb_of(terms: &Merge<BString>) -> Vec<u8> {
        virtual_base_content(text_hunks(terms, &options()), NAMES, &HashSet::new()).unwrap()
    }

    /// `terms` with the changes that turn their VB into `new`, the base's new
    /// text, made.
    fn edited(terms: &Merge<BString>, new: &Text) -> Vec<String> {
        let vb = vb_of(terms);
        let blocks = blocks(&vb).unwrap();
        let finder = BlockFinder::new(&vb, &blocks);
        let layout = Layout::new(&finder, terms, &options()).unwrap();
        // The base as the file that jj materialized showed it: VB, each of
        // its blocks in the base of a conflict, so no hunk that the
        // conflict's merge resolved cuts one.
        let mut text = Text::default();
        let mut at = 0;
        for block in &blocks {
            text.push(&vb[at..block.start], false);
            text.push(&vb[block.start..block.end], true);
            at = block.end;
        }
        text.push(&vb[at..], false);
        let seen = ShownBase {
            text,
            side_edits: vec![],
        };
        let edited = layout.edited(&base_edits(&finder, &seen, new));
        edited
            .iter()
            .map(|content| String::from_utf8(content.clone()).unwrap())
            .collect()
    }

    /// `text` as the base's new text, each of its lines in the base of a
    /// conflict that the person kept if `of_base` holds for it.
    fn new_text(text: &str, of_base: impl Fn(&str) -> bool) -> Text {
        let mut new = Text::default();
        for line in text.split_inclusive('\n') {
            new.push(line.as_bytes(), of_base(line));
        }
        new
    }

    /// The lines of `vb`'s blocks, in order, each block's lines apart.
    fn block_lines(vb: &[u8]) -> Vec<String> {
        blocks(vb)
            .unwrap()
            .iter()
            .map(|block| String::from_utf8(vb[block.start..block.end].to_vec()).unwrap())
            .collect()
    }

    /// Whether `line` is a line of a block of the VB of [`terms`], which is
    /// in the base of a conflict that the person kept: a marker line, or one
    /// of the lines `x`, `y`, `a6` and `b6`. [`MARKER_SHAPED`] is a line
    /// that the person wrote.
    fn in_a_block(line: &str) -> bool {
        let is_marker = ["<<<<<<<", "=======", ">>>>>>>"]
            .iter()
            .any(|marker| line.starts_with(marker));
        (is_marker && line != MARKER_SHAPED) || ["x\n", "y\n", "a6\n", "b6\n"].contains(&line)
    }

    #[test]
    fn a_marker_shaped_line_is_text() {
        let terms = terms();
        let vb = String::from_utf8(vb_of(&terms)).unwrap();
        let new = vb.replace("\n3\n", &format!("\n{MARKER_SHAPED}"));
        let edited = edited(&terms, &new_text(&new, in_a_block));
        for (old, new) in zip(terms.iter(), edited) {
            assert_eq!(new, old.to_string().replace("3\n", MARKER_SHAPED));
        }
    }

    #[test]
    fn a_change_inside_a_side_goes_into_its_term() {
        let terms = terms();
        let vb = String::from_utf8(vb_of(&terms)).unwrap();
        let new = vb.replace("\ny\n", "\nY\n");
        let edited = edited(&terms, &new_text(&new, in_a_block));
        assert_eq!(
            edited,
            [
                "x\n2\n3\n4\n5\na6\n",
                "o\n2\n3\n4\n5\n6\n",
                "Y\n2\n3\n4\n5\nb6\n"
            ]
        );
    }

    #[test]
    fn a_change_that_reaches_past_a_block_goes_into_every_term() {
        // The second side's `y` and the line `2` after the block are replaced
        // by `Y` and `TWO`: the block and that line take what is left of them,
        // without the block's marker lines, in every term.
        let terms = terms();
        let vb = String::from_utf8(vb_of(&terms)).unwrap();
        let [first, _] = &block_lines(vb.as_bytes())[..] else {
            panic!("expected two blocks in {vb}");
        };
        let close = first.lines().last().unwrap();
        let new = vb.replace(&format!("y\n{close}\n2\n"), "Y\nTWO\n");
        let edited = edited(&terms, &new_text(&new, in_a_block));
        assert_eq!(
            edited,
            [
                "x\nY\nTWO\n3\n4\n5\na6\n",
                "x\nY\nTWO\n3\n4\n5\n6\n",
                "x\nY\nTWO\n3\n4\n5\nb6\n"
            ]
        );
    }

    #[test]
    fn a_change_across_two_sides_goes_into_every_term() {
        let terms = terms();
        let vb = String::from_utf8(vb_of(&terms)).unwrap();
        let [first, _] = &block_lines(vb.as_bytes())[..] else {
            panic!("expected two blocks in {vb}");
        };
        let separator = first.lines().nth(2).unwrap();
        let new = vb.replace(&format!("x\n{separator}\ny\n"), "z\n");
        let edited = edited(&terms, &new_text(&new, in_a_block));
        assert_eq!(
            edited,
            [
                "z\n2\n3\n4\n5\na6\n",
                "z\n2\n3\n4\n5\n6\n",
                "z\n2\n3\n4\n5\nb6\n"
            ]
        );
    }

    #[test]
    fn a_deleted_marker_line_leaves_no_marker_line_in_a_term() {
        // The first block's opening line is deleted from the base of the
        // conflict that the person kept: the block's sides are text for every
        // term, and its other marker lines go into none.
        let terms = terms();
        let vb = String::from_utf8(vb_of(&terms)).unwrap();
        let (_, new) = vb.split_once('\n').unwrap();
        let edited = edited(&terms, &new_text(new, in_a_block));
        assert_eq!(
            edited,
            [
                "x\ny\n2\n3\n4\n5\na6\n",
                "x\ny\n2\n3\n4\n5\n6\n",
                "x\ny\n2\n3\n4\n5\nb6\n"
            ]
        );
    }

    #[test]
    fn a_change_next_to_text_that_only_a_base_of_v_holds_keeps_that_text() {
        // V's base holds a line 2 that both of V's sides deleted, so VB holds
        // none of it: a change to line 1 leaves it in V's base.
        let terms = Merge::from_vec(vec![
            BString::from("1\n3\nx\n"),
            BString::from("1\n2\n3\no\n"),
            BString::from("1\n3\ny\n"),
        ]);
        let vb = String::from_utf8(vb_of(&terms)).unwrap();
        let new = vb.replacen("1\n", "ONE\n", 1);
        let edited = edited(&terms, &new_text(&new, in_a_block));
        assert_eq!(edited, ["ONE\n3\nx\n", "ONE\n2\n3\no\n", "ONE\n3\ny\n"]);
    }

    #[test]
    fn a_resolved_conflict_and_a_deleted_one_go_into_every_term() {
        let terms = terms();
        let vb = String::from_utf8(vb_of(&terms)).unwrap();
        let [first, second] = &block_lines(vb.as_bytes())[..] else {
            panic!("expected two blocks in {vb}");
        };
        // The first conflict resolved to `R1`.
        let new = vb.replace(first, "R1\n");
        let resolved = edited(&terms, &new_text(&new, in_a_block));
        assert_eq!(
            resolved,
            [
                "R1\n2\n3\n4\n5\na6\n",
                "R1\n2\n3\n4\n5\n6\n",
                "R1\n2\n3\n4\n5\nb6\n"
            ]
        );
        // The first conflict deleted with the lines up to the second one,
        // which the person kept.
        let deleted = edited(&terms, &new_text(second, in_a_block));
        assert_eq!(deleted, ["a6\n", "6\n", "b6\n"]);
    }

    #[test]
    fn a_moved_conflict_takes_its_terms_with_it() {
        let terms = terms();
        let vb = String::from_utf8(vb_of(&terms)).unwrap();
        let [first, second] = &block_lines(vb.as_bytes())[..] else {
            panic!("expected two blocks in {vb}");
        };
        // The two conflicts swapped.
        let swapped = format!("{second}2\n3\n4\n5\n{first}");
        let edited_terms = edited(&terms, &new_text(&swapped, in_a_block));
        assert_eq!(
            edited_terms,
            [
                "a6\n2\n3\n4\n5\nx\n",
                "6\n2\n3\n4\n5\no\n",
                "b6\n2\n3\n4\n5\ny\n"
            ]
        );
        // Line 2 moved below the second conflict.
        let moved = format!("{first}3\n4\n5\n{second}2\n");
        let edited_terms = edited(&terms, &new_text(&moved, in_a_block));
        assert_eq!(
            edited_terms,
            [
                "x\n3\n4\n5\na6\n2\n",
                "o\n3\n4\n5\n6\n2\n",
                "y\n3\n4\n5\nb6\n2\n"
            ]
        );
    }

    #[test]
    fn a_pasted_block_is_the_persons_text() {
        // The person pastes the first block's opening line just above the
        // block, and the whole block at the end, as text of their own: both
        // go into every term as they are.
        let terms = terms();
        let vb = String::from_utf8(vb_of(&terms)).unwrap();
        let [first, _] = &block_lines(vb.as_bytes())[..] else {
            panic!("expected two blocks in {vb}");
        };
        let opening = first.split_inclusive('\n').next().unwrap();
        let mut new = Text::default();
        new.push(opening.as_bytes(), false);
        for line in vb.split_inclusive('\n') {
            new.push(line.as_bytes(), in_a_block(line));
        }
        new.push(first.as_bytes(), false);
        let edited = edited(&terms, &new);
        for (old, new) in zip(terms.iter(), edited) {
            assert_eq!(new, format!("{opening}{old}{first}"));
        }
    }

    #[test]
    fn a_snapshot_reads_the_blocks_of_vb_a_bounded_number_of_times() {
        // V conflicts at N hunks, each between the sides `x` and `y`, so VB
        // has N blocks with the same sides, and the file that jj
        // materialized shows each in the base of a conflict that the person
        // kept. The person changes one line between two of them, and `y` to
        // `Y` in each base, so that no block has the sides that a base
        // holds. The count is of the blocks whose sides the finders read:
        // each block's once to build the finder, and, for each base that a
        // pass over the bases looks up, each block whose sides it compares
        // with the base's (see `BlockFinder::holds`). The finder compares the
        // block of the base's opening line and finds no other by those sides,
        // in three passes, so it reads 4N in all. Comparing a base's sides
        // with those of each block of VB in turn reads N for each base that
        // no block holds, N² in all.
        const N: usize = 64;
        let text = |side: &str| BString::from((0..N).map(|i| format!("{side}\ns{i}\n")).join(""));
        let terms = Merge::from_vec(vec![text("x"), text("o"), text("y")]);
        let vb = String::from_utf8(vb_of(&terms)).unwrap();
        let new = vb.replacen("\ns0\n", "\nS0\n", 1).replace("\ny\n", "\nY\n");
        let before = blocks_read();
        let in_a_base = |line: &str| line == "Y\n" || in_a_block(line);
        let edited = edited(&terms, &new_text(&new, in_a_base));
        let read = blocks_read() - before;
        for (old, new) in zip(terms.iter(), edited) {
            assert_eq!(new, old.to_string().replacen("\ns0\n", "\nS0\n", 1));
        }
        assert!(
            (N..=8 * N).contains(&read),
            "{read} blocks read for {N} blocks"
        );
    }

    #[test]
    fn changes_to_shared_text_reach_the_terms_of_an_m() {
        // The conflict of `p` and `q` over M, which stands for `terms`: the
        // person changes line 4 to `FOUR` in both sides, and line 2 to `TWO`
        // in the first side only. Every side shares lines 2 to 5 with V's VB,
        // so `FOUR` goes into every term, and `TWO`, which only one side
        // holds, into none.
        let terms = terms();
        let reference = vb_of(&terms);
        let reference_blocks = blocks(&reference).unwrap();
        let finder = BlockFinder::new(&reference, &reference_blocks);
        let layout = Layout::new(&finder, &terms, &options()).unwrap();
        let side = |first: &str, two: &str, four: &str| {
            BString::from(format!("{first}\n{two}\n3\n{four}\n5\n{first}6\n"))
        };
        let old_contents = Merge::from_vec(vec![
            side("p", "2", "4"),
            BString::from("<<<<<<< m\n"),
            side("q", "2", "4"),
        ]);
        let new_hunks = [Merge::from_vec(vec![
            side("p", "TWO", "FOUR"),
            BString::from("<<<<<<< m\n"),
            side("q", "2", "FOUR"),
        ])];
        let edits = shared_text_edits(&reference, &old_contents, 0, &new_hunks, &options());
        let edited = layout.edited(&edits);
        let expected = terms.map(|content| content.to_string().replace("\n4\n", "\nFOUR\n"));
        assert_eq!(
            edited.map(|content| String::from_utf8(content.clone()).unwrap()),
            expected
        );
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
            text: Text::literal(MARKER_SHAPED.as_bytes()),
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
