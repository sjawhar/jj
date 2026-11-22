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

//! The merge of two trees against V, the merge of their merge bases, where V
//! does not resolve (<https://github.com/jj-vcs/jj/issues/6369>).
//!
//! Where V does not resolve at a path, a side that holds V's own conflict
//! counts as unchanged, and two sides that both changed it are merged against
//! a virtual base (VB): for a conflict of text files, jj's line merge of V's
//! terms with V's sides between marker lines where they do not merge, as git
//! builds one; otherwise M, a block of marker lines around one line for each of
//! V's terms, so the merge is a conflict. VB and M are only ever terms of a
//! conflict: in any merge jj makes, a path that would resolve to one of them
//! stays a conflict.

use std::collections::BTreeSet;
use std::collections::HashSet;
use std::sync::Arc;

use blake2::Blake2b512;
use bstr::BString;
use bstr::ByteSlice as _;
use digest::Digest as _;
use futures::StreamExt as _;
use futures::TryStreamExt as _;
use futures::future::try_join_all;
use futures::try_join;
use itertools::Itertools as _;

use crate::backend::BackendResult;
use crate::backend::CopyId;
use crate::backend::FileId;
use crate::backend::MergedTreeValue;
use crate::backend::MergedTreeValueExt as _;
use crate::backend::TreeValue;
use crate::conflicts::extract_as_single_hunk;
use crate::content_hash::ContentHash;
use crate::files;
use crate::files::MergeResult;
use crate::files::is_binary_content;
use crate::hex_util;
use crate::matchers::EverythingMatcher;
use crate::merge::Merge;
use crate::merge::SameChange;
use crate::merged_tree::MergedTree;
use crate::merged_tree_builder::MergedTreeBuilder;
use crate::object_id::ObjectId as _;
use crate::repo_path::RepoPath;
use crate::repo_path::RepoPathBuf;
use crate::store::Store;
use crate::tree_merge::MergeOptions;

/// Merges `side1` and `side2` against `base`, V, the merge of their merge
/// bases, and resolves the result if `resolve` is set.
///
/// Each path where V does not resolve, and, under `merge.same-change=accept`,
/// each path where both sides hold the same conflict, takes the value
/// [`merge_parent_values`] gives; below a conflict between a directory and
/// something else in either side or in V, the path of the highest such
/// conflict takes it instead. Each term of such a value keeps the label of the
/// side or the merge base it comes from. Every other path is merged as
/// [`MergedTree::merge`] merges it.
pub(crate) async fn merge_parent_trees(
    store: &Arc<Store>,
    side1: (MergedTree, String),
    base: (MergedTree, String),
    side2: (MergedTree, String),
    resolve: bool,
) -> BackendResult<MergedTree> {
    let overrides = rule_overrides(store, &side1.0, &base.0, &side2.0).await?;
    let merge = Merge::from_vec(vec![side1, base, side2]);
    merge_keeping_virtual_bases(merge, &overrides, resolve).await
}

/// Merges the trees of `merge`, with `overrides` written over their terms, and
/// resolves the result if `resolve` is set. Each override has one term for each
/// term of the trees, in the order [`MergedTree::merge_unsimplified`] gives.
///
/// VB and M are only ever terms of a conflict, but a merge can cancel every
/// other term of a conflict whose base is one: a rebase of a merge commit
/// whose own tree resolved the rule's conflict `[x, VB, y]` merges that
/// conflict as its base, so that `x` and `y` cancel and VB is left. Where the
/// merge would hold VB or M as a path's value, the path keeps a conflict
/// instead, which [`keep_conflicted`] builds from the merge's terms there; the
/// other paths are merged as they would be.
pub(crate) async fn merge_keeping_virtual_bases(
    merge: Merge<(MergedTree, String)>,
    overrides: &[(RepoPathBuf, MergedTreeValue)],
    resolve: bool,
) -> BackendResult<MergedTree> {
    // Only a term of a conflicted tree can be VB or M.
    if !merge.iter().any(|(tree, _)| tree.has_conflict()) {
        return write_merge(merge, overrides, resolve).await;
    }
    let tree = write_merge(merge.clone(), overrides, resolve).await?;
    let kept = virtual_base_conflicts(&merge, overrides, &tree).await?;
    if kept.is_empty() {
        return Ok(tree);
    }
    let overrides = overrides.iter().cloned().chain(kept).collect_vec();
    write_merge(merge, &overrides, resolve).await
}

/// The merge of the trees of `merge`, with `overrides` written over their
/// terms, resolved if `resolve` is set.
async fn write_merge(
    merge: Merge<(MergedTree, String)>,
    overrides: &[(RepoPathBuf, MergedTreeValue)],
    resolve: bool,
) -> BackendResult<MergedTree> {
    let tree = if overrides.is_empty() {
        MergedTree::merge_no_resolve(merge)
    } else {
        // Each override has one term for each term of the trees, so it is set
        // before their terms are simplified, and the result is simplified once,
        // with its labels.
        let mut builder = MergedTreeBuilder::new(MergedTree::merge_unsimplified(merge));
        for (path, value) in overrides {
            builder.set_or_remove(path.clone(), value.clone());
        }
        builder.write_tree_no_resolve().await?
    };
    if resolve {
        tree.resolve().await
    } else {
        Ok(tree)
    }
}

/// The paths where `tree`, the merge of `merge` with `overrides`, holds VB or
/// M as a resolved value, each with the conflict that the path keeps instead.
///
/// VB or M can only come from a term of a conflicted tree of `merge`, so only
/// the paths where the terms of such a tree differ are read, and only a file
/// that no tree of `merge` holds resolved there is read from the store. The
/// paths of `overrides` hold a parent's value or a conflict whose base is VB
/// or M, and are not read.
async fn virtual_base_conflicts(
    merge: &Merge<(MergedTree, String)>,
    overrides: &[(RepoPathBuf, MergedTreeValue)],
    tree: &MergedTree,
) -> BackendResult<Vec<(RepoPathBuf, MergedTreeValue)>> {
    let store = tree.store();
    let mut paths = BTreeSet::new();
    for (input, _) in merge {
        let term_ids = input.tree_ids();
        let first = MergedTree::resolved(store.clone(), term_ids.first().clone());
        for term_id in term_ids.iter().skip(1) {
            let term = MergedTree::resolved(store.clone(), term_id.clone());
            let mut diff = first.diff_stream(&term, &EverythingMatcher);
            while let Some(entry) = diff.next().await {
                entry.values?;
                paths.insert(entry.path);
            }
        }
    }
    let mut conflicts = vec![];
    for path in paths {
        if overrides
            .iter()
            .any(|(override_path, _)| path.starts_with(override_path))
        {
            continue;
        }
        let value = tree.path_value(&path).await?;
        let Some(Some(file @ TreeValue::File { .. })) = value.as_resolved() else {
            continue;
        };
        let input_terms =
            try_join_all(merge.iter().map(|(input, _)| term_values(input, &path))).await?;
        let held_resolved = input_terms
            .iter()
            .any(|terms| terms.iter().all(|term| term.as_ref() == Some(file)));
        if held_resolved || !is_virtual_base_value(store, &path, file).await? {
            continue;
        }
        let terms = Merge::from_vec(input_terms).flatten();
        conflicts.push((path, keep_conflicted(terms, file)));
    }
    Ok(conflicts)
}

/// The conflict that a path keeps where the merge of `terms`, the unsimplified
/// terms of a merge at the path, resolves to `value`, VB or M: each add that
/// holds `value`'s file is absent, the first remove is `value`, and every other
/// remove is absent. The other adds keep their places, and so their labels: a
/// rebase of a merge commit that resolved the rule's conflict `[x, VB, y]` to
/// `y`, onto the parent that holds `x`, keeps `[x, VB, y]`, with `x` labeled as
/// the rebase destination and `y` as the rebased commit.
fn keep_conflicted(terms: MergedTreeValue, value: &TreeValue) -> MergedTreeValue {
    fn file_id(value: &TreeValue) -> Option<&FileId> {
        match value {
            TreeValue::File { id, .. } => Some(id),
            _ => None,
        }
    }

    let values = terms
        .into_iter()
        .enumerate()
        .map(|(index, term)| match index {
            1 => Some(value.clone()),
            _ if index % 2 == 1 => None,
            _ if term.as_ref().and_then(file_id) == file_id(value) => None,
            _ => term,
        })
        .collect_vec();
    Merge::from_vec(values)
}

/// The values that [`merge_parent_trees`] writes over the merge of `side1`
/// and `side2` against `base`, with their paths. Each value has one term for
/// each term of the three trees, as [`merge_parent_values`] gives it.
async fn rule_overrides(
    store: &Arc<Store>,
    side1: &MergedTree,
    base: &MergedTree,
    side2: &MergedTree,
) -> BackendResult<Vec<(RepoPathBuf, MergedTreeValue)>> {
    let same_change = store.merge_options().same_change;
    let base_conflicts = conflicts(base)?;
    let side1_conflicts = if same_change == SameChange::Accept || !base_conflicts.is_empty() {
        conflicts(side1)?
    } else {
        vec![]
    };
    let mut paths: HashSet<&RepoPath> = base_conflicts
        .iter()
        .map(|(path, _)| path.as_ref())
        .collect();
    // A flattened merge against a resolved V leaves a conflict that both sides
    // hold unresolved.
    if same_change == SameChange::Accept {
        for (path, value1) in &side1_conflicts {
            if side2.path_value(path).await?.same_terms_as(value1) {
                paths.insert(path);
            }
        }
    }
    if paths.is_empty() {
        return Ok(vec![]);
    }
    // Below a conflict between a directory and something else, which
    // `conflicts()` yields at the directory's path, `path_value()` reads every
    // path as absent, so the rule applies at the highest such conflict above a
    // path.
    let side2_conflicts = conflicts(side2)?;
    let conflict_paths: HashSet<&RepoPath> = [&base_conflicts, &side1_conflicts, &side2_conflicts]
        .into_iter()
        .flatten()
        .map(|(path, _)| path.as_ref())
        .collect();
    let paths: HashSet<&RepoPath> = paths
        .into_iter()
        .map(|path| {
            path.ancestors()
                .skip(1)
                .filter(|ancestor| conflict_paths.contains(ancestor))
                .last()
                .unwrap_or(path)
        })
        .collect();
    // The value at a path decides everything inside it. The futures are made
    // before they are buffered: with the closure that makes them inside the
    // stream, rustc cannot prove that the merge's future is `Send`.
    let values = paths
        .iter()
        .filter(|path| {
            !path
                .ancestors()
                .skip(1)
                .any(|ancestor| paths.contains(ancestor))
        })
        .map(|path| rule_override(store, side1, base, side2, path))
        .collect_vec();
    futures::stream::iter(values)
        .buffered(store.concurrency())
        .try_collect()
        .await
}

/// The value that [`merge_parent_trees`] writes at `path`, with the path.
async fn rule_override(
    store: &Store,
    side1: &MergedTree,
    base: &MergedTree,
    side2: &MergedTree,
    path: &RepoPath,
) -> BackendResult<(RepoPathBuf, MergedTreeValue)> {
    let (terms1, base_terms, terms2) = try_join!(
        term_values(side1, path),
        term_values(base, path),
        term_values(side2, path),
    )?;
    let value = merge_parent_values(store, path, &base_terms, &terms1, &terms2).await?;
    Ok((path.to_owned(), value))
}

/// The paths where `tree` does not resolve, with their values, as
/// [`MergedTree::conflicts`] yields them.
fn conflicts(tree: &MergedTree) -> BackendResult<Vec<(RepoPathBuf, MergedTreeValue)>> {
    if !tree.has_conflict() {
        return Ok(vec![]);
    }
    tree.conflicts()
        .map(|(path, value)| Ok((path, value?)))
        .collect()
}

/// The value that each term of `tree` holds at `path`, one for each term.
async fn term_values(tree: &MergedTree, path: &RepoPath) -> BackendResult<MergedTreeValue> {
    let trees = tree.trees().await?;
    trees.try_map_async(|tree| tree.path_value(path)).await
}

/// `terms`, resolved where they merge trivially: the value that
/// [`MergedTree::path_value`] gives at a path below no conflict between a
/// directory and something else.
fn trivially_merged(terms: &MergedTreeValue, same_change: SameChange) -> MergedTreeValue {
    match terms.resolve_trivial(same_change) {
        Some(value) => Merge::resolved(value.clone()),
        None => terms.clone(),
    }
}

/// How the rule of [`merge_parent_values`] merges two parents at a path.
enum ParentMerge {
    /// The first parent's value.
    Parent1,
    /// The second parent's value.
    Parent2,
    /// The three-way merge of the parents against the base.
    ThreeWay,
    /// The merge of the parents against this virtual base.
    VirtualBase(TreeValue),
}

/// How the values that two parents hold at `path` merge against `base`, each
/// merged trivially, as [`merge_parent_values`] says.
async fn parent_merge(
    store: &Store,
    path: &RepoPath,
    base: &MergedTreeValue,
    parent1: &MergedTreeValue,
    parent2: &MergedTreeValue,
) -> BackendResult<ParentMerge> {
    let same_change = store.merge_options().same_change;
    if base.resolve_trivial(same_change).is_some() {
        if same_change == SameChange::Accept && parent1.same_terms_as(parent2) {
            return Ok(ParentMerge::Parent1);
        }
        return Ok(ParentMerge::ThreeWay);
    }
    if parent1.same_terms_as(parent2) || parent2.same_terms_as(base) {
        Ok(ParentMerge::Parent1)
    } else if parent1.same_terms_as(base) {
        Ok(ParentMerge::Parent2)
    } else {
        let virtual_base = virtual_merge_base(store, path, base, parent1, parent2).await?;
        Ok(ParentMerge::VirtualBase(virtual_base))
    }
}

/// Merges the values that two parents, `parent1` and `parent2`, hold at
/// `path`, against `base`, the value that V, the merge of all their merge
/// bases, holds there. Each of the three holds one value for each term of its
/// tree, and the result holds one value for each of their terms, in the order
/// of `[parent1, base, parent2]`, flattened, so that each keeps its label in
/// the merge of the trees.
///
/// The rule reads the three values merged trivially. Where `base` resolves,
/// two parents that hold the same value give that value under the store's
/// `merge.same-change=accept`, and otherwise the result is the three-way merge
/// `[parent1, base, parent2]`. Where `base` does not resolve, under any
/// `merge.same-change` setting: two parents that hold the same value give that
/// value; a parent whose value has the same adds and the same removes as
/// `base` (as multisets, after `simplify()`) counts as unchanged, so the result
/// is the other parent's value, or `base` itself when both parents leave it;
/// and two parents that differ from `base` and from each other give
/// `[parent1, virtual base, parent2]`, left unresolved, where the virtual base
/// is the file [`virtual_merge_base`] writes. A parent's value takes its
/// parent's terms, and every other term is absent; the virtual base takes the
/// place of `base`'s first term.
pub async fn merge_parent_values(
    store: &Store,
    path: &RepoPath,
    base: &MergedTreeValue,
    parent1: &MergedTreeValue,
    parent2: &MergedTreeValue,
) -> BackendResult<MergedTreeValue> {
    let same_change = store.merge_options().same_change;
    let [base_value, value1, value2] =
        [base, parent1, parent2].map(|terms| trivially_merged(terms, same_change));
    let absent = |terms: &MergedTreeValue| Merge::repeated(None, terms.num_sides());
    let (segment1, base_segment, segment2) =
        match parent_merge(store, path, &base_value, &value1, &value2).await? {
            ParentMerge::Parent1 => (parent1.clone(), absent(base), absent(parent2)),
            ParentMerge::Parent2 => (absent(parent1), absent(base), parent2.clone()),
            ParentMerge::ThreeWay => (parent1.clone(), base.clone(), parent2.clone()),
            ParentMerge::VirtualBase(virtual_base) => {
                let mut virtual_base_terms = Merge::normal(virtual_base);
                virtual_base_terms.pad_to(base.num_sides(), &None);
                (parent1.clone(), virtual_base_terms, parent2.clone())
            }
        };
    Ok(Merge::from_vec(vec![segment1, base_segment, segment2]).flatten())
}

/// Writes to the store the virtual base against which [`merge_parent_values`]
/// merges two parents that both changed `base`, an unresolved merge, at
/// `path`, and returns it as a file value.
///
/// Where `base` is a conflict of text files (each add a regular file whose
/// first 8,000 bytes hold no NUL byte, each remove such a file or absent, read
/// as an empty file), the file is jj's line merge of `base`'s terms: each hunk
/// that merges keeps its merged text, and each hunk that does not is a block
/// that holds `base`'s sides at that hunk, in `base`'s order. Otherwise the
/// file is M, one block whose sides are one line for each term of `base`, in
/// `base`'s order, that names the term. A block is the marker line
/// `<<<<<<< jj virtual merge base <hex>`, its sides, with `======= <hex>`
/// between two sides, and `>>>>>>> <hex>`. `<hex>` is 32 hex digits of a hash
/// over a fixed tag, the block's ordinal among the file's blocks, its sides as
/// the block encloses them, and a salt, which goes up from 0 until no line of
/// any term of either parent equals a marker line; so the file shows that a
/// merge wrote it, wherever it is (see [`is_virtual_base_content`]). The file
/// is executable where the executable bits of `base` merge to `true`, and
/// keeps the copy id of `base` where its copy ids merge, else the placeholder.
/// The result does not depend on the order of the parents.
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
    // Only a line that starts like a marker line can equal one.
    let parent_lines: HashSet<&[u8]> = contents1
        .iter()
        .chain(contents2.iter())
        .flat_map(|content| content.lines())
        .filter(|line| {
            [
                VIRTUAL_BASE_OPEN,
                VIRTUAL_BASE_SEPARATOR,
                VIRTUAL_BASE_CLOSE,
            ]
            .iter()
            .any(|prefix| line.starts_with(prefix.as_bytes()))
        })
        .collect();
    let hunks = match base_contents {
        Some(base_contents) => text_hunks(&base_contents, store.merge_options()),
        None => {
            let lines = base.iter().map(|term| term_line(term.as_ref())).collect();
            vec![VirtualBaseHunk::Block(lines)]
        }
    };
    let content = virtual_base_content(&hunks, &parent_lines);
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

/// Whether `value` is VB or M, which a merge writes only as a term of a
/// conflict: a file whose content [`is_virtual_base_content`] accepts.
pub async fn is_virtual_base_value(
    store: &Store,
    path: &RepoPath,
    value: &TreeValue,
) -> BackendResult<bool> {
    let TreeValue::File { id, .. } = value else {
        return Ok(false);
    };
    let content = extract_as_single_hunk(&Merge::normal(id.clone()), store, path).await?;
    Ok(is_virtual_base_content(content.first()))
}

/// Whether `content` is VB or M as [`virtual_merge_base`] writes it: it holds
/// at least one block, and at one salt, from 0 up to its number of lines, the
/// hex of each block is the hash of the block's ordinal and the sides it
/// encloses. Text whose marker lines no merge wrote is neither, unless it is a
/// byte copy of one.
pub fn is_virtual_base_content(content: &[u8]) -> bool {
    let Some(blocks) = virtual_base_blocks(content) else {
        return false;
    };
    if blocks.is_empty() {
        return false;
    }
    let max_salt = content.lines_with_terminator().count() as u64;
    (0..=max_salt).any(|salt| {
        blocks.iter().enumerate().all(|(ordinal, block)| {
            block.hex == block_hex(ordinal, block.sides.as_slice(), salt).as_bytes()
        })
    })
}

/// The tag that the hex of every block's marker lines hashes first.
const VIRTUAL_BASE_TAG: &str = "jj virtual merge base";
/// The start of the opening marker line of a block.
const VIRTUAL_BASE_OPEN: &str = "<<<<<<< jj virtual merge base ";
/// The start of the marker line between two sides of a block.
const VIRTUAL_BASE_SEPARATOR: &str = "======= ";
/// The start of the closing marker line of a block.
const VIRTUAL_BASE_CLOSE: &str = ">>>>>>> ";

/// 32 hex digits of a hash over [`VIRTUAL_BASE_TAG`], `ordinal`, the ordinal
/// of a block among the blocks of a file, `sides`, the sides it encloses, and
/// `salt`.
fn block_hex(ordinal: usize, sides: &[impl AsRef<[u8]>], salt: u64) -> String {
    let mut hasher = Blake2b512::default();
    ContentHash::hash(VIRTUAL_BASE_TAG, &mut hasher);
    ContentHash::hash(&(ordinal as u64), &mut hasher);
    ContentHash::hash(&(sides.len() as u64), &mut hasher);
    for side in sides {
        ContentHash::hash(side.as_ref(), &mut hasher);
    }
    ContentHash::hash(&salt, &mut hasher);
    hex_util::encode_hex(&hasher.finalize()[..16])
}

/// The opening, separator and closing marker lines of a block whose hex is
/// `hex`, without their newlines.
fn marker_lines(hex: &str) -> [String; 3] {
    [
        VIRTUAL_BASE_OPEN,
        VIRTUAL_BASE_SEPARATOR,
        VIRTUAL_BASE_CLOSE,
    ]
    .map(|prefix| format!("{prefix}{hex}"))
}

/// A block of a file that holds VB or M: the hex of its marker lines, and its
/// sides, each with its newlines.
struct Block<'a> {
    hex: &'a [u8],
    sides: Vec<&'a [u8]>,
}

/// The blocks of `content`, in order, or `None` if a block does not close. A
/// block opens at a line that is [`VIRTUAL_BASE_OPEN`] and a hex, and each of
/// its sides ends at the next line that is [`VIRTUAL_BASE_SEPARATOR`] or
/// [`VIRTUAL_BASE_CLOSE`] and the same hex.
fn virtual_base_blocks(content: &[u8]) -> Option<Vec<Block<'_>>> {
    let mut blocks = vec![];
    // The block being read, and the offset at which its current side starts.
    let mut current: Option<(Block<'_>, usize)> = None;
    let mut offset = 0;
    for line in content.lines_with_terminator() {
        let line_start = offset;
        offset += line.len();
        match current.take() {
            None => {
                if let Some(hex) = marker_hex(line, VIRTUAL_BASE_OPEN) {
                    let block = Block { hex, sides: vec![] };
                    current = Some((block, offset));
                }
            }
            Some((mut block, side_start)) => {
                let is_separator = marker_hex(line, VIRTUAL_BASE_SEPARATOR) == Some(block.hex);
                let is_close = marker_hex(line, VIRTUAL_BASE_CLOSE) == Some(block.hex);
                if is_separator || is_close {
                    block.sides.push(&content[side_start..line_start]);
                }
                if is_close {
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

/// The hex of `line` if `line` is `prefix`, 32 lowercase hex digits and a
/// newline.
fn marker_hex<'a>(line: &'a [u8], prefix: &str) -> Option<&'a [u8]> {
    let hex = line.strip_prefix(prefix.as_bytes())?.strip_suffix(b"\n")?;
    let is_hex = hex.len() == 32
        && hex
            .iter()
            .all(|&byte| matches!(byte, b'0'..=b'9' | b'a'..=b'f'));
    is_hex.then_some(hex)
}

/// A hunk of a file that holds VB or M: text that V's terms merge to, or the
/// sides of a block, each as the block encloses it.
enum VirtualBaseHunk {
    Resolved(Vec<u8>),
    Block(Vec<Vec<u8>>),
}

/// The hunks of the virtual base of `contents`, the terms of a conflict of
/// text files.
fn text_hunks(contents: &Merge<BString>, options: &MergeOptions) -> Vec<VirtualBaseHunk> {
    match files::merge_hunks(contents, options) {
        MergeResult::Resolved(content) => vec![VirtualBaseHunk::Resolved(content.into())],
        MergeResult::Conflict(hunks) => hunks
            .into_iter()
            .map(|hunk| match hunk.into_resolved() {
                Ok(content) => VirtualBaseHunk::Resolved(content.into()),
                Err(hunk) => {
                    VirtualBaseHunk::Block(hunk.adds().map(|side| enclosed(side)).collect())
                }
            })
            .collect(),
    }
}

/// `side` as a block encloses it: with a newline at its end, unless it is
/// empty.
fn enclosed(side: &[u8]) -> Vec<u8> {
    let mut side = side.to_vec();
    if side.last().is_some_and(|&byte| byte != b'\n') {
        side.push(b'\n');
    }
    side
}

/// The line of M that names `term`, a term of V.
fn term_line(term: Option<&TreeValue>) -> Vec<u8> {
    let name = match term {
        None => "absent".to_owned(),
        Some(TreeValue::File {
            id,
            executable,
            copy_id,
        }) => format!(
            "file {} executable={executable} copy-id={}",
            id.hex(),
            copy_id.hex()
        ),
        Some(TreeValue::Symlink(id)) => format!("symlink {}", id.hex()),
        Some(TreeValue::Tree(id)) => format!("tree {}", id.hex()),
        Some(TreeValue::GitSubmodule(id)) => format!("submodule {}", id.hex()),
    };
    format!("{name}\n").into_bytes()
}

/// The content of a file made of `hunks`: the text of each resolved hunk, and
/// each block's sides between marker lines whose hex is [`block_hex`] of the
/// block at the lowest salt at which no marker line is one of `parent_lines`.
fn virtual_base_content(hunks: &[VirtualBaseHunk], parent_lines: &HashSet<&[u8]>) -> Vec<u8> {
    let blocks = hunks
        .iter()
        .filter_map(|hunk| match hunk {
            VirtualBaseHunk::Resolved(_) => None,
            VirtualBaseHunk::Block(sides) => Some(sides),
        })
        .collect_vec();
    let hexes = (0..)
        .map(|salt| {
            blocks
                .iter()
                .enumerate()
                .map(|(ordinal, sides)| block_hex(ordinal, sides.as_slice(), salt))
                .collect_vec()
        })
        .find(|hexes| {
            hexes
                .iter()
                .flat_map(|hex| marker_lines(hex))
                .all(|line| !parent_lines.contains(line.as_bytes()))
        })
        .expect("some salt gives marker lines that no parent holds");
    let mut hexes = hexes.iter();
    let mut content = vec![];
    for hunk in hunks {
        match hunk {
            VirtualBaseHunk::Resolved(text) => content.extend_from_slice(text),
            VirtualBaseHunk::Block(sides) => {
                let hex = hexes.next().expect("a hex for each block");
                let [open, separator, close] = marker_lines(hex);
                push_line(&mut content, &open);
                for (index, side) in sides.iter().enumerate() {
                    if index > 0 {
                        push_line(&mut content, &separator);
                    }
                    content.extend_from_slice(side);
                }
                push_line(&mut content, &close);
            }
        }
    }
    content
}

/// Appends `line` and a newline to `content`, after a newline if `content`
/// does not end with one.
fn push_line(content: &mut Vec<u8>, line: &str) {
    if content.last().is_some_and(|&byte| byte != b'\n') {
        content.push(b'\n');
    }
    content.extend_from_slice(line.as_bytes());
    content.push(b'\n');
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
    let contents = extract_as_single_hunk(&file_ids, store, path).await?;
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
    extract_as_single_hunk(&file_ids, store, path).await
}
