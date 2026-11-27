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
//! a virtual merge base that stands for V ([`crate::virtual_base`]): VB for a
//! conflict of text files, as git builds one, and otherwise M, so that the
//! merge is a conflict.

use std::collections::HashSet;
use std::sync::Arc;

use futures::StreamExt as _;
use futures::TryStreamExt as _;
use futures::try_join;
use itertools::Itertools as _;

use crate::backend::BackendResult;
use crate::backend::MergedTreeValue;
use crate::backend::TreeValue;
use crate::conflict_labels::ConflictLabels;
use crate::merge::Merge;
use crate::merge::SameChange;
use crate::merged_tree::MergedTree;
use crate::repo_path::RepoPath;
use crate::repo_path::RepoPathBuf;
use crate::store::Store;
use crate::virtual_base;

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
///
/// Returns with the merge the paths where it replaced a virtual merge base, as
/// [`MergedTree::merge_with_overrides`] says. The merge is resolved as
/// [`MergedTree::resolve_unchecked`] resolves it.
pub(crate) async fn merge_parent_trees(
    store: &Arc<Store>,
    side1: (MergedTree, String),
    base: (MergedTree, String),
    side2: (MergedTree, String),
    resolve: bool,
) -> BackendResult<(MergedTree, Vec<RepoPathBuf>)> {
    let overrides = rule_overrides(store, &side1.0, &base.0, &side2.0).await?;
    let merge = Merge::from_vec(vec![side1, base, side2]);
    let merge = merge.into_map(|(tree, label)| (tree, label, None));
    let (tree, replaced) = MergedTree::merge_replacing(merge, &overrides, false).await?;
    let tree = if resolve {
        tree.resolve_unchecked().await?
    } else {
        tree
    };
    Ok((tree, replaced))
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
    let paths = override_paths(
        store,
        side1.has_conflict(),
        || conflicts(side1),
        base,
        side2,
    )
    .await?;
    // The futures are made before they are buffered: with the closure that
    // makes them inside the stream, rustc cannot prove that the merge's future
    // is `Send`.
    let values = paths
        .iter()
        .map(|path| rule_override(store, side1, base, side2, path))
        .collect_vec();
    futures::stream::iter(values)
        .buffered(store.concurrency())
        .try_collect()
        .await
}

/// The paths where [`merge_parent_trees`] writes the value that
/// [`merge_parent_values`] gives over the merge of `side1` and `side2` against
/// `base`, where `side1_has_conflict` says whether `side1` has a conflict and
/// `side1_conflicts` gives its conflicts, as [`MergedTree::conflicts`] yields
/// them. No path is inside another.
pub(crate) async fn override_paths(
    store: &Store,
    side1_has_conflict: bool,
    side1_conflicts: impl FnOnce() -> BackendResult<Vec<(RepoPathBuf, MergedTreeValue)>>,
    base: &MergedTree,
    side2: &MergedTree,
) -> BackendResult<Vec<RepoPathBuf>> {
    let same_change = store.merge_options().same_change;
    let base_conflicts = conflicts(base)?;
    // Under `merge.same-change=accept`, a path where both sides hold the same
    // conflict takes the rule's value too. A side that holds no conflict at
    // all holds none of them, so the first side, which can be the merge of
    // many commits, is not read for a second side that has none.
    let both_sides_conflict =
        same_change == SameChange::Accept && side1_has_conflict && side2.has_conflict();
    let side1_conflicts = if both_sides_conflict || !base_conflicts.is_empty() {
        side1_conflicts()?
    } else {
        vec![]
    };
    let mut paths: HashSet<&RepoPath> = base_conflicts
        .iter()
        .map(|(path, _)| path.as_ref())
        .collect();
    // A flattened merge against a resolved V leaves a conflict that both sides
    // hold unresolved.
    if both_sides_conflict {
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
    // The value at a path decides everything inside it.
    Ok(paths
        .iter()
        .filter(|path| {
            !path
                .ancestors()
                .skip(1)
                .any(|ancestor| paths.contains(ancestor))
        })
        .map(|path| (*path).to_owned())
        .collect())
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
        side1.term_values(path),
        base.term_values(path),
        side2.term_values(path),
    )?;
    let value =
        merge_parent_values(store, path, &base_terms, base.labels(), &terms1, &terms2).await?;
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
    base_labels: &ConflictLabels,
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
        let virtual_base =
            virtual_base::virtual_merge_base(store, path, base, base_labels, parent1, parent2)
                .await?;
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
/// is the file [`virtual_base::virtual_merge_base`] writes, which names each
/// term of `base` with its label of `base_labels`. A parent's value takes its
/// parent's terms, and every other term is absent; the virtual base takes the
/// place of `base`'s first term.
pub async fn merge_parent_values(
    store: &Store,
    path: &RepoPath,
    base: &MergedTreeValue,
    base_labels: &ConflictLabels,
    parent1: &MergedTreeValue,
    parent2: &MergedTreeValue,
) -> BackendResult<MergedTreeValue> {
    let same_change = store.merge_options().same_change;
    let [base_value, value1, value2] =
        [base, parent1, parent2].map(|terms| trivially_merged(terms, same_change));
    let absent = |terms: &MergedTreeValue| Merge::repeated(None, terms.num_sides());
    let (segment1, base_segment, segment2) =
        match parent_merge(store, path, &base_value, base_labels, &value1, &value2).await? {
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
