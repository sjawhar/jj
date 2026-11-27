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

//! The merge of many commits, one commit at a time, carried from one commit
//! to the next without writing the trees of its terms.
//!
//! [`crate::rewrite`] merges the commits of a merge one at a time: each commit
//! is merged into the merge of the commits before it, against V, the merge of
//! their merge bases, by the rule of [`crate::merge_rule`], and the result is
//! resolved. The merge gains the terms of V and of the commit at each step, so
//! the merge of a hundred commits has a hundred and more terms, and resolving
//! a step writes a tree for each term at each directory the step changes. That
//! is done again at every step.
//!
//! [`FoldedMerge`] is such a merge kept as one tree, which holds what every
//! term holds wherever the merge resolves, and, at each path where it does
//! not, that path's terms. A step merges only the paths where its result can
//! differ from the merge it starts from: where the rule writes a value, and
//! where the terms that the step adds do not cancel. At every other path the
//! merge keeps its value, and a path that does not resolve takes the added
//! terms after its own. The trees of the terms are written once, when the merge
//! is done. Each step gives, term for term and label for label, the merge that
//! [`crate::merge_rule::merge_parent_trees`] writes and resolves; a step this
//! module does not carry is left to it (see [`FoldedMerge::merge_next`]).

use std::collections::BTreeMap;
use std::collections::BTreeSet;
use std::collections::HashMap;
use std::iter;
use std::ops::Bound;
use std::sync::Arc;

use futures::StreamExt as _;
use futures::future::try_join_all;
use futures::try_join;
use itertools::Itertools as _;

use crate::backend::BackendResult;
use crate::backend::FileId;
use crate::backend::MergedTreeValue;
use crate::backend::TreeId;
use crate::backend::TreeValue;
use crate::conflict_labels::ConflictLabels;
use crate::matchers::EverythingMatcher;
use crate::merge::Merge;
use crate::merge_rule;
use crate::merged_tree::MergedTree;
use crate::merged_tree::inverted_bases;
use crate::merged_tree_builder::MergedTreeBuilder;
use crate::repo_path::RepoPath;
use crate::repo_path::RepoPathBuf;
use crate::store::Store;
use crate::tree::Tree;
use crate::tree_builder::TreeBuilder;
use crate::tree_merge::merge_simplified_file_values;
use crate::virtual_base;

/// A path where a [`FoldedMerge`] does not resolve.
#[derive(Clone, Debug)]
struct Conflict {
    /// The value of each term of the merge at the path.
    terms: MergedTreeValue,
    /// `terms`, simplified, where that is known.
    simplified: Option<MergedTreeValue>,
    /// The keys of the terms whose files a merge of the path failed to merge.
    /// A merge of terms with the same key gives what it gave.
    failed: Vec<FileMergeKey>,
}

/// What merging a path gives.
enum Outcome {
    /// The path's value, the same in every term.
    Resolved(Option<TreeValue>),
    /// The path does not resolve.
    Conflict(Conflict),
}

/// A merge of trees, as [`crate::merged_tree::MergedTree::resolve`] leaves it,
/// whose trees are not written: the tree `resolved`, which holds what every
/// term holds wherever the merge resolves and holds nothing at the paths of
/// `conflicts`, and, at each of those, the value of each term. The tree of a
/// term is `resolved` with that term's value written at each path of
/// `conflicts`. No path of `conflicts` is inside another, inside a file of
/// `resolved`, or holds a directory in any term.
#[derive(Clone, Debug)]
pub(crate) struct FoldedMerge {
    store: Arc<Store>,
    resolved: TreeId,
    conflicts: BTreeMap<RepoPathBuf, Conflict>,
    /// The number of terms, one where the merge resolves.
    num_terms: usize,
    labels: ConflictLabels,
}

impl FoldedMerge {
    /// `tree` as a folded merge, or none where its terms differ at a path
    /// where any of them holds a directory. `merged` says that `tree` is what
    /// [`MergedTree::resolve`] gave and that its terms were not simplified
    /// after the merge of its trees: at each path where its terms differ, that
    /// merge tried to merge the files of exactly those terms, and failed.
    pub(crate) async fn from_tree(tree: &MergedTree, merged: bool) -> BackendResult<Option<Self>> {
        let store = tree.store().clone();
        let tree_ids = tree.tree_ids();
        let num_terms = tree_ids.as_slice().len();
        let first = tree_ids.first().clone();
        if num_terms == 1 {
            return Ok(Some(Self {
                store,
                resolved: first,
                conflicts: BTreeMap::new(),
                num_terms,
                labels: ConflictLabels::unlabeled(),
            }));
        }
        let first_tree = MergedTree::resolved(store.clone(), first.clone());
        let mut paths = BTreeSet::new();
        for id in tree_ids.iter().skip(1).unique() {
            let other = MergedTree::resolved(store.clone(), id.clone());
            let mut diff = first_tree.diff_stream(&other, &EverythingMatcher);
            while let Some(entry) = diff.next().await {
                entry.values?;
                paths.insert(entry.path);
            }
        }
        // Terms that are the same tree, which the merge would drop in pairs.
        if paths.is_empty() {
            return Ok(None);
        }
        let mut builder = TreeBuilder::new(store.clone(), first);
        let mut conflicts = BTreeMap::new();
        for path in paths {
            let terms = tree.term_values(&path).await?;
            if holds_tree(&terms) {
                return Ok(None);
            }
            builder.remove(path.clone());
            let simplified = terms.simplify();
            let conflict = Conflict {
                failed: if merged {
                    FileMergeKey::new(&simplified).into_iter().collect()
                } else {
                    vec![]
                },
                terms,
                simplified: Some(simplified),
            };
            conflicts.insert(path, conflict);
        }
        Ok(Some(Self {
            store,
            resolved: builder.write_tree().await?,
            conflicts,
            num_terms,
            labels: tree.labels().clone(),
        }))
    }

    /// The merge, with a tree written for each of its terms.
    pub(crate) async fn into_tree(self) -> BackendResult<MergedTree> {
        if self.conflicts.is_empty() {
            return Ok(MergedTree::resolved(self.store, self.resolved));
        }
        let tree_ids = Merge::from_vec(vec![self.resolved; self.num_terms]);
        let tree = MergedTree::new(self.store, tree_ids, self.labels);
        let mut builder = MergedTreeBuilder::new(tree);
        for (path, conflict) in self.conflicts {
            builder.set_or_remove(path, conflict.terms);
        }
        builder.write_tree_no_resolve().await
    }

    /// Merges `next`, the tree of the next commit with its label, into this
    /// merge, whose label is `label` where it resolves, against `base`, V with
    /// its label, and resolves the result, as
    /// [`merge_rule::merge_parent_trees`] does.
    ///
    /// None where the step is one that this merge does not carry: where a path
    /// that the step merges holds a directory in some term and something else
    /// in another, or holds a directory with a conflict inside it; where the
    /// merge would replace a virtual merge base with the terms it names; and
    /// where, before it resolves, the merge would drop a pair of terms of which
    /// one is this merge's.
    pub(crate) async fn merge_next(
        &self,
        label: &str,
        (base, base_label): (&MergedTree, &str),
        (next, next_label): (&MergedTree, &str),
    ) -> BackendResult<Option<Self>> {
        let store = &self.store;
        let num_terms = self.num_terms;
        let resolved = store.get_tree(RepoPathBuf::root(), &self.resolved).await?;

        // The values of the merge rule, written over the terms of the merge.
        let paths = merge_rule::override_paths(
            store,
            num_terms > 1,
            || Ok(self.conflict_values()),
            base,
            next,
        )
        .await?;
        let mut overrides = BTreeMap::new();
        for path in paths {
            let Some(own) = self.term_values(&resolved, &path).await? else {
                return Ok(None);
            };
            // A directory of this merge where the rule writes a value.
            if holds_tree(&own) {
                return Ok(None);
            }
            let (base_terms, next_terms) =
                try_join!(base.term_values(&path), next.term_values(&path))?;
            let value = merge_rule::merge_parent_values(
                store,
                &path,
                &base_terms,
                base.labels(),
                &own,
                &next_terms,
            )
            .await?;
            if holds_tree(&value) {
                return Ok(None);
            }
            overrides.insert(path, value);
        }

        // Where the merge would replace a virtual merge base with the terms
        // it names.
        let override_paths: Vec<&RepoPath> = overrides.keys().map(|path| path.as_ref()).collect();
        let candidates = inverted_bases([base], override_paths)?;
        for (path, candidates) in &candidates {
            let Some(own) = self.term_values(&resolved, path).await? else {
                return Ok(None);
            };
            let (base_terms, next_terms) =
                try_join!(base.term_values(path), next.term_values(path))?;
            let terms = Merge::from_vec(vec![own, base_terms, next_terms]).flatten();
            let mut named = virtual_base::NamedTermsCache::new();
            if virtual_base::replaces_at(store, path, &terms, candidates, &mut named).await? {
                return Ok(None);
            }
        }

        // The terms of the base and of the next commit, after this merge's,
        // in the order the merge flattens them, with the rule's values.
        let mut others = Merge::from_vec(vec![
            Merge::resolved(store.empty_tree_id().clone()),
            base.tree_ids().clone(),
            next.tree_ids().clone(),
        ])
        .flatten()
        .into_iter()
        .skip(1)
        .collect_vec();
        if !overrides.is_empty() {
            for (index, id) in others.iter_mut().enumerate() {
                let mut builder = TreeBuilder::new(store.clone(), id.clone());
                for (path, value) in &overrides {
                    builder
                        .set_or_remove(path.clone(), value.as_slice()[num_terms + index].clone());
                }
                *id = builder.write_tree().await?;
            }
        }

        // The merge simplifies its terms, by their trees, before it resolves.
        // This merge's terms can differ only at the paths of its conflicts and
        // of the rule's values.
        let paths: Vec<&RepoPath> = self
            .conflicts
            .keys()
            .merge(overrides.keys())
            .dedup()
            .map(|path| path.as_ref())
            .collect();
        let own_columns = (0..num_terms)
            .map(|term| {
                paths
                    .iter()
                    .map(|path| own_value(&overrides, &self.conflicts, path, term))
                    .collect_vec()
            })
            .collect_vec();
        for id in others.iter().unique() {
            if let Some(values) = self.values_where_alike(id, &paths).await?
                && own_columns
                    .iter()
                    .any(|column| column.iter().copied().eq(values.iter().map(Option::as_ref)))
            {
                return Ok(None);
            }
        }
        let keys = classes(&own_columns)
            .into_iter()
            .map(TermKey::Own)
            .chain(others.iter().map(TermKey::Other))
            .collect_vec();
        let order =
            Merge::from_vec((0..keys.len()).collect_vec()).simplify_by(|&index| &keys[index]);
        if !order.iter().take(num_terms).copied().eq(0..num_terms) {
            return Ok(None);
        }
        let others = order.as_slice()[num_terms..]
            .iter()
            .map(|&index| others[index - num_terms].clone())
            .collect_vec();
        let own_labels = if num_terms == 1 {
            Merge::resolved(label.to_owned())
        } else if self.labels.has_labels() {
            self.labels.as_merge().clone()
        } else {
            Merge::repeated(String::new(), num_terms.div_ceil(2))
        };
        let flattened_labels = ConflictLabels::from_merge(
            Merge::from_vec(vec![
                own_labels,
                base.labels_by_term(base_label)
                    .map(|&label| label.to_owned()),
                next.labels_by_term(next_label)
                    .map(|&label| label.to_owned()),
            ])
            .flatten(),
        );
        let labels = if flattened_labels.has_labels() {
            ConflictLabels::from_merge(
                order.map(|&index| flattened_labels.as_merge().as_slice()[index].clone()),
            )
        } else {
            ConflictLabels::unlabeled()
        };

        let Some((resolved, conflicts)) = self.resolve(&resolved, &overrides, &others).await?
        else {
            return Ok(None);
        };
        let num_terms = num_terms + others.len();
        Ok(Some(
            self.simplified(resolved, conflicts, num_terms, labels),
        ))
    }

    /// The tree and the conflicts of this merge with the terms `others` after
    /// its own and with `overrides` written over them, resolved as
    /// [`MergedTree::resolve`] resolves it, before it simplifies the terms; or
    /// none where a path to merge holds a directory in some term and
    /// something else in another.
    async fn resolve(
        &self,
        resolved: &Tree,
        overrides: &BTreeMap<RepoPathBuf, MergedTreeValue>,
        others: &[TreeId],
    ) -> BackendResult<Option<(TreeId, BTreeMap<RepoPathBuf, Conflict>)>> {
        let store = &self.store;
        let num_terms = self.num_terms;
        let other_trees = try_join_all(
            others
                .iter()
                .map(|id| store.get_tree(RepoPathBuf::root(), id)),
        )
        .await?;
        // The paths where the added terms may not cancel.
        let mut changed: BTreeSet<RepoPathBuf> = overrides.keys().cloned().collect();
        if let Some((first, rest)) = others.split_first() {
            let first_tree = MergedTree::resolved(store.clone(), first.clone());
            for id in rest.iter().unique().filter(|&id| id != first) {
                let other = MergedTree::resolved(store.clone(), id.clone());
                let mut diff = first_tree.diff_stream(&other, &EverythingMatcher);
                while let Some(entry) = diff.next().await {
                    entry.values?;
                    changed.insert(entry.path);
                }
            }
        }

        let mut builder = TreeBuilder::new(store.clone(), self.resolved.clone());
        let mut conflicts = BTreeMap::new();
        for path in &changed {
            let inside_conflict = path
                .ancestors()
                .skip(1)
                .any(|ancestor| self.conflicts.contains_key(ancestor));
            if inside_conflict || self.has_conflict_below(path) {
                return Ok(None);
            }
            let Some(resolved_value) = leaf_value(resolved, path).await? else {
                return Ok(None);
            };
            let own = match (overrides.get(path), self.conflicts.get(path)) {
                (Some(value), _) => Merge::from_vec(value.as_slice()[..num_terms].to_vec()),
                (None, Some(conflict)) => conflict.terms.clone(),
                // A directory of this merge where another term holds a file.
                (None, None) if matches!(resolved_value, Some(TreeValue::Tree(_))) => {
                    return Ok(None);
                }
                (None, None) => Merge::from_vec(vec![resolved_value; num_terms]),
            };
            let mut terms = own.into_iter().collect_vec();
            for tree in &other_trees {
                let Some(value) = leaf_value(tree, path).await? else {
                    return Ok(None);
                };
                terms.push(value);
            }
            let terms = Merge::from_vec(terms);
            if holds_tree(&terms) {
                return Ok(None);
            }
            let failed = self
                .conflicts
                .get(path)
                .map(|conflict| conflict.failed.clone())
                .unwrap_or_default();
            match merge_path(store, path, terms, None, failed).await? {
                Outcome::Resolved(value) => builder.set_or_remove(path.clone(), value),
                Outcome::Conflict(conflict) => {
                    if !self.conflicts.contains_key(path) {
                        builder.remove(path.clone());
                    }
                    conflicts.insert(path.clone(), conflict);
                }
            }
        }

        // Every other conflict takes the added terms, which are the same at
        // its path, so they cancel in pairs.
        for (path, conflict) in &self.conflicts {
            if changed.contains(path) {
                continue;
            }
            let (terms, simplified) = match other_trees.first() {
                None => (conflict.terms.clone(), conflict.simplified.clone()),
                Some(tree) => {
                    let Some(value) = leaf_value(tree, path).await? else {
                        return Ok(None);
                    };
                    let terms = conflict
                        .terms
                        .iter()
                        .cloned()
                        .chain(iter::repeat_n(value.clone(), others.len()))
                        .collect_vec();
                    let simplified = conflict.simplified.as_ref().map(|simplified| {
                        (0..others.len() / 2).fold(simplified.clone(), |simplified, _| {
                            simplified_with_pair(&simplified, value.as_ref())
                        })
                    });
                    (Merge::from_vec(terms), simplified)
                }
            };
            // Terms whose files failed to merge before fail again, and terms
            // that cancel do not resolve them trivially either, so
            // `merge_path` reads the files at such a path no more than once.
            let outcome =
                merge_path(store, path, terms, simplified, conflict.failed.clone()).await?;
            match outcome {
                Outcome::Resolved(value) => builder.set_or_remove(path.clone(), value),
                Outcome::Conflict(conflict) => {
                    conflicts.insert(path.clone(), conflict);
                }
            }
        }
        Ok(Some((builder.write_tree().await?, conflicts)))
    }

    /// The merge of `num_terms` terms whose resolved tree is `resolved`, with
    /// `conflicts` and `labels`, with its terms simplified as
    /// [`MergedTree::resolve`] simplifies them: by their trees, which differ
    /// only at the paths of `conflicts`.
    fn simplified(
        &self,
        resolved: TreeId,
        conflicts: BTreeMap<RepoPathBuf, Conflict>,
        num_terms: usize,
        labels: ConflictLabels,
    ) -> Self {
        let store = self.store.clone();
        if conflicts.is_empty() {
            return Self {
                store,
                resolved,
                conflicts,
                num_terms: 1,
                labels: ConflictLabels::unlabeled(),
            };
        }
        let keys = {
            let columns = (0..num_terms)
                .map(|term| {
                    conflicts
                        .values()
                        .map(|conflict| conflict.terms.as_slice()[term].as_ref())
                        .collect_vec()
                })
                .collect_vec();
            classes(&columns)
        };
        let order = Merge::from_vec((0..num_terms).collect_vec()).simplify_by(|&index| keys[index]);
        if order.as_slice().len() == num_terms {
            return Self {
                store,
                resolved,
                conflicts,
                num_terms,
                labels,
            };
        }
        let conflicts = conflicts
            .into_iter()
            .map(|(path, conflict)| {
                let conflict = Conflict {
                    terms: order.map(|&index| conflict.terms.as_slice()[index].clone()),
                    simplified: None,
                    failed: conflict.failed,
                };
                (path, conflict)
            })
            .collect();
        let labels = if labels.has_labels() {
            ConflictLabels::from_merge(
                order.map(|&index| labels.as_merge().as_slice()[index].clone()),
            )
        } else {
            labels
        };
        Self {
            store,
            resolved,
            conflicts,
            num_terms: order.as_slice().len(),
            labels,
        }
    }

    /// The paths where the merge does not resolve, with their terms, as
    /// [`MergedTree::conflicts`] yields them for the merge's trees.
    fn conflict_values(&self) -> Vec<(RepoPathBuf, MergedTreeValue)> {
        self.conflicts
            .iter()
            .map(|(path, conflict)| (path.clone(), conflict.terms.clone()))
            .collect()
    }

    /// Whether a path of `conflicts` is inside `path`.
    fn has_conflict_below(&self, path: &RepoPath) -> bool {
        self.conflicts
            .range::<RepoPath, _>((Bound::Excluded(path), Bound::Unbounded))
            .next()
            .is_some_and(|(conflict_path, _)| conflict_path.starts_with(path))
    }

    /// The value of each term at `path`, as [`MergedTree::term_values`] gives
    /// it for the merge's trees, where `resolved` is the merge's resolved
    /// tree; none where `path` holds a directory with a conflict in it, or is
    /// inside a conflict's path.
    async fn term_values(
        &self,
        resolved: &Tree,
        path: &RepoPath,
    ) -> BackendResult<Option<MergedTreeValue>> {
        if let Some(conflict) = self.conflicts.get(path) {
            return Ok(Some(conflict.terms.clone()));
        }
        let inside_conflict = path
            .ancestors()
            .skip(1)
            .any(|ancestor| self.conflicts.contains_key(ancestor));
        if inside_conflict || self.has_conflict_below(path) {
            return Ok(None);
        }
        let value = resolved.path_value(path).await?;
        Ok(Some(Merge::from_vec(vec![value; self.num_terms])))
    }

    /// The values that the tree `id` holds at `paths`, which are sorted, where
    /// at every other path it holds what the merge's resolved tree holds; none
    /// where it holds something else at another path.
    async fn values_where_alike(
        &self,
        id: &TreeId,
        paths: &[&RepoPath],
    ) -> BackendResult<Option<Vec<Option<TreeValue>>>> {
        let resolved = MergedTree::resolved(self.store.clone(), self.resolved.clone());
        let other = MergedTree::resolved(self.store.clone(), id.clone());
        let mut diff = resolved.diff_stream(&other, &EverythingMatcher);
        while let Some(entry) = diff.next().await {
            entry.values?;
            let path: &RepoPath = &entry.path;
            if paths.binary_search(&path).is_err() {
                return Ok(None);
            }
        }
        let tree = self.store.get_tree(RepoPathBuf::root(), id).await?;
        let mut values = vec![];
        for path in paths {
            let Some(value) = leaf_value(&tree, path).await? else {
                return Ok(None);
            };
            values.push(value);
        }
        Ok(Some(values))
    }
}

/// How the merge tells terms apart before it resolves: a term of the merge
/// that the fold carries, by the class of its values, or another term, by its
/// tree.
#[derive(PartialEq)]
enum TermKey<'a> {
    Own(usize),
    Other(&'a TreeId),
}

/// The value that the term `term` of a merge holds at `path`, one of the paths
/// of `conflicts` or of `overrides`, with `overrides` written over the merge.
fn own_value<'a>(
    overrides: &'a BTreeMap<RepoPathBuf, MergedTreeValue>,
    conflicts: &'a BTreeMap<RepoPathBuf, Conflict>,
    path: &RepoPath,
    term: usize,
) -> Option<&'a TreeValue> {
    match overrides.get(path) {
        Some(value) => value.as_slice()[term].as_ref(),
        None => conflicts[path].terms.as_slice()[term].as_ref(),
    }
}

/// A class for each of `columns`: columns that hold the same values have the
/// same class.
fn classes(columns: &[Vec<Option<&TreeValue>>]) -> Vec<usize> {
    let mut seen: HashMap<&[Option<&TreeValue>], usize> = HashMap::new();
    columns
        .iter()
        .map(|column| {
            let next = seen.len();
            *seen.entry(column.as_slice()).or_insert(next)
        })
        .collect()
}

/// Whether any term of `terms` is a directory.
fn holds_tree(terms: &MergedTreeValue) -> bool {
    terms
        .iter()
        .flatten()
        .any(|value| matches!(value, TreeValue::Tree(_)))
}

/// The value at `path` in `tree`, a root tree, or none where something that is
/// not a directory holds `path`'s place above it.
async fn leaf_value(tree: &Tree, path: &RepoPath) -> BackendResult<Option<Option<TreeValue>>> {
    let Some((dir, basename)) = path.split() else {
        return Ok(None);
    };
    let mut current = tree.clone();
    for name in dir.components() {
        match current.value(name) {
            None => return Ok(Some(None)),
            Some(TreeValue::Tree(_)) => {
                current = current.sub_tree(name).await?.expect("a directory");
            }
            Some(_) => return Ok(None),
        }
    }
    Ok(Some(current.value(basename).cloned()))
}

/// Merges `terms`, the terms of a path that is a directory in none of them, as
/// [`crate::tree_merge::merge_trees`] merges such a path: trivially if it can,
/// and otherwise by merging the files of the simplified terms, unless their
/// key is among `failed`, whose files failed to merge before. `simplified` is
/// the simplified terms, where they are known.
async fn merge_path(
    store: &Store,
    path: &RepoPath,
    terms: MergedTreeValue,
    simplified: Option<MergedTreeValue>,
    mut failed: Vec<FileMergeKey>,
) -> BackendResult<Outcome> {
    if let Some(value) = terms.resolve_trivial(store.merge_options().same_change) {
        return Ok(Outcome::Resolved(value.clone()));
    }
    let simplified = simplified.unwrap_or_else(|| terms.simplify());
    let key = FileMergeKey::new(&simplified);
    if key.as_ref().is_none_or(|key| !failed.contains(key)) {
        if let Some(value) = merge_simplified_file_values(store, path, &simplified).await? {
            return Ok(Outcome::Resolved(Some(value)));
        }
        failed.extend(key);
    }
    Ok(Outcome::Conflict(Conflict {
        terms,
        simplified: Some(simplified),
        failed,
    }))
}

/// What the merge of the files of a path's terms, which are simplified, gives:
/// the terms' first base, and the number of sides and of bases that hold each
/// value. Terms with the same key give the same merge, whatever their order.
///
/// The merge of the files reads the terms as two sets, the bases and the
/// sides, and merges the contents of the files of the terms it cannot resolve
/// by counting them ([`crate::files::try_merge`]): it diffs every other term
/// against the first base, keeps the lines they all leave as they are, and
/// merges each hunk the diff leaves by counting the terms that hold each of
/// its texts, by line and then by word (`core/src/diff.rs`, the unchanged
/// regions of `ContentDiff::with_inputs_and_token_ranges`, intersected over
/// the other terms, and `files::resolve_diff_hunk`). The intersection does not
/// depend on the order of the terms it intersects over, two terms that hold
/// the same file give the same lines, and counting does not depend on order
/// either, so the merge depends on the first base and on the two sets alone.
/// The merge of the contents reads the terms simplified by their files, not by
/// their values, which can take another base: a key is not given for terms
/// where a file is another term's file with another executable bit or copy id.
#[derive(Clone, Debug, PartialEq)]
struct FileMergeKey {
    base: Option<TreeValue>,
    counts: HashMap<Option<TreeValue>, (usize, usize)>,
}

impl FileMergeKey {
    /// The key of `simplified`, terms that are simplified and do not resolve,
    /// or none where the merge of their files would simplify them again.
    fn new(simplified: &MergedTreeValue) -> Option<Self> {
        let mut counts: HashMap<Option<TreeValue>, (usize, usize)> = HashMap::new();
        let mut files: HashMap<&FileId, (usize, usize)> = HashMap::new();
        for (index, value) in simplified.iter().enumerate() {
            let is_add = index % 2 == 0;
            let count = counts.entry(value.clone()).or_default();
            *(if is_add { &mut count.0 } else { &mut count.1 }) += 1;
            if let Some(TreeValue::File { id, .. }) = value {
                let count = files.entry(id).or_default();
                *(if is_add { &mut count.0 } else { &mut count.1 }) += 1;
            }
        }
        if files
            .values()
            .any(|&(adds, removes)| adds > 0 && removes > 0)
        {
            return None;
        }
        let base = simplified.removes().next()?.clone();
        Some(Self { base, counts })
    }
}

/// `simplified`, the simplified terms of a path, followed by a remove and an
/// add of `value`, and simplified again: what [`Merge::simplify`] gives for the
/// path's terms followed by that pair.
///
/// [`Merge::simplify`] takes the adds in order, and each add cancels the first
/// remove that equals it; the add after that remove takes the place of the
/// canceled add. An add of the path's terms that equals `value` and that no
/// remove of theirs cancels cancels the pair's remove, and the pair's add, the
/// same value, takes its place. Otherwise the pair's add, which is taken last,
/// cancels the first remaining remove that equals `value`, and the add after
/// that remove goes to the end, after the pair's remove; and where no remove
/// equals it, the pair cancels itself.
fn simplified_with_pair(
    simplified: &MergedTreeValue,
    value: Option<&TreeValue>,
) -> MergedTreeValue {
    if simplified.adds().any(|add| add.as_ref() == value) {
        return simplified.clone();
    }
    let terms = simplified.as_slice();
    let first_remove = (1..terms.len())
        .step_by(2)
        .find(|&index| terms[index].as_ref() == value);
    match first_remove {
        None => simplified.clone(),
        Some(index) => {
            let mut moved = terms[..index].to_vec();
            moved.extend_from_slice(&terms[index + 2..]);
            moved.push(value.cloned());
            moved.push(terms[index + 1].clone());
            Merge::from_vec(moved)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn symlink(name: u8) -> Option<TreeValue> {
        Some(TreeValue::Symlink(crate::backend::SymlinkId::new(vec![
            name,
        ])))
    }

    /// The key tells apart terms that differ in their first base, which the
    /// merge of the contents takes as the base of its diff, and not terms
    /// that differ only in the order of the rest.
    #[test]
    fn test_file_merge_key_reads_the_first_base() {
        let [a, p, b, q, c] = [1, 2, 3, 4, 5].map(symlink);
        let terms = |values: [&Option<TreeValue>; 5]| {
            Merge::from_vec(values.map(|value| value.clone()).to_vec())
        };
        let first = FileMergeKey::new(&terms([&a, &p, &b, &q, &c]));
        let reordered = FileMergeKey::new(&terms([&c, &p, &b, &q, &a]));
        let another_base = FileMergeKey::new(&terms([&a, &q, &b, &p, &c]));
        assert!(first.is_some());
        assert_eq!(first, reordered, "the same terms with the same first base");
        assert_ne!(
            first, another_base,
            "the same terms with another first base"
        );
    }

    /// `simplified_with_pair` gives what simplifying the whole merge gives.
    #[test]
    fn test_simplified_with_pair() {
        let values = [None, symlink(1), symlink(2), symlink(3)];
        // Every merge of up to seven terms over four values, each followed by
        // a pair of each value.
        for len in [1, 3, 5, 7] {
            let merges = iter::repeat_n(values.iter(), len).multi_cartesian_product();
            for terms in merges {
                let terms = Merge::from_vec(terms.into_iter().cloned().collect_vec());
                for value in &values {
                    let whole = Merge::from_vec(
                        terms
                            .iter()
                            .cloned()
                            .chain([value.clone(), value.clone()])
                            .collect_vec(),
                    );
                    assert_eq!(
                        simplified_with_pair(&terms.simplify(), value.as_ref()),
                        whole.simplify(),
                        "{terms:?} followed by a pair of {value:?}"
                    );
                }
            }
        }
    }
}
