// Copyright 2023 The Jujutsu Authors
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

//! A lazily merged view of a set of trees.

use std::collections::BTreeMap;
use std::collections::HashMap;
use std::collections::HashSet;
use std::collections::VecDeque;
use std::fmt;
use std::iter;
use std::iter::zip;
use std::ops::Range;
use std::pin::Pin;
use std::sync::Arc;
use std::task::Context;
use std::task::Poll;
use std::task::ready;
use std::vec;

use either::Either;
use futures::Stream;
use futures::StreamExt as _;
use futures::future::BoxFuture;
use futures::future::try_join;
use futures::stream::BoxStream;
use itertools::EitherOrBoth;
use itertools::Itertools as _;
use pollster::FutureExt as _;

use crate::backend::BackendError;
use crate::backend::BackendResult;
use crate::backend::CopyId;
use crate::backend::MergedTreeVal;
use crate::backend::MergedTreeValue;
use crate::backend::MergedTreeValueExt as _;
use crate::backend::TreeId;
use crate::backend::TreeValue;
use crate::conflict_labels::ConflictLabels;
use crate::copies::CopiesTreeDiffEntry;
use crate::copies::CopiesTreeDiffStream;
use crate::copies::CopyHistoryDiffStream;
use crate::copies::CopyHistoryTreeDiffEntry;
use crate::copies::CopyRecords;
use crate::matchers::EverythingMatcher;
use crate::matchers::Matcher;
use crate::merge::Diff;
use crate::merge::Merge;
use crate::merge::MergeBuilder;
use crate::merged_tree_builder::MergedTreeBuilder;
use crate::repo_path::RepoPath;
use crate::repo_path::RepoPathBuf;
use crate::repo_path::RepoPathComponent;
use crate::store::Store;
use crate::tree::ToTreeMergeExt as _;
use crate::tree::Tree;
use crate::tree::TreeMergeExt as _;
use crate::tree_merge::merge_trees;
use crate::virtual_base;

/// Presents a view of a merged set of trees at the root directory, as well as
/// conflict labels.
#[derive(Clone)]
pub struct MergedTree {
    store: Arc<Store>,
    tree_ids: Merge<TreeId>,
    labels: ConflictLabels,
}

impl fmt::Debug for MergedTree {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("MergedTree")
            .field("tree_ids", &self.tree_ids)
            .field("labels", &self.labels)
            .finish_non_exhaustive()
    }
}

impl MergedTree {
    /// Creates a `MergedTree` with the given resolved tree ID.
    pub fn resolved(store: Arc<Store>, tree_id: TreeId) -> Self {
        Self {
            store,
            tree_ids: Merge::resolved(tree_id),
            labels: ConflictLabels::unlabeled(),
        }
    }

    /// Creates a `MergedTree` with the given tree IDs.
    pub fn new(store: Arc<Store>, tree_ids: Merge<TreeId>, labels: ConflictLabels) -> Self {
        if let Some(num_sides) = labels.num_sides() {
            assert_eq!(tree_ids.num_sides(), num_sides);
        }
        Self {
            store,
            tree_ids,
            labels,
        }
    }

    /// The `Store` associated with this tree.
    pub fn store(&self) -> &Arc<Store> {
        &self.store
    }

    /// The underlying tree IDs for this `MergedTree`. If there are file changes
    /// between two trees, then the tree IDs will be different.
    pub fn tree_ids(&self) -> &Merge<TreeId> {
        &self.tree_ids
    }

    /// Extracts the underlying tree IDs for this `MergedTree`, discarding any
    /// conflict labels.
    pub fn into_tree_ids(self) -> Merge<TreeId> {
        self.tree_ids
    }

    /// Returns this merge's conflict labels, if any.
    pub fn labels(&self) -> &ConflictLabels {
        &self.labels
    }

    /// Returns both the underlying tree IDs and any conflict labels. This can
    /// be used to check whether there are changes in files to be materialized
    /// in the working copy.
    pub fn tree_ids_and_labels(&self) -> (&Merge<TreeId>, &ConflictLabels) {
        (&self.tree_ids, &self.labels)
    }

    /// Extracts the underlying tree IDs and conflict labels.
    pub fn into_tree_ids_and_labels(self) -> (Merge<TreeId>, ConflictLabels) {
        (self.tree_ids, self.labels)
    }

    /// Reads the merge of tree objects represented by this `MergedTree`.
    pub async fn trees(&self) -> BackendResult<Merge<Tree>> {
        self.tree_ids
            .try_map_async(|id| self.store.get_tree(RepoPathBuf::root(), id))
            .await
    }

    /// Returns a label for each term in a merge. Resolved merges use the
    /// provided label, while conflicted merges keep their original labels.
    /// Missing labels are indicated by empty strings.
    pub fn labels_by_term<'a>(&'a self, label: &'a str) -> Merge<&'a str> {
        if self.tree_ids.is_resolved() {
            assert!(!self.labels.has_labels());
            Merge::resolved(label)
        } else if self.labels.has_labels() {
            // If the merge is conflicted and it already has labels, then we want to use
            // those labels instead of the provided label. This ensures that rebasing
            // conflicted commits keeps meaningful labels.
            let labels = self.labels.as_merge();
            assert_eq!(labels.num_sides(), self.tree_ids.num_sides());
            labels.map(|label| label.as_str())
        } else {
            // If the merge is conflicted but it doesn't have labels (e.g. conflicts created
            // before labels were added), then we use empty strings to indicate missing
            // labels. We could consider using `label` for all the sides instead, but it
            // might be confusing.
            Merge::repeated("", self.tree_ids.num_sides())
        }
    }

    /// Tries to resolve any conflicts, resolving any conflicts that can be
    /// automatically resolved and leaving the rest unresolved.
    pub async fn resolve(self) -> BackendResult<Self> {
        let merged = merge_trees(&self.store, self.tree_ids).await?;
        // If the result can be resolved, then `merge_trees()` above would have returned
        // a resolved merge. However, that function will always preserve the arity of
        // conflicts it cannot resolve. So we simplify the conflict again
        // here to possibly reduce a complex conflict to a simpler one.
        let (simplified_labels, simplified) = if merged.is_resolved() {
            (ConflictLabels::unlabeled(), merged)
        } else {
            self.labels.simplify_with(&merged)
        };
        // If debug assertions are enabled, check that the merge was idempotent. In
        // particular, that this last simplification doesn't enable further automatic
        // resolutions
        if cfg!(debug_assertions) {
            let re_merged = merge_trees(&self.store, simplified.clone()).await.unwrap();
            debug_assert_eq!(re_merged, simplified);
        }
        Ok(Self {
            store: self.store,
            tree_ids: simplified,
            labels: simplified_labels,
        })
    }

    /// An iterator over the conflicts in this tree, including subtrees.
    /// Recurses into subtrees and yields conflicts in those, but only if
    /// all sides are trees, so tree/file conflicts will be reported as a single
    /// conflict, not one for each path in the tree.
    pub fn conflicts(
        &self,
    ) -> impl Iterator<Item = (RepoPathBuf, BackendResult<MergedTreeValue>)> + use<> {
        self.conflicts_matching(&EverythingMatcher)
    }

    /// Like `conflicts()` but restricted by a matcher.
    pub fn conflicts_matching<'matcher>(
        &self,
        matcher: &'matcher dyn Matcher,
    ) -> impl Iterator<Item = (RepoPathBuf, BackendResult<MergedTreeValue>)> + use<'matcher> {
        ConflictIterator::new(self, matcher)
    }

    /// Whether this tree has conflicts.
    pub fn has_conflict(&self) -> bool {
        !self.tree_ids.is_resolved()
    }

    /// The value at the given path. The value can be `Resolved` even if
    /// `self` is a `Conflict`, which happens if the value at the path can be
    /// trivially merged.
    pub async fn path_value(&self, path: &RepoPath) -> BackendResult<MergedTreeValue> {
        match path.split() {
            Some((dir, basename)) => {
                let trees = self.trees().await?;
                match trees.sub_tree_recursive(dir).await? {
                    None => Ok(Merge::absent()),
                    Some(tree) => Ok(tree.value(basename).cloned()),
                }
            }
            None => Ok(self.to_merged_tree_value()),
        }
    }

    /// Returns the `TreeValue` associated with `id` if it exists at the
    /// expected path and is resolved.
    pub async fn copy_value(&self, id: &CopyId) -> BackendResult<Option<TreeValue>> {
        let copy = self.store().backend().read_copy(id).await?;
        let merged_val = self.path_value(&copy.current_path).await?;
        match merged_val.into_resolved() {
            Ok(Some(val)) if val.copy_id() == Some(id) => Ok(Some(val)),
            _ => Ok(None),
        }
    }

    fn to_merged_tree_value(&self) -> MergedTreeValue {
        self.tree_ids
            .map(|tree_id| Some(TreeValue::Tree(tree_id.clone())))
    }

    /// Iterator over the entries matching the given matcher. Subtrees are
    /// visited recursively. Subtrees that differ between the current
    /// `MergedTree`'s terms are merged on the fly. Missing terms are treated as
    /// empty directories. Subtrees that conflict with non-trees are not
    /// visited. For example, if current tree is a merge of 3 trees, and the
    /// entry for 'foo' is a conflict between a change subtree and a symlink
    /// (i.e. the subdirectory was replaced by symlink in one side of the
    /// conflict), then the entry for `foo` itself will be emitted, but no
    /// entries from inside `foo/` from either of the trees will be.
    pub fn entries(&self) -> TreeEntriesIterator<'static> {
        self.entries_matching(&EverythingMatcher)
    }

    /// Like `entries()` but restricted by a matcher.
    pub fn entries_matching<'matcher>(
        &self,
        matcher: &'matcher dyn Matcher,
    ) -> TreeEntriesIterator<'matcher> {
        TreeEntriesIterator::new(self, matcher)
    }

    /// Stream of the differences between this tree and another tree.
    fn diff_stream_internal<'matcher>(
        &self,
        other: &Self,
        matcher: &'matcher dyn Matcher,
    ) -> TreeDiffStream<'matcher> {
        let concurrency = self.store().concurrency();
        if concurrency <= 1 {
            futures::stream::iter(TreeDiffIterator::new(self, other, matcher)).boxed()
        } else {
            TreeDiffStreamImpl::new(self, other, matcher, concurrency).boxed()
        }
    }

    /// Stream of the differences between this tree and another tree.
    pub fn diff_stream<'matcher>(
        &self,
        other: &Self,
        matcher: &'matcher dyn Matcher,
    ) -> TreeDiffStream<'matcher> {
        stream_without_trees(self.diff_stream_internal(other, matcher))
    }

    /// Like `diff_stream()` but trees with diffs themselves are also included.
    pub fn diff_stream_with_trees<'matcher>(
        &self,
        other: &Self,
        matcher: &'matcher dyn Matcher,
    ) -> TreeDiffStream<'matcher> {
        self.diff_stream_internal(other, matcher)
    }

    /// Like `diff_stream()` but files in a removed tree will be returned before
    /// a file that replaces it.
    pub fn diff_stream_for_file_system<'matcher>(
        &self,
        other: &Self,
        matcher: &'matcher dyn Matcher,
    ) -> TreeDiffStream<'matcher> {
        DiffStreamForFileSystem::new(self.diff_stream_internal(other, matcher)).boxed()
    }

    /// Like `diff_stream()` but takes the given copy records into account.
    pub fn diff_stream_with_copies<'a>(
        &self,
        other: &Self,
        matcher: &'a dyn Matcher,
        copy_records: &'a CopyRecords,
    ) -> BoxStream<'a, CopiesTreeDiffEntry> {
        let stream = self.diff_stream(other, matcher);
        CopiesTreeDiffStream::new(stream, self.clone(), other.clone(), copy_records).boxed()
    }

    /// Like `diff_stream()` but takes CopyHistory into account.
    pub fn diff_stream_with_copy_history<'a>(
        &'a self,
        other: &'a Self,
        matcher: &'a dyn Matcher,
    ) -> BoxStream<'a, CopyHistoryTreeDiffEntry> {
        let stream = self.diff_stream(other, matcher);
        CopyHistoryDiffStream::new(stream, self, other).boxed()
    }

    /// Merges the provided trees into a single `MergedTree`. Any conflicts will
    /// be resolved recursively if possible. The provided labels are used if a
    /// conflict arises. However, if one of the input trees is already
    /// conflicted, the corresponding label will be ignored, and its existing
    /// labels will be used instead.
    ///
    /// A virtual merge base ([`crate::virtual_base`]) is only ever a base of a
    /// conflict: where the merge leaves one as a side of a path, the merge
    /// replaces every one at that path, sides and bases alike, with the terms
    /// of the conflict that it stands for.
    pub async fn merge(merge: Merge<(Self, String)>) -> BackendResult<Self> {
        Self::merge_with_overrides(merge, &[], &[], true).await
    }

    /// Merges the provided trees as [`Self::merge`] does, where each tree may
    /// come with `unresolved`, the merge that it is the resolution of, such as
    /// the merge of a commit's parents: at a path where the merge replaces a
    /// virtual merge base, each such tree enters as the terms of its
    /// unresolved merge there (each parent and the virtual merge base of their
    /// merge bases), not as the value they resolve to.
    pub(crate) async fn merge_with_unresolved(
        merge: Merge<(Self, String)>,
        unresolved: &[Option<Self>],
    ) -> BackendResult<Self> {
        Self::merge_with_overrides(merge, &[], unresolved, true).await
    }

    /// Merges the provided trees into a single `MergedTree`, without attempting
    /// to resolve file conflicts.
    ///
    /// Unlike [`Self::merge`], this keeps a virtual merge base that the merge
    /// leaves as a side, so `merge_no_resolve(..).resolve()` can differ from
    /// `merge(..)` where an input tree at a base of the merge holds a conflict
    /// whose base is a virtual merge base.
    pub fn merge_no_resolve(merge: Merge<(Self, String)>) -> Self {
        Self::merge_unsimplified(merge).simplified()
    }

    /// The merge of the provided trees with each term of each tree kept, as
    /// [`Self::merge_no_resolve`] builds it before it simplifies the terms:
    /// the terms of the first tree, then those of the second, inverted, and so
    /// on, each with its label.
    fn merge_unsimplified(merge: Merge<(Self, String)>) -> Self {
        debug_assert!(
            merge
                .iter()
                .map(|(tree, _)| Arc::as_ptr(tree.store()))
                .all_equal()
        );
        let store = merge.first().0.store().clone();
        let flattened_labels = ConflictLabels::from_merge(
            merge
                .map(|(tree, label)| tree.labels_by_term(label))
                .flatten()
                .map(|&label| label.to_owned()),
        );
        let flattened_tree_ids: Merge<TreeId> = merge
            .into_map(|(tree, _label)| tree.into_tree_ids())
            .flatten();
        Self::new(store, flattened_tree_ids, flattened_labels)
    }

    /// This merge with its terms simplified, with their labels.
    fn simplified(self) -> Self {
        let (labels, tree_ids) = self.labels.simplify_with(&self.tree_ids);
        Self::new(self.store, tree_ids, labels)
    }

    /// Merges the trees of `merge`, with `overrides` written over their terms,
    /// and resolves the result if `resolve` is set. Each override has one term
    /// for each term of the trees, in the order [`Self::merge_unsimplified`]
    /// gives. `unresolved` holds, by index in `merge`, the unresolved merge
    /// that a tree is the resolution of, if any.
    ///
    /// A virtual merge base, VB or M ([`crate::virtual_base`]), is only ever a
    /// base of a conflict, but a tree at a base of `merge` brings the bases of
    /// its conflicts into the merge as sides. At each path outside `overrides`
    /// where such a tree has a conflict, and where, once the merge's terms are
    /// simplified, such a base is a side and is VB or M, each tree that comes
    /// with its unresolved merge enters as the terms of that merge at the path,
    /// and every VB and M term at that path, sides and bases alike, is
    /// replaced with the terms of V that it names, inverted where it is a
    /// base, and so on for the VB and M among those, until none is left. The
    /// terms then merge as the terms of the merge bases' own conflicts would,
    /// whatever the order of `merge`. A path where VB or M is only a base is
    /// merged as it is. The decision reads the merge's terms and the files that
    /// may be VB or M, never merged text. Each term that replaces a VB or M
    /// takes the label that it names for that term, the label of the commit it
    /// comes from.
    pub(crate) async fn merge_with_overrides(
        merge: Merge<(Self, String)>,
        overrides: &[(RepoPathBuf, MergedTreeValue)],
        unresolved: &[Option<Self>],
        resolve: bool,
    ) -> BackendResult<Self> {
        let inverted_bases = inverted_bases(&merge, overrides)?;
        let mut substitutes = vec![];
        let mut start = 0;
        for (index, (tree, label)) in merge.iter().enumerate() {
            let end = start + tree.tree_ids().as_slice().len();
            if let Some(Some(unresolved)) = unresolved.get(index) {
                substitutes.push(Substitute {
                    slots: start..end,
                    tree: tree.clone(),
                    unresolved,
                    label: label.clone(),
                });
            }
            start = end;
        }
        let tree = Self::merge_unsimplified(merge);
        let store = tree.store();
        let mut replaced = vec![];
        for (path, candidates) in inverted_bases {
            let terms = tree.term_values(&path).await?;
            let mut named = HashMap::new();
            if replaces_at(store, &path, &terms, &candidates, &mut named).await? {
                replaced.push((path, terms, named));
            }
        }
        let mut replacements = Replacements {
            in_merge: 0,
            replaced_paths: replaced.len(),
        };
        let mut expansions = vec![];
        for (path, terms, named) in replaced {
            let expansion =
                Expansion::new(store, &path, terms, &substitutes, named, &mut replacements).await?;
            expansions.push((path, expansion));
        }
        let tree = if overrides.is_empty() && expansions.is_empty() {
            tree.simplified()
        } else {
            write_expanded(tree, overrides, expansions).await?
        };
        if resolve {
            tree.resolve().await
        } else {
            Ok(tree)
        }
    }

    /// The value that each term of this merge holds at `path`, one for each
    /// term, none of them merged with the others.
    pub(crate) async fn term_values(&self, path: &RepoPath) -> BackendResult<MergedTreeValue> {
        let trees = self.trees().await?;
        trees.try_map_async(|tree| tree.path_value(path)).await
    }
}

/// The files that the trees at the bases of `merge` hold as bases of their
/// conflicts, by path, outside the paths of `overrides`: in the merge, each is
/// a side.
fn inverted_bases(
    merge: &Merge<(MergedTree, String)>,
    overrides: &[(RepoPathBuf, MergedTreeValue)],
) -> BackendResult<BTreeMap<RepoPathBuf, HashSet<TreeValue>>> {
    let override_paths: HashSet<&RepoPath> =
        overrides.iter().map(|(path, _)| path.as_ref()).collect();
    let mut bases: BTreeMap<RepoPathBuf, HashSet<TreeValue>> = BTreeMap::new();
    for (tree, _) in merge.removes().filter(|(tree, _)| tree.has_conflict()) {
        for (path, value) in tree.conflicts() {
            let value = value?;
            if path
                .ancestors()
                .any(|ancestor| override_paths.contains(ancestor))
            {
                continue;
            }
            let files = value
                .removes()
                .flatten()
                .filter(|term| matches!(term, TreeValue::File { .. }));
            bases.entry(path).or_default().extend(files.cloned());
        }
    }
    bases.retain(|_, files| !files.is_empty());
    Ok(bases)
}

/// A term of a merge, with its label.
type LabeledTerm = (Option<TreeValue>, String);

/// The terms of V that a VB or M names, with their labels.
type NamedTerms = (MergedTreeValue, Merge<String>);

/// The most VB and M terms that a merge replaces at one path. A VB names at
/// most one VB or M for each level of merge bases below it, so a rewrite of the
/// top merge of a braid of 8,000 levels of merge bases replaces 7,999 at its
/// one path; a planted set of files that name one another can stand for more,
/// and the merge then fails, as a user error, rather than spend time and
/// memory on them.
const MAX_REPLACEMENTS_AT_A_PATH: usize = 10_000;

/// The most VB and M terms that a merge replaces in all is this, and
/// [`MAX_REPLACEMENTS_PER_REPLACED_PATH`] more for each path where it replaces
/// one: a real merge of many conflicted paths replaces about one at each, and
/// planted files at many paths get about this many each.
const MAX_REPLACEMENTS_IN_A_MERGE: usize = 10_000;

/// See [`MAX_REPLACEMENTS_IN_A_MERGE`].
const MAX_REPLACEMENTS_PER_REPLACED_PATH: usize = 100;

/// How many VB and M terms a merge has replaced, and at how many paths it
/// replaces them.
struct Replacements {
    in_merge: usize,
    replaced_paths: usize,
}

impl Replacements {
    /// Counts one more replacement at `path`, where `at_path` were made
    /// before it; fails, as a user error, once it passes the bound at a path
    /// or the bound of the merge.
    fn count(&mut self, path: &RepoPath, at_path: &mut usize) -> BackendResult<()> {
        *at_path += 1;
        self.in_merge += 1;
        let path = path.as_internal_file_string();
        if *at_path > MAX_REPLACEMENTS_AT_A_PATH {
            return Err(BackendError::Unsupported(format!(
                "The merge at {path} names more than {MAX_REPLACEMENTS_AT_A_PATH} virtual bases \
                 at that path, the bound at one path"
            )));
        }
        let bound =
            MAX_REPLACEMENTS_IN_A_MERGE + MAX_REPLACEMENTS_PER_REPLACED_PATH * self.replaced_paths;
        if self.in_merge > bound {
            let paths = self.replaced_paths;
            return Err(BackendError::Unsupported(format!(
                "The merge at {path} names more than {bound} virtual bases in all, the bound of a \
                 merge that replaces them at {paths} paths"
            )));
        }
        Ok(())
    }
}

/// A tree of a merge that comes with the unresolved merge that it resolves
/// (see [`MergedTree::merge_with_unresolved`]): the range of its terms among
/// the merge's terms, the tree, and that unresolved merge, with the tree's
/// label.
struct Substitute<'a> {
    slots: Range<usize>,
    tree: MergedTree,
    unresolved: &'a MergedTree,
    label: String,
}

/// The terms of a merge at a path once every VB and M among them is replaced
/// with the terms of V that it names (see [`MergedTree::merge_with_overrides`]):
/// the merge's own terms, each in its place, a replaced one absent, and the
/// terms that the replaced ones bring, as pairs of a remove and an add with
/// their labels, by the index of the merge's term that each came from.
struct Expansion {
    terms: Vec<Option<TreeValue>>,
    extra: BTreeMap<usize, Vec<[LabeledTerm; 2]>>,
    /// Where each term that is present is, by whether it is an add and by its
    /// value.
    present: HashMap<(bool, TreeValue), Vec<Slot>>,
}

/// Where a term of an [`Expansion`] is: the merge's own term at an index, or
/// the remove (0) or the add (1) of a pair among those that came from the
/// merge's term at an index.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Slot {
    Own(usize),
    Extra(usize, usize, usize),
}

impl Slot {
    /// Whether the term is an add.
    fn is_add(self) -> bool {
        match self {
            Self::Own(index) => index.is_multiple_of(2),
            Self::Extra(_, _, side) => side == 1,
        }
    }

    /// The index of the merge's own term that the term came from.
    fn origin(self) -> usize {
        match self {
            Self::Own(index) | Self::Extra(index, _, _) => index,
        }
    }
}

impl Expansion {
    /// `terms` at `path`, where [`replaces_at`] holds, with every VB and M
    /// replaced, sides and bases alike, nested ones too, with `named` the
    /// terms already looked up. Each of `substitutes` first enters as the
    /// terms of its unresolved merge at `path` in place of its own terms.
    /// Each VB and M replaced counts in `replacements`, which fails past its
    /// bounds.
    async fn new(
        store: &Store,
        path: &RepoPath,
        terms: MergedTreeValue,
        substitutes: &[Substitute<'_>],
        mut named: HashMap<TreeValue, Option<NamedTerms>>,
        replacements: &mut Replacements,
    ) -> BackendResult<Self> {
        let mut expansion = Self {
            terms: vec![None; terms.as_slice().len()],
            extra: BTreeMap::new(),
            present: HashMap::new(),
        };
        let mut pending = VecDeque::new();
        for (index, term) in terms.into_iter().enumerate() {
            expansion.set(Slot::Own(index), term);
            pending.push_back(Slot::Own(index));
        }
        // A substituted tree whose terms at `path` are those of its unresolved
        // merge keeps them, with their places and labels. The terms of each
        // other one are made absent first: they are one more of the tree's
        // sign than of the other, so they leave one absent term of that sign,
        // which the absent term that starts the unresolved merge's pairs
        // cancels (see [`Self::replace`]). The tree's first term has the
        // tree's sign, so the unresolved merge takes its place. No unresolved
        // term can then cancel a term that is being replaced.
        let mut entering = vec![];
        for substitute in substitutes {
            let own_terms = substitute.tree.term_values(path).await?;
            let unresolved_terms = substitute.unresolved.term_values(path).await?;
            if own_terms.simplify() != unresolved_terms.simplify() {
                entering.push((substitute, unresolved_terms));
            }
        }
        for (substitute, _) in &entering {
            for index in substitute.slots.clone() {
                expansion.set(Slot::Own(index), None);
            }
        }
        for (substitute, unresolved_terms) in entering {
            let labels = substitute
                .unresolved
                .labels_by_term(&substitute.label)
                .map(|label| (*label).to_owned());
            let first = Slot::Own(substitute.slots.start);
            pending.extend(expansion.replace(first, unresolved_terms, labels));
        }
        let mut at_path = 0;
        while let Some(slot) = pending.pop_front() {
            let Some(value) = expansion.term(slot).cloned() else {
                continue;
            };
            let Some((terms, labels)) = named_terms(store, path, &mut named, &value).await? else {
                continue;
            };
            replacements.count(path, &mut at_path)?;
            pending.extend(expansion.replace(slot, terms, labels));
        }
        Ok(expansion)
    }

    /// The term at `slot`.
    fn term(&self, slot: Slot) -> Option<&TreeValue> {
        match slot {
            Slot::Own(index) => self.terms[index].as_ref(),
            Slot::Extra(origin, pair, side) => self.extra[&origin][pair][side].0.as_ref(),
        }
    }

    /// Sets the term at `slot` to `value`.
    fn set(&mut self, slot: Slot, value: Option<TreeValue>) {
        if let Some(old) = self.term(slot).cloned()
            && let Some(slots) = self.present.get_mut(&(slot.is_add(), old))
        {
            slots.retain(|&other| other != slot);
        }
        if let Some(new) = &value {
            let key = (slot.is_add(), new.clone());
            self.present.entry(key).or_default().push(slot);
        }
        match slot {
            Slot::Own(index) => self.terms[index] = value,
            Slot::Extra(origin, pair, side) => {
                self.extra.entry(origin).or_default()[pair][side].0 = value;
            }
        }
    }

    /// Replaces the VB or M at `slot` with `terms`, the terms of V that it
    /// names, with their `labels`, and returns where the terms that it brings
    /// are. The slot becomes absent, and `terms`, inverted where the slot is a
    /// remove, go to pairs after the merge's own terms, with the slot's absent
    /// term cancelling the absent term that starts them. Each of `terms` that
    /// equals a term of the merge of the other sign cancels it first, both
    /// made absent, so that where the two would cancel, the merge's own term
    /// is left, with its label.
    fn replace(&mut self, slot: Slot, terms: MergedTreeValue, labels: Merge<String>) -> Vec<Slot> {
        self.set(slot, None);
        // Each term as it enters the merge, and whether it is an add there.
        let mut entering = zip(terms, labels)
            .enumerate()
            .map(|(index, labeled)| (labeled, index.is_multiple_of(2) == slot.is_add()))
            .collect_vec();
        for ((term, _), is_add) in &mut entering {
            let Some(value) = term.clone() else {
                continue;
            };
            let opposite = self
                .present
                .get(&(!*is_add, value))
                .and_then(|slots| slots.first().copied());
            if let Some(other) = opposite {
                self.set(other, None);
                *term = None;
            }
        }
        let (adds, removes): (Vec<_>, Vec<_>) =
            entering.into_iter().partition(|(_, is_add)| *is_add);
        let adds = adds.into_iter().map(|(labeled, _)| labeled);
        let removes = removes.into_iter().map(|(labeled, _)| labeled);
        let absent = (None, String::new());
        let pairs = if slot.is_add() {
            zip(iter::once(absent).chain(removes), adds).collect_vec()
        } else {
            zip(removes, iter::once(absent).chain(adds)).collect_vec()
        };
        let origin = slot.origin();
        let mut new_slots = vec![];
        for ((remove, remove_label), (add, add_label)) in pairs {
            let pairs = self.extra.entry(origin).or_default();
            let pair = pairs.len();
            pairs.push([(None, remove_label), (None, add_label)]);
            for (side, value) in [(0, remove), (1, add)] {
                if value.is_some() {
                    let new_slot = Slot::Extra(origin, pair, side);
                    self.set(new_slot, value);
                    new_slots.push(new_slot);
                }
            }
        }
        new_slots
    }

    /// The terms, with `num_pairs` pairs after the merge's own terms for each
    /// of the merge's terms that a replaced term came from at any path.
    fn into_merge(mut self, num_pairs: &BTreeMap<usize, usize>) -> MergedTreeValue {
        let merge = Merge::from_vec(self.terms);
        append_pairs(merge, num_pairs, |origin, n| {
            let mut pairs = self
                .extra
                .remove(&origin)
                .unwrap_or_default()
                .into_iter()
                .map(|[(remove, _), (add, _)]| (remove, add))
                .collect_vec();
            pairs.resize(n, (None, None));
            pairs
        })
    }
}

/// Whether a merge replaces the VB and M at `path`, whose terms are `terms`:
/// whether a side that is one of `candidates` is VB or M once `terms` are
/// simplified. The terms it looks up go to `named`.
async fn replaces_at(
    store: &Store,
    path: &RepoPath,
    terms: &MergedTreeValue,
    candidates: &HashSet<TreeValue>,
    named: &mut HashMap<TreeValue, Option<NamedTerms>>,
) -> BackendResult<bool> {
    for side in terms.simplify().adds().flatten() {
        if candidates.contains(side) && named_terms(store, path, named, side).await?.is_some() {
            return Ok(true);
        }
    }
    Ok(false)
}

/// The terms that `value` names, with their labels, if it is VB or M, looked
/// up once for each value in `named`.
async fn named_terms(
    store: &Store,
    path: &RepoPath,
    named: &mut HashMap<TreeValue, Option<NamedTerms>>,
    value: &TreeValue,
) -> BackendResult<Option<NamedTerms>> {
    if !matches!(value, TreeValue::File { .. }) {
        return Ok(None);
    }
    if let Some(found) = named.get(value) {
        return Ok(found.clone());
    }
    let found = virtual_base::stored_virtual_base_terms(store, path, value).await?;
    named.insert(value.clone(), found.clone());
    Ok(found)
}

/// `tree`, an unsimplified merge, with `overrides` and `expansions` written
/// over its terms, and simplified once, with its labels. The terms that
/// replace VB and M go after the merge's own terms, as many pairs for each of
/// the merge's terms that they came from as at the path where it brings the
/// most. Each such term takes the label that the first path to put a term
/// there gives it, or the label of the merge's term it came from.
async fn write_expanded(
    tree: MergedTree,
    overrides: &[(RepoPathBuf, MergedTreeValue)],
    expansions: Vec<(RepoPathBuf, Expansion)>,
) -> BackendResult<MergedTree> {
    let mut pair_labels: BTreeMap<usize, Vec<[Option<String>; 2]>> = BTreeMap::new();
    for (_, expansion) in &expansions {
        for (&origin, pairs) in &expansion.extra {
            let labels = pair_labels.entry(origin).or_default();
            if labels.len() < pairs.len() {
                labels.resize(pairs.len(), [None, None]);
            }
            for (slots, pair) in zip(labels.iter_mut(), pairs) {
                for (slot, (term, label)) in zip(slots.iter_mut(), pair) {
                    if slot.is_none() && term.is_some() && !label.is_empty() {
                        *slot = Some(label.clone());
                    }
                }
            }
        }
    }
    let num_pairs: BTreeMap<usize, usize> = pair_labels
        .iter()
        .map(|(&origin, labels)| (origin, labels.len()))
        .collect();
    let MergedTree {
        store,
        tree_ids,
        labels,
    } = tree;
    let own_labels = if labels.has_labels() {
        labels.into_merge()
    } else {
        Merge::repeated(String::new(), tree_ids.num_sides())
    };
    let empty_tree_id = store.empty_tree_id().clone();
    let tree_ids = append_pairs(tree_ids, &num_pairs, |_, n| {
        vec![(empty_tree_id.clone(), empty_tree_id.clone()); n]
    });
    let labels = append_pairs(own_labels.clone(), &num_pairs, |origin, n| {
        let origin_label = own_labels.as_slice()[origin].clone();
        let label = |label: Option<&String>| label.cloned().unwrap_or_else(|| origin_label.clone());
        pair_labels[&origin][..n]
            .iter()
            .map(|[remove, add]| (label(remove.as_ref()), label(add.as_ref())))
            .collect()
    });
    let mut builder = MergedTreeBuilder::new(MergedTree::new(
        store,
        tree_ids,
        ConflictLabels::from_merge(labels),
    ));
    for (path, value) in overrides {
        builder.set_or_remove(path.clone(), value.clone());
    }
    for (path, expansion) in expansions {
        builder.set_or_remove(path, expansion.into_merge(&num_pairs));
    }
    builder.write_tree_no_resolve().await
}

/// `merge` with pairs of a remove and an add after its terms: for each origin
/// of `num_pairs` in order, the number of pairs it maps to, which `pairs`
/// gives for the origin and that number.
fn append_pairs<T>(
    merge: Merge<T>,
    num_pairs: &BTreeMap<usize, usize>,
    mut pairs: impl FnMut(usize, usize) -> Vec<(T, T)>,
) -> Merge<T> {
    let (removes, adds) = merge.into_removes_adds();
    let (mut removes, mut adds) = (removes.collect_vec(), adds.collect_vec());
    for (&origin, &num) in num_pairs {
        for (remove, add) in pairs(origin, num) {
            removes.push(remove);
            adds.push(add);
        }
    }
    Merge::from_removes_adds(removes, adds)
}

/// A single entry in a tree diff.
#[derive(Debug)]
pub struct TreeDiffEntry {
    /// The path.
    pub path: RepoPathBuf,
    /// The resolved tree values if available.
    pub values: BackendResult<Diff<MergedTreeValue>>,
}

/// Type alias for the result from `MergedTree::diff_stream()`. We use a
/// `Stream` instead of an `Iterator` so high-latency backends (e.g. cloud-based
/// ones) can fetch trees asynchronously.
pub type TreeDiffStream<'matcher> = BoxStream<'matcher, TreeDiffEntry>;

fn all_tree_entries(
    trees: &Merge<Tree>,
) -> impl Iterator<Item = (&RepoPathComponent, MergedTreeVal<'_>)> {
    if let Some(tree) = trees.as_resolved() {
        let iter = tree
            .entries_non_recursive()
            .map(|entry| (entry.name(), Merge::normal(entry.value())));
        Either::Left(iter)
    } else {
        let same_change = trees.first().store().merge_options().same_change;
        let iter = all_merged_tree_entries(trees).map(move |(name, values)| {
            // TODO: move resolve_trivial() to caller?
            let values = match values.resolve_trivial(same_change) {
                Some(resolved) => Merge::resolved(*resolved),
                None => values,
            };
            (name, values)
        });
        Either::Right(iter)
    }
}

/// Suppose the given `trees` aren't resolved, iterates `(name, values)` pairs
/// non-recursively. This also works if `trees` are resolved, but is more costly
/// than `tree.entries_non_recursive()`.
pub fn all_merged_tree_entries(
    trees: &Merge<Tree>,
) -> impl Iterator<Item = (&RepoPathComponent, MergedTreeVal<'_>)> {
    let mut entries_iters = trees
        .iter()
        .map(|tree| tree.entries_non_recursive().peekable())
        .collect_vec();
    iter::from_fn(move || {
        let next_name = entries_iters
            .iter_mut()
            .filter_map(|iter| iter.peek())
            .map(|entry| entry.name())
            .min()?;
        let values: MergeBuilder<_> = entries_iters
            .iter_mut()
            .map(|iter| {
                let entry = iter.next_if(|entry| entry.name() == next_name)?;
                Some(entry.value())
            })
            .collect();
        Some((next_name, values.build()))
    })
}

fn merged_tree_entry_diff<'a>(
    trees1: &'a Merge<Tree>,
    trees2: &'a Merge<Tree>,
) -> impl Iterator<Item = (&'a RepoPathComponent, Diff<MergedTreeVal<'a>>)> {
    itertools::merge_join_by(
        all_tree_entries(trees1),
        all_tree_entries(trees2),
        |(name1, _), (name2, _)| name1.cmp(name2),
    )
    .map(|entry| match entry {
        EitherOrBoth::Both((name, value1), (_, value2)) => (name, Diff::new(value1, value2)),
        EitherOrBoth::Left((name, value1)) => (name, Diff::new(value1, Merge::absent())),
        EitherOrBoth::Right((name, value2)) => (name, Diff::new(Merge::absent(), value2)),
    })
    .filter(|(_, diff)| diff.is_changed())
}

/// Recursive iterator over the entries in a tree.
pub struct TreeEntriesIterator<'matcher> {
    store: Arc<Store>,
    stack: Vec<TreeEntriesDirItem>,
    matcher: &'matcher dyn Matcher,
}

struct TreeEntriesDirItem {
    entries: Vec<(RepoPathBuf, MergedTreeValue)>,
}

impl TreeEntriesDirItem {
    fn new(trees: &Merge<Tree>, matcher: &dyn Matcher) -> Self {
        let mut entries = vec![];
        let dir = trees.first().dir();
        for (name, value) in all_tree_entries(trees) {
            let path = dir.join(name);
            if value.is_tree() {
                // TODO: Handle the other cases (specific files and trees)
                if matcher.visit(&path).is_nothing() {
                    continue;
                }
            } else if !matcher.matches(&path) {
                continue;
            }
            entries.push((path, value.cloned()));
        }
        entries.reverse();
        Self { entries }
    }
}

impl<'matcher> TreeEntriesIterator<'matcher> {
    fn new(trees: &MergedTree, matcher: &'matcher dyn Matcher) -> Self {
        Self {
            store: trees.store.clone(),
            stack: vec![TreeEntriesDirItem {
                entries: vec![(RepoPathBuf::root(), trees.to_merged_tree_value())],
            }],
            matcher,
        }
    }
}

impl Iterator for TreeEntriesIterator<'_> {
    type Item = (RepoPathBuf, BackendResult<MergedTreeValue>);

    fn next(&mut self) -> Option<Self::Item> {
        while let Some(top) = self.stack.last_mut() {
            if let Some((path, value)) = top.entries.pop() {
                let maybe_trees = match value.to_tree_merge(&self.store, &path).block_on() {
                    Ok(maybe_trees) => maybe_trees,
                    Err(err) => return Some((path, Err(err))),
                };
                if let Some(trees) = maybe_trees {
                    self.stack
                        .push(TreeEntriesDirItem::new(&trees, self.matcher));
                } else {
                    return Some((path, Ok(value)));
                }
            } else {
                self.stack.pop();
            }
        }
        None
    }
}

/// The state for the non-recursive iteration over the conflicted entries in a
/// single directory.
struct ConflictsDirItem {
    entries: Vec<(RepoPathBuf, MergedTreeValue)>,
}

impl ConflictsDirItem {
    fn new(trees: &Merge<Tree>, matcher: &dyn Matcher) -> Self {
        if trees.is_resolved() {
            return Self { entries: vec![] };
        }

        let dir = trees.first().dir();
        let mut entries = vec![];
        for (basename, value) in all_tree_entries(trees) {
            if value.is_resolved() {
                continue;
            }
            let path = dir.join(basename);
            if value.is_tree() {
                if matcher.visit(&path).is_nothing() {
                    continue;
                }
            } else if !matcher.matches(&path) {
                continue;
            }
            entries.push((path, value.cloned()));
        }
        entries.reverse();
        Self { entries }
    }
}

struct ConflictIterator<'matcher> {
    store: Arc<Store>,
    stack: Vec<ConflictsDirItem>,
    matcher: &'matcher dyn Matcher,
}

impl<'matcher> ConflictIterator<'matcher> {
    fn new(tree: &MergedTree, matcher: &'matcher dyn Matcher) -> Self {
        Self {
            store: tree.store().clone(),
            stack: vec![ConflictsDirItem {
                entries: vec![(RepoPathBuf::root(), tree.to_merged_tree_value())],
            }],
            matcher,
        }
    }
}

impl Iterator for ConflictIterator<'_> {
    type Item = (RepoPathBuf, BackendResult<MergedTreeValue>);

    fn next(&mut self) -> Option<Self::Item> {
        while let Some(top) = self.stack.last_mut() {
            if let Some((path, tree_values)) = top.entries.pop() {
                match tree_values.to_tree_merge(&self.store, &path).block_on() {
                    Ok(Some(trees)) => {
                        // If all sides are trees or missing, descend into the merged tree
                        self.stack.push(ConflictsDirItem::new(&trees, self.matcher));
                    }
                    Ok(None) => {
                        // Otherwise this is a conflict between files, trees, etc. If they could
                        // be automatically resolved, they should have been when the top-level
                        // tree conflict was written, so we assume that they can't be.
                        return Some((path, Ok(tree_values)));
                    }
                    Err(err) => {
                        return Some((path, Err(err)));
                    }
                }
            } else {
                self.stack.pop();
            }
        }
        None
    }
}

/// Iterator over the differences between two trees.
pub struct TreeDiffIterator<'matcher> {
    store: Arc<Store>,
    stack: Vec<TreeDiffDir>,
    matcher: &'matcher dyn Matcher,
}

struct TreeDiffDir {
    entries: Vec<(RepoPathBuf, Diff<MergedTreeValue>)>,
}

impl<'matcher> TreeDiffIterator<'matcher> {
    /// Creates a iterator over the differences between two trees.
    pub fn new(tree1: &MergedTree, tree2: &MergedTree, matcher: &'matcher dyn Matcher) -> Self {
        assert!(Arc::ptr_eq(tree1.store(), tree2.store()));
        let root_dir = RepoPath::root();
        let mut stack = Vec::new();
        let root_diff = Diff::new(tree1.to_merged_tree_value(), tree2.to_merged_tree_value());
        if root_diff.is_changed() && !matcher.visit(root_dir).is_nothing() {
            stack.push(TreeDiffDir {
                entries: vec![(root_dir.to_owned(), root_diff)],
            });
        }
        Self {
            store: tree1.store().clone(),
            stack,
            matcher,
        }
    }

    /// Gets the given trees if `values` are trees, otherwise an empty tree.
    fn trees(
        store: &Arc<Store>,
        dir: &RepoPath,
        values: &MergedTreeValue,
    ) -> BackendResult<Merge<Tree>> {
        if let Some(trees) = values.to_tree_merge(store, dir).block_on()? {
            Ok(trees)
        } else {
            Ok(Merge::resolved(Tree::empty(store.clone(), dir.to_owned())))
        }
    }
}

impl TreeDiffDir {
    fn from_trees(
        dir: &RepoPath,
        trees1: &Merge<Tree>,
        trees2: &Merge<Tree>,
        matcher: &dyn Matcher,
    ) -> Self {
        let mut entries = vec![];
        for (name, diff) in merged_tree_entry_diff(trees1, trees2) {
            let path = dir.join(name);
            let tree_before = diff.before.is_tree();
            let tree_after = diff.after.is_tree();
            // Check if trees and files match, but only if either side is a tree or a file
            // (don't query the matcher unnecessarily).
            let tree_matches = (tree_before || tree_after) && !matcher.visit(&path).is_nothing();
            let file_matches = (!tree_before || !tree_after) && matcher.matches(&path);

            // Replace trees or files that don't match by `Merge::absent()`
            let before = if (tree_before && tree_matches) || (!tree_before && file_matches) {
                diff.before
            } else {
                Merge::absent()
            };
            let after = if (tree_after && tree_matches) || (!tree_after && file_matches) {
                diff.after
            } else {
                Merge::absent()
            };
            if before.is_absent() && after.is_absent() {
                continue;
            }
            entries.push((path, Diff::new(before.cloned(), after.cloned())));
        }
        entries.reverse();
        Self { entries }
    }
}

impl Iterator for TreeDiffIterator<'_> {
    type Item = TreeDiffEntry;

    fn next(&mut self) -> Option<Self::Item> {
        while let Some(top) = self.stack.last_mut() {
            let Some((path, diff)) = top.entries.pop() else {
                self.stack.pop().unwrap();
                continue;
            };

            if diff.before.is_tree() || diff.after.is_tree() {
                let (before_tree, after_tree) = match (
                    Self::trees(&self.store, &path, &diff.before),
                    Self::trees(&self.store, &path, &diff.after),
                ) {
                    (Ok(before_tree), Ok(after_tree)) => (before_tree, after_tree),
                    (Err(before_err), _) => {
                        return Some(TreeDiffEntry {
                            path,
                            values: Err(before_err),
                        });
                    }
                    (_, Err(after_err)) => {
                        return Some(TreeDiffEntry {
                            path,
                            values: Err(after_err),
                        });
                    }
                };
                let subdir =
                    TreeDiffDir::from_trees(&path, &before_tree, &after_tree, self.matcher);
                self.stack.push(subdir);
            }
            if diff.before.is_file_like()
                || diff.after.is_file_like()
                || self.matcher.matches(&path)
            {
                return Some(TreeDiffEntry {
                    path,
                    values: Ok(diff),
                });
            }
        }
        None
    }
}

/// Stream of differences between two trees.
pub struct TreeDiffStreamImpl<'matcher> {
    store: Arc<Store>,
    matcher: &'matcher dyn Matcher,
    /// Pairs of tree values that may or may not be ready to emit, sorted in the
    /// order we want to emit them. If either side is a tree, there will be
    /// a corresponding entry in `pending_trees`. The item is ready to emit
    /// unless there's a smaller or equal path in `pending_trees`.
    items: BTreeMap<RepoPathBuf, BackendResult<Diff<MergedTreeValue>>>,
    // TODO: Is it better to combine this and `items` into a single map?
    #[expect(clippy::type_complexity)]
    pending_trees:
        BTreeMap<RepoPathBuf, BoxFuture<'matcher, BackendResult<(Merge<Tree>, Merge<Tree>)>>>,
    /// The maximum number of trees to request concurrently. However, we do the
    /// accounting per path, so there will often be twice as many pending
    /// `Backend::read_tree()` calls - for the "before" and "after" sides. For
    /// conflicts, there will be even more.
    max_concurrent_reads: usize,
    /// The maximum number of items in `items`. However, we will always add the
    /// full differences from a particular pair of trees, so it may temporarily
    /// go over the limit (until we emit those items). It may also go over the
    /// limit because we have a file item that's blocked by pending subdirectory
    /// items.
    max_queued_items: usize,
}

impl<'matcher> TreeDiffStreamImpl<'matcher> {
    /// Creates a iterator over the differences between two trees. Generally
    /// prefer `MergedTree::diff_stream()` of calling this directly.
    pub fn new(
        tree1: &MergedTree,
        tree2: &MergedTree,
        matcher: &'matcher dyn Matcher,
        max_concurrent_reads: usize,
    ) -> Self {
        assert!(Arc::ptr_eq(tree1.store(), tree2.store()));
        let store = tree1.store().clone();
        let mut stream = Self {
            store: store.clone(),
            matcher,
            items: BTreeMap::new(),
            pending_trees: BTreeMap::new(),
            max_concurrent_reads,
            max_queued_items: 10000,
        };
        let dir = RepoPathBuf::root();
        let merged_tree1 = tree1.to_merged_tree_value();
        let merged_tree2 = tree2.to_merged_tree_value();
        let root_diff = Diff::new(merged_tree1.clone(), merged_tree2.clone());
        if root_diff.is_changed() && matcher.matches(&dir) {
            stream.items.insert(dir.clone(), Ok(root_diff));
        }
        let root_tree_fut = Box::pin(try_join(
            Self::trees(store.clone(), dir.clone(), merged_tree1),
            Self::trees(store, dir.clone(), merged_tree2),
        ));
        stream.pending_trees.insert(dir, root_tree_fut);
        stream
    }

    async fn single_tree(
        store: &Arc<Store>,
        dir: RepoPathBuf,
        value: Option<&TreeValue>,
    ) -> BackendResult<Tree> {
        match value {
            Some(TreeValue::Tree(tree_id)) => store.get_tree(dir, tree_id).await,
            _ => Ok(Tree::empty(store.clone(), dir.clone())),
        }
    }

    /// Gets the given trees if `values` are trees, otherwise an empty tree.
    async fn trees(
        store: Arc<Store>,
        dir: RepoPathBuf,
        values: MergedTreeValue,
    ) -> BackendResult<Merge<Tree>> {
        if values.is_tree() {
            values
                .try_map_async(|value| Self::single_tree(&store, dir.clone(), value.as_ref()))
                .await
        } else {
            Ok(Merge::resolved(Tree::empty(store, dir)))
        }
    }

    fn add_dir_diff_items(&mut self, dir: &RepoPath, trees1: &Merge<Tree>, trees2: &Merge<Tree>) {
        for (basename, diff) in merged_tree_entry_diff(trees1, trees2) {
            let path = dir.join(basename);
            let tree_before = diff.before.is_tree();
            let tree_after = diff.after.is_tree();
            // Check if trees and files match, but only if either side is a tree or a file
            // (don't query the matcher unnecessarily).
            let tree_matches =
                (tree_before || tree_after) && !self.matcher.visit(&path).is_nothing();
            let file_matches = (!tree_before || !tree_after) && self.matcher.matches(&path);

            // Replace trees or files that don't match by `Merge::absent()`
            let before = if (tree_before && tree_matches) || (!tree_before && file_matches) {
                diff.before
            } else {
                Merge::absent()
            };
            let after = if (tree_after && tree_matches) || (!tree_after && file_matches) {
                diff.after
            } else {
                Merge::absent()
            };
            if before.is_absent() && after.is_absent() {
                continue;
            }

            // If the path was a tree on either side of the diff, read those trees.
            if tree_matches {
                let before_tree_future =
                    Self::trees(self.store.clone(), path.clone(), before.cloned());
                let after_tree_future =
                    Self::trees(self.store.clone(), path.clone(), after.cloned());
                let both_trees_future = try_join(before_tree_future, after_tree_future);
                self.pending_trees
                    .insert(path.clone(), Box::pin(both_trees_future));
            }

            if file_matches || self.matcher.matches(&path) {
                self.items
                    .insert(path, Ok(Diff::new(before.cloned(), after.cloned())));
            }
        }
    }

    fn poll_tree_futures(&mut self, cx: &mut Context<'_>) {
        loop {
            let mut tree_diffs = vec![];
            let mut some_pending = false;
            let mut all_pending = true;
            for (dir, future) in self
                .pending_trees
                .iter_mut()
                .take(self.max_concurrent_reads)
            {
                if let Poll::Ready(tree_diff) = future.as_mut().poll(cx) {
                    all_pending = false;
                    tree_diffs.push((dir.clone(), tree_diff));
                } else {
                    some_pending = true;
                }
            }

            for (dir, tree_diff) in tree_diffs {
                drop(self.pending_trees.remove_entry(&dir).unwrap());
                match tree_diff {
                    Ok((trees1, trees2)) => {
                        self.add_dir_diff_items(&dir, &trees1, &trees2);
                    }
                    Err(err) => {
                        self.items.insert(dir, Err(err));
                    }
                }
            }

            // If none of the futures have been polled and returned `Poll::Pending`, we must
            // not return. If we did, nothing would call the waker so we might never get
            // polled again.
            if all_pending || (some_pending && self.items.len() >= self.max_queued_items) {
                return;
            }
        }
    }
}

impl Stream for TreeDiffStreamImpl<'_> {
    type Item = TreeDiffEntry;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        // Go through all pending tree futures and poll them.
        self.poll_tree_futures(cx);

        // Now emit the first file, or the first tree that completed with an error
        if let Some((path, _)) = self.items.first_key_value() {
            // Check if there are any pending trees before this item that we need to finish
            // polling before we can emit this item.
            if let Some((dir, _)) = self.pending_trees.first_key_value()
                && dir < path
            {
                return Poll::Pending;
            }

            let (path, values) = self.items.pop_first().unwrap();
            Poll::Ready(Some(TreeDiffEntry { path, values }))
        } else if self.pending_trees.is_empty() {
            Poll::Ready(None)
        } else {
            Poll::Pending
        }
    }
}

fn stream_without_trees(stream: TreeDiffStream) -> TreeDiffStream {
    stream
        .filter_map(|mut entry| async move {
            let skip_tree = |merge: MergedTreeValue| {
                if merge.is_tree() {
                    Merge::absent()
                } else {
                    merge
                }
            };
            entry.values = entry.values.map(|diff| diff.map(skip_tree));

            // Filter out entries where neither side is present.
            let any_present = entry.values.as_ref().map_or(true, |diff| {
                diff.before.is_present() || diff.after.is_present()
            });
            any_present.then_some(entry)
        })
        .boxed()
}

/// Adapts a `TreeDiffStream` to emit a added file at a given path after a
/// removed directory at the same path.
struct DiffStreamForFileSystem<'a> {
    inner: TreeDiffStream<'a>,
    next_item: Option<TreeDiffEntry>,
    held_file: Option<TreeDiffEntry>,
}

impl<'a> DiffStreamForFileSystem<'a> {
    fn new(inner: TreeDiffStream<'a>) -> Self {
        Self {
            inner,
            next_item: None,
            held_file: None,
        }
    }
}

impl Stream for DiffStreamForFileSystem<'_> {
    type Item = TreeDiffEntry;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        while let Some(next) = match self.next_item.take() {
            Some(next) => Some(next),
            None => ready!(self.inner.as_mut().poll_next(cx)),
        } {
            // Filter out changes where neither side (before or after) is_file_like.
            // This ensures we only process file-level changes and transitions.
            if let Ok(diff) = &next.values
                && !diff.before.is_file_like()
                && !diff.after.is_file_like()
            {
                continue;
            }

            // If there's a held file "foo" and the next item to emit is not "foo/...", then
            // we must be done with the "foo/" directory and it's time to emit "foo" as a
            // removed file.
            if let Some(held_entry) = self
                .held_file
                .take_if(|held_entry| !next.path.starts_with(&held_entry.path))
            {
                self.next_item = Some(next);
                return Poll::Ready(Some(held_entry));
            }

            match next.values {
                Ok(diff) if diff.before.is_tree() => {
                    assert!(diff.after.is_present());
                    assert!(self.held_file.is_none());
                    self.held_file = Some(TreeDiffEntry {
                        path: next.path,
                        values: Ok(Diff::new(Merge::absent(), diff.after)),
                    });
                }
                Ok(diff) if diff.after.is_tree() => {
                    assert!(diff.before.is_present());
                    return Poll::Ready(Some(TreeDiffEntry {
                        path: next.path,
                        values: Ok(Diff::new(diff.before, Merge::absent())),
                    }));
                }
                _ => {
                    return Poll::Ready(Some(next));
                }
            }
        }
        Poll::Ready(self.held_file.take())
    }
}
