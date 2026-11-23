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

//! The replacement of VB and M at one path of a merge that leaves one as a
//! side: every VB and M term there, sides and bases alike, nested ones too,
//! replaced with the terms of V that it names (see
//! `MergedTree::merge_with_overrides`).

use std::collections::BTreeMap;
use std::collections::HashMap;
use std::collections::HashSet;
use std::collections::VecDeque;
use std::iter;
use std::iter::zip;
use std::ops::Range;

use itertools::Itertools as _;

use super::NamedTerms;
use super::stored_virtual_base_terms;
use crate::backend::BackendError;
use crate::backend::BackendResult;
use crate::backend::MergedTreeValue;
use crate::backend::TreeValue;
use crate::merge::Merge;
use crate::repo_path::RepoPath;
use crate::store::Store;

/// The most VB and M terms that a merge replaces at one path. A VB names at
/// most one VB or M for each level of merge bases below it, so a rewrite of the
/// top merge of a braid of 8,000 levels of merge bases replaces 7,999 at its
/// one path; a planted set of files that name one another can stand for more,
/// and the merge then fails, as a user error, rather than spend time and
/// memory on them.
const MAX_REPLACEMENTS_AT_A_PATH: usize = 10_000;

/// The most VB and M terms that a merge replaces in all is this, and
/// [`MAX_REPLACEMENTS_PER_REPLACED_PATH`] more for each path where it replaces
/// one. A merge of a history one level deep replaces about one at each
/// conflicted path, but a nested history replaces about one for each level at
/// each path: a 202-level braid over 100 files replaces about 201 at each, over
/// 20,000 in all, and its merge is refused, a stated limit. Planted files at
/// many paths get about this many each.
const MAX_REPLACEMENTS_IN_A_MERGE: usize = 10_000;

/// See [`MAX_REPLACEMENTS_IN_A_MERGE`].
const MAX_REPLACEMENTS_PER_REPLACED_PATH: usize = 100;

/// How many VB and M terms a merge has replaced, and at how many paths it
/// replaces them.
pub(crate) struct Replacements {
    in_merge: usize,
    replaced_paths: usize,
}

impl Replacements {
    /// A count for a merge that replaces VB and M at `replaced_paths` paths.
    pub(crate) fn new(replaced_paths: usize) -> Self {
        Self {
            in_merge: 0,
            replaced_paths,
        }
    }

    /// Counts one more replacement at `path`, where `at_path` were made
    /// before it; fails once it passes the limit at a path or the limit of the
    /// merge. The error is `Unsupported`, the variant that jj reports as a
    /// user error: what the merge reads, not the backend, is at fault, and no
    /// other variant says that.
    fn count(&mut self, path: &RepoPath, at_path: &mut usize) -> BackendResult<()> {
        *at_path += 1;
        self.in_merge += 1;
        if *at_path > MAX_REPLACEMENTS_AT_A_PATH {
            let path = path.as_internal_file_string();
            return Err(BackendError::Unsupported(format!(
                "Refused to merge: at {path} the merge would replace more than \
                 {MAX_REPLACEMENTS_AT_A_PATH} virtual merge bases (the limit at one path). Files \
                 that pass as virtual merge bases can stand for that many, and so can a history \
                 that is both deep and wide; no change was made"
            )));
        }
        let bound =
            MAX_REPLACEMENTS_IN_A_MERGE + MAX_REPLACEMENTS_PER_REPLACED_PATH * self.replaced_paths;
        if self.in_merge > bound {
            let paths = self.replaced_paths;
            let path = path.as_internal_file_string();
            return Err(BackendError::Unsupported(format!(
                "Refused to merge: across {paths} paths the merge would replace more than {bound} \
                 virtual merge bases (the limit is {MAX_REPLACEMENTS_IN_A_MERGE} and \
                 {MAX_REPLACEMENTS_PER_REPLACED_PATH} for each path where it replaces one); it \
                 stopped at {path}. Files that pass as virtual merge bases can stand for that \
                 many, and so can a history that is both deep and wide; no change was made"
            )));
        }
        Ok(())
    }
}

/// The terms that values name, with their labels, where they are VB or M,
/// each value looked up once.
pub(crate) type NamedTermsCache = HashMap<TreeValue, Option<NamedTerms>>;

/// A tree of a merge that comes with the unresolved merge that it resolves,
/// at one path: the range of its terms among the merge's terms, its own terms
/// there, and the unresolved merge's terms there, with their labels.
pub(crate) struct Substitute {
    pub(crate) slots: Range<usize>,
    pub(crate) own_terms: MergedTreeValue,
    pub(crate) unresolved: NamedTerms,
}

/// A term of a merge, with its label.
#[derive(Clone, Debug)]
struct LabeledTerm {
    term: Option<TreeValue>,
    label: String,
}

/// A remove and an add that the terms that replace a VB or M bring into a
/// merge.
#[derive(Clone, Debug)]
struct TermPair {
    remove: LabeledTerm,
    add: LabeledTerm,
}

/// The labels of a pair of terms that the terms that replace a VB or M bring
/// into a merge, where a path gives them.
#[derive(Clone, Debug, Default)]
pub(crate) struct PairLabels {
    pub(crate) remove: Option<String>,
    pub(crate) add: Option<String>,
}

/// The terms of a merge at a path once every VB and M among them is replaced
/// with the terms of V that it names: the merge's own terms, each in its
/// place, a replaced one absent, and the terms that the replaced ones bring,
/// as pairs of a remove and an add with their labels, by the index of the
/// merge's term that each came from.
pub(crate) struct Expansion {
    terms: Vec<Option<TreeValue>>,
    pairs: BTreeMap<usize, Vec<TermPair>>,
    /// Where each term that is present is, by whether it is an add and by its
    /// value.
    present: HashMap<(bool, TreeValue), Vec<Slot>>,
}

/// Where a term of an [`Expansion`] is: the merge's own term at an index, or
/// the remove or the add of a pair among those that came from the merge's
/// term at an index.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Slot {
    Own(usize),
    Extra {
        origin: usize,
        pair: usize,
        is_add: bool,
    },
}

impl Slot {
    /// Whether the term is an add.
    fn is_add(self) -> bool {
        match self {
            Self::Own(index) => index.is_multiple_of(2),
            Self::Extra { is_add, .. } => is_add,
        }
    }

    /// The index of the merge's own term that the term came from.
    fn origin(self) -> usize {
        match self {
            Self::Own(index) | Self::Extra { origin: index, .. } => index,
        }
    }
}

impl Expansion {
    /// `terms` at `path`, where [`replaces_at`] holds, with every VB and M
    /// replaced, sides and bases alike, nested ones too, with `named` the
    /// terms already looked up. Each of `substitutes` whose own terms differ
    /// from its unresolved merge's first enters as the terms of that merge in
    /// place of its own terms. Each VB and M replaced counts in
    /// `replacements`, which fails past its limits.
    pub(crate) async fn new(
        store: &Store,
        path: &RepoPath,
        terms: MergedTreeValue,
        substitutes: Vec<Substitute>,
        mut named: NamedTermsCache,
        replacements: &mut Replacements,
    ) -> BackendResult<Self> {
        let mut expansion = Self {
            terms: vec![None; terms.as_slice().len()],
            pairs: BTreeMap::new(),
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
        let entering = substitutes
            .into_iter()
            .filter(|substitute| {
                substitute.own_terms.simplify() != substitute.unresolved.terms.simplify()
            })
            .collect_vec();
        for substitute in &entering {
            for index in substitute.slots.clone() {
                expansion.set(Slot::Own(index), None);
            }
        }
        for substitute in entering {
            let first = Slot::Own(substitute.slots.start);
            pending.extend(expansion.replace(first, substitute.unresolved));
        }
        let mut at_path = 0;
        while let Some(slot) = pending.pop_front() {
            let Some(value) = expansion.term(slot).cloned() else {
                continue;
            };
            let Some(named_terms) = cached_named_terms(store, path, &mut named, &value).await?
            else {
                continue;
            };
            replacements.count(path, &mut at_path)?;
            pending.extend(expansion.replace(slot, named_terms));
        }
        Ok(expansion)
    }

    /// The labels of the pairs of terms that `expansions` bring, by the index
    /// of the merge's term that each came from: as many pairs for each as at
    /// the path where it brings the most, each term with the label that the
    /// first path to put a term there gives it.
    pub(crate) fn pair_labels<'a>(
        expansions: impl IntoIterator<Item = &'a Self>,
    ) -> BTreeMap<usize, Vec<PairLabels>> {
        let mut pair_labels: BTreeMap<usize, Vec<PairLabels>> = BTreeMap::new();
        for expansion in expansions {
            for (&origin, pairs) in &expansion.pairs {
                let labels = pair_labels.entry(origin).or_default();
                if labels.len() < pairs.len() {
                    labels.resize(pairs.len(), PairLabels::default());
                }
                for (labels, pair) in zip(labels.iter_mut(), pairs) {
                    let sides = [
                        (&mut labels.remove, &pair.remove),
                        (&mut labels.add, &pair.add),
                    ];
                    for (label, side) in sides {
                        if label.is_none() && side.term.is_some() && !side.label.is_empty() {
                            *label = Some(side.label.clone());
                        }
                    }
                }
            }
        }
        pair_labels
    }

    /// The term at `slot`.
    fn term(&self, slot: Slot) -> Option<&TreeValue> {
        match slot {
            Slot::Own(index) => self.terms[index].as_ref(),
            Slot::Extra {
                origin,
                pair,
                is_add,
            } => {
                let pair = &self.pairs[&origin][pair];
                let side = if is_add { &pair.add } else { &pair.remove };
                side.term.as_ref()
            }
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
            Slot::Extra {
                origin,
                pair,
                is_add,
            } => {
                let pair = &mut self.pairs.entry(origin).or_default()[pair];
                let side = if is_add {
                    &mut pair.add
                } else {
                    &mut pair.remove
                };
                side.term = value;
            }
        }
    }

    /// Replaces the VB or M at `slot` with `named`, the terms of V that it
    /// names, with their labels, and returns where the terms that it brings
    /// are. The slot becomes absent, and the terms, inverted where the slot is
    /// a remove, go to pairs after the merge's own terms, with the slot's
    /// absent term cancelling the absent term that starts them. Each of the
    /// terms that equals a term of the merge of the other sign cancels it
    /// first, both made absent, so that where the two would cancel, the
    /// merge's own term is left, with its label.
    fn replace(&mut self, slot: Slot, named: NamedTerms) -> Vec<Slot> {
        self.set(slot, None);
        // Each term as it enters the merge, and whether it is an add there.
        let mut entering = zip(named.terms, named.labels)
            .enumerate()
            .map(|(index, (term, label))| {
                let is_add = index.is_multiple_of(2) == slot.is_add();
                (LabeledTerm { term, label }, is_add)
            })
            .collect_vec();
        for (labeled, is_add) in &mut entering {
            let Some(value) = labeled.term.clone() else {
                continue;
            };
            let opposite = self
                .present
                .get(&(!*is_add, value))
                .and_then(|slots| slots.first().copied());
            if let Some(other) = opposite {
                self.set(other, None);
                labeled.term = None;
            }
        }
        let (adds, removes): (Vec<_>, Vec<_>) =
            entering.into_iter().partition(|(_, is_add)| *is_add);
        let adds = adds.into_iter().map(|(labeled, _)| labeled);
        let removes = removes.into_iter().map(|(labeled, _)| labeled);
        let absent = LabeledTerm {
            term: None,
            label: String::new(),
        };
        let pairs = if slot.is_add() {
            zip(iter::once(absent).chain(removes), adds).collect_vec()
        } else {
            zip(removes, iter::once(absent).chain(adds)).collect_vec()
        };
        let origin = slot.origin();
        let mut new_slots = vec![];
        for (remove, add) in pairs {
            let pairs = self.pairs.entry(origin).or_default();
            let pair = pairs.len();
            pairs.push(TermPair {
                remove: LabeledTerm {
                    term: None,
                    label: remove.label,
                },
                add: LabeledTerm {
                    term: None,
                    label: add.label,
                },
            });
            for (is_add, value) in [(false, remove.term), (true, add.term)] {
                if value.is_some() {
                    let new_slot = Slot::Extra {
                        origin,
                        pair,
                        is_add,
                    };
                    self.set(new_slot, value);
                    new_slots.push(new_slot);
                }
            }
        }
        new_slots
    }

    /// The terms, with `num_pairs` pairs after the merge's own terms for each
    /// of the merge's terms that a replaced term came from at any path.
    pub(crate) fn into_merge(mut self, num_pairs: &BTreeMap<usize, usize>) -> MergedTreeValue {
        let merge = Merge::from_vec(self.terms);
        append_pairs(merge, num_pairs, |origin, n| {
            let mut pairs = self
                .pairs
                .remove(&origin)
                .unwrap_or_default()
                .into_iter()
                .map(|pair| (pair.remove.term, pair.add.term))
                .collect_vec();
            pairs.resize(n, (None, None));
            pairs
        })
    }
}

/// Whether a merge replaces the VB and M at `path`, whose terms are `terms`:
/// whether a side that is one of `candidates` is VB or M once `terms` are
/// simplified. The terms it looks up go to `named`.
pub(crate) async fn replaces_at(
    store: &Store,
    path: &RepoPath,
    terms: &MergedTreeValue,
    candidates: &HashSet<TreeValue>,
    named: &mut NamedTermsCache,
) -> BackendResult<bool> {
    for side in terms.simplify().adds().flatten() {
        if candidates.contains(side)
            && cached_named_terms(store, path, named, side)
                .await?
                .is_some()
        {
            return Ok(true);
        }
    }
    Ok(false)
}

/// The terms that `value` names, with their labels, if it is VB or M, looked
/// up once for each value in `named`.
async fn cached_named_terms(
    store: &Store,
    path: &RepoPath,
    named: &mut NamedTermsCache,
    value: &TreeValue,
) -> BackendResult<Option<NamedTerms>> {
    if !matches!(value, TreeValue::File { .. }) {
        return Ok(None);
    }
    if let Some(found) = named.get(value) {
        return Ok(found.clone());
    }
    let found = stored_virtual_base_terms(store, path, value).await?;
    named.insert(value.clone(), found.clone());
    Ok(found)
}

/// `merge` with pairs of a remove and an add after its terms: for each origin
/// of `num_pairs` in order, the number of pairs it maps to, which `pairs`
/// gives for the origin and that number.
pub(crate) fn append_pairs<T>(
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
