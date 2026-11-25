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

//! The rules a merge of two commits follows at each path, where V is the merge
//! of all their merge bases at that path
//! (https://github.com/jj-vcs/jj/issues/6369).
//!
//! Where V resolves, the merge is the three-way merge of the parents against
//! V. Where V does not resolve, a parent that holds V's own conflict (the same
//! adds and the same removes, in any order, after `simplify()`) counts as
//! unchanged, and two parents that both changed it are merged against a
//! virtual base (VB): for a conflict of text files, jj's line merge of V's
//! terms with V's sides between marker lines where they do not merge, as git
//! builds one; otherwise M, a block of marker lines around one line for each
//! of V's terms, so the merge is a conflict. VB and M name V's terms, and are
//! only ever bases of conflicts: where a later merge, such as a rebase, an
//! abandon or a squash, leaves one as a side, it merges the terms of V that it
//! names instead (VB1 to VB5). Merge bases, and the parents of a merge of three
//! or more commits, are merged in the order of their commit ids.
//!
//! The case tests are named for the histories they build: K (the merge bases'
//! own merge is a conflict), D and MD (a merge base deletes the path), T (three
//! merge bases), H, E and X (text that git's recursive merge merges line by
//! line) and BIN (binary files). Each builds its history commit by commit, as
//! `jj new <parents>` followed by file writes makes it, in the order in which
//! a user would make the commits, so that the index lists the merge bases in
//! that order.

use std::collections::HashMap;
use std::fmt::Write as _;
use std::process::Command;
use std::sync::Arc;

use blake2::Blake2b512;
use digest::Digest as _;
use itertools::Itertools as _;
use jj_lib::backend::CommitId;
use jj_lib::backend::CopyId;
use jj_lib::backend::MergedTreeValue;
use jj_lib::backend::MergedTreeValueExt as _;
use jj_lib::backend::TreeValue;
use jj_lib::commit::Commit;
use jj_lib::config::ConfigLayer;
use jj_lib::config::ConfigSource;
use jj_lib::conflict_labels::ConflictLabels;
use jj_lib::conflicts::ConflictMarkerStyle;
use jj_lib::conflicts::ConflictMaterializeOptions;
use jj_lib::conflicts::choose_materialized_conflict_marker_len;
use jj_lib::conflicts::extract_as_single_hunk;
use jj_lib::conflicts::materialize_merge_result_to_bytes;
use jj_lib::conflicts::update_from_content;
use jj_lib::files;
use jj_lib::merge::Merge;
use jj_lib::merge::SameChange;
use jj_lib::merge_rule::merge_parent_values;
use jj_lib::merged_tree::MergedTree;
use jj_lib::merged_tree_builder::MergedTreeBuilder;
use jj_lib::object_id::ObjectId as _;
use jj_lib::repo::Repo as _;
use jj_lib::repo_path::RepoPath;
use jj_lib::repo_path::RepoPathBuf;
use jj_lib::rewrite::CommitWithSelection;
use jj_lib::rewrite::RebaseOptions;
use jj_lib::rewrite::duplicate_commits;
use jj_lib::rewrite::merge_commit_trees;
use jj_lib::rewrite::merge_commit_trees_no_resolve;
use jj_lib::rewrite::rebase_commit;
use jj_lib::rewrite::squash_commits;
use jj_lib::settings::UserSettings;
use jj_lib::store::Store;
use jj_lib::transaction::Transaction;
use jj_lib::tree_merge::MergeOptions;
use jj_lib::virtual_base::virtual_base_terms;
use jj_lib::virtual_base::virtual_merge_base;
use pollster::FutureExt as _;
use testutils::CommitBuilderExt as _;
use testutils::TestRepo;
use testutils::TestResult;
use testutils::TestTreeBuilder;
use testutils::create_tree;
use testutils::read_file;
use testutils::rebase_descendants_with_options_return_map;
use testutils::repo_path;
use testutils::repo_path_buf;
use testutils::settings_with_same_change;

/// The opening marker line of a block of a virtual base, before its hex, its
/// salt and the names of V's terms.
const OPEN: &str = "<<<<<<< jj virtual merge base ";
/// The marker line between two sides of a block of a virtual base.
const SEPARATOR: &str = "======= ";
/// The closing marker line of a block of a virtual base.
const CLOSE: &str = ">>>>>>> ";

/// The path every case merges.
fn f() -> &'static RepoPath {
    repo_path("f")
}

/// A path and what a commit writes there: a file's content, or `None` to
/// remove the path.
type Write = (&'static str, Option<Vec<u8>>);

/// `f=o` in a case: a file that holds the value and a newline.
fn line(value: &str) -> Option<Vec<u8>> {
    Some(format!("{value}\n").into_bytes())
}

/// A file that holds each word of `words` on a line of its own.
fn lines(words: &str) -> Option<Vec<u8>> {
    let mut content = words.replace(' ', "\n");
    content.push('\n');
    Some(content.into_bytes())
}

/// A binary file: the value, a NUL byte and `bin`.
fn binary(value: &str) -> Option<Vec<u8>> {
    Some(format!("{value}\0bin\n").into_bytes())
}

/// Which of the two commits A and B a case makes first.
#[derive(Clone, Copy, Debug)]
enum Made {
    AFirst,
    BFirst,
}

/// The order in which a case names the two parents of its merge.
#[derive(Clone, Copy, Debug)]
enum Order {
    AsNamed,
    Swapped,
}

/// A test repository with one open transaction, in which a test builds a
/// history as a user does with `jj new <parents>` and file writes: each new
/// commit's tree is the merge of its parents' trees, conflicts included, with
/// the writes on top.
struct Case {
    tx: Transaction,
    _test_repo: TestRepo,
}

impl Case {
    fn new() -> Self {
        Self::with_settings(&testutils::user_settings())
    }

    fn with_settings(settings: &UserSettings) -> Self {
        let test_repo = TestRepo::init_with_settings(settings);
        let tx = test_repo.repo.start_transaction();
        Self {
            tx,
            _test_repo: test_repo,
        }
    }

    fn store(&self) -> &Arc<Store> {
        self.tx.repo().store()
    }

    fn root(&self) -> Commit {
        self.store().root_commit()
    }

    /// The merge of the parents' trees, as `jj new <parents>` makes it.
    fn merge(&self, parents: &[&Commit]) -> TestResult<MergedTree> {
        let parents = parents.iter().map(|&commit| commit.clone()).collect_vec();
        Ok(merge_commit_trees(self.tx.repo(), &parents).block_on()?)
    }

    /// A file value that holds `content`, written to the store.
    fn file(
        &self,
        path: &RepoPath,
        content: &[u8],
        executable: bool,
        copy_id: CopyId,
    ) -> Option<TreeValue> {
        let id = self
            .store()
            .write_file(path, &mut &content[..])
            .block_on()
            .unwrap();
        Some(TreeValue::File {
            id,
            executable,
            copy_id,
        })
    }

    /// The value `f=<value>` writes at `f`.
    fn text(&self, value: &str) -> Option<TreeValue> {
        self.file(f(), &line(value).unwrap(), false, CopyId::placeholder())
    }

    /// A directory value that holds the file `x` with `content`.
    fn directory(&self, content: &str) -> Option<TreeValue> {
        let mut builder = TestTreeBuilder::new(self.store().clone());
        builder.file(repo_path("x"), content);
        Some(TreeValue::Tree(builder.write_single_tree().id().clone()))
    }

    /// The content of a file value.
    fn read(&self, value: Option<&TreeValue>) -> Option<Vec<u8>> {
        match value {
            Some(TreeValue::File { id, .. }) => Some(read_file(self.store(), f(), id)),
            _ => None,
        }
    }

    /// A commit made as `jj new <parents>`, then `writes`.
    fn commit(&mut self, parents: &[&Commit], writes: &[Write]) -> TestResult<Commit> {
        let values = writes
            .iter()
            .map(|(path, content)| {
                let value = match content {
                    Some(content) => Merge::resolved(self.file(
                        repo_path(path),
                        content,
                        false,
                        CopyId::placeholder(),
                    )),
                    None => Merge::absent(),
                };
                (repo_path_buf(*path), value)
            })
            .collect_vec();
        self.commit_with_values(parents, values)
    }

    /// A commit made as `jj new <parents>`, then with `values` at their paths,
    /// as `jj restore` can set a conflict.
    fn commit_with_values(
        &mut self,
        parents: &[&Commit],
        values: Vec<(RepoPathBuf, MergedTreeValue)>,
    ) -> TestResult<Commit> {
        let tree = self.merge(parents)?;
        let num_sides = tree.tree_ids().num_sides();
        let mut builder = MergedTreeBuilder::new(tree);
        for (path, mut value) in values {
            if !value.is_resolved() {
                value.pad_to(num_sides, &None);
            }
            builder.set_or_remove(path, value);
        }
        let tree = builder.write_tree().block_on()?;
        let parent_ids = parents.iter().map(|commit| commit.id().clone()).collect();
        Ok(self
            .tx
            .repo_mut()
            .new_commit(parent_ids, tree)
            .set_description("case commit")
            .write_unwrap())
    }

    /// Makes A and B, children of `parent`, in the order `made` gives.
    fn commit_a_and_b(
        &mut self,
        made: Made,
        parent: &Commit,
        a_writes: &[Write],
        b_writes: &[Write],
    ) -> TestResult<(Commit, Commit)> {
        Ok(match made {
            Made::AFirst => {
                let a = self.commit(&[parent], a_writes)?;
                let b = self.commit(&[parent], b_writes)?;
                (a, b)
            }
            Made::BFirst => {
                let b = self.commit(&[parent], b_writes)?;
                let a = self.commit(&[parent], a_writes)?;
                (a, b)
            }
        })
    }

    /// This history in its repository loaded again with `settings`, so that
    /// its store merges with their merge options, with `commits` read from
    /// that store.
    fn reloaded(
        self,
        settings: &UserSettings,
        commits: &[&Commit],
    ) -> TestResult<(Self, Vec<Commit>)> {
        let Self {
            tx,
            _test_repo: test_repo,
        } = self;
        tx.commit("case").block_on()?;
        let repo = test_repo
            .env
            .load_repo_at_head(settings, test_repo.repo_path());
        let commits = commits
            .iter()
            .map(|commit| repo.store().get_commit(commit.id()))
            .collect::<Result<Vec<_>, _>>()?;
        let tx = repo.start_transaction();
        Ok((
            Self {
                tx,
                _test_repo: test_repo,
            },
            commits,
        ))
    }
}

/// The value a tree holds at `path`.
fn value_at(tree: &MergedTree, path: &str) -> TestResult<MergedTreeValue> {
    Ok(tree.path_value(repo_path(path)).block_on()?)
}

/// A history and the merge a case makes of two (or more) of its commits.
struct Merged {
    case: Case,
    parents: Vec<Commit>,
    tree: MergedTree,
}

impl Merged {
    fn new(case: Case, parents: &[&Commit]) -> TestResult<Self> {
        let tree = case.merge(parents)?;
        let parents = parents.iter().map(|&commit| commit.clone()).collect();
        Ok(Self {
            case,
            parents,
            tree,
        })
    }

    /// The merge of `p1` and `p2` in `order`. A swapped case follows the case
    /// that merges them as named, which made that merge a commit first.
    fn pair(mut case: Case, p1: &Commit, p2: &Commit, order: Order) -> TestResult<Self> {
        match order {
            Order::AsNamed => Self::new(case, &[p1, p2]),
            Order::Swapped => {
                case.commit(&[p1, p2], &[])?;
                Self::new(case, &[p2, p1])
            }
        }
    }

    /// The merge's value at `f`.
    fn value(&self) -> TestResult<MergedTreeValue> {
        value_at(&self.tree, "f")
    }
}

/// Whether two merged values have the same adds and the same removes, as
/// multisets, after `simplify()`.
fn same_terms(a: &MergedTreeValue, b: &MergedTreeValue) -> bool {
    let (a, b) = (a.simplify(), b.simplify());
    a.adds().counts() == b.adds().counts() && a.removes().counts() == b.removes().counts()
}

/// A merge of the given terms.
fn conflict(terms: &[&Option<TreeValue>]) -> MergedTreeValue {
    Merge::from_vec(terms.iter().map(|&term| term.clone()).collect_vec())
}

/// A resolved merge of the given term.
fn resolved(term: Option<&TreeValue>) -> MergedTreeValue {
    Merge::resolved(term.cloned())
}

/// No labels for the terms of a merge.
fn unlabeled() -> ConflictLabels {
    ConflictLabels::unlabeled()
}

/// `value` with each file term shown as its content, for failure messages.
fn describe(case: &Case, value: &MergedTreeValue) -> String {
    let term = |term: &Option<TreeValue>| match (term, case.read(term.as_ref())) {
        (_, Some(content)) => format!("{:?}", String::from_utf8_lossy(&content)),
        (None, None) => "absent".to_owned(),
        (Some(other), None) => format!("{other:?}"),
    };
    let value = value.simplify();
    match value.as_resolved() {
        Some(resolved) => format!("resolved {}", term(resolved)),
        None => format!(
            "a {}-sided conflict, adds [{}], removes [{}]",
            value.num_sides(),
            value.adds().map(term).join(", "),
            value.removes().map(term).join(", ")
        ),
    }
}

/// Asserts that `actual` and `expected` have the same adds and the same
/// removes, with file contents in the failure message.
#[track_caller]
fn assert_same_value(case: &Case, actual: &MergedTreeValue, expected: &MergedTreeValue) {
    assert!(
        same_terms(actual, expected),
        "expected {}, got {}",
        describe(case, expected),
        describe(case, actual)
    );
}

/// The hex of `line` if it is a marker line that starts with `prefix`. An
/// opening marker line carries a salt and the names of V's terms after its
/// hex.
fn marker_hex<'a>(line: &'a str, prefix: &str) -> Option<&'a str> {
    let rest = line.strip_prefix(prefix)?;
    let hex = if prefix == OPEN {
        let (hex, salt_and_terms) = rest.split_once(' ')?;
        let (salt, terms) = salt_and_terms.split_once(' ')?;
        (salt.parse::<u64>().is_ok() && !terms.is_empty()).then_some(hex)?
    } else {
        rest
    };
    (hex.len() == 32 && hex.bytes().all(|byte| byte.is_ascii_hexdigit())).then_some(hex)
}

/// The hex of the opening marker line that starts `lines`.
#[track_caller]
fn opening_hex<'a>(lines: &[&'a str]) -> &'a str {
    marker_hex(lines[0], OPEN)
        .unwrap_or_else(|| panic!("expected an opening marker line first, got {lines:?}"))
}

/// Asserts that `content` is VB or M of `v`: it names V's terms, after
/// `simplify()`, and returns its opening marker line.
#[track_caller]
fn assert_names_terms<'a>(content: &'a [u8], v: &MergedTreeValue) -> &'a str {
    assert_eq!(virtual_base_terms(content), Some(v.simplify()));
    let lines = text_lines(content);
    opening_hex(&lines);
    lines[0]
}

/// Whether `content` holds a line that starts with the virtual base's opening
/// marker.
fn holds_marker(content: &[u8]) -> bool {
    content
        .split(|&byte| byte == b'\n')
        .any(|line| line.starts_with(OPEN.trim_end().as_bytes()))
}

/// The lines of a text file.
fn text_lines(content: &[u8]) -> Vec<&str> {
    std::str::from_utf8(content)
        .expect("the file is text")
        .lines()
        .collect()
}

/// Whether `content` is M: one block, whose sides are each one line that names
/// a term of V, and that the verifier of virtual bases accepts.
fn is_marker_file(content: &[u8]) -> bool {
    let Ok(text) = std::str::from_utf8(content) else {
        return false;
    };
    let lines = text.lines().collect_vec();
    let [first, inner @ .., last] = &lines[..] else {
        return false;
    };
    let Some(hex) = marker_hex(first, OPEN) else {
        return false;
    };
    let separator = format!("{SEPARATOR}{hex}");
    // A name may end with `@` and the label of the term.
    let names_a_term = |line: &str| {
        let name = line.split('@').next().unwrap_or(line);
        name == "absent"
            || ["file:", "symlink:", "tree:", "submodule:"]
                .iter()
                .any(|kind| name.starts_with(kind))
    };
    *last == format!("{CLOSE}{hex}")
        && inner.len() % 2 == 1
        && inner.iter().enumerate().all(|(index, &line)| {
            if index % 2 == 1 {
                line == separator
            } else {
                names_a_term(line)
            }
        })
        && virtual_base_terms(content).is_some()
}

/// Whether `content` is a virtual base of a conflict of text files: more than
/// one line, an opening marker line, a separator and a closing marker line,
/// and the verifier of virtual bases accepts it.
fn is_virtual_base(content: &[u8]) -> bool {
    let Ok(text) = std::str::from_utf8(content) else {
        return false;
    };
    let lines = text.lines().collect_vec();
    lines.len() > 1
        && lines.iter().any(|line| marker_hex(line, OPEN).is_some())
        && lines
            .iter()
            .any(|line| marker_hex(line, SEPARATOR).is_some())
        && lines.iter().any(|line| marker_hex(line, CLOSE).is_some())
        && virtual_base_terms(content).is_some()
}

#[track_caller]
fn assert_virtual_base(content: &[u8], sides: &[&str]) {
    assert!(
        is_virtual_base(content),
        "expected a virtual base with marker lines, got {:?}",
        String::from_utf8_lossy(content)
    );
    let lines = text_lines(content);
    for side in sides {
        assert!(
            lines.contains(side),
            "the virtual base {lines:?} does not hold V's side {side:?}"
        );
    }
}

#[track_caller]
fn assert_marker_file(content: &[u8]) {
    assert!(
        is_marker_file(content),
        "expected M, a block that names V's terms, got {:?}",
        String::from_utf8_lossy(content)
    );
}

/// Asserts that the merge holds `expected` at `f`, resolved.
fn assert_resolved_text(merged: &Merged, expected: &str) -> TestResult {
    let value = merged.value()?;
    let Ok(Some(TreeValue::File { id, .. })) = value.clone().into_resolved() else {
        panic!(
            "expected {expected:?} at f, got {}",
            describe(&merged.case, &value)
        );
    };
    let content = read_file(merged.case.store(), f(), &id);
    assert_eq!(String::from_utf8_lossy(&content), expected);
    Ok(())
}

/// Asserts that each parent's value at `f` (a conflicted parent's first add)
/// is among the adds of `value`.
fn assert_parents_among_adds(merged: &Merged, value: &MergedTreeValue) -> TestResult {
    for parent in &merged.parents {
        let parent_value = value_at(&parent.tree(), "f")?;
        let side = parent_value.first();
        assert!(
            value.adds().any(|add| add == side),
            "the conflict does not hold the parent's value {}: {}",
            describe(&merged.case, &Merge::resolved(side.clone())),
            describe(&merged.case, value)
        );
    }
    Ok(())
}

/// The base term a conflict at `f` holds as its one remove.
enum Base {
    /// The virtual base of a conflict of text files, which holds these lines
    /// of V's sides between its marker lines.
    Virtual(&'static [&'static str]),
    /// M, the block that names V's terms, for a V that is not a conflict of
    /// text files.
    Marker,
    /// A file of this text, which no virtual base is: a term of V.
    Text(String),
}

/// Asserts that the merge is a conflict at `f`, and returns its value there,
/// simplified.
fn conflict_at_f(merged: &Merged) -> TestResult<MergedTreeValue> {
    let value = merged.value()?;
    assert!(
        !value.is_resolved(),
        "expected a conflict at f, got {}",
        describe(&merged.case, &value)
    );
    Ok(value.simplify())
}

/// Asserts that the merge is a conflict at `f` whose adds hold each parent's
/// value there, and no marker line of a virtual base, and whose one remove,
/// after `simplify()`, is `base`.
fn assert_conflict(merged: &Merged, base: Base) -> TestResult {
    let simplified = conflict_at_f(merged)?;
    assert_parents_among_adds(merged, &simplified)?;
    assert!(
        !simplified.adds().any(|add| merged
            .case
            .read(add.as_ref())
            .is_some_and(|content| holds_marker(&content))),
        "a side of the conflict holds a virtual base's marker line: {}",
        describe(&merged.case, &simplified)
    );
    let removes = simplified.removes().collect_vec();
    let [remove] = removes[..] else {
        panic!(
            "expected one remove in the conflict, got {}",
            describe(&merged.case, &simplified)
        );
    };
    let content = merged
        .case
        .read(remove.as_ref())
        .unwrap_or_else(|| panic!("expected a file as the conflict's remove, got {remove:?}"));
    match base {
        Base::Virtual(sides) => assert_virtual_base(&content, sides),
        Base::Marker => assert_marker_file(&content),
        Base::Text(text) => assert_eq!(
            String::from_utf8_lossy(&content),
            text,
            "expected the conflict's base to be a term of V"
        ),
    }
    Ok(())
}

/// The history most cases share: o; A and B, children of o; P1 and P2, two
/// merges of A and B, which have the two merge bases A and B.
struct TwoBases {
    case: Case,
    a: Commit,
    b: Commit,
    p1: Commit,
    p2: Commit,
}

fn two_bases(
    o_writes: &[Write],
    a_writes: &[Write],
    b_writes: &[Write],
    made: Made,
    p1_writes: &[Write],
    p2_writes: &[Write],
) -> TestResult<TwoBases> {
    let mut case = Case::new();
    let root = case.root();
    let o = case.commit(&[&root], o_writes)?;
    let (a, b) = case.commit_a_and_b(made, &o, a_writes, b_writes)?;
    let p1 = case.commit(&[&a, &b], p1_writes)?;
    let p2 = case.commit(&[&a, &b], p2_writes)?;
    Ok(TwoBases { case, a, b, p1, p2 })
}

/// K4: A sets `f` to `x` and B to `y`; P1 keeps `x` and P2 `y`.
fn k4_history(made: Made) -> TestResult<TwoBases> {
    two_bases(
        &[("f", line("o"))],
        &[("f", line("x"))],
        &[("f", line("y"))],
        made,
        &[("f", line("x"))],
        &[("f", line("y"))],
    )
}

/// VB6's history: A replaces line 1 of ten by `x1` and `x2`, B by `y1`; P1
/// keeps A's text, and P2 adds `Q` and `W` to it.
fn vb6_history() -> TestResult<TwoBases> {
    two_bases(
        &[("f", lines("1 2 3 4 5 6 7 8 9 10"))],
        &[("f", lines("x1 x2 2 3 4 5 6 7 8 9 10"))],
        &[("f", lines("y1 2 3 4 5 6 7 8 9 10"))],
        Made::AFirst,
        &[("f", lines("x1 x2 2 3 4 5 6 7 8 9 10"))],
        &[("f", lines("x1 Q x2 W 2 3 4 5 6 7 8 9 10"))],
    )
}

/// The history in which m is a conflicted merge of a and b, which change `f`
/// from `o` to `a_value` and to `b_value`.
struct ConflictedBase {
    case: Case,
    o: Commit,
    m: Commit,
}

fn conflicted_base(a_value: &str, b_value: &str) -> TestResult<ConflictedBase> {
    let mut case = Case::new();
    let root = case.root();
    let o = case.commit(&[&root], &[("f", line("o"))])?;
    let a = case.commit(&[&o], &[("f", line(a_value))])?;
    let b = case.commit(&[&o], &[("f", line(b_value))])?;
    let m = case.commit(&[&a, &b], &[])?;
    assert!(
        !value_at(&m.tree(), "f")?.is_resolved(),
        "the case needs m to be conflicted at f"
    );
    Ok(ConflictedBase { case, o, m })
}

/// K1: r1 resolves m's conflict at `f` to `R`, and r2 changes another file;
/// the merge of r1 and r2 has the one merge base m.
fn case_k1(order: Order) -> TestResult<Merged> {
    let ConflictedBase { mut case, m, .. } = conflicted_base("a", "b")?;
    let r1 = case.commit(&[&m], &[("f", line("R"))])?;
    let r2 = case.commit(&[&m], &[("g", line("x"))])?;
    match order {
        Order::AsNamed => Merged::new(case, &[&r1, &r2]),
        Order::Swapped => {
            // The swapped case follows K1's merge and K3's commits.
            case.commit(&[&r1, &r2], &[])?;
            let r3 = case.commit(&[&m], &[("h", line("x"))])?;
            let r4 = case.commit(&[&m], &[("k", line("x"))])?;
            case.commit(&[&r3, &r4], &[])?;
            Merged::new(case, &[&r2, &r1])
        }
    }
}

/// K3: K1's m, with two children r3 and r4 that leave `f` alone. Returns the
/// merge of r3 and r4, and m's value at `f`.
fn case_k3() -> TestResult<(Merged, MergedTreeValue)> {
    let ConflictedBase { mut case, m, .. } = conflicted_base("a", "b")?;
    let r1 = case.commit(&[&m], &[("f", line("R"))])?;
    let r2 = case.commit(&[&m], &[("g", line("x"))])?;
    case.commit(&[&r1, &r2], &[])?;
    let r3 = case.commit(&[&m], &[("h", line("x"))])?;
    let r4 = case.commit(&[&m], &[("k", line("x"))])?;
    let m_value = value_at(&m.tree(), "f")?;
    Ok((Merged::new(case, &[&r3, &r4])?, m_value))
}

/// K2: a criss-cross in which X, a merge of A and B, resolves `f` to `R`, and
/// Y, another merge of A and B, leaves the conflict.
fn case_k2(made: Made, order: Order) -> TestResult<Merged> {
    let mut case = Case::new();
    let root = case.root();
    let o = case.commit(&[&root], &[("f", line("o"))])?;
    let (a, b) = case.commit_a_and_b(made, &o, &[("f", line("a"))], &[("f", line("b"))])?;
    let x = case.commit(&[&a, &b], &[("f", line("R"))])?;
    let y = case.commit(&[&a, &b], &[("g", line("x"))])?;
    Merged::pair(case, &x, &y, order)
}

fn case_k4(made: Made, order: Order) -> TestResult<Merged> {
    let TwoBases { case, p1, p2, .. } = k4_history(made)?;
    Merged::pair(case, &p1, &p2, order)
}

/// K5: A sets `f` to `x` and B to `y`; P1 goes back to `o`, P2 keeps `y`.
fn case_k5() -> TestResult<Merged> {
    let TwoBases { case, p1, p2, .. } = two_bases(
        &[("f", line("o"))],
        &[("f", line("x"))],
        &[("f", line("y"))],
        Made::AFirst,
        &[("f", line("o"))],
        &[("f", line("y"))],
    )?;
    Merged::new(case, &[&p1, &p2])
}

/// D1: A deletes `f` and B sets it to `y`; P1 deletes it, P2 keeps `y`.
fn case_d1(order: Order) -> TestResult<Merged> {
    let TwoBases { case, p1, p2, .. } = two_bases(
        &[("f", line("o")), ("g", line("g"))],
        &[("f", None)],
        &[("f", line("y"))],
        Made::AFirst,
        &[("f", None)],
        &[("f", line("y"))],
    )?;
    Merged::pair(case, &p1, &p2, order)
}

/// o, then three merge bases A (`x`), B (`y`) and C (`x`, with `c=1`), made in
/// the order `made` gives for A and B, and C last.
fn three_bases(made: Made) -> TestResult<(Case, [Commit; 3])> {
    let mut case = Case::new();
    let root = case.root();
    let o = case.commit(&[&root], &[("f", line("o"))])?;
    let (a, b) = case.commit_a_and_b(made, &o, &[("f", line("x"))], &[("f", line("y"))])?;
    let c = case.commit(&[&o], &[("f", line("x")), ("c", line("1"))])?;
    Ok((case, [a, b, c]))
}

/// T3: the three merge bases of `three_bases`; P1 keeps `x`, P2 `y`.
fn case_t3() -> TestResult<Merged> {
    let (mut case, [a, b, c]) = three_bases(Made::AFirst)?;
    let p1 = case.commit(&[&a, &b, &c], &[("f", line("x"))])?;
    let p2 = case.commit(&[&a, &b, &c], &[("f", line("y"))])?;
    Merged::new(case, &[&p1, &p2])
}

/// T4: three merge bases that all hold `b`, whose own merge bases hold `c` and
/// `d`; P1 keeps `b`, P2 sets `z`.
fn case_t4(order: Order) -> TestResult<Merged> {
    let mut case = Case::new();
    let root = case.root();
    let o = case.commit(&[&root], &[("f", line("o"))])?;
    let d = case.commit(&[&o], &[("f", line("d"))])?;
    let c = case.commit(&[&d], &[("f", line("c"))])?;
    let b1 = case.commit(&[&c], &[("f", line("b"))])?;
    let b2 = case.commit(&[&c], &[("f", line("b")), ("g", line("2"))])?;
    let b3 = case.commit(&[&d], &[("f", line("b")), ("h", line("3"))])?;
    let p1 = case.commit(&[&b1, &b2, &b3], &[("f", line("b"))])?;
    let p2 = case.commit(&[&b1, &b2, &b3], &[("f", line("z"))])?;
    Merged::pair(case, &p1, &p2, order)
}

/// T5: the three merge bases of `three_bases`; X resolves `f` to `R`, Y leaves
/// the conflict.
fn case_t5(made: Made) -> TestResult<Merged> {
    let (mut case, [a, b, c]) = three_bases(made)?;
    let x = case.commit(&[&a, &b, &c], &[("f", line("R"))])?;
    let y = case.commit(&[&a, &b, &c], &[("g", line("1"))])?;
    Merged::new(case, &[&x, &y])
}

/// T6: three merge bases A (`x`), B (`y`) and C (`x`); X is a merge of A, B and
/// C that resolves `f` to `R`, Y a merge of C, B and A that leaves the
/// conflict.
fn case_t6(made: Made) -> TestResult<Merged> {
    let mut case = Case::new();
    let root = case.root();
    let o = case.commit(&[&root], &[("f", line("o"))])?;
    let [a, b, c] = match made {
        Made::AFirst => {
            let a = case.commit(&[&o], &[("f", line("x"))])?;
            let b = case.commit(&[&o], &[("f", line("y"))])?;
            let c = case.commit(&[&o], &[("f", line("x"))])?;
            [a, b, c]
        }
        Made::BFirst => {
            let b = case.commit(&[&o], &[("f", line("y"))])?;
            let c = case.commit(&[&o], &[("f", line("x"))])?;
            let a = case.commit(&[&o], &[("f", line("x"))])?;
            [a, b, c]
        }
    };
    let x = case.commit(&[&a, &b, &c], &[("f", line("R"))])?;
    let y = case.commit(&[&c, &b, &a], &[("g", line("1"))])?;
    Merged::new(case, &[&x, &y])
}

/// H1: A and B change line 1 to `x` and `y`; both parents resolve it to `z`,
/// and P2 also changes line 9.
fn case_h1() -> TestResult<Merged> {
    let TwoBases { case, p1, p2, .. } = two_bases(
        &[("f", lines("1 2 3 4 5 6 7 8 9 10"))],
        &[("f", lines("x 2 3 4 5 6 7 8 9 10"))],
        &[("f", lines("y 2 3 4 5 6 7 8 9 10"))],
        Made::AFirst,
        &[("f", lines("z 2 3 4 5 6 7 8 9 10"))],
        &[("f", lines("z 2 3 4 5 6 7 8 Q 10"))],
    )?;
    Merged::new(case, &[&p1, &p2])
}

/// E1: A changes line 1 to `x` and line 9 to `9a`, B changes line 1 to `y`; P1
/// resolves line 1 to `z` and reverts `9a`, P2 resolves line 1 to `z` and keeps
/// `9a`.
fn case_e1(order: Order) -> TestResult<Merged> {
    let TwoBases { case, p1, p2, .. } = two_bases(
        &[("f", lines("1 2 3 4 5 6 7 8 9 10"))],
        &[("f", lines("x 2 3 4 5 6 7 8 9a 10"))],
        &[("f", lines("y 2 3 4 5 6 7 8 9 10"))],
        Made::AFirst,
        &[("f", lines("z 2 3 4 5 6 7 8 9 10"))],
        &[("f", lines("z 2 3 4 5 6 7 8 9a 10"))],
    )?;
    Merged::pair(case, &p1, &p2, order)
}

/// E2: A changes line 1 to `x`, B to `y`; P1 keeps A's text, P2 goes back to
/// `1` at line 1 and changes line 9 to `Q`.
fn case_e2(order: Order) -> TestResult<Merged> {
    let TwoBases { case, p1, p2, .. } = two_bases(
        &[("f", lines("1 2 3 4 5 6 7 8 9 10"))],
        &[("f", lines("x 2 3 4 5 6 7 8 9 10"))],
        &[("f", lines("y 2 3 4 5 6 7 8 9 10"))],
        Made::AFirst,
        &[("f", lines("x 2 3 4 5 6 7 8 9 10"))],
        &[("f", lines("1 2 3 4 5 6 7 8 Q 10"))],
    )?;
    Merged::pair(case, &p1, &p2, order)
}

/// E4: Z is K4's merge of P1 and P2, as a commit; W is a child of P2 that
/// deletes `f`.
fn case_e4() -> TestResult<Merged> {
    let TwoBases {
        mut case, p1, p2, ..
    } = two_bases(
        &[("f", line("o")), ("g", line("g"))],
        &[("f", line("x"))],
        &[("f", line("y"))],
        Made::AFirst,
        &[("f", line("x"))],
        &[("f", line("y"))],
    )?;
    let z = case.commit(&[&p1, &p2], &[])?;
    let w = case.commit(&[&p2], &[("f", None)])?;
    Merged::new(case, &[&z, &w])
}

/// MD1's history: A deletes `f`, B sets it to `y`; P1 deletes it, P2 goes back
/// to `o`.
fn md1_history() -> TestResult<TwoBases> {
    two_bases(
        &[("f", line("o")), ("g", line("g"))],
        &[("f", None)],
        &[("f", line("y"))],
        Made::AFirst,
        &[("f", None)],
        &[("f", line("o"))],
    )
}

fn case_md1() -> TestResult<Merged> {
    let TwoBases { case, p1, p2, .. } = md1_history()?;
    Merged::new(case, &[&p1, &p2])
}

/// MD2: MD1's history, then P3, a merge of A and B that keeps `y`; the merge of
/// P3 and P2.
fn case_md2() -> TestResult<Merged> {
    let TwoBases {
        mut case,
        a,
        b,
        p1,
        p2,
    } = md1_history()?;
    case.commit(&[&p1, &p2], &[])?;
    let p3 = case.commit(&[&a, &b], &[("f", line("y"))])?;
    Merged::new(case, &[&p3, &p2])
}

/// BIN1: K5 with binary files.
fn case_bin1() -> TestResult<Merged> {
    let TwoBases { case, p1, p2, .. } = two_bases(
        &[("f", binary("o"))],
        &[("f", binary("x"))],
        &[("f", binary("y"))],
        Made::AFirst,
        &[("f", binary("o"))],
        &[("f", binary("y"))],
    )?;
    Merged::new(case, &[&p1, &p2])
}

/// X1: A replaces line 1 by `x1` and `x2`, B by `y1`; P1 keeps A's text, P2 adds
/// `Q` between `x1` and `x2`.
fn case_x1(order: Order) -> TestResult<Merged> {
    let TwoBases { case, p1, p2, .. } = two_bases(
        &[("f", lines("1 2 3 4 5 6 7 8 9 10"))],
        &[("f", lines("x1 x2 2 3 4 5 6 7 8 9 10"))],
        &[("f", lines("y1 2 3 4 5 6 7 8 9 10"))],
        Made::AFirst,
        &[("f", lines("x1 x2 2 3 4 5 6 7 8 9 10"))],
        &[("f", lines("x1 Q x2 2 3 4 5 6 7 8 9 10"))],
    )?;
    Merged::pair(case, &p1, &p2, order)
}

/// The minimal case with three merge bases: B1, B2 and B3 each change `f` from
/// `c` to `b`, and m1 and m2 are two merges of all three.
fn case_minimal_three_bases() -> TestResult<Merged> {
    let mut case = Case::new();
    let root = case.root();
    let o = case.commit(&[&root], &[("f", line("c"))])?;
    let b1 = case.commit(&[&o], &[("f", line("b"))])?;
    let b2 = case.commit(&[&o], &[("f", line("b"))])?;
    let b3 = case.commit(&[&o], &[("f", line("b"))])?;
    let m1 = case.commit(&[&b1, &b2, &b3], &[])?;
    let m2 = case.commit(&[&b1, &b2, &b3], &[])?;
    Merged::new(case, &[&m1, &m2])
}

/// m is a conflicted merge of a and b; p2, a child of o, takes `f` from m, as
/// `jj restore --from m f` does. Returns the merge of m and p2, and m's value.
fn case_same_conflict_in_both_parents() -> TestResult<(Merged, MergedTreeValue)> {
    let ConflictedBase { mut case, o, m } = conflicted_base("a", "b")?;
    let m_value = value_at(&m.tree(), "f")?;
    let p2 = case.commit_with_values(&[&o], vec![(repo_path_buf("f"), m_value.clone())])?;
    Ok((Merged::new(case, &[&m, &p2])?, m_value))
}

/// MD1's conflict, as a commit Z, merged with W, a child of P2 that deletes
/// `f`.
fn case_md1_then_a_deletion() -> TestResult<Merged> {
    let TwoBases {
        mut case, p1, p2, ..
    } = md1_history()?;
    let z = case.commit(&[&p1, &p2], &[])?;
    let w = case.commit(&[&p2], &[("f", None)])?;
    Merged::new(case, &[&z, &w])
}

/// Z is K4's merge of P1 and P2, as a commit; W is a third merge of A and B,
/// which sets `f` to `w`.
fn case_k4_result_then_a_resolution_of_the_bases() -> TestResult<Merged> {
    let TwoBases {
        mut case,
        a,
        b,
        p1,
        p2,
    } = k4_history(Made::AFirst)?;
    let z = case.commit(&[&p1, &p2], &[])?;
    let w = case.commit(&[&a, &b], &[("f", line("w"))])?;
    Merged::new(case, &[&z, &w])
}

/// Q1 is a merge of K4's P1 and P2 that leaves `f`, Q2 one that sets `f` to
/// `R`.
fn case_two_level_resolution_of_a_k4_merge() -> TestResult<Merged> {
    let TwoBases {
        mut case, p1, p2, ..
    } = k4_history(Made::AFirst)?;
    let q1 = case.commit(&[&p1, &p2], &[("g", line("x"))])?;
    let q2 = case.commit(&[&p1, &p2], &[("f", line("R"))])?;
    Merged::new(case, &[&q1, &q2])
}

/// m is a conflicted merge of a and b; r1 resolves `f` to `a`, r2 to `b`.
fn case_single_conflicted_base_two_resolutions() -> TestResult<Merged> {
    let ConflictedBase { mut case, m, .. } = conflicted_base("a", "b")?;
    let r1 = case.commit(&[&m], &[("f", line("a"))])?;
    let r2 = case.commit(&[&m], &[("f", line("b"))])?;
    Merged::new(case, &[&r1, &r2])
}

/// a and b set `f` to `A` and `B`; r1 resolves m's conflict to `A`, m's first
/// side; r2 adds `g2`.
fn case_k1_resolved_to_the_bases_first_side() -> TestResult<Merged> {
    let ConflictedBase { mut case, m, .. } = conflicted_base("A", "B")?;
    let r1 = case.commit(&[&m], &[("f", line("A"))])?;
    let r2 = case.commit(&[&m], &[("g2", line("g2"))])?;
    Merged::new(case, &[&r1, &r2])
}

/// r1, a child of the conflicted m, resolves `f` to `R`; the merge of m and r1.
fn case_merge_of_a_conflicted_commit_and_its_descendant() -> TestResult<Merged> {
    let ConflictedBase { mut case, m, .. } = conflicted_base("a", "b")?;
    let r1 = case.commit(&[&m], &[("f", line("R"))])?;
    Merged::new(case, &[&m, &r1])
}

/// A adds `f` as a file and makes `e` executable; B adds `f/x`, so `f` is a
/// directory, and changes `e`. X and Y are two merges of A and B: X resolves
/// `f` to A's file where `x_resolves`, else changes another file; Y changes
/// another file.
fn case_k2_file_directory(x_resolves: bool) -> TestResult<Merged> {
    let mut case = Case::new();
    let root = case.root();
    let o = case.commit(&[&root], &[("g", line("o")), ("e", line("e"))])?;
    let file_f = case.file(f(), b"f\n", false, CopyId::placeholder());
    let executable_e = case.file(repo_path("e"), b"e\n", true, CopyId::placeholder());
    let a = case.commit_with_values(
        &[&o],
        vec![
            (repo_path_buf("f"), Merge::resolved(file_f)),
            (repo_path_buf("e"), Merge::resolved(executable_e)),
        ],
    )?;
    let b = case.commit(&[&o], &[("f/x", line("x")), ("e", line("e2"))])?;
    let x_writes = if x_resolves {
        [("f", line("f"))]
    } else {
        [("h", line("h"))]
    };
    let x = case.commit(&[&a, &b], &x_writes)?;
    let y = case.commit(&[&a, &b], &[("k", line("k"))])?;
    Merged::new(case, &[&x, &y])
}

/// m is a conflicted merge of a and b; p and q are children of m that change
/// other files; X, a merge of p and q, resolves `f` to `R`, and Y, another
/// merge of p and q, adds `h`.
fn case_criss_cross_bases_inherit_a_conflict() -> TestResult<Merged> {
    let ConflictedBase { mut case, m, .. } = conflicted_base("a", "b")?;
    let p = case.commit(&[&m], &[("g1", line("g1"))])?;
    let q = case.commit(&[&m], &[("g2", line("g2"))])?;
    let x = case.commit(&[&p, &q], &[("f", line("R"))])?;
    let y = case.commit(&[&p, &q], &[("h", line("h"))])?;
    Merged::new(case, &[&x, &y])
}

/// A sets `f` to `a`, B to `b`; X, a merge of A and B, resolves `f` to `c`, and
/// Y, another, to `b`.
fn case_resolved_differently(made: Made) -> TestResult<Merged> {
    let TwoBases { case, p1, p2, .. } = two_bases(
        &[("f", line("o"))],
        &[("f", line("a"))],
        &[("f", line("b"))],
        made,
        &[("f", line("c"))],
        &[("f", line("b"))],
    )?;
    Merged::new(case, &[&p1, &p2])
}

/// o holds `d/f` and `g`; a and b change `d/f`, and m, their merge, is
/// conflicted there; c, a child of o, replaces the directory `d` with a file;
/// P1 merges m and c, so it holds a conflict between a directory and a file at
/// `d`; P2, a child of m, changes `g`. Returns the merge of P1 and P2, and P1's
/// value at `d`.
fn case_file_directory_conflict_in_a_side() -> TestResult<(Merged, MergedTreeValue)> {
    let mut case = Case::new();
    let root = case.root();
    let o = case.commit(&[&root], &[("d/f", line("o")), ("g", line("g"))])?;
    let a = case.commit(&[&o], &[("d/f", line("a"))])?;
    let b = case.commit(&[&o], &[("d/f", line("b"))])?;
    let m = case.commit(&[&a, &b], &[])?;
    let c = case.commit(&[&o], &[("d", line("c"))])?;
    let p1 = case.commit(&[&m, &c], &[])?;
    let p2 = case.commit(&[&m], &[("g", line("g2"))])?;
    let p1_value = value_at(&p1.tree(), "d")?;
    assert!(
        !p1_value.is_resolved(),
        "the case needs P1 to hold a conflict at d"
    );
    Ok((Merged::new(case, &[&p1, &p2])?, p1_value))
}

/// K4's history and d, a merge of P1 and P2 that resolves `f` to `y` and
/// makes `d_writes`. d's parent tree holds the rule's conflict `[x, VB, y]`,
/// which a rewrite of d inverts: as the base of a merge, its VB is an add.
fn resolved_k4_merge(d_writes: &[Write]) -> TestResult<(TwoBases, Commit)> {
    resolved_merge(k4_history(Made::AFirst)?, d_writes)
}

/// `history` and d, a merge of its P1 and P2 that makes `d_writes`.
fn resolved_merge(mut history: TwoBases, d_writes: &[Write]) -> TestResult<(TwoBases, Commit)> {
    let d = history.case.commit(&[&history.p1, &history.p2], d_writes)?;
    Ok((history, d))
}

/// The six lines of `f` that o holds in the histories whose merge bases change
/// some of them.
const SIX_LINES: &str = "1\n2\n3\n4\n5\n6\n";

/// K4's history with six lines at `f`: A changes line 1 to `a_line1`, and B
/// changes lines 1 to 3 to `b_lines`; P1 and P2 keep A's and B's text.
fn six_line_history(a_line1: &str, b_lines: &str) -> TestResult<TwoBases> {
    let a_text = lines(&format!("{a_line1} 2 3 4 5 6"));
    let b_text = lines(&format!("{b_lines} 4 5 6"));
    two_bases(
        &[("f", lines("1 2 3 4 5 6"))],
        &[("f", a_text.clone())],
        &[("f", b_text.clone())],
        Made::AFirst,
        &[("f", a_text)],
        &[("f", b_text)],
    )
}

/// VB2's history: A sets line 1 to `a`, B lines 1 to 3 to `b1 b2 b3`, and d
/// resolves `f` to B's text with `b2` changed to `B2`, inside the hunk where A
/// and B disagree. d's parent tree holds `[P1, VB, P2]`, and d's change from it
/// leaves VB's marker lines in place, so no merge of VB's text with d's
/// change can tell that it is VB.
fn b2_history() -> TestResult<(TwoBases, Commit)> {
    resolved_merge(
        six_line_history("a", "b1 b2 b3")?,
        &[("f", lines("b1 B2 b3 4 5 6"))],
    )
}

/// d rebased onto P1, as `jj rebase -r d -d P1` rebases it.
fn rebase_onto_p1((history, d): (TwoBases, Commit)) -> TestResult<Merged> {
    let TwoBases { mut case, p1, .. } = history;
    let rebased = rebase_commit(case.tx.repo_mut(), d.clone(), vec![p1.id().clone()]).block_on()?;
    Ok(Merged {
        case,
        parents: vec![p1, d],
        tree: rebased.tree(),
    })
}

/// d rebased onto P2's parents once P2 is abandoned, as `jj abandon P2`
/// rebases it.
fn abandon_p2((history, d): (TwoBases, Commit)) -> TestResult<Merged> {
    let TwoBases {
        mut case, p1, p2, ..
    } = history;
    case.tx.repo_mut().record_abandoned_commit(&p2);
    let rebased =
        rebase_descendants_with_options_return_map(case.tx.repo_mut(), &RebaseOptions::default());
    let new_d = case.store().get_commit(&rebased[d.id()])?;
    Ok(Merged {
        case,
        parents: vec![p1, d],
        tree: new_d.tree(),
    })
}

/// P1's tree once d is squashed into it, as `jj squash --from d --into P1`
/// computes it: P1's tree with d's change from d's parent tree.
fn squash_into_p1((history, d): (TwoBases, Commit)) -> TestResult<Merged> {
    let TwoBases { case, p1, .. } = history;
    let parent_tree = d.parent_tree(case.tx.repo()).block_on()?;
    let tree = MergedTree::merge(Merge::from_vec(vec![
        (p1.tree(), "P1".to_owned()),
        (parent_tree, "parents of d".to_owned()),
        (d.tree(), "d".to_owned()),
    ]))
    .block_on()?;
    Ok(Merged {
        case,
        parents: vec![p1, d],
        tree,
    })
}

/// VB1a: d rebased onto P1. The result's conflict must hold P1's `x` and d's
/// `y`.
fn case_vb1a(d_writes: &[Write]) -> TestResult<Merged> {
    rebase_onto_p1(resolved_k4_merge(d_writes)?)
}

/// VB1b: d rebased once P2 is abandoned.
fn case_vb1b() -> TestResult<Merged> {
    abandon_p2(resolved_k4_merge(&[("f", line("y"))])?)
}

/// VB1c: d squashed into P1.
fn case_vb1c() -> TestResult<Merged> {
    squash_into_p1(resolved_k4_merge(&[("f", line("y"))])?)
}

/// Z is the merge of `history`'s P1 and P2, as a commit; c and q are children
/// of Z that make `c_writes` and `q_writes`. Returns c rebased onto q, as `jj
/// rebase -r c -d q` rebases it.
fn case_sibling_rebase(
    history: TwoBases,
    c_writes: &[Write],
    q_writes: &[Write],
) -> TestResult<Merged> {
    let TwoBases {
        mut case, p1, p2, ..
    } = history;
    let z = case.commit(&[&p1, &p2], &[])?;
    let c = case.commit(&[&z], c_writes)?;
    let q = case.commit(&[&z], q_writes)?;
    let rebased = rebase_commit(case.tx.repo_mut(), c.clone(), vec![q.id().clone()]).block_on()?;
    Ok(Merged {
        case,
        parents: vec![q, c],
        tree: rebased.tree(),
    })
}

/// The sibling rebase of K4's conflict: c resolves it to `x`, q to `y`.
fn case_vb1_sibling() -> TestResult<Merged> {
    case_sibling_rebase(
        k4_history(Made::AFirst)?,
        &[("f", line("x"))],
        &[("f", line("y"))],
    )
}

/// The sibling rebase of D1's conflict, whose base is M: c deletes `f`, q
/// keeps `y`.
fn case_vb1_sibling_d1() -> TestResult<Merged> {
    let history = two_bases(
        &[("f", line("o")), ("g", line("g"))],
        &[("f", None)],
        &[("f", line("y"))],
        Made::AFirst,
        &[("f", None)],
        &[("f", line("y"))],
    )?;
    case_sibling_rebase(history, &[("f", None)], &[("f", line("y"))])
}

/// The sibling rebase of VB2's conflict: c resolves it to A's text, q to B's
/// text with `B2`.
fn case_vb2_sibling() -> TestResult<Merged> {
    case_sibling_rebase(
        six_line_history("a", "b1 b2 b3")?,
        &[("f", lines("a 2 3 4 5 6"))],
        &[("f", lines("b1 B2 b3 4 5 6"))],
    )
}

/// VB3: d resolves K4's hunk to `y` and changes line 5, outside that hunk; d
/// rebased onto P1.
fn case_vb3() -> TestResult<Merged> {
    rebase_onto_p1(resolved_merge(
        six_line_history("x", "y 2 3")?,
        &[("f", lines("y 2 3 4 FIVE 6"))],
    )?)
}

#[test]
fn k1_single_base() -> TestResult {
    assert_resolved_text(&case_k1(Order::AsNamed)?, "R\n")
}

#[test]
fn k1_single_base_swapped() -> TestResult {
    assert_resolved_text(&case_k1(Order::Swapped)?, "R\n")
}

#[test]
fn k2_criss_cross() -> TestResult {
    assert_resolved_text(&case_k2(Made::AFirst, Order::AsNamed)?, "R\n")
}

#[test]
fn k2_criss_cross_swapped() -> TestResult {
    assert_resolved_text(&case_k2(Made::AFirst, Order::Swapped)?, "R\n")
}

#[test]
fn k2_b_made_first() -> TestResult {
    assert_resolved_text(&case_k2(Made::BFirst, Order::AsNamed)?, "R\n")
}

/// Both parents leave V's conflict, so the merge holds V's own conflict, with
/// V's sides.
#[test]
fn k3_both_leave_the_base_conflict() -> TestResult {
    let (merged, m_value) = case_k3()?;
    let value = merged.value()?;
    assert_eq!(m_value.simplify().num_sides(), 2);
    assert_same_value(&merged.case, &value, &m_value);
    Ok(())
}

#[test]
fn k4a() -> TestResult {
    assert_conflict(
        &case_k4(Made::AFirst, Order::AsNamed)?,
        Base::Virtual(&["x", "y"]),
    )
}

#[test]
fn k4b() -> TestResult {
    assert_conflict(
        &case_k4(Made::AFirst, Order::Swapped)?,
        Base::Virtual(&["x", "y"]),
    )
}

#[test]
fn k4c() -> TestResult {
    assert_conflict(
        &case_k4(Made::BFirst, Order::AsNamed)?,
        Base::Virtual(&["x", "y"]),
    )
}

#[test]
fn k4d() -> TestResult {
    assert_conflict(
        &case_k4(Made::BFirst, Order::Swapped)?,
        Base::Virtual(&["x", "y"]),
    )
}

#[test]
fn k5_one_parent_returns_to_the_bases_base() -> TestResult {
    assert_conflict(&case_k5()?, Base::Virtual(&["x", "y"]))
}

#[test]
fn d1_delete_and_modify() -> TestResult {
    assert_conflict(&case_d1(Order::AsNamed)?, Base::Marker)
}

#[test]
fn d1_swapped() -> TestResult {
    assert_conflict(&case_d1(Order::Swapped)?, Base::Marker)
}

#[test]
fn t3_three_bases() -> TestResult {
    assert_conflict(&case_t3()?, Base::Virtual(&["x", "y"]))
}

/// Every merge base holds `b`, so V is `b` whatever their own merge bases hold,
/// and the merge takes P2's `z`.
#[test]
fn t4_three_bases_all_b() -> TestResult {
    assert_resolved_text(&case_t4(Order::AsNamed)?, "z\n")
}

#[test]
fn t4_three_bases_all_b_swapped() -> TestResult {
    assert_resolved_text(&case_t4(Order::Swapped)?, "z\n")
}

#[test]
fn t5_a_made_first() -> TestResult {
    assert_resolved_text(&case_t5(Made::AFirst)?, "R\n")
}

#[test]
fn t5_b_made_first() -> TestResult {
    assert_resolved_text(&case_t5(Made::BFirst)?, "R\n")
}

#[test]
fn t6_a_made_first() -> TestResult {
    assert_resolved_text(&case_t6(Made::AFirst)?, "R\n")
}

#[test]
fn t6_b_made_first() -> TestResult {
    assert_resolved_text(&case_t6(Made::BFirst)?, "R\n")
}

#[test]
fn h1_git_merge() -> TestResult {
    assert_resolved_text(&case_h1()?, "z\n2\n3\n4\n5\n6\n7\n8\nQ\n10\n")
}

#[test]
fn e1_git_merge() -> TestResult {
    assert_resolved_text(&case_e1(Order::AsNamed)?, "z\n2\n3\n4\n5\n6\n7\n8\n9\n10\n")
}

#[test]
fn e1_swapped() -> TestResult {
    assert_resolved_text(&case_e1(Order::Swapped)?, "z\n2\n3\n4\n5\n6\n7\n8\n9\n10\n")
}

#[test]
fn e2_conflict() -> TestResult {
    assert_conflict(&case_e2(Order::AsNamed)?, Base::Virtual(&["x", "y"]))
}

#[test]
fn e2_swapped() -> TestResult {
    assert_conflict(&case_e2(Order::Swapped)?, Base::Virtual(&["x", "y"]))
}

/// No side of K4's conflict is a deletion, so a later merge with a deletion
/// keeps a conflict.
#[test]
fn e4_deletion_after_a_k4_merge() -> TestResult {
    let merged = case_e4()?;
    let simplified = conflict_at_f(&merged)?;
    assert!(
        simplified.adds().any(|add| add.is_none()),
        "expected a deletion among the conflict's sides, got {}",
        describe(&merged.case, &simplified)
    );
    assert_parents_among_adds(&merged, &simplified)
}

#[test]
fn md1_delete_and_back_to_the_base() -> TestResult {
    assert_conflict(&case_md1()?, Base::Marker)
}

#[test]
fn md2_kept_and_back_to_the_base() -> TestResult {
    assert_conflict(&case_md2()?, Base::Marker)
}

#[test]
fn bin1_binary() -> TestResult {
    assert_conflict(&case_bin1()?, Base::Marker)
}

#[test]
fn x1_git_merge() -> TestResult {
    assert_resolved_text(
        &case_x1(Order::AsNamed)?,
        "x1\nQ\nx2\n2\n3\n4\n5\n6\n7\n8\n9\n10\n",
    )
}

#[test]
fn x1_swapped() -> TestResult {
    assert_resolved_text(
        &case_x1(Order::Swapped)?,
        "x1\nQ\nx2\n2\n3\n4\n5\n6\n7\n8\n9\n10\n",
    )
}

#[test]
fn minimal_three_bases() -> TestResult {
    assert_resolved_text(&case_minimal_three_bases()?, "b\n")
}

/// V resolves, and both parents hold the same conflict K, which the merge
/// keeps as it is.
#[test]
fn same_conflict_in_both_parents_with_a_resolved_base() -> TestResult {
    let (merged, m_value) = case_same_conflict_in_both_parents()?;
    assert_same_value(&merged.case, &merged.value()?, &m_value);
    Ok(())
}

/// A stated limit: a conflict with a deletion among its sides, merged with a
/// commit that deletes the path, resolves to the deletion under
/// `merge.same-change=accept`.
#[test]
fn md1_then_a_deletion_resolves_to_the_deletion() -> TestResult {
    let merged = case_md1_then_a_deletion()?;
    let value = merged.value()?;
    assert!(
        value.is_absent(),
        "expected f to be deleted, got {}",
        describe(&merged.case, &value)
    );
    Ok(())
}

/// K4's result `[x, VB, y]` differs from V, as does a third merge of A and B
/// that sets `w`, so their merge drops none of the three.
#[test]
fn k4_result_then_a_resolution_of_the_bases() -> TestResult {
    let merged = case_k4_result_then_a_resolution_of_the_bases()?;
    let simplified = conflict_at_f(&merged)?;
    for side in ["x", "y", "w"] {
        let term = merged.case.text(side);
        assert!(
            simplified.adds().any(|add| *add == term),
            "the conflict does not hold {side:?}: {}",
            describe(&merged.case, &simplified)
        );
    }
    Ok(())
}

/// V of Q1 and Q2 is K4's `[x, VB, y]`, which Q1 holds unchanged, so the merge
/// takes Q2's `R`.
#[test]
fn two_level_resolution_of_a_k4_merge() -> TestResult {
    assert_resolved_text(&case_two_level_resolution_of_a_k4_merge()?, "R\n")
}

/// r1 and r2 resolve m's conflict differently: a conflict whose adds are `a`
/// and `b` (the parents' values, beside the one remove `assert_conflict`
/// checks).
#[test]
fn single_conflicted_base_two_resolutions() -> TestResult {
    assert_conflict(
        &case_single_conflicted_base_two_resolutions()?,
        Base::Virtual(&["a", "b"]),
    )
}

#[test]
fn k1_resolved_to_the_bases_first_side() -> TestResult {
    assert_resolved_text(&case_k1_resolved_to_the_bases_first_side()?, "A\n")
}

#[test]
fn merge_of_a_conflicted_commit_and_its_descendant() -> TestResult {
    assert_resolved_text(
        &case_merge_of_a_conflicted_commit_and_its_descendant()?,
        "R\n",
    )
}

/// V at `f` is a file against a directory, which jj reads as one conflicted
/// path. Where X resolves it and Y leaves it, the merge takes X's file; where
/// both leave it, the merge holds V's own conflict.
#[test]
fn k2_file_directory_conflict() -> TestResult {
    let merged = case_k2_file_directory(true)?;
    let x_value = value_at(&merged.parents[0].tree(), "f")?;
    assert!(
        x_value.is_resolved(),
        "the case needs X to resolve the conflict"
    );
    assert_same_value(&merged.case, &merged.value()?, &x_value);
    assert!(value_at(&merged.tree, "f/x")?.is_absent());

    let merged = case_k2_file_directory(false)?;
    let x_value = value_at(&merged.parents[0].tree(), "f")?;
    assert!(
        !x_value.is_resolved(),
        "the case needs X to leave the conflict"
    );
    assert_same_value(&merged.case, &merged.value()?, &x_value);
    Ok(())
}

#[test]
fn criss_cross_bases_inherit_a_conflict() -> TestResult {
    assert_resolved_text(&case_criss_cross_bases_inherit_a_conflict()?, "R\n")
}

/// X resolves `f` to `c` and Y to `b`: a conflict, whichever merge base was
/// made first.
#[test]
fn resolved_differently_in_both_base_orders() -> TestResult {
    for made in [Made::AFirst, Made::BFirst] {
        assert_conflict(
            &case_resolved_differently(made)?,
            Base::Virtual(&["a", "b"]),
        )?;
    }
    Ok(())
}

/// V does not resolve at `d/f`, P1 holds a conflict between a directory and a
/// file at `d`, and P2 leaves `d` as V holds it: the merge holds P1's conflict
/// at `d`.
#[test]
fn file_directory_conflict_above_a_conflicted_base() -> TestResult {
    let (merged, p1_value) = case_file_directory_conflict_in_a_side()?;
    let value = value_at(&merged.tree, "d")?;
    assert!(
        same_terms(&value, &p1_value),
        "expected P1's value at d, {p1_value:?}, got {value:?}"
    );
    Ok(())
}

/// Where both parents changed V's conflict at `f`, each parent's value in the
/// merge's conflict carries that parent's label, as jj's materialization reads
/// the labels: the tree's labels, simplified with the file ids at `f`.
#[test]
fn conflict_labels_name_each_parent() -> TestResult {
    let cases = [
        ("K4", case_k4(Made::AFirst, Order::AsNamed)?),
        (
            "resolved differently, A first",
            case_resolved_differently(Made::AFirst)?,
        ),
        (
            "resolved differently, B first",
            case_resolved_differently(Made::BFirst)?,
        ),
    ];
    for (name, merged) in cases {
        let file_ids = merged
            .value()?
            .to_file_merge()
            .unwrap_or_else(|| panic!("{name}: expected files at f"));
        let (labels, file_ids) = merged.tree.labels().simplify_with(&file_ids);
        for parent in &merged.parents {
            let parent_ids = value_at(&parent.tree(), "f")?
                .to_file_merge()
                .unwrap_or_else(|| panic!("{name}: expected the parent to hold a file at f"));
            let index = file_ids
                .adds()
                .position(|id| id == parent_ids.first())
                .unwrap_or_else(|| panic!("{name}: {file_ids:?} lacks {parent_ids:?}"));
            assert_eq!(
                labels.get_add(index),
                Some(parent.conflict_label().as_str()),
                "{name}: the label of the parent's value in the conflict, with the labels \
                 {labels:?}"
            );
        }
    }
    Ok(())
}

/// `merge_commit_trees_no_resolve` keeps its contract: the rule's value at
/// `f`, unresolved, and at `g`, a conflict that resolves, what
/// `MergedTree::merge_no_resolve` of the parents and V holds there.
#[test]
fn no_resolve_leaves_other_paths_unresolved() -> TestResult {
    let TwoBases { case, a, b, p1, p2 } = two_bases(
        &[("f", line("o")), ("g", lines("1 2 3"))],
        &[("f", line("x"))],
        &[("f", line("y"))],
        Made::AFirst,
        &[("f", line("x")), ("g", lines("1a 2 3"))],
        &[("f", line("y")), ("g", lines("1 2 3a"))],
    )?;
    let tree =
        merge_commit_trees_no_resolve(case.tx.repo(), &[p1.clone(), p2.clone()]).block_on()?;
    let merged = Merged {
        case,
        parents: vec![p1.clone(), p2.clone()],
        tree,
    };
    assert_conflict(&merged, Base::Virtual(&["x", "y"]))?;
    let g_value = value_at(&merged.tree, "g")?;
    assert!(
        !g_value.is_resolved(),
        "expected g unresolved, got {g_value:?}"
    );
    let base = merged.case.merge(&[&a, &b])?;
    let expected = MergedTree::merge_no_resolve(Merge::from_vec(vec![
        (p1.tree(), "p1".to_owned()),
        (base, "base".to_owned()),
        (p2.tree(), "p2".to_owned()),
    ]));
    assert_same_value(&merged.case, &g_value, &value_at(&expected, "g")?);
    Ok(())
}

/// No resolved value of any case holds a marker line of the virtual base, and
/// in each conflict only removes do.
#[test]
fn virtual_base_never_in_a_resolved_tree() -> TestResult {
    type Build = fn() -> TestResult<Merged>;
    let cases: Vec<(&str, Build)> = vec![
        ("K1", || case_k1(Order::AsNamed)),
        ("K1 swapped", || case_k1(Order::Swapped)),
        ("K2", || case_k2(Made::AFirst, Order::AsNamed)),
        ("K2 swapped", || case_k2(Made::AFirst, Order::Swapped)),
        ("K2 B first", || case_k2(Made::BFirst, Order::AsNamed)),
        ("K3", || Ok(case_k3()?.0)),
        ("K4a", || case_k4(Made::AFirst, Order::AsNamed)),
        ("K4b", || case_k4(Made::AFirst, Order::Swapped)),
        ("K4c", || case_k4(Made::BFirst, Order::AsNamed)),
        ("K4d", || case_k4(Made::BFirst, Order::Swapped)),
        ("K5", case_k5),
        ("D1", || case_d1(Order::AsNamed)),
        ("D1 swapped", || case_d1(Order::Swapped)),
        ("T3", case_t3),
        ("T4", || case_t4(Order::AsNamed)),
        ("T4 swapped", || case_t4(Order::Swapped)),
        ("T5 A first", || case_t5(Made::AFirst)),
        ("T5 B first", || case_t5(Made::BFirst)),
        ("T6 A first", || case_t6(Made::AFirst)),
        ("T6 B first", || case_t6(Made::BFirst)),
        ("H1", case_h1),
        ("E1", || case_e1(Order::AsNamed)),
        ("E1 swapped", || case_e1(Order::Swapped)),
        ("E2", || case_e2(Order::AsNamed)),
        ("E2 swapped", || case_e2(Order::Swapped)),
        ("E4", case_e4),
        ("MD1", case_md1),
        ("MD2", case_md2),
        ("BIN1", case_bin1),
        ("X1", || case_x1(Order::AsNamed)),
        ("X1 swapped", || case_x1(Order::Swapped)),
        ("minimal, three bases", case_minimal_three_bases),
        ("same conflict in both parents", || {
            Ok(case_same_conflict_in_both_parents()?.0)
        }),
        ("MD1 then a deletion", case_md1_then_a_deletion),
        (
            "K4 result then a resolution of the bases",
            case_k4_result_then_a_resolution_of_the_bases,
        ),
        (
            "two-level resolution of a K4 merge",
            case_two_level_resolution_of_a_k4_merge,
        ),
        (
            "single conflicted base, two resolutions",
            case_single_conflicted_base_two_resolutions,
        ),
        (
            "K1 resolved to the bases' first side",
            case_k1_resolved_to_the_bases_first_side,
        ),
        (
            "merge of a conflicted commit and its descendant",
            case_merge_of_a_conflicted_commit_and_its_descendant,
        ),
        ("K2 file/directory, X resolves", || {
            case_k2_file_directory(true)
        }),
        ("K2 file/directory, both leave", || {
            case_k2_file_directory(false)
        }),
        (
            "criss-cross bases inherit a conflict",
            case_criss_cross_bases_inherit_a_conflict,
        ),
        ("resolved differently, A first", || {
            case_resolved_differently(Made::AFirst)
        }),
        ("resolved differently, B first", || {
            case_resolved_differently(Made::BFirst)
        }),
        ("file/directory conflict in a side", || {
            Ok(case_file_directory_conflict_in_a_side()?.0)
        }),
        ("VB1a, d rebased onto P1", || case_vb1a(&[("f", line("y"))])),
        ("VB1a, d with another change rebased onto P1", || {
            case_vb1a(&[("f", line("y")), ("h", line("h"))])
        }),
        ("VB1b, P2 abandoned", case_vb1b),
        ("VB1c, d squashed into P1", case_vb1c),
        ("sibling rebase of K4's conflict", case_vb1_sibling),
        ("sibling rebase of D1's conflict", case_vb1_sibling_d1),
        ("VB2a, d rebased onto P1", || rebase_onto_p1(b2_history()?)),
        ("VB2b, P2 abandoned", || abandon_p2(b2_history()?)),
        ("VB2, d squashed into P1", || squash_into_p1(b2_history()?)),
        ("sibling rebase of VB2's conflict", case_vb2_sibling),
        ("VB3, d rebased onto P1", case_vb3),
    ];
    for (name, build) in cases {
        let merged = build()?;
        let store = merged.case.store();
        for (path, value) in merged.tree.entries() {
            let value = value?;
            let holds = |term: &Option<TreeValue>| match term {
                Some(TreeValue::File { id, .. }) => holds_marker(&read_file(store, &path, id)),
                _ => false,
            };
            match value.into_resolved() {
                Ok(term) => assert!(
                    !holds(&term),
                    "{name}: the resolved value at {path:?} holds a virtual base's marker line"
                ),
                Err(terms) => assert!(
                    !terms.simplify().adds().any(holds),
                    "{name}: a side of the conflict at {path:?} holds a virtual base's marker \
                     line: {terms:?}"
                ),
            }
        }
    }
    Ok(())
}

/// V does not depend on the order in which its merge bases are given.
#[test]
fn v_independent_of_base_order() -> TestResult {
    for (name, made) in [("A first", Made::AFirst), ("B first", Made::BFirst)] {
        let (case, bases) = three_bases(made)?;
        let mut values: Vec<(Vec<CommitId>, MergedTreeValue, MergedTreeValue)> = vec![];
        for order in bases.iter().permutations(bases.len()) {
            let base = case.merge(&order)?;
            let ids = order.iter().map(|commit| commit.id().clone()).collect();
            values.push((ids, value_at(&base, "f")?, value_at(&base, "c")?));
        }
        let (_, first_f, first_c) = &values[0];
        for (order, value_f, value_c) in &values {
            assert!(
                same_terms(value_f, first_f) && same_terms(value_c, first_c),
                "{name}: V for the bases in the order {order:?} is {value_f:?} at f and \
                 {value_c:?} at c, not {first_f:?} and {first_c:?}"
            );
        }
    }
    Ok(())
}

/// A merge of three commits is the same tree in whatever order they are named.
#[test]
fn octopus_folds_in_commit_id_order() -> TestResult {
    let (case, [a, b, c]) = three_bases(Made::AFirst)?;
    let trees = [&a, &b, &c]
        .into_iter()
        .permutations(3)
        .map(|parents| case.merge(&parents))
        .collect::<TestResult<Vec<_>>>()?;
    for tree in &trees {
        assert_eq!(tree.tree_ids(), trees[0].tree_ids());
    }
    Ok(())
}

/// A parent whose stored conflict an older jj folded in another order than V
/// counts as changed, and the merge drops none of its change.
#[test]
fn older_fold_counts_as_changed() -> TestResult {
    let (mut case, [a, b, c]) = three_bases(Made::AFirst)?;
    let x = case.commit(&[&a, &b, &c], &[("f", line("R"))])?;
    let base = case.merge(&[&a, &b, &c])?;
    let base_value = value_at(&base, "f")?;
    let [term_x, term_o, term_y] = ["x", "o", "y"].map(|value| case.text(value));
    let older_value = if base_value.simplify().num_sides() == 2 {
        conflict(&[&term_x, &term_o, &term_y, &term_o, &term_x])
    } else {
        conflict(&[&term_x, &term_o, &term_y])
    };
    assert!(!same_terms(&older_value, &base_value));
    let g_value = Merge::resolved(case.file(repo_path("g"), b"1\n", false, CopyId::placeholder()));
    let y = case.commit_with_values(
        &[&a, &b, &c],
        vec![
            (repo_path_buf("g"), g_value),
            (repo_path_buf("f"), older_value),
        ],
    )?;
    let merged = Merged::new(case, &[&x, &y])?;
    let simplified = conflict_at_f(&merged)?;
    let term_r = merged.case.text("R");
    assert!(
        simplified.adds().any(|add| *add == term_r),
        "the conflict does not hold R: {}",
        describe(&merged.case, &simplified)
    );
    Ok(())
}

/// VB1a: d, K4's merge resolved to `y`, rebased onto P1. The rebase cancels
/// every term of `[x, x - VB + y, y]` but VB, a side, which the merge replaces
/// with V's terms: the path is a conflict of P1's `x` and d's `y` against `o`,
/// as jj 0.45.1 gives.
#[test]
fn vb1a_rebase_onto_one_parent() -> TestResult {
    assert_conflict(
        &case_vb1a(&[("f", line("y"))])?,
        Base::Text("o\n".to_owned()),
    )
}

/// VB1a where d also changes another file, so that the rebase cancels the
/// terms of the trees in part and VB stands alone only at `f`.
#[test]
fn vb1a_rebase_onto_one_parent_with_another_change() -> TestResult {
    let merged = case_vb1a(&[("f", line("y")), ("h", line("h"))])?;
    assert_conflict(&merged, Base::Text("o\n".to_owned()))?;
    let h_value = value_at(&merged.tree, "h")?;
    let Ok(Some(TreeValue::File { id, .. })) = h_value.clone().into_resolved() else {
        panic!("expected d's file h, resolved, got {h_value:?}");
    };
    assert_eq!(read_file(merged.case.store(), repo_path("h"), &id), b"h\n");
    Ok(())
}

/// VB1b: d, K4's merge resolved to `y`, rebased onto P2's parents once P2 is
/// abandoned.
#[test]
fn vb1b_abandon_a_parent() -> TestResult {
    assert_conflict(&case_vb1b()?, Base::Text("o\n".to_owned()))
}

/// VB1c: d, K4's merge resolved to `y`, squashed into P1.
#[test]
fn vb1c_squash_into_a_parent() -> TestResult {
    assert_conflict(&case_vb1c()?, Base::Text("o\n".to_owned()))
}

/// c and q, children of a commit that holds K4's conflict, resolve it to `x`
/// and to `y`; c rebased onto q holds a conflict of the two against `o`.
#[test]
fn vb1_sibling_rebase() -> TestResult {
    assert_conflict(&case_vb1_sibling()?, Base::Text("o\n".to_owned()))
}

/// The sibling rebase of D1's conflict, whose base is M: a conflict of `y`
/// and the deletion against `o`.
#[test]
fn vb1_sibling_rebase_d1() -> TestResult {
    assert_conflict(&case_vb1_sibling_d1()?, Base::Text("o\n".to_owned()))
}

/// VB2a: d, a merge of P1 and P2 that edits a line inside the hunk where A and
/// B disagree, rebased onto P1: a conflict of d's and A's changes from o, as
/// jj 0.45.1 gives.
#[test]
fn vb2a_rebase_with_an_edit_in_the_disputed_hunk() -> TestResult {
    assert_conflict(
        &rebase_onto_p1(b2_history()?)?,
        Base::Text(SIX_LINES.to_owned()),
    )
}

/// VB2b: VB2's d once P2 is abandoned.
#[test]
fn vb2b_abandon_with_an_edit_in_the_disputed_hunk() -> TestResult {
    assert_conflict(
        &abandon_p2(b2_history()?)?,
        Base::Text(SIX_LINES.to_owned()),
    )
}

/// VB2's d squashed into P1.
#[test]
fn vb2_squash_with_an_edit_in_the_disputed_hunk() -> TestResult {
    assert_conflict(
        &squash_into_p1(b2_history()?)?,
        Base::Text(SIX_LINES.to_owned()),
    )
}

/// The sibling rebase of VB2's conflict.
#[test]
fn vb2_sibling_rebase_with_an_edit_in_the_disputed_hunk() -> TestResult {
    assert_conflict(&case_vb2_sibling()?, Base::Text(SIX_LINES.to_owned()))
}

/// VB3: d resolves K4's hunk to `y` and changes line 5, outside that hunk.
/// Rebased onto P1, d holds a conflict of its own and A's changes from o, whose
/// terms do not merge, so that jj shows it with conflict markers.
#[test]
fn vb3_rebase_with_an_edit_outside_the_disputed_hunk() -> TestResult {
    let merged = case_vb3()?;
    assert_conflict(&merged, Base::Text(SIX_LINES.to_owned()))?;
    let contents = merged
        .value()?
        .simplify()
        .map(|term| merged.case.read(term.as_ref()).unwrap_or_default());
    assert_eq!(
        files::try_merge(&contents, merged.case.store().merge_options()),
        None
    );
    Ok(())
}

/// VB4a and VB4b: K4's d, resolved to `y`, once P1 takes d's `f`, as `jj
/// restore --from d --into P1 f` gives it. d's rebase merges `[y, -x, VB]`,
/// which expands to `[y, -x, x, -o, y]`: d holds `y`, resolved, as jj 0.45.1
/// gives, so that `jj resolve` with either side's tool has nothing to resolve.
#[test]
fn vb4a_restore_into_one_parent() -> TestResult {
    let (TwoBases { mut case, p1, .. }, d) = resolved_k4_merge(&[("f", line("y"))])?;
    let mut builder = MergedTreeBuilder::new(p1.tree());
    builder.set_or_remove(repo_path_buf("f"), value_at(&d.tree(), "f")?);
    let p1_tree = builder.write_tree().block_on()?;
    case.tx
        .repo_mut()
        .rewrite_commit(&p1)
        .set_tree(p1_tree)
        .write_unwrap();
    let rebased =
        rebase_descendants_with_options_return_map(case.tx.repo_mut(), &RebaseOptions::default());
    let tree = case.store().get_commit(&rebased[d.id()])?.tree();
    let merged = Merged {
        case,
        parents: vec![],
        tree,
    };
    assert_resolved_text(&merged, "y\n")
}

/// VB4c and VB4d: K4's d, resolved to `y`, once P2 is rebased onto A, as `jj
/// rebase -s P2 -d A` does. P2 then holds `o`, and d's rebase merges `[o, -x,
/// VB]`, which expands to `[o, -x, x, -o, y]`: d holds `y`, resolved, as jj
/// 0.45.1 gives.
#[test]
fn vb4c_rebase_one_parent_onto_a_merge_base() -> TestResult {
    let (
        TwoBases {
            mut case, a, p2, ..
        },
        d,
    ) = resolved_k4_merge(&[("f", line("y"))])?;
    let new_p2 = rebase_commit(case.tx.repo_mut(), p2, vec![a.id().clone()]).block_on()?;
    let p2_value = value_at(&new_p2.tree(), "f")?;
    assert_eq!(
        p2_value
            .as_resolved()
            .and_then(|term| case.read(term.as_ref())),
        Some(b"o\n".to_vec()),
        "expected P2 to hold o, got {}",
        describe(&case, &p2_value)
    );
    let rebased =
        rebase_descendants_with_options_return_map(case.tx.repo_mut(), &RebaseOptions::default());
    let tree = case.store().get_commit(&rebased[d.id()])?.tree();
    let merged = Merged {
        case,
        parents: vec![],
        tree,
    };
    assert_resolved_text(&merged, "y\n")
}

/// VB5: w, K4's merge with six lines at `f`, left conflicted, with line 5,
/// outside the conflict, changed to `FIVE` in the working copy. The snapshot
/// applies the change to each term of w's conflict, its virtual base too, so
/// a merge that later replaces that base with the terms of V must find the
/// change in them. s and u, children of w, resolve `f` to `x` and to `y`, each
/// with `FIVE`: u rebased onto s is a conflict whose every side holds `FIVE`,
/// `xF + yF - oF`, as jj 0.45.1 gives.
#[test]
fn vb5_an_edit_outside_the_conflict_survives_its_expansion() -> TestResult {
    assert_edits_survive_expansion(&[("5", "FIVE")])
}

/// VB5 with a second edit, line 6 changed to `SIX`, to the same working copy
/// after the first was snapshotted: the working copy still holds the marker
/// lines of the virtual base that the first snapshot replaced, and the second
/// snapshot applies the change to each term of V too.
#[test]
fn vb5_a_second_edit_of_the_same_working_copy_survives_expansion() -> TestResult {
    assert_edits_survive_expansion(&[("5", "FIVE"), ("6", "SIX")])
}

/// Makes VB5's history, with each of `edits`, a line of `f` outside the
/// conflict and its replacement, made to the working copy of w and
/// snapshotted in turn, and asserts that u rebased onto s, whose texts make
/// the same edits, is a conflict whose every side holds them.
fn assert_edits_survive_expansion(edits: &[(&str, &str)]) -> TestResult {
    let TwoBases {
        mut case, p1, p2, ..
    } = six_line_history("x", "y 2 3")?;
    let w = case.commit(&[&p1, &p2], &[])?;
    let new_value = snapshot_edits(&case, &w, edits)?;
    let words = edits
        .iter()
        .fold("2 3 4 5 6".to_owned(), |words, (line, replacement)| {
            words.replacen(line, replacement, 1)
        });
    let w = case.commit_with_values(&[&p1, &p2], vec![(repo_path_buf("f"), new_value)])?;
    let s = case.commit(&[&w], &[("f", lines(&format!("x {words}")))])?;
    let u = case.commit(&[&w], &[("f", lines(&format!("y {words}")))])?;
    let rebased = rebase_commit(case.tx.repo_mut(), u, vec![s.id().clone()]).block_on()?;
    let value = value_at(&rebased.tree(), "f")?;
    let [x_edited, o_edited, y_edited] = ["x", "1", "y"].map(|first| {
        let content = lines(&format!("{first} {words}")).unwrap();
        case.file(f(), &content, false, CopyId::placeholder())
    });
    assert_same_value(&case, &value, &conflict(&[&x_edited, &o_edited, &y_edited]));
    Ok(())
}

/// The value at `f` of `w`, a conflict, once each of `edits`, a line of `f`
/// and its replacement, or an empty line and a line to append, is made to the
/// file that jj materializes for the conflict and snapshotted in turn, as jj
/// snapshots a working copy.
fn snapshot_edits(case: &Case, w: &Commit, edits: &[(&str, &str)]) -> TestResult<MergedTreeValue> {
    snapshot_edits_materialized(case, w, case.store().merge_options().clone(), edits)
}

/// The value at `f` of `w` once `edits` are made to the file that jj
/// materializes for its conflict with `materialize`, the merge options of the
/// checkout that wrote the file, which can be other than the store's now (see
/// [`snapshot_edits`]).
fn snapshot_edits_materialized(
    case: &Case,
    w: &Commit,
    materialize: MergeOptions,
    edits: &[(&str, &str)],
) -> TestResult<MergedTreeValue> {
    let mut edits = edits.iter();
    snapshot_in_turn(case, w, materialize, |working_copy, _| {
        let (line, replacement) = edits.next()?;
        Some(if line.is_empty() {
            format!("{working_copy}{replacement}\n")
        } else {
            let (from, to) = (format!("\n{line}\n"), format!("\n{replacement}\n"));
            working_copy.replacen(&from, &to, 1)
        })
    })
}

/// The value at `f` of `w`, a conflict, once `edit` has rewritten the file
/// that jj materializes for the conflict, given with its conflict markers'
/// length, and it is snapshotted, as jj snapshots a working copy.
fn snapshot_text(
    case: &Case,
    w: &Commit,
    edit: impl FnOnce(&str, usize) -> String,
) -> TestResult<MergedTreeValue> {
    let mut edit = Some(edit);
    let materialize = case.store().merge_options().clone();
    snapshot_in_turn(case, w, materialize, |working_copy, marker_len| {
        edit.take().map(|edit| edit(working_copy, marker_len))
    })
}

/// The value at `f` of `w`, a conflict, once the file that jj materializes
/// for the conflict with `materialize` is rewritten by `next`, given the file
/// and its conflict markers' length, and snapshotted, in turn until `next`
/// gives `None`.
fn snapshot_in_turn(
    case: &Case,
    w: &Commit,
    materialize: MergeOptions,
    mut next: impl FnMut(&str, usize) -> Option<String>,
) -> TestResult<MergedTreeValue> {
    let store = case.store().clone();
    let mut file_ids = value_at(&w.tree(), "f")?
        .to_file_merge()
        .expect("files at f");
    let contents = extract_as_single_hunk(&file_ids, &store, f()).block_on()?;
    let options = ConflictMaterializeOptions {
        marker_style: ConflictMarkerStyle::Diff,
        marker_len: None,
        merge: materialize,
    };
    let mut working_copy = String::from_utf8(
        materialize_merge_result_to_bytes(&contents, &ConflictLabels::unlabeled(), &options).into(),
    )?;
    let marker_len = choose_materialized_conflict_marker_len(&contents);
    while let Some(edited) = next(&working_copy, marker_len) {
        assert_ne!(edited, working_copy);
        working_copy = edited;
        let content = working_copy.as_bytes();
        file_ids = update_from_content(&file_ids, &store, f(), content, marker_len).block_on()?;
    }
    Ok(file_ids.map(|id| {
        id.clone().map(|id| TreeValue::File {
            id,
            executable: false,
            copy_id: CopyId::placeholder(),
        })
    }))
}

/// Whether `content` holds a marker line of a virtual base, whole or cut: an
/// opening line, or a separator or closing line with a hex.
fn holds_any_marker(content: &[u8]) -> bool {
    String::from_utf8_lossy(content).lines().any(|line| {
        line.starts_with(OPEN.trim_end())
            || marker_hex(line, SEPARATOR).is_some()
            || marker_hex(line, CLOSE).is_some()
    })
}

/// VB5b: w, the merge of P1 and P2, left conflicted, where the merge bases
/// A and B merge at line 9 (A's `9a` against o's and B's `9`), and line 9,
/// `9a` in VB and in the working copy, changed to `9b` there. The snapshot
/// writes `9b` into each term of V at that line, so that u rebased onto s,
/// which both hold `9b`, is `xF + yF - oF`, as jj 0.45.1 gives, whose every
/// side holds `9b`.
#[test]
fn vb5b_an_edit_where_the_merge_bases_merge_survives_expansion() -> TestResult {
    let text = |first: &str, nine: &str| lines(&format!("{first} 2 3 4 5 6 7 8 {nine} 10"));
    let TwoBases {
        mut case, p1, p2, ..
    } = two_bases(
        &[("f", text("o", "9"))],
        &[("f", text("x", "9a"))],
        &[("f", text("y", "9"))],
        Made::AFirst,
        &[("f", text("x", "9a"))],
        &[("f", text("y", "9a"))],
    )?;
    let w = case.commit(&[&p1, &p2], &[])?;
    let new_value = snapshot_edits(&case, &w, &[("9a", "9b")])?;
    let w = case.commit_with_values(&[&p1, &p2], vec![(repo_path_buf("f"), new_value)])?;
    let s = case.commit(&[&w], &[("f", text("x", "9b"))])?;
    let u = case.commit(&[&w], &[("f", text("y", "9b"))])?;
    let rebased = rebase_commit(case.tx.repo_mut(), u, vec![s.id().clone()]).block_on()?;
    let value = value_at(&rebased.tree(), "f")?;
    let [x_edited, o_edited, y_edited] = ["x", "o", "y"].map(|first| {
        let content = text(first, "9b").unwrap();
        case.file(f(), &content, false, CopyId::placeholder())
    });
    assert_same_value(&case, &value, &conflict(&[&x_edited, &o_edited, &y_edited]));
    Ok(())
}

/// VB6: P2 adds `Q` and `W` inside the lines of A's side of the block where
/// the merge bases disagree, so w, the merge of P1 and P2, left conflicted,
/// conflicts only at the block's tail, and VB's opening line is in a hunk
/// that resolves. Line 5 changed to `FIVE` in the working copy, then s, a
/// child of w that resolves `f` to P2's text with `FIVE`, rebased onto P1: no
/// term of the result holds a marker line of a virtual base, whole or cut.
#[test]
fn vb6_an_edit_where_the_opening_line_resolves_keeps_every_marker_line() -> TestResult {
    let TwoBases {
        mut case, p1, p2, ..
    } = vb6_history()?;
    let w = case.commit(&[&p1, &p2], &[])?;
    let new_value = snapshot_edits(&case, &w, &[("5", "FIVE")])?;
    let w_value = new_value.clone();
    let w = case.commit_with_values(&[&p1, &p2], vec![(repo_path_buf("f"), new_value)])?;
    let s = case.commit(&[&w], &[("f", lines("x1 Q x2 W 2 3 4 FIVE 6 7 8 9 10"))])?;
    let rebased = rebase_commit(case.tx.repo_mut(), s, vec![p1.id().clone()]).block_on()?;
    for (name, value) in [("w", w_value), ("s", value_at(&rebased.tree(), "f")?)] {
        let simplified = value.simplify();
        let marked = |term: &Option<TreeValue>| {
            case.read(term.as_ref())
                .is_some_and(|content| holds_any_marker(&content))
        };
        assert!(
            !simplified.adds().any(&marked),
            "{name}: a side holds a marker line of a virtual base: {}",
            describe(&case, &simplified)
        );
        // w's base is the virtual base of the edited terms: its every marker
        // line is whole, so that it names them.
        for remove in simplified.removes().filter(|&term| marked(term)) {
            let content = case.read(remove.as_ref()).unwrap();
            assert!(
                virtual_base_terms(&content).is_some(),
                "{name}: a base holds cut marker lines: {}",
                describe(&case, &simplified)
            );
        }
    }
    Ok(())
}

/// VB6: the merge of P1 and P2 resolves two hunks of VB's block to its
/// sides' text (both drop the opening line, and P2 puts `Q` inside the first
/// side), so the file that jj materializes for w shows the base without the
/// one and with the other. Neither is a change of the person's, but jj's
/// snapshot writes `Q` into side 1, P1's text, which lacked it: a snapshot
/// with line 5 changed to `FIVE` makes that change to V's terms, and makes
/// side 1's change, `Q` between `x1` and `x2`, inside A's side of the block,
/// to A's term. The opening line and B's side stay. s, which resolves `f` to
/// P2's text, rebased onto P1, is a conflict.
#[test]
fn vb6_a_snapshot_changes_what_the_person_and_the_cut_block_changed() -> TestResult {
    let TwoBases {
        mut case, p1, p2, ..
    } = vb6_history()?;
    let w = case.commit(&[&p1, &p2], &[])?;
    let new_value = snapshot_edits(&case, &w, &[("5", "FIVE")])?;
    assert_eq!(
        named_term_texts(&case, &new_value.simplify())
            .into_iter()
            .sorted()
            .collect_vec(),
        [
            "1\n2\n3\n4\nFIVE\n6\n7\n8\n9\n10\n",
            "x1\nQ\nx2\n2\n3\n4\nFIVE\n6\n7\n8\n9\n10\n",
            "y1\n2\n3\n4\nFIVE\n6\n7\n8\n9\n10\n",
        ]
    );
    let w = case.commit_with_values(&[&p1, &p2], vec![(repo_path_buf("f"), new_value)])?;
    let s = case.commit(&[&w], &[("f", lines("x1 Q x2 W 2 3 4 FIVE 6 7 8 9 10"))])?;
    let rebased = rebase_commit(case.tx.repo_mut(), s, vec![p1.id().clone()]).block_on()?;
    let value = value_at(&rebased.tree(), "f")?.simplify();
    assert!(!value.is_resolved(), "{}", describe(&case, &value));
    Ok(())
}

/// VB6 with a line appended in the working copy: the line goes into every
/// term of V, and V's terms hold no other change but side 1's `Q` in A's
/// term.
#[test]
fn vb6_a_line_appended_goes_into_every_term() -> TestResult {
    let TwoBases {
        mut case, p1, p2, ..
    } = vb6_history()?;
    let w = case.commit(&[&p1, &p2], &[])?;
    let new_value = snapshot_edits(&case, &w, &[("", "END")])?;
    assert_eq!(
        named_term_texts(&case, &new_value.simplify())
            .into_iter()
            .sorted()
            .collect_vec(),
        [
            "1\n2\n3\n4\n5\n6\n7\n8\n9\n10\nEND\n",
            "x1\nQ\nx2\n2\n3\n4\n5\n6\n7\n8\n9\n10\nEND\n",
            "y1\n2\n3\n4\n5\n6\n7\n8\n9\n10\nEND\n",
        ]
    );
    Ok(())
}

/// A parent's own change: A sets line 1 of `1 2 3 4 5 6` to `x` and B to
/// `y`; P1, a merge of A and B, holds `x` and its own change `Z` at line 6,
/// where every term of V holds `6`, and P2 holds `y`. w, the merge of P1 and
/// P2, left conflicted: line 1 conflicts with VB as the base, and w's own
/// merge resolves line 6, outside VB's block, to P1's `Z`. Line 3 is changed
/// to `EDIT` in w's working copy and snapshotted, and s and u, children of w,
/// hold w's file with `x` and with `y` at line 1.
struct OwnChange {
    case: Case,
    p1: Commit,
    w: Commit,
    s: Commit,
    u: Commit,
}

fn own_change_history() -> TestResult<OwnChange> {
    let TwoBases {
        mut case, p1, p2, ..
    } = two_bases(
        &[("f", lines("1 2 3 4 5 6"))],
        &[("f", lines("x 2 3 4 5 6"))],
        &[("f", lines("y 2 3 4 5 6"))],
        Made::AFirst,
        &[("f", lines("x 2 3 4 5 Z"))],
        &[("f", lines("y 2 3 4 5 6"))],
    )?;
    let w = case.commit(&[&p1, &p2], &[])?;
    let new_value = snapshot_edits(&case, &w, &[("3", "EDIT")])?;
    let w = case.commit_with_values(&[&p1, &p2], vec![(repo_path_buf("f"), new_value)])?;
    let s = case.commit(&[&w], &[("f", lines("x 2 EDIT 4 5 Z"))])?;
    let u = case.commit(&[&w], &[("f", lines("y 2 EDIT 4 5 Z"))])?;
    Ok(OwnChange { case, p1, w, s, u })
}

/// A parent's own change: the snapshot writes the text that w's own merge
/// resolved outside VB's block, P1's `Z`, into every term of V, as jj's
/// snapshot writes it into a plain base, so u rebased onto s, as `jj rebase
/// -r u -o s` does, is a conflict at line 1 whose every term holds `Z` and
/// the edit, as jj 0.45.1 gives.
#[test]
fn a_parents_own_change_survives_a_sibling_rebase_after_a_snapshot() -> TestResult {
    let OwnChange { mut case, s, u, .. } = own_change_history()?;
    let rebased = rebase_commit(case.tx.repo_mut(), u, vec![s.id().clone()]).block_on()?;
    let value = value_at(&rebased.tree(), "f")?;
    let [x, one, y] = ["x", "1", "y"].map(|first| {
        let content = lines(&format!("{first} 2 EDIT 4 5 Z")).unwrap();
        case.file(f(), &content, false, CopyId::placeholder())
    });
    assert_same_value(&case, &value, &conflict(&[&x, &one, &y]));
    Ok(())
}

/// A parent's own change: s rebased onto P1, as `jj rebase -r s -o P1` does,
/// holds P1's own text, `Z` with it, as jj 0.45.1 gives, not the merge bases'
/// `6`.
#[test]
fn a_parents_own_change_survives_a_rebase_onto_it_after_a_snapshot() -> TestResult {
    let OwnChange {
        mut case, p1, s, ..
    } = own_change_history()?;
    let rebased = rebase_commit(case.tx.repo_mut(), s.clone(), vec![p1.id().clone()]).block_on()?;
    let merged = Merged {
        case,
        parents: vec![p1, s],
        tree: rebased.tree(),
    };
    assert_resolved_text(&merged, "x\n2\n3\n4\n5\nZ\n")
}

/// A parent's own change: u once w is abandoned, as `jj abandon w` rebases it
/// onto P1 and P2, holds `y 2 3 4 5 Z`, as jj 0.45.1 gives, not the merge
/// bases' `6`.
#[test]
fn a_parents_own_change_survives_an_abandon_of_the_snapshotted_merge() -> TestResult {
    let OwnChange { mut case, w, u, .. } = own_change_history()?;
    case.tx.repo_mut().record_abandoned_commit(&w);
    let merged = rebased_descendant(case, &u)?;
    assert_resolved_text(&merged, "y\n2\n3\n4\n5\nZ\n")
}

/// A third parent's own change: w merges P1 and P2, which hold `x` and `y`
/// over the merge bases A and B, and Q, a child of o that holds its own
/// change `Q6` at line 6. Line 3 is changed to `EDIT` in w's working copy and
/// snapshotted; s and u, children of w, hold `x` and `y` with the edit and
/// `Q6`. u rebased onto s is a conflict whose every term holds `Q6` and the
/// edit.
#[test]
fn a_third_parents_own_change_survives_a_sibling_rebase_after_a_snapshot() -> TestResult {
    let mut case = Case::new();
    let root = case.root();
    let o = case.commit(&[&root], &[("f", lines("1 2 3 4 5 6"))])?;
    let a = case.commit(&[&o], &[("f", lines("x 2 3 4 5 6"))])?;
    let b = case.commit(&[&o], &[("f", lines("y 2 3 4 5 6"))])?;
    let p1 = case.commit(&[&a, &b], &[("f", lines("x 2 3 4 5 6"))])?;
    let p2 = case.commit(&[&a, &b], &[("f", lines("y 2 3 4 5 6"))])?;
    let q = case.commit(&[&o], &[("f", lines("1 2 3 4 5 Q6"))])?;
    let w = case.commit(&[&p1, &p2, &q], &[])?;
    let new_value = snapshot_edits(&case, &w, &[("3", "EDIT")])?;
    let w = case.commit_with_values(&[&p1, &p2, &q], vec![(repo_path_buf("f"), new_value)])?;
    let s = case.commit(&[&w], &[("f", lines("x 2 EDIT 4 5 Q6"))])?;
    let u = case.commit(&[&w], &[("f", lines("y 2 EDIT 4 5 Q6"))])?;
    let rebased = rebase_commit(case.tx.repo_mut(), u, vec![s.id().clone()]).block_on()?;
    let value = value_at(&rebased.tree(), "f")?.simplify();
    assert!(!value.is_resolved(), "{}", describe(&case, &value));
    for line in ["EDIT", "Q6"] {
        assert_every_term_holds(&case, &value, line);
    }
    Ok(())
}

/// K4s: P1 and P2, merges of A and B, which set line 1 of `o 2 3 4 5 6 7 8`
/// to `x` and `y`, both set line 6 to `SAME`, so w's own merge resolves line
/// 6, outside VB's block, to `SAME`, where no side changes. jj's snapshot
/// writes the text of a hunk that resolves into every term of a conflict, a
/// plain base among them, and the snapshot of an edit of line 3 writes
/// `SAME` into V's terms too: u rebased onto s, children of w that hold its
/// file with `x` and with `y`, holds `SAME` in every term, and s rebased onto
/// P1 is P1's text, clean.
#[test]
fn k4s_a_line_that_both_parents_add_survives_rewrites_after_a_snapshot() -> TestResult {
    let text =
        |first: &str, three: &str, six: &str| lines(&format!("{first} 2 {three} 4 5 {six} 7 8"));
    let TwoBases {
        mut case, p1, p2, ..
    } = two_bases(
        &[("f", text("o", "3", "6"))],
        &[("f", text("x", "3", "6"))],
        &[("f", text("y", "3", "6"))],
        Made::AFirst,
        &[("f", text("x", "3", "SAME"))],
        &[("f", text("y", "3", "SAME"))],
    )?;
    let w = case.commit(&[&p1, &p2], &[])?;
    let new_value = snapshot_edits(&case, &w, &[("3", "EDIT")])?;
    let w = case.commit_with_values(&[&p1, &p2], vec![(repo_path_buf("f"), new_value)])?;
    let s = case.commit(&[&w], &[("f", text("x", "EDIT", "SAME"))])?;
    let u = case.commit(&[&w], &[("f", text("y", "EDIT", "SAME"))])?;
    let rebased = rebase_commit(case.tx.repo_mut(), u, vec![s.id().clone()]).block_on()?;
    let value = value_at(&rebased.tree(), "f")?.simplify();
    assert!(!value.is_resolved(), "{}", describe(&case, &value));
    assert_every_term_holds(&case, &value, "SAME");
    let rebased = rebase_commit(case.tx.repo_mut(), s.clone(), vec![p1.id().clone()]).block_on()?;
    let merged = Merged {
        case,
        parents: vec![p1, s],
        tree: rebased.tree(),
    };
    assert_resolved_text(&merged, "x\n2\n3\n4\n5\nSAME\n7\n8\n")
}

/// A history whose merge cuts a block of VB: o holds `1 2 3 4 5 6 7 8`; A
/// and B, its children, and P1 and P2, merges of A and B, hold the texts of
/// a [`CutTexts`], where a parent's own line sits inside one side of VB's
/// block at line 1. w, the merge of P1 and P2, is left conflicted: its own
/// merge resolves that line's hunk, which cuts the block, and conflicts at
/// the last line. With `edit`, a line of `f` and its replacement, that line
/// is changed in w's working copy and snapshotted. s and u, children of w,
/// hold the file that jj materializes for w with side 1 and with side 2 of
/// its conflict.
struct CutHistory {
    case: Case,
    a: Commit,
    p1: Commit,
    p2: Commit,
    w: Commit,
    s: Commit,
    u: Commit,
}

/// The texts of a [`CutHistory`], each word a line.
struct CutTexts {
    a: &'static str,
    b: &'static str,
    p1: &'static str,
    p2: &'static str,
    /// What the file that jj materializes for w shows before its conflict
    /// at the last line, between A's `xA` and B's `yB`.
    shown: &'static str,
}

/// cutq: A sets line 1 to `x1 x2` and line 8 to `xA`, and B to `y1` and
/// `yB`; P1 holds A's text, and P2 holds A's lines with its own line `Q`
/// between `x1` and `x2`, inside A's side of the block, and B's `yB`.
const CUTQ: CutTexts = CutTexts {
    a: "x1 x2 2 3 4 5 6 7 xA",
    b: "y1 2 3 4 5 6 7 yB",
    p1: "x1 x2 2 3 4 5 6 7 xA",
    p2: "x1 Q x2 2 3 4 5 6 7 yB",
    shown: "x1 Q x2 2 3 4 5 6 7",
};

/// cutqz: cutq where P1 holds its own `Z` at line 6 too, outside every block.
const CUTQZ: CutTexts = CutTexts {
    p1: "x1 x2 2 3 4 Z 6 7 xA",
    shown: "x1 Q x2 2 3 4 Z 6 7",
    ..CUTQ
};

/// cqp, cutq's mirror: A sets line 1 to `x1` and line 8 to `xA`, and B to
/// `y1 y2` and `yB`; P1 holds B's lines with its own line `Q` between `y1`
/// and `y2`, inside B's side of the block, and A's `xA`, and P2 holds B's
/// text.
const CQP: CutTexts = CutTexts {
    a: "x1 2 3 4 5 6 7 xA",
    b: "y1 y2 2 3 4 5 6 7 yB",
    p1: "y1 Q y2 2 3 4 5 6 7 xA",
    p2: "y1 y2 2 3 4 5 6 7 yB",
    shown: "y1 Q y2 2 3 4 5 6 7",
};

fn cut_history(texts: &CutTexts, edit: Option<(&str, &str)>) -> TestResult<CutHistory> {
    let TwoBases {
        mut case,
        a,
        p1,
        p2,
        ..
    } = two_bases(
        &[("f", lines("1 2 3 4 5 6 7 8"))],
        &[("f", lines(texts.a))],
        &[("f", lines(texts.b))],
        Made::AFirst,
        &[("f", lines(texts.p1))],
        &[("f", lines(texts.p2))],
    )?;
    let mut w = case.commit(&[&p1, &p2], &[])?;
    if let Some(edit) = edit {
        let new_value = snapshot_edits(&case, &w, &[edit])?;
        w = case.commit_with_values(&[&p1, &p2], vec![(repo_path_buf("f"), new_value)])?;
    }
    let side = |last: &str| {
        let words = format!("{} {last}", texts.shown);
        lines(&match edit {
            Some((line, replacement)) => words
                .split(' ')
                .map(|word| if word == line { replacement } else { word })
                .join(" "),
            None => words,
        })
    };
    let s = case.commit(&[&w], &[("f", side("xA"))])?;
    let u = case.commit(&[&w], &[("f", side("yB"))])?;
    Ok(CutHistory {
        case,
        a,
        p1,
        p2,
        w,
        s,
        u,
    })
}

/// The values at `f` that a rewrite reads, each with the commit's name.
type Reads = Vec<(&'static str, MergedTreeValue)>;

/// A value at `f` hunk by hunk: for each hunk, its adds and its removes,
/// each sorted; a hunk that resolves is its text alone.
type Hunks = Vec<(Vec<String>, Vec<String>)>;

/// `value` hunk by hunk, as jj shows it: the hunks of the merge of the
/// contents of its terms after `simplify()`, each simplified.
fn hunks(case: &Case, value: &MergedTreeValue) -> Hunks {
    let contents = value
        .simplify()
        .map(|term| case.read(term.as_ref()).unwrap_or_default());
    let hunks = match files::merge_hunks(&contents, case.store().merge_options()) {
        files::MergeResult::Resolved(content) => vec![Merge::resolved(content)],
        files::MergeResult::Conflict(hunks) => hunks,
    };
    let texts = |terms: Vec<String>| terms.into_iter().sorted().collect_vec();
    hunks
        .iter()
        .map(|hunk| {
            let hunk = hunk
                .simplify()
                .map(|text| String::from_utf8_lossy(text).into_owned());
            let (removes, adds) = hunk.into_removes_adds();
            (texts(adds.collect()), texts(removes.collect()))
        })
        .collect()
}

/// A hunk of [`Hunks`] that resolves to `words`, one a line.
fn resolved_hunk(words: &str) -> (Vec<String>, Vec<String>) {
    conflict_hunk(&[words], &[])
}

/// A hunk of [`Hunks`] with the sides `adds` and the bases `removes`, each
/// words, one a line.
fn conflict_hunk(adds: &[&str], removes: &[&str]) -> (Vec<String>, Vec<String>) {
    let texts = |terms: &[&str]| {
        terms
            .iter()
            .map(|words| format!("{}\n", words.replace(' ', "\n")))
            .sorted()
            .collect_vec()
    };
    (texts(adds), texts(removes))
}

/// Asserts that `rewrite`, run on the [`CutHistory`] of `texts`, gives, at
/// each commit that it reads, the hunks that `expected` gives for the text
/// of the line that `edit` changes, a line of `f` and its replacement: the
/// line with no edit, and its replacement after the edit of w's working copy.
/// Where the rewrite takes w's own change away, both are the same.
fn assert_cut_rewrite(
    texts: &CutTexts,
    edit: (&str, &str),
    rewrite: impl Fn(CutHistory) -> TestResult<(Case, Reads)>,
    expected: impl Fn(&str) -> Vec<(&'static str, Hunks)>,
) -> TestResult {
    let (line, replacement) = edit;
    let (mut actual, mut wanted) = (vec![], vec![]);
    for (half, edit, text) in [("no edit", None, line), ("edit", Some(edit), replacement)] {
        let (case, reads) = rewrite(cut_history(texts, edit)?)?;
        let reads = reads
            .iter()
            .map(|(name, value)| (*name, hunks(&case, value)))
            .collect_vec();
        actual.push((half, reads));
        wanted.push((half, expected(text)));
    }
    assert_eq!(actual, wanted);
    Ok(())
}

/// The value at `f` of each of `commits`, with its name, once the
/// descendants of the rewritten commits are rebased.
fn rebased_values(case: &mut Case, commits: &[(&'static str, &Commit)]) -> TestResult<Reads> {
    let rebased =
        rebase_descendants_with_options_return_map(case.tx.repo_mut(), &RebaseOptions::default());
    commits
        .iter()
        .map(|&(name, commit)| {
            let commit = case.store().get_commit(&rebased[commit.id()])?;
            Ok((name, value_at(&commit.tree(), "f")?))
        })
        .collect()
}

/// u rebased onto s, as `jj rebase -r u -o s` does.
fn sibling_rebase(history: CutHistory) -> TestResult<(Case, Reads)> {
    let CutHistory { mut case, s, u, .. } = history;
    let rebased = rebase_commit(case.tx.repo_mut(), u, vec![s.id().clone()]).block_on()?;
    let value = value_at(&rebased.tree(), "f")?;
    Ok((case, vec![("u", value)]))
}

/// u once w is abandoned, as `jj abandon w` rebases it onto P1 and P2.
fn abandon_w(history: CutHistory) -> TestResult<(Case, Reads)> {
    let CutHistory { mut case, w, u, .. } = history;
    case.tx.repo_mut().record_abandoned_commit(&w);
    let reads = rebased_values(&mut case, &[("u", &u)])?;
    Ok((case, reads))
}

/// w once P2 is abandoned, as `jj abandon P2` rebases it onto P1 and P2's
/// parents.
fn abandon_p2_under_w(history: CutHistory) -> TestResult<(Case, Reads)> {
    let CutHistory {
        mut case, p2, w, ..
    } = history;
    case.tx.repo_mut().record_abandoned_commit(&p2);
    let reads = rebased_values(&mut case, &[("w", &w)])?;
    Ok((case, reads))
}

/// The duplicate of w onto P1, as `jj duplicate -r w -o P1` makes it.
fn duplicate_w_onto_p1(history: CutHistory) -> TestResult<(Case, Reads)> {
    let CutHistory {
        mut case, p1, w, ..
    } = history;
    let stats = duplicate_commits(
        case.tx.repo_mut(),
        &[w.id().clone()],
        &HashMap::new(),
        &[p1.id().clone()],
        &[],
    )
    .block_on()?;
    let duplicate = &stats.duplicated_commits[w.id()];
    Ok((case, vec![("dup", value_at(&duplicate.tree(), "f")?)]))
}

/// cutq through `jj abandon w`: u holds P2's `Q` and B's `yB`, clean, with no
/// edit and after line 4 of w's working copy is changed to `EDIT` alike, as
/// jj 0.45.1 gives. jj's snapshot writes w's merge of Q's hunk, which cuts
/// VB's block, into each side, so side 1, P1's text, gains `Q` between `x1`
/// and `x2`, inside A's side of the block; the snapshot makes that change to
/// A's term of V too, so the abandon, which takes w's sides and V's terms
/// away and puts P1 and P2 back, gives back what their merge held there.
#[test]
fn abandon_after_a_snapshot_keeps_a_parents_line_in_a_block_the_merge_cuts() -> TestResult {
    assert_cut_rewrite(&CUTQ, ("4", "EDIT"), abandon_w, |_| {
        vec![("u", vec![resolved_hunk("x1 Q x2 2 3 4 5 6 7 yB")])]
    })
}

/// cutq: u rebased onto s is the conflict of V's terms, `x1 Q x2` and `y1`
/// against `1` at line 1 and `xA` and `yB` against `8` at the last line, with
/// no edit and after the edit alike, as jj 0.45.1 gives.
#[test]
fn sibling_rebase_after_a_snapshot_keeps_a_parents_line_in_a_block_the_merge_cuts() -> TestResult {
    assert_cut_rewrite(&CUTQ, ("4", "EDIT"), sibling_rebase, |four| {
        vec![(
            "u",
            vec![
                conflict_hunk(&["x1 Q x2", "y1"], &["1"]),
                resolved_hunk(&format!("2 3 {four} 5 6 7")),
                conflict_hunk(&["xA", "yB"], &["8"]),
            ],
        )]
    })
}

/// cutq: s rebased onto P1, as `jj rebase -r s -o P1` does, which takes w's
/// change away, is `x1 x2` and `y1` against `1` at line 1, with no edit and
/// after the edit alike, as jj 0.45.1 gives: the `Q` that s holds from w's
/// merge cancels against P2's.
#[test]
fn rebase_onto_p1_after_a_snapshot_keeps_a_parents_line_in_a_block_the_merge_cuts() -> TestResult {
    let rebase_s_onto_p1 = |history: CutHistory| -> TestResult<(Case, Reads)> {
        let CutHistory {
            mut case, p1, s, ..
        } = history;
        let rebased = rebase_commit(case.tx.repo_mut(), s, vec![p1.id().clone()]).block_on()?;
        let value = value_at(&rebased.tree(), "f")?;
        Ok((case, vec![("s", value)]))
    };
    assert_cut_rewrite(&CUTQ, ("4", "EDIT"), rebase_s_onto_p1, |_| {
        vec![(
            "s",
            vec![
                conflict_hunk(&["x1 x2", "y1"], &["1"]),
                resolved_hunk("2 3 4 5 6 7 xA"),
            ],
        )]
    })
}

/// cutq: P2 rebased onto A, as `jj rebase -s P2 -o A` does, with w and u
/// rebased: w is P2's `x1 Q x2` and o's `1` against B's `y1` at line 1, and u
/// holds its own text, clean, with no edit and after the edit alike, as jj
/// 0.45.1 gives.
#[test]
fn rebase_of_p2_after_a_snapshot_keeps_a_parents_line_in_a_block_the_merge_cuts() -> TestResult {
    let rebase_p2_onto_a = |history: CutHistory| -> TestResult<(Case, Reads)> {
        let CutHistory {
            mut case,
            a,
            p2,
            w,
            u,
            ..
        } = history;
        rebase_commit(case.tx.repo_mut(), p2, vec![a.id().clone()]).block_on()?;
        let reads = rebased_values(&mut case, &[("w", &w), ("u", &u)])?;
        Ok((case, reads))
    };
    assert_cut_rewrite(&CUTQ, ("4", "EDIT"), rebase_p2_onto_a, |four| {
        vec![
            (
                "w",
                vec![
                    conflict_hunk(&["x1 Q x2", "1"], &["y1"]),
                    resolved_hunk(&format!("2 3 {four} 5 6 7 8")),
                ],
            ),
            (
                "u",
                vec![resolved_hunk(&format!("x1 Q x2 2 3 {four} 5 6 7 yB"))],
            ),
        ]
    })
}

/// cutq: s squashed into P1, as `jj squash --from s --into P1` does, with w
/// and u rebased: P1 is `x1 x2` and `y1` against `1` at line 1, and u is the
/// conflict that u rebased onto s is, with no edit and after the edit alike,
/// as jj 0.45.1 gives.
#[test]
fn squash_into_p1_after_a_snapshot_keeps_a_parents_line_in_a_block_the_merge_cuts() -> TestResult {
    let squash_s_into_p1 = |history: CutHistory| -> TestResult<(Case, Reads)> {
        let CutHistory {
            mut case, p1, s, u, ..
        } = history;
        let p1 = squash_all_of(&mut case, s, &p1)?;
        let mut reads = vec![("P1", value_at(&p1.tree(), "f")?)];
        reads.extend(rebased_values(&mut case, &[("u", &u)])?);
        Ok((case, reads))
    };
    assert_cut_rewrite(&CUTQ, ("4", "EDIT"), squash_s_into_p1, |four| {
        vec![
            (
                "P1",
                vec![
                    conflict_hunk(&["x1 x2", "y1"], &["1"]),
                    resolved_hunk("2 3 4 5 6 7 xA"),
                ],
            ),
            (
                "u",
                vec![
                    conflict_hunk(&["x1 Q x2", "y1"], &["1"]),
                    resolved_hunk(&format!("2 3 {four} 5 6 7")),
                    conflict_hunk(&["xA", "yB"], &["8"]),
                ],
            ),
        ]
    })
}

/// cutqz: cutq where P1 holds its own `Z` at line 6 too, outside every block,
/// and line 3 is edited: u rebased onto s is V's conflict with `Z` in every
/// term, and u once w is abandoned holds `Q` and `Z`, clean, with no edit and
/// after the edit alike, as jj 0.45.1 gives.
#[test]
fn rewrites_after_a_snapshot_keep_a_parents_lines_in_and_outside_a_cut_block() -> TestResult {
    assert_cut_rewrite(&CUTQZ, ("3", "EDIT"), sibling_rebase, |three| {
        vec![(
            "u",
            vec![
                conflict_hunk(&["x1 Q x2", "y1"], &["1"]),
                resolved_hunk(&format!("2 {three} 4 Z 6 7")),
                conflict_hunk(&["xA", "yB"], &["8"]),
            ],
        )]
    })?;
    assert_cut_rewrite(&CUTQZ, ("3", "EDIT"), abandon_w, |_| {
        vec![("u", vec![resolved_hunk("x1 Q x2 2 3 4 Z 6 7 yB")])]
    })
}

/// cqp, cutq's mirror, through `jj abandon P2`: w, rebased onto P1 and P2's
/// parents, is P1's text, `Q` with it, clean, with no edit and after line 4
/// of w's working copy is changed to `EDIT` alike, as jj 0.45.1 gives. jj's
/// snapshot writes `Q` into side 2, P2's text, which lacked it, inside B's
/// side of the block that w's merge cuts, and the snapshot makes that change
/// to B's term of V.
#[test]
fn abandon_of_p2_after_a_snapshot_keeps_p1s_line_in_a_block_the_merge_cuts() -> TestResult {
    assert_cut_rewrite(&CQP, ("4", "EDIT"), abandon_p2_under_w, |four| {
        vec![(
            "w",
            vec![resolved_hunk(&format!("y1 Q y2 2 3 {four} 5 6 7 xA"))],
        )]
    })
}

/// cqp through `jj duplicate -r w -o P1`: the duplicate is P1's text, `Q`
/// with it, clean, with no edit and after the edit alike, as jj 0.45.1 gives.
#[test]
fn duplicate_after_a_snapshot_keeps_p1s_line_in_a_block_the_merge_cuts() -> TestResult {
    assert_cut_rewrite(&CQP, ("4", "EDIT"), duplicate_w_onto_p1, |four| {
        vec![(
            "dup",
            vec![resolved_hunk(&format!("y1 Q y2 2 3 {four} 5 6 7 xA"))],
        )]
    })
}

/// Appends `END` to the working copy of k, the merge of `history`'s P1 and P2,
/// left conflicted, and snapshots it; s and u, children of k, hold `s_text`
/// and `u_text`. Returns u rebased onto s.
fn append_then_rebase(
    history: TwoBases,
    s_text: Option<Vec<u8>>,
    u_text: Option<Vec<u8>>,
) -> TestResult<Merged> {
    let TwoBases {
        mut case, p1, p2, ..
    } = history;
    let k = case.commit(&[&p1, &p2], &[])?;
    let new_value = snapshot_edits(&case, &k, &[("", "END")])?;
    let k = case.commit_with_values(&[&p1, &p2], vec![(repo_path_buf("f"), new_value)])?;
    let s = case.commit(&[&k], &[("f", s_text)])?;
    let u = case.commit(&[&k], &[("f", u_text)])?;
    let rebased = rebase_commit(case.tx.repo_mut(), u.clone(), vec![s.id().clone()]).block_on()?;
    Ok(Merged {
        case,
        parents: vec![s, u],
        tree: rebased.tree(),
    })
}

/// Asserts that no term of the merge's value at `f` holds a marker line of a
/// virtual base, whole or cut.
fn assert_no_marker_line(merged: &Merged) -> TestResult {
    let value = merged.value()?;
    let marked = |term: &Option<TreeValue>| {
        merged
            .case
            .read(term.as_ref())
            .is_some_and(|content| holds_any_marker(&content))
    };
    assert!(
        !value.iter().any(marked),
        "a term holds a marker line of a virtual base: {}",
        describe(&merged.case, &value)
    );
    Ok(())
}

/// K4-append: a line appended after K4's kept conflict goes, outside the
/// virtual base's block, into each term of V, so that u rebased onto s, which
/// both hold it, is `[x END, -o END, y END]`, as jj 0.45.1 gives.
#[test]
fn k4_append_goes_into_each_term_of_v() -> TestResult {
    let merged = append_then_rebase(k4_history(Made::AFirst)?, lines("x END"), lines("y END"))?;
    let [x_end, o_end, y_end] = ["x", "o", "y"].map(|first| {
        let content = lines(&format!("{first} END")).unwrap();
        merged
            .case
            .file(f(), &content, false, CopyId::placeholder())
    });
    assert_same_value(
        &merged.case,
        &merged.value()?,
        &conflict(&[&x_end, &o_end, &y_end]),
    );
    Ok(())
}

/// X1-append: VB6's history, a line appended after the kept conflict, and u,
/// which holds P2's text with it, rebased onto s, which holds P1's: no term
/// holds a marker line.
#[test]
fn x1_append_keeps_every_marker_line() -> TestResult {
    let history = vb6_history()?;
    let merged = append_then_rebase(
        history,
        lines("x1 x2 2 3 4 5 6 7 8 9 10 END"),
        lines("x1 Q x2 W 2 3 4 5 6 7 8 9 10 END"),
    )?;
    assert_no_marker_line(&merged)
}

/// X1-restore: X1's history, where P2 adds `Q` and `W` and d, the merge of P1
/// and P2, adds `FIVE` too. d's `f` restored into P1, as `jj restore --from d
/// --into P1 f` does, and d rebased: the merges of d's new parents and of its
/// old ones enter as their terms, each with VB, so d holds `d + P1' - P1`
/// with `P1' = d`, its own text, as jj 0.45.1 gives.
#[test]
fn x1_restore_into_p1_keeps_d() -> TestResult {
    let TwoBases {
        mut case, p1, p2, ..
    } = vb6_history()?;
    let d_text = "x1 Q x2 W 2 3 4 FIVE 6 7 8 9 10";
    let d = case.commit(&[&p1, &p2], &[("f", lines(d_text))])?;
    let merged = amend_and_rebase(case, &p1, &[("f", lines(d_text))], &d)?;
    assert_resolved_text(&merged, &format!("{}\n", d_text.replace(' ', "\n")))
}

/// MD-append: A deletes `f` and B changes it, so the kept conflict's base is
/// M; a line appended after the conflict, and u, which holds B's text with it,
/// rebased onto s, which holds only the line: no term holds a marker line.
#[test]
fn md_append_keeps_every_marker_line() -> TestResult {
    let history = two_bases(
        &[("f", line("o"))],
        &[("f", None)],
        &[("f", line("y"))],
        Made::AFirst,
        &[("f", None)],
        &[("f", line("y"))],
    )?;
    let merged = append_then_rebase(history, line("END"), lines("y END"))?;
    assert_no_marker_line(&merged)
}

/// Rewrites `commit` with `writes`, as `jj squash` into it does, and returns
/// `descendant` once the descendants are rebased.
fn amend_and_rebase(
    mut case: Case,
    commit: &Commit,
    writes: &[Write],
    descendant: &Commit,
) -> TestResult<Merged> {
    let mut builder = MergedTreeBuilder::new(commit.tree());
    for (path, content) in writes {
        let value = match content {
            Some(content) => {
                Merge::resolved(case.file(repo_path(path), content, false, CopyId::placeholder()))
            }
            None => Merge::absent(),
        };
        builder.set_or_remove(repo_path_buf(*path), value);
    }
    let tree = builder.write_tree().block_on()?;
    case.tx
        .repo_mut()
        .rewrite_commit(commit)
        .set_tree(tree)
        .write_unwrap();
    rebased_descendant(case, descendant)
}

/// `descendant` once the descendants of the rewritten commits are rebased.
fn rebased_descendant(mut case: Case, descendant: &Commit) -> TestResult<Merged> {
    let rebased =
        rebase_descendants_with_options_return_map(case.tx.repo_mut(), &RebaseOptions::default());
    let tree = case.store().get_commit(&rebased[descendant.id()])?.tree();
    Ok(Merged {
        case,
        parents: vec![],
        tree,
    })
}

/// Amending a merge base of a resolved merge: six-line K4 with d, a merge of
/// P1 and P2 that resolves `f` to `y` and changes line 5; A amended at line
/// 3, outside the hunk where A and B disagree. d's new parents hold the
/// virtual base of the amended merge bases, and d's rebase replaces it and
/// d's old one alike: d holds `y 2 THREE 4 FIVE 6`, as jj 0.45.1 gives.
#[test]
fn amend_a_merge_base_of_a_resolved_merge() -> TestResult {
    let (TwoBases { case, a, .. }, d) = resolved_merge(
        six_line_history("x", "y 2 3")?,
        &[("f", lines("y 2 3 4 FIVE 6"))],
    )?;
    let merged = amend_and_rebase(case, &a, &[("f", lines("x 2 THREE 4 5 6"))], &d)?;
    assert_resolved_text(&merged, "y\n2\nTHREE\n4\nFIVE\n6\n")
}

/// K4's d resolves `f` to `y`; B amended to `Y`, as `jj squash` of a child
/// that sets `Y` into B does: d holds `Y`, as jj 0.45.1 gives.
#[test]
fn amend_the_other_merge_base_of_a_resolved_merge() -> TestResult {
    let (TwoBases { case, b, .. }, d) = resolved_k4_merge(&[("f", line("y"))])?;
    let merged = amend_and_rebase(case, &b, &[("f", line("Y"))], &d)?;
    assert_resolved_text(&merged, "Y\n")
}

/// Each term that replaces a virtual base takes the label of the commit it
/// comes from: in VB1a's conflict, `o` is labeled as commit o, not as the
/// first merge base.
#[test]
fn terms_that_replace_a_virtual_base_name_their_commits() -> TestResult {
    let (history, d) = resolved_k4_merge(&[("f", line("y"))])?;
    let o = history
        .case
        .store()
        .get_commit(&history.a.parent_ids()[0])?;
    let o_id = value_at(&o.tree(), "f")?
        .to_file_merge()
        .expect("a file at f")
        .into_resolved()
        .expect("o holds a file");
    let merged = rebase_onto_p1((history, d))?;
    let file_ids = merged.value()?.to_file_merge().expect("files at f");
    let (labels, file_ids) = merged.tree.labels().simplify_with(&file_ids);
    let index = file_ids
        .removes()
        .position(|id| *id == o_id)
        .unwrap_or_else(|| panic!("{file_ids:?} lacks o's {o_id:?}"));
    assert_eq!(labels.get_remove(index), Some(o.conflict_label().as_str()));
    Ok(())
}

/// The nest history, whose virtual bases name virtual bases: o; X and Y,
/// children of o, hold `x` and `y`; L1 and L2, merges of X and Y, hold `x` and
/// `y`; A and B, merges of L1 and L2, hold `a` and `b`; P1 and P2, merges of A
/// and B, hold `p1` and `p2`; d, a merge of P1 and P2, holds `d_value`.
/// Returns d once P2 is rebased onto A, as `jj rebase -s P2 -o A` does.
fn nest_rebase_p2_onto_a(p1: &str, p2: &str, d_value: &str) -> TestResult<Merged> {
    let mut case = Case::new();
    let root = case.root();
    let o = case.commit(&[&root], &[("f", line("o"))])?;
    let x = case.commit(&[&o], &[("f", line("x"))])?;
    let y = case.commit(&[&o], &[("f", line("y"))])?;
    let l1 = case.commit(&[&x, &y], &[("f", line("x"))])?;
    let l2 = case.commit(&[&x, &y], &[("f", line("y"))])?;
    let a = case.commit(&[&l1, &l2], &[("f", line("a"))])?;
    let b = case.commit(&[&l1, &l2], &[("f", line("b"))])?;
    let p1 = case.commit(&[&a, &b], &[("f", line(p1))])?;
    let p2 = case.commit(&[&a, &b], &[("f", line(p2))])?;
    let d = case.commit(&[&p1, &p2], &[("f", line(d_value))])?;
    rebase_commit(case.tx.repo_mut(), p2, vec![a.id().clone()]).block_on()?;
    rebased_descendant(case, &d)
}

/// nest: d holds `b`, as jj 0.45.1 gives.
#[test]
fn nest_rebase_of_a_parent_onto_a_merge_base() -> TestResult {
    assert_resolved_text(&nest_rebase_p2_onto_a("a", "b", "b")?, "b\n")
}

/// nestv: P1 and P2 hold `p` and `q`, and d `r`: d holds `r`, as jj 0.45.1
/// gives.
#[test]
fn nestv_rebase_of_a_parent_onto_a_merge_base() -> TestResult {
    assert_resolved_text(&nest_rebase_p2_onto_a("p", "q", "r")?, "r\n")
}

/// A file that passes as VB or M, as anyone can write one: one block, with
/// `sides`, whose opening line names `names`, at salt 0.
fn crafted_virtual_base(names: &str, sides: &[&[u8]]) -> Vec<u8> {
    crafted_block(names, b"", sides, b"")
}

/// A file that passes as VB or M: the text `before`, one block, with
/// `sides`, whose opening line holds `tail` after its salt, 0, and the text
/// `after`.
fn crafted_block(tail: &str, before: &[u8], sides: &[&[u8]], after: &[u8]) -> Vec<u8> {
    fn hash_bytes(hasher: &mut Blake2b512, bytes: &[u8]) {
        hasher.update((bytes.len() as u64).to_le_bytes());
        hasher.update(bytes);
    }
    let mut hasher = Blake2b512::new();
    hash_bytes(&mut hasher, b"jj virtual merge base");
    hasher.update(0_u64.to_le_bytes());
    hash_bytes(&mut hasher, tail.as_bytes());
    hash_bytes(&mut hasher, before);
    hasher.update((sides.len() as u64).to_le_bytes());
    for side in sides {
        hash_bytes(&mut hasher, side);
    }
    hash_bytes(&mut hasher, after);
    hasher.update(0_u64.to_le_bytes());
    let hex = hasher.finalize()[..16]
        .iter()
        .fold(String::new(), |mut hex, byte| {
            write!(hex, "{byte:02x}").unwrap();
            hex
        });
    let mut content = before.to_vec();
    content.extend_from_slice(format!("{OPEN}{hex} 0 {tail}\n").as_bytes());
    for (index, side) in sides.iter().enumerate() {
        if index > 0 {
            content.extend_from_slice(format!("{SEPARATOR}{hex}\n").as_bytes());
        }
        content.extend_from_slice(side);
    }
    content.extend_from_slice(format!("{CLOSE}{hex}\n").as_bytes());
    content.extend_from_slice(after);
    content
}

/// A file value of `content` at `f`.
fn file_value(case: &Case, content: &[u8]) -> TestResult<Option<TreeValue>> {
    file_value_at(case, f(), content)
}

/// A file value of `content` at `path`.
fn file_value_at(case: &Case, path: &RepoPath, content: &[u8]) -> TestResult<Option<TreeValue>> {
    let id = case
        .store()
        .write_file(path, &mut &content[..])
        .block_on()?;
    Ok(Some(TreeValue::File {
        id,
        executable: false,
        copy_id: CopyId::placeholder(),
    }))
}

/// The name that a VB gives a plain file with no label.
fn file_name(value: Option<&TreeValue>) -> String {
    match value {
        Some(TreeValue::File { id, .. }) => format!("file:{}", id.hex()),
        _ => panic!("expected a file, got {value:?}"),
    }
}

/// A planted chain of files that each pass as VB, each naming the one before
/// as both of its sides, `[file i-1, absent, file i-1]`, fifteen deep. No
/// merge writes a VB that names a VB or M as a side, so the top file is a
/// plain file: d, which resolves the conflict over it, rebased onto A holds
/// `[c, -b, file 15]`. The top file is a side of the result, so the rebase
/// replaced no file of the chain.
#[test]
fn planted_chain_of_virtual_bases_is_plain_text() -> TestResult {
    let mut case = Case::new();
    let mut top = file_value(&case, b"leaf\n")?;
    for _ in 0..15 {
        let name = file_name(top.as_ref());
        top = file_value(
            &case,
            &crafted_virtual_base(&format!("{name} absent {name}"), &[b"s\n"]),
        )?;
    }
    assert!(virtual_base_terms(&case.read(top.as_ref()).unwrap()).is_some());
    let root = case.root();
    let o =
        case.commit_with_values(&[&root], vec![(repo_path_buf("f"), resolved(top.as_ref()))])?;
    let a = case.commit(&[&o], &[("f", line("a"))])?;
    let b = case.commit(&[&o], &[("f", line("b"))])?;
    let m = case.commit(&[&a, &b], &[])?;
    let d = case.commit(&[&m], &[("f", line("c"))])?;
    let rebased = rebase_commit(case.tx.repo_mut(), d, vec![a.id().clone()]).block_on()?;
    let [b_value, c_value] = ["b", "c"].map(|value| case.text(value));
    let value = value_at(&rebased.tree(), "f")?;
    assert_same_value(&case, &value, &conflict(&[&c_value, &b_value, &top]));
    Ok(())
}

/// CH2: at each of `chains`, a path and a depth, a planted chain of files that
/// each pass as VB, file i naming `[leaf, file i-1, leaf, file i-1, leaf]`,
/// that deep over the plain file `leaf`, as `contrib/vb-chain.py base` writes
/// it, so that a full replacement doubles at each level. o holds each top
/// file, A and B hold `a` and `b` at each path, m merges them, and d, its
/// child, resolves each conflict to `c`. Returns the history and d rebased
/// onto A.
fn base_chain_rebase(
    chains: &[(&'static str, usize)],
) -> TestResult<(Case, jj_lib::backend::BackendResult<Commit>)> {
    let mut case = Case::new();
    let mut tops = vec![];
    for &(path, depth) in chains {
        let leaf = file_value_at(&case, repo_path(path), b"leaf\n")?;
        let leaf_name = file_name(leaf.as_ref());
        let mut top = leaf;
        for _ in 0..depth {
            let name = file_name(top.as_ref());
            let names = format!("{leaf_name} {name} {leaf_name} {name} {leaf_name}");
            let content = crafted_virtual_base(&names, &[b"s\n"]);
            top = file_value_at(&case, repo_path(path), &content)?;
        }
        tops.push((repo_path_buf(path), resolved(top.as_ref())));
    }
    let root = case.root();
    let o = case.commit_with_values(&[&root], tops)?;
    let paths = chains.iter().map(|&(path, _)| path).collect_vec();
    let writes = |text: &str| paths.iter().map(|&path| (path, line(text))).collect_vec();
    let a = case.commit(&[&o], &writes("a"))?;
    let b = case.commit(&[&o], &writes("b"))?;
    let m = case.commit(&[&a, &b], &[])?;
    let d = case.commit(&[&m], &writes("c"))?;
    let result = rebase_commit(case.tx.repo_mut(), d, vec![a.id().clone()]).block_on();
    Ok((case, result))
}

/// Two chains of 13 levels at two paths, 8,191 replacements each, under the
/// limit at one path: one alone is rewritten, but together they pass the
/// limit of the merge, 10,000 and 100 for each of its 2 paths, so the rebase
/// fails, naming that limit and the path where it stopped.
#[test]
fn planted_chains_at_two_paths_are_refused_together() -> TestResult {
    let (_case, alone) = base_chain_rebase(&[("f", 13)])?;
    assert!(alone.is_ok(), "expected one chain alone to be rewritten");
    let (_case, result) = base_chain_rebase(&[("f", 13), ("g", 13)])?;
    let Err(err) = result else {
        panic!("expected the rebase to fail");
    };
    assert_eq!(
        err.to_string(),
        "Refused to merge: across 2 paths the merge would replace more than 10200 virtual merge \
         bases (the limit is 10000 and 100 for each path where it replaces one); it stopped at g. \
         Files that pass as virtual merge bases can stand for that many, and so can a history \
         that is both deep and wide; no change was made"
    );
    Ok(())
}

/// CH2 twenty deep: a full replacement would replace 2^20 - 1 files, more than
/// a merge replaces at one path, so the rebase fails, naming the path and that
/// bound, 10,000. The error is raised as the count of replacements passes the
/// bound: the rebase refuses the 10,001st replacement, after the bound's
/// 10,000, whatever the depth of the chain.
#[test]
fn planted_chain_named_as_bases_is_refused() -> TestResult {
    let (_case, result) = base_chain_rebase(&[("f", 20)])?;
    let Err(err) = result else {
        panic!("expected the rebase to fail");
    };
    assert_eq!(
        err.to_string(),
        "Refused to merge: at f the merge would replace more than 10000 virtual merge bases (the \
         limit at one path). Files that pass as virtual merge bases can stand for that many, and \
         so can a history that is both deep and wide; no change was made"
    );
    Ok(())
}

/// CH2 four deep: the rebase replaces all 15 files, which stand for `leaf`, so
/// d holds `[c, -b, leaf]`.
#[test]
fn planted_chain_named_as_bases_four_deep_is_replaced() -> TestResult {
    let (case, result) = base_chain_rebase(&[("f", 4)])?;
    let rebased = result?;
    let [b, c, leaf] = ["b", "c", "leaf"].map(|value| case.text(value));
    let value = value_at(&rebased.tree(), "f")?;
    assert_same_value(&case, &value, &conflict(&[&c, &b, &leaf]));
    Ok(())
}

/// WIDE: K4's history over 10,001 files, d resolving every file to `y`,
/// rebased onto P1. The merge replaces one VB at each of 10,001 paths, more
/// than 10,000 in all, but each path adds 100 to the bound of the merge, so
/// the rebase completes, with K4's conflict `[x, -o, y]` at every path.
#[test]
fn wide_merge_replaces_one_virtual_base_at_each_path() -> TestResult {
    const NUM_PATHS: usize = 10_001;
    let mut case = Case::new();
    let paths = (0..NUM_PATHS)
        .map(|index| repo_path_buf(format!("f{index}")))
        .collect_vec();
    let values = |case: &Case, text: &str| {
        paths
            .iter()
            .map(|path| {
                let file = case.file(path, text.as_bytes(), false, CopyId::placeholder());
                (path.clone(), resolved(file.as_ref()))
            })
            .collect_vec()
    };
    let root = case.root();
    let o = case.commit_with_values(&[&root], values(&case, "o\n"))?;
    let a = case.commit_with_values(&[&o], values(&case, "x\n"))?;
    let b = case.commit_with_values(&[&o], values(&case, "y\n"))?;
    let p1 = case.commit_with_values(&[&a, &b], values(&case, "x\n"))?;
    let p2 = case.commit_with_values(&[&a, &b], values(&case, "y\n"))?;
    let d = case.commit_with_values(&[&p1, &p2], values(&case, "y\n"))?;
    let rebased = rebase_commit(case.tx.repo_mut(), d, vec![p1.id().clone()]).block_on()?;
    let [o, x, y] = ["o", "x", "y"].map(|value| case.text(value));
    for path in ["f0", "f10000"] {
        let value = value_at(&rebased.tree(), path)?;
        assert_same_value(&case, &value, &conflict(&[&x, &o, &y]));
    }
    Ok(())
}

/// t3x: three merge bases, A `x` and B `y` over o1, and C `x` over o2, so that
/// V names the plain file `x` twice, `[x, -o1, y, -r, x]`. P1 and P2, merges
/// of A, B and C, hold `x` and `y`, and d, their merge, `y`. d rebased onto
/// P1: VB still counts as VB, so the result is a conflict and no term holds
/// its text.
#[test]
fn t3x_virtual_base_naming_a_plain_side_twice() -> TestResult {
    let mut case = Case::new();
    let root = case.root();
    let r = case.commit(&[&root], &[("f", line("r"))])?;
    let o1 = case.commit(&[&r], &[("f", line("o1"))])?;
    let o2 = case.commit(&[&r], &[("f", line("o2"))])?;
    let a = case.commit(&[&o1], &[("f", line("x"))])?;
    let b = case.commit(&[&o1], &[("f", line("y"))])?;
    let c = case.commit(&[&o2], &[("f", line("x"))])?;
    let p1 = case.commit(&[&a, &b, &c], &[("f", line("x"))])?;
    let p2 = case.commit(&[&a, &b, &c], &[("f", line("y"))])?;
    let d = case.commit(&[&p1, &p2], &[("f", line("y"))])?;
    let merged = rebase_onto_p1((TwoBases { case, a, b, p1, p2 }, d))?;
    conflict_at_f(&merged)?;
    assert_no_marker_line(&merged)
}

/// The snapshot's rewrite of a virtual base keeps it where the person changed
/// only a side: after line 5 of VB5's conflict is changed and snapshotted, an
/// edit of P2's side in the same working copy keeps the rewritten VB.
#[test]
fn snapshot_of_a_side_edit_keeps_the_virtual_base() -> TestResult {
    let TwoBases {
        mut case, p1, p2, ..
    } = six_line_history("x", "y 2 3")?;
    let w = case.commit(&[&p1, &p2], &[])?;
    let first = snapshot_edits(&case, &w, &[("5", "FIVE")])?.simplify();
    let second = snapshot_edits(&case, &w, &[("5", "FIVE"), ("y", "Y")])?.simplify();
    assert_eq!(
        second.removes().collect_vec(),
        first.removes().collect_vec()
    );
    let y_five = case.file(
        f(),
        &lines("Y 2 3 4 FIVE 6").unwrap(),
        false,
        CopyId::placeholder(),
    );
    assert!(
        second.adds().any(|add| *add == y_five),
        "expected P2's edit among the sides: {}",
        describe(&case, &second)
    );
    Ok(())
}

/// An edit of a line of a block's side, in VB's lines in the base of a
/// conflict that the person kept, is not made: those lines are VB's, which
/// V's terms give, and the base stays the VB that it was.
#[test]
fn snapshot_of_an_edit_of_the_base_of_a_kept_conflict_keeps_the_virtual_base() -> TestResult {
    let TwoBases {
        mut case, p1, p2, ..
    } = six_line_history("x", "y 2 3")?;
    let w = case.commit(&[&p1, &p2], &[])?;
    let before = value_at(&w.tree(), "f")?.simplify();
    let value = snapshot_edits(&case, &w, &[("-y", "-z")])?.simplify();
    assert_eq!(
        value.removes().collect_vec(),
        before.removes().collect_vec(),
        "{}",
        describe(&case, &value)
    );
    Ok(())
}

/// An edit of M's lines, which hold none of V's text, is not made: the base
/// stays M.
#[test]
fn snapshot_of_an_edit_of_m_keeps_m() -> TestResult {
    let TwoBases {
        mut case, p1, p2, ..
    } = two_bases(
        &[("f", line("o"))],
        &[("f", None)],
        &[("f", line("y"))],
        Made::AFirst,
        &[("f", None)],
        &[("f", line("y"))],
    )?;
    let w = case.commit(&[&p1, &p2], &[])?;
    let value = value_at(&w.tree(), "f")?.simplify();
    let m = value.removes().next().expect("a base");
    let m_content = case.read(m.as_ref()).expect("M is a file");
    let from = format!("-{}", text_lines(&m_content)[1]);
    let edited = snapshot_edits(&case, &w, &[(from.as_str(), "-edited")])?.simplify();
    assert_eq!(
        edited.removes().collect_vec(),
        value.removes().collect_vec()
    );
    Ok(())
}

/// Asserts that no side of `value` holds a marker line of a virtual base.
fn assert_no_side_marked(case: &Case, value: &MergedTreeValue) {
    assert!(
        !value.adds().any(|side| {
            case.read(side.as_ref())
                .is_some_and(|content| holds_any_marker(&content))
        }),
        "a side holds a marker line: {}",
        describe(case, value)
    );
}

/// MD1's conflict, whose base is M, with a line appended in the working copy:
/// the line goes into each of V's terms, the absent one read as empty, so the
/// base names `[END, -o END, y END]`, and no side holds M's text.
#[test]
fn snapshot_of_an_append_after_m_edits_the_terms_of_v() -> TestResult {
    let TwoBases {
        mut case, p1, p2, ..
    } = md1_history()?;
    let w = case.commit(&[&p1, &p2], &[])?;
    let value = snapshot_edits(&case, &w, &[("", "END")])?.simplify();
    assert_no_side_marked(&case, &value);
    let removes = value.removes().collect_vec();
    let [base] = removes[..] else {
        panic!("expected one base, got {}", describe(&case, &value));
    };
    let content = case.read(base.as_ref()).expect("the base is a file");
    let terms = virtual_base_terms(&content).expect("the base is VB or M");
    let [end, o_end, y_end] = ["END", "o END", "y END"]
        .map(|text| case.file(f(), &lines(text).unwrap(), false, CopyId::placeholder()));
    assert_same_value(&case, &terms, &conflict(&[&end, &o_end, &y_end]));
    Ok(())
}

/// K4 with binary files, whose conflict's base is M, with a line appended in
/// the working copy: jj's line merge splits binary content at newline bytes
/// too, so the line goes at the end of each of V's file terms, and the base
/// names `[x END, -o END, y END]`, binary; no side holds M's text.
#[test]
fn snapshot_of_an_append_after_a_binary_m_edits_every_file_term() -> TestResult {
    let TwoBases {
        mut case, p1, p2, ..
    } = two_bases(
        &[("f", binary("o"))],
        &[("f", binary("x"))],
        &[("f", binary("y"))],
        Made::AFirst,
        &[("f", binary("x"))],
        &[("f", binary("y"))],
    )?;
    let w = case.commit(&[&p1, &p2], &[])?;
    let edited = snapshot_edits(&case, &w, &[("", "END")])?.simplify();
    assert_no_side_marked(&case, &edited);
    let removes = edited.removes().collect_vec();
    let [base] = removes[..] else {
        panic!("expected one base, got {}", describe(&case, &edited));
    };
    let content = case.read(base.as_ref()).expect("the base is a file");
    let terms = virtual_base_terms(&content).expect("the base is VB or M");
    let [x_end, o_end, y_end] = ["x", "o", "y"].map(|value| {
        let content = [binary(value).unwrap(), b"END\n".to_vec()].concat();
        case.file(f(), &content, false, CopyId::placeholder())
    });
    assert_same_value(&case, &terms, &conflict(&[&x_end, &o_end, &y_end]));
    Ok(())
}

/// A change to a line of a block's side that no side of the conflict holds,
/// which jj shows only in the conflict's base, is a change of the base's own
/// lines too, and is not made. A and B hold `x` and `y` in line 1, and P1 and
/// P2 `p` and `q`.
#[test]
fn snapshot_of_an_edit_of_a_line_only_vb_holds_keeps_the_virtual_base() -> TestResult {
    let TwoBases {
        mut case, p1, p2, ..
    } = two_bases(
        &[("f", lines("1 2 3 4 5 6"))],
        &[("f", lines("x 2 3 4 5 6"))],
        &[("f", lines("y 2 3 4 5 6"))],
        Made::AFirst,
        &[("f", lines("p 2 3 4 5 6"))],
        &[("f", lines("q 2 3 4 5 6"))],
    )?;
    let w = case.commit(&[&p1, &p2], &[])?;
    let before = value_at(&w.tree(), "f")?.simplify();
    let edited = snapshot_edits(&case, &w, &[("-y", "-z")])?.simplify();
    assert_eq!(
        edited.removes().collect_vec(),
        before.removes().collect_vec(),
        "{}",
        describe(&case, &edited)
    );
    Ok(())
}

/// The text of each term of V that the one base of `value`, a virtual base,
/// names.
fn named_term_texts(case: &Case, value: &MergedTreeValue) -> Vec<String> {
    let removes = value.removes().collect_vec();
    let [base] = removes[..] else {
        panic!("expected one base, got {}", describe(case, value));
    };
    let content = case.read(base.as_ref()).expect("the base is a file");
    let terms = virtual_base_terms(&content).expect("the base is VB or M");
    terms
        .iter()
        .map(|term| case.read(term.as_ref()).unwrap_or_default())
        .map(|content| String::from_utf8_lossy(&content).into_owned())
        .collect()
}

/// `words` as lines, the last one with no final newline.
fn unterminated(words: &str) -> Option<Vec<u8>> {
    let mut content = lines(words)?;
    content.pop();
    Some(content)
}

/// VB5d: VB5b's history where every file but o's ends without a final
/// newline, so V's terms conflict at lines 9 and 10 too, and VB's second block
/// holds sides whose last line has no final newline in the terms. w, the
/// merge of P1 and P2, which both hold `9a 10` there, is snapshotted with `9a`
/// edited to `9b`: that resolved hunk covers the whole block, so its text goes
/// into every term of V. s and u, children of w, hold P1's and P2's text with
/// `9b`; u rebased onto s is a conflict at line 1 only, and every side holds
/// `9b`.
#[test]
fn vb5d_an_edit_reaches_v_where_the_files_end_without_a_newline() -> TestResult {
    let TwoBases {
        mut case, p1, p2, ..
    } = two_bases(
        &[("f", lines("o 2 3 4 5 6 7 8 9 10"))],
        &[("f", unterminated("x 2 3 4 5 6 7 8 9a 10"))],
        &[("f", unterminated("y 2 3 4 5 6 7 8 9 10"))],
        Made::AFirst,
        &[("f", unterminated("x 2 3 4 5 6 7 8 9a 10"))],
        &[("f", unterminated("y 2 3 4 5 6 7 8 9a 10"))],
    )?;
    let w = case.commit(&[&p1, &p2], &[])?;
    let new_value = snapshot_edits(&case, &w, &[("9a", "9b")])?;
    let w = case.commit_with_values(&[&p1, &p2], vec![(repo_path_buf("f"), new_value)])?;
    let s = case.commit(&[&w], &[("f", unterminated("x 2 3 4 5 6 7 8 9b 10"))])?;
    let u = case.commit(&[&w], &[("f", unterminated("y 2 3 4 5 6 7 8 9b 10"))])?;
    let rebased = rebase_commit(case.tx.repo_mut(), u, vec![s.id().clone()]).block_on()?;
    let value = value_at(&rebased.tree(), "f")?.simplify();
    let [x, o, y] = ["x", "o", "y"].map(|first| {
        let content = unterminated(&format!("{first} 2 3 4 5 6 7 8 9b 10")).unwrap();
        case.file(f(), &content, false, CopyId::placeholder())
    });
    assert_same_value(&case, &value, &conflict(&[&x, &o, &y]));
    Ok(())
}

/// Asserts that each term of `value` holds the line `line`.
fn assert_every_term_holds(case: &Case, value: &MergedTreeValue, line: &str) {
    for term in value {
        let content = case.read(term.as_ref()).unwrap_or_default();
        assert!(
            text_lines(&content).contains(&line),
            "a term without {line:?}: {}",
            describe(case, value)
        );
    }
}

/// A line in the form of a VB opening line, which opens no block of any VB:
/// no closing line with its hex follows it.
const OPENING_FORM: &str =
    "<<<<<<< jj virtual merge base 0123456789abcdef0123456789abcdef 0 quoted";

/// The lines of `content` in the form of a marker line: seven `<`, `=` or `>`
/// signs, then a space or the end of the line.
fn marker_form_lines(content: &[u8]) -> Vec<String> {
    String::from_utf8_lossy(content)
        .lines()
        .filter(|line| {
            ["<<<<<<<", "=======", ">>>>>>>"].iter().any(|marker| {
                line.strip_prefix(marker)
                    .is_some_and(|rest| rest.is_empty() || rest.starts_with(' '))
            })
        })
        .map(str::to_owned)
        .collect()
}

/// Asserts that `value` is a conflict of two sides, that every term holds
/// `line`, and that no side holds another line in the form of a marker line.
fn assert_line_on_every_term(case: &Case, value: &MergedTreeValue, line: &str) {
    assert_eq!(value.num_sides(), 2, "{}", describe(case, value));
    assert_every_term_holds(case, value, line);
    for side in value.adds() {
        let content = case.read(side.as_ref()).unwrap_or_default();
        assert_eq!(
            marker_form_lines(&content),
            [line],
            "{}",
            describe(case, value)
        );
    }
}

/// K4 with its conflict at the last line of six; k, the merge of P1 and P2,
/// left conflicted, with line 2 of the file that jj materializes for it
/// edited to `line(case, k)`; and s and u, children of k that hold P1's and
/// P2's text with that line: u rebased onto s, with the line.
fn line_written_into_k_then_rebased(
    line: impl FnOnce(&Case, &Commit) -> TestResult<String>,
) -> TestResult<(Case, MergedTreeValue, String)> {
    let TwoBases {
        mut case, p1, p2, ..
    } = two_bases(
        &[("f", lines("1 2 3 4 5 6"))],
        &[("f", lines("1 2 3 4 5 x"))],
        &[("f", lines("1 2 3 4 5 y"))],
        Made::AFirst,
        &[("f", lines("1 2 3 4 5 x"))],
        &[("f", lines("1 2 3 4 5 y"))],
    )?;
    let k = case.commit(&[&p1, &p2], &[])?;
    let line = line(&case, &k)?;
    let new_value = snapshot_edits(&case, &k, &[("2", line.as_str())])?;
    let k = case.commit_with_values(&[&p1, &p2], vec![(repo_path_buf("f"), new_value)])?;
    let text = |last: &str| Some(format!("1\n{line}\n3\n4\n5\n{last}\n").into_bytes());
    let s = case.commit(&[&k], &[("f", text("x"))])?;
    let u = case.commit(&[&k], &[("f", text("y"))])?;
    let rebased = rebase_commit(case.tx.repo_mut(), u, vec![s.id().clone()]).block_on()?;
    let value = value_at(&rebased.tree(), "f")?.simplify();
    Ok((case, value, line))
}

/// A line the person writes is text, whatever it looks like: a line in the
/// form of a separator line of VB, written into k's working copy, reaches
/// every term after the rebase, as jj 0.45.1 gives.
#[test]
fn a_line_in_the_form_of_a_separator_line_reaches_every_term() -> TestResult {
    let separator = "======= 0123456789abcdef0123456789abcdef";
    let (case, value, line) = line_written_into_k_then_rebased(|_, _| Ok(separator.to_owned()))?;
    assert_line_on_every_term(&case, &value, &line);
    Ok(())
}

/// VB5e with the opening form: a line in the form of an opening line of VB,
/// written into k's working copy, reaches every term after the rebase, and no
/// side holds VB.
#[test]
fn a_line_in_the_form_of_an_opening_line_reaches_every_term() -> TestResult {
    let (case, value, line) = line_written_into_k_then_rebased(|_, _| Ok(OPENING_FORM.to_owned()))?;
    assert_line_on_every_term(&case, &value, &line);
    Ok(())
}

/// The opening line of k's own VB, pasted byte for byte into k's working
/// copy, is the person's text: it reaches every term after the rebase, and no
/// side holds VB.
#[test]
fn a_pasted_opening_line_of_the_virtual_base_reaches_every_term() -> TestResult {
    let (case, value, line) = line_written_into_k_then_rebased(|case, k| {
        let base = virtual_base_term(case, &value_at(&k.tree(), "f")?);
        let content = String::from_utf8(case.read(base.as_ref()).expect("VB is a file"))?;
        let opening = content.lines().find(|line| line.starts_with(OPEN));
        Ok(opening.expect("VB's opening line").to_owned())
    })?;
    assert_line_on_every_term(&case, &value, &line);
    Ok(())
}

/// VB7: every commit's `f` holds, at line 2 of six, a line in the form of an
/// opening line of VB; A and B change line 6, P1 and P2 keep their texts, and
/// d, their merge, resolves `f` to P2's. d rebased onto `new_parents(P1, A,
/// B)`: a conflict of P1's and d's texts, each with the line and no other line
/// in the form of a marker line, as jj 0.45.1 gives, never a clean file that
/// holds VB.
fn assert_vb7_conflicts(new_parents: impl FnOnce([Commit; 3]) -> Vec<Commit>) -> TestResult {
    let text = |last: &str| Some(format!("1\n{OPENING_FORM}\n3\n4\n5\n{last}\n").into_bytes());
    let TwoBases {
        mut case,
        a,
        b,
        p1,
        p2,
    } = two_bases(
        &[("f", text("6"))],
        &[("f", text("x"))],
        &[("f", text("y"))],
        Made::AFirst,
        &[("f", text("x"))],
        &[("f", text("y"))],
    )?;
    let d = case.commit(&[&p1, &p2], &[("f", text("y"))])?;
    let parents = new_parents([p1, a, b])
        .iter()
        .map(|commit| commit.id().clone())
        .collect();
    let rebased = rebase_commit(case.tx.repo_mut(), d, parents).block_on()?;
    let value = value_at(&rebased.tree(), "f")?.simplify();
    assert_line_on_every_term(&case, &value, OPENING_FORM);
    Ok(())
}

/// VB7a: d rebased onto P1.
#[test]
fn vb7a_a_quoted_opening_line_then_a_rebase_onto_p1_conflicts() -> TestResult {
    assert_vb7_conflicts(|[p1, _, _]| vec![p1])
}

/// VB7b: d left by an abandon of P2, which rebases it onto P1, A and B.
#[test]
fn vb7b_a_quoted_opening_line_then_an_abandon_of_p2_conflicts() -> TestResult {
    assert_vb7_conflicts(|[p1, a, b]| vec![p1, a, b])
}

/// User settings with `merge.hunk-level` set to `level`.
fn settings_at_hunk_level(level: &str) -> TestResult<UserSettings> {
    let mut config = testutils::base_user_config();
    let mut layer = ConfigLayer::empty(ConfigSource::User);
    layer.set_value("merge.hunk-level", level)?;
    config.add_layer(layer);
    Ok(testutils::user_settings_from_config(config))
}

/// VB5h: VB written at `write_level` and snapshotted at `snapshot_level`. V
/// conflicts at line 1 only (A `x`, B `y`), so VB is one block there; line 3
/// of P1 and P2, `A b c` against `a b C` against VB's `a b c`, conflicts by
/// line and merges by word, so the file that jj materializes for k, the merge
/// of P1 and P2, holds two conflicts at line level and one at word level. k
/// is made at `write_level`, so VB records it, and the file for its conflict
/// is written at `materialize_level`; then, at `snapshot_level`, line 4 of
/// that file is edited to `FOUR`. V's terms then hold `FOUR`, and at line 3
/// what the file held there outside its conflicts. u, a child of k that
/// holds P2's text with `FOUR`, is rebased onto s, which holds P1's: every
/// term holds `FOUR`, as the snapshot lays VB out at the level it records.
fn assert_snapshot_across_hunk_levels(
    write_level: &str,
    snapshot_level: &str,
    materialize_level: &str,
) -> TestResult {
    let write_settings = settings_at_hunk_level(write_level)?;
    let mut case = Case::with_settings(&write_settings);
    let text = |words: &str| {
        Some(format!("{}\n", words.replace(' ', "\n").replace('_', " ")).into_bytes())
    };
    let root = case.root();
    let o = case.commit(&[&root], &[("f", text("1 2 a_b_c 4 5 6"))])?;
    let a = case.commit(&[&o], &[("f", text("x 2 a_b_c 4 5 6"))])?;
    let b = case.commit(&[&o], &[("f", text("y 2 a_b_c 4 5 6"))])?;
    let p1 = case.commit(&[&a, &b], &[("f", text("x 2 A_b_c 4 5 6"))])?;
    let p2 = case.commit(&[&a, &b], &[("f", text("y 2 a_b_C 4 5 6"))])?;
    let k = case.commit(&[&p1, &p2], &[])?;
    // The checkout that wrote the file for k's conflict merged at
    // `materialize_level`; the snapshot that reads it back merges at the
    // store's.
    let materialize = MergeOptions::from_settings(&settings_at_hunk_level(materialize_level)?)?;
    let snapshot_settings = settings_at_hunk_level(snapshot_level)?;
    let (mut case, commits) = case.reloaded(&snapshot_settings, &[&p1, &p2, &k])?;
    let [p1, p2, k] = <[Commit; 3]>::try_from(commits).expect("three commits");
    let new_value = snapshot_edits_materialized(&case, &k, materialize, &[("4", "FOUR")])?;
    // Written by word, the file holds line 3 as jj's word merge of it,
    // `A b C`, in a hunk that resolves: the text goes into V's terms, as
    // jj's snapshot writes the text of a hunk that resolves into every term
    // of a conflict, whether the merge that wrote VB left a conflict there
    // (by line) or resolved it outside VB's block (by word). Written by line,
    // the file holds line 3 as a conflict whose base V's terms keep.
    let line_3 = match materialize_level {
        "word" => "A b C",
        _ => "a b c",
    };
    for term in named_term_texts(&case, &new_value.simplify()) {
        let lines = term.lines().collect_vec();
        assert_eq!((lines[2], lines[3]), (line_3, "FOUR"), "{term:?}");
    }
    let k = case.commit_with_values(&[&p1, &p2], vec![(repo_path_buf("f"), new_value)])?;
    let s = case.commit(&[&k], &[("f", text("x 2 A_b_c FOUR 5 6"))])?;
    let u = case.commit(&[&k], &[("f", text("y 2 a_b_C FOUR 5 6"))])?;
    let rebased = rebase_commit(case.tx.repo_mut(), u, vec![s.id().clone()]).block_on()?;
    let value = value_at(&rebased.tree(), "f")?.simplify();
    assert!(!value.is_resolved(), "{}", describe(&case, &value));
    assert_every_term_holds(&case, &value, "FOUR");
    Ok(())
}

/// VB5h1: VB written by line, snapshotted by word.
#[test]
fn vb5h1_a_snapshot_by_word_of_vb_written_by_line_reaches_every_term() -> TestResult {
    assert_snapshot_across_hunk_levels("line", "word", "line")
}

/// VB5h2: VB written by word, snapshotted by line.
#[test]
fn vb5h2_a_snapshot_by_line_of_vb_written_by_word_reaches_every_term() -> TestResult {
    assert_snapshot_across_hunk_levels("word", "line", "word")
}

/// VB written by line and the file written again by a checkout at word
/// level, where the snapshot also runs: the file holds jj's word merge of
/// line 3 in a hunk that resolves, which goes into every term of V, as jj's
/// snapshot writes a resolved hunk into every term of a conflict.
#[test]
fn a_snapshot_of_a_file_written_at_the_store_level_reaches_every_term() -> TestResult {
    assert_snapshot_across_hunk_levels("line", "word", "word")
}

/// VB written by word and the file written again by a checkout at line
/// level, where the snapshot also runs: the file holds line 3 as a conflict,
/// and V's terms keep their own line 3.
#[test]
fn a_snapshot_of_a_file_written_by_line_of_vb_written_by_word_reaches_every_term() -> TestResult {
    assert_snapshot_across_hunk_levels("word", "line", "line")
}

/// The snapshot splits a conflict into hunks as every merge of contents does,
/// at the store's hunk level. With `merge.hunk-level=word`: A and B change
/// line 1 of `o` to `x` and `y`, P1 also changes `a` to `A` in line 3 and P2
/// `c` to `C`, so that line merges by word in k, the merge of P1 and P2, left
/// conflicted with line 5 edited to `FIVE`. s and u, children of k, hold P1's
/// and P2's text with `FIVE`; u rebased onto s is a conflict, and every term
/// holds `FIVE`, as jj 0.45.1 gives.
#[test]
fn a_snapshot_at_word_level_reaches_every_term() -> TestResult {
    let mut case = Case::with_settings(&settings_at_hunk_level("word")?);
    let text = |text: &str| Some(text.replace(' ', "\n").replace('_', " ").into_bytes());
    let root = case.root();
    let o = case.commit(&[&root], &[("f", text("1 2 a_b_c 4 5 6 "))])?;
    let a = case.commit(&[&o], &[("f", text("x 2 a_b_c 4 5 6 "))])?;
    let b = case.commit(&[&o], &[("f", text("y 2 a_b_c 4 5 6 "))])?;
    let p1 = case.commit(&[&a, &b], &[("f", text("x 2 A_b_c 4 5 6 "))])?;
    let p2 = case.commit(&[&a, &b], &[("f", text("y 2 a_b_C 4 5 6 "))])?;
    let k = case.commit(&[&p1, &p2], &[])?;
    let new_value = snapshot_edits(&case, &k, &[("5", "FIVE")])?;
    let k = case.commit_with_values(&[&p1, &p2], vec![(repo_path_buf("f"), new_value)])?;
    let s = case.commit(&[&k], &[("f", text("x 2 A_b_c 4 FIVE 6 "))])?;
    let u = case.commit(&[&k], &[("f", text("y 2 a_b_C 4 FIVE 6 "))])?;
    let rebased = rebase_commit(case.tx.repo_mut(), u, vec![s.id().clone()]).block_on()?;
    let value = value_at(&rebased.tree(), "f")?.simplify();
    assert!(!value.is_resolved(), "{}", describe(&case, &value));
    assert_every_term_holds(&case, &value, "FIVE");
    Ok(())
}

/// A snapshot of a conflict whose base passes as VB but names another such
/// file as a side, which no merge writes: the base is not VB, so the snapshot
/// writes it as the plain text the person left, here with a line appended.
#[test]
fn snapshot_of_a_base_naming_a_virtual_base_as_a_side_writes_plain_text() -> TestResult {
    let mut case = Case::new();
    let [x, o, y, s1, s2] = ["x", "o", "y", "s1", "s2"].map(|value| case.text(value));
    let names = |first: &Option<TreeValue>| {
        format!(
            "{} {} {}",
            file_name(first.as_ref()),
            file_name(o.as_ref()),
            file_name(y.as_ref())
        )
    };
    let inner = file_value(&case, &crafted_virtual_base(&names(&x), &[b"x\n", b"y\n"]))?;
    let outer_content = crafted_virtual_base(&names(&inner), &[b"s\n"]);
    let outer = file_value(&case, &outer_content)?;
    let root = case.root();
    let base_value = conflict(&[&s1, &outer, &s2]);
    let w = case.commit_with_values(&[&root], vec![(repo_path_buf("f"), base_value)])?;
    let value = snapshot_edits(&case, &w, &[("", "END")])?.simplify();
    let removes = value.removes().collect_vec();
    let [base] = removes[..] else {
        panic!("expected one base, got {}", describe(&case, &value));
    };
    let expected = [outer_content, b"END\n".to_vec()].concat();
    assert_eq!(case.read(base.as_ref()), Some(expected));
    assert_no_side_marked(&case, &value);
    Ok(())
}

/// K4 over six lines where A and B change lines 1 and 6 to `a1`, `a6` and
/// `b1`, `b6`, so k, the merge of P1 and P2, holds two conflicts; the first
/// conflict of the file that jj materializes for k replaced with
/// `replacement`; and s and u, children of k that hold `replacement` and
/// lines 2 to 5 with P1's and P2's line 6: u rebased onto s.
fn first_of_two_conflicts_replaced(replacement: &str) -> TestResult<(Case, MergedTreeValue)> {
    let TwoBases {
        mut case, p1, p2, ..
    } = two_bases(
        &[("f", lines("1 2 3 4 5 6"))],
        &[("f", lines("a1 2 3 4 5 a6"))],
        &[("f", lines("b1 2 3 4 5 b6"))],
        Made::AFirst,
        &[("f", lines("a1 2 3 4 5 a6"))],
        &[("f", lines("b1 2 3 4 5 b6"))],
    )?;
    let k = case.commit(&[&p1, &p2], &[])?;
    let new_value = snapshot_text(&case, &k, |working_copy, marker_len| {
        let start = working_copy.find(&"<".repeat(marker_len)).unwrap();
        let close = start + working_copy[start..].find(&">".repeat(marker_len)).unwrap();
        let end = close + working_copy[close..].find('\n').unwrap() + 1;
        format!(
            "{}{replacement}{}",
            &working_copy[..start],
            &working_copy[end..]
        )
    })?;
    let k = case.commit_with_values(&[&p1, &p2], vec![(repo_path_buf("f"), new_value)])?;
    let text = |last: &str| Some(format!("{replacement}2\n3\n4\n5\n{last}\n").into_bytes());
    let s = case.commit(&[&k], &[("f", text("a6"))])?;
    let u = case.commit(&[&k], &[("f", text("b6"))])?;
    let rebased = rebase_commit(case.tx.repo_mut(), u, vec![s.id().clone()]).block_on()?;
    let value = value_at(&rebased.tree(), "f")?.simplify();
    Ok((case, value))
}

/// VB5g: a snapshot that resolves one of two conflicts to `R1` writes `R1`
/// into every term of V there, so u rebased onto s is a conflict at line 6
/// only, and every term holds `R1`.
#[test]
fn vb5g_a_snapshot_that_resolves_one_of_two_conflicts_keeps_the_resolution() -> TestResult {
    let (case, value) = first_of_two_conflicts_replaced("R1\n")?;
    assert!(!value.is_resolved(), "{}", describe(&case, &value));
    assert_every_term_holds(&case, &value, "R1");
    assert_no_side_marked(&case, &value);
    Ok(())
}

/// VB5g2: a snapshot that deletes one of two conflicts deletes its lines from
/// every term of V there, so every term of u rebased onto s starts at line 2.
#[test]
fn vb5g2_a_snapshot_that_deletes_one_of_two_conflicts_keeps_the_deletion() -> TestResult {
    let (case, value) = first_of_two_conflicts_replaced("")?;
    assert!(!value.is_resolved(), "{}", describe(&case, &value));
    for term in &value {
        let content = case.read(term.as_ref()).unwrap_or_default();
        assert_eq!(
            text_lines(&content).first(),
            Some(&"2"),
            "{}",
            describe(&case, &value)
        );
    }
    assert_no_side_marked(&case, &value);
    Ok(())
}

/// K4 over `num_lines` lines, `1` up, where A and B change each line of
/// `conflicting` to `a<n>` and `b<n>`, so k, the merge of P1 and P2, holds a
/// conflict at each. The file that jj materializes for k is rewritten as
/// `plan` says, the parts that the person left, in order: `C<n>` is the n-th
/// conflict, kept, and `=<line>` and `+<text>` are a line of text. s and u,
/// children of k, hold the person's text with P1's and P2's line in place of
/// each conflict kept, and u is rebased onto s. Asserts that u is then the
/// conflict of those two texts over V's base with the same plan, the line
/// `<n>` in place of each conflict kept: every change of the plan is in
/// every term.
fn assert_planned_snapshot_reaches_every_term(
    num_lines: usize,
    conflicting: &[usize],
    plan: &str,
) -> TestResult {
    let text = |prefix: &str| {
        let line = |n: usize| {
            if conflicting.contains(&n) {
                format!("{prefix}{n}")
            } else {
                n.to_string()
            }
        };
        lines(&(1..=num_lines).map(line).join(" "))
    };
    let TwoBases {
        mut case, p1, p2, ..
    } = two_bases(
        &[("f", text(""))],
        &[("f", text("a"))],
        &[("f", text("b"))],
        Made::AFirst,
        &[("f", text("a"))],
        &[("f", text("b"))],
    )?;
    let k = case.commit(&[&p1, &p2], &[])?;
    let parts = plan.split(' ').collect_vec();
    // The conflict that a part keeps, counted from 0, or the line that it is.
    let part = |part: &str| -> Result<usize, String> {
        match part.split_at(1) {
            ("C", number) => Ok(number.parse::<usize>().unwrap() - 1),
            (_, text) => Err(text.to_owned()),
        }
    };
    let new_value = snapshot_text(&case, &k, |working_copy, marker_len| {
        let (open, close) = ("<".repeat(marker_len), ">".repeat(marker_len));
        let mut conflicts = vec![];
        let mut rest = working_copy;
        while let Some(start) = rest.find(&open) {
            let end = start + rest[start..].find(&close).unwrap();
            let end = end + rest[end..].find('\n').unwrap() + 1;
            conflicts.push(&rest[start..end]);
            rest = &rest[end..];
        }
        let mut texts = parts.iter().map(|text| match part(text) {
            Ok(conflict) => conflicts[conflict].to_owned(),
            Err(text) => text + "\n",
        });
        texts.join("")
    })?;
    let k = case.commit_with_values(&[&p1, &p2], vec![(repo_path_buf("f"), new_value)])?;
    let planned = |prefix: &str| {
        let words = parts.iter().map(|text| match part(text) {
            Ok(conflict) => format!("{prefix}{}", conflicting[conflict]),
            Err(text) => text,
        });
        lines(&words.collect_vec().join(" "))
    };
    let s = case.commit(&[&k], &[("f", planned("a"))])?;
    let u = case.commit(&[&k], &[("f", planned("b"))])?;
    let rebased = rebase_commit(case.tx.repo_mut(), u, vec![s.id().clone()]).block_on()?;
    let value = value_at(&rebased.tree(), "f")?;
    let [a, o, b] = ["a", "", "b"].map(|prefix| {
        let content = planned(prefix).unwrap();
        case.file(f(), &content, false, CopyId::placeholder())
    });
    assert_same_value(&case, &value, &conflict(&[&a, &o, &b]));
    Ok(())
}

/// VB5i: a snapshot that deletes a conflict together with the text up to the
/// next conflict keeps that next conflict: each block of VB lines up with
/// itself, not with the conflict that was deleted before it. Over five lines
/// that conflict at 1, 3 and 5, the first conflict and line 2 are deleted.
#[test]
fn vb5i1_a_snapshot_that_deletes_a_conflict_and_the_next_line_keeps_the_rest() -> TestResult {
    assert_planned_snapshot_reaches_every_term(5, &[1, 3, 5], "C2 =4 C3")
}

/// VB5i: the second conflict and line 4 are deleted.
#[test]
fn vb5i2_a_snapshot_that_deletes_the_middle_conflict_and_a_line_keeps_the_rest() -> TestResult {
    assert_planned_snapshot_reaches_every_term(5, &[1, 3, 5], "C1 =2 C3")
}

/// VB5i: over three lines that conflict at 1 and 3, only the second conflict
/// is kept.
#[test]
fn vb5i3_a_snapshot_that_keeps_only_the_last_conflict_keeps_it() -> TestResult {
    assert_planned_snapshot_reaches_every_term(3, &[1, 3], "C2")
}

/// VB5i: over eight lines that conflict at 2 and 6, lines 2 to 5 are deleted,
/// the first conflict among them.
#[test]
fn vb5i4_a_snapshot_that_deletes_lines_with_a_conflict_among_them_keeps_the_next() -> TestResult {
    assert_planned_snapshot_reaches_every_term(8, &[2, 6], "=1 C2 =7 =8")
}

/// VB5k: a snapshot that moves a line of text across a conflict that it
/// keeps moves the line in every term.
#[test]
fn vb5k1_a_snapshot_that_moves_a_line_across_a_conflict_keeps_the_move() -> TestResult {
    assert_planned_snapshot_reaches_every_term(5, &[1, 3, 5], "C1 C2 =2 =4 C3")
}

/// VB5k: a snapshot that swaps two conflicts swaps what each term of V holds
/// at them: a block that the person moved takes V's conflict with it.
#[test]
fn vb5k2_a_snapshot_that_swaps_two_conflicts_keeps_the_swap() -> TestResult {
    assert_planned_snapshot_reaches_every_term(5, &[1, 3, 5], "C3 =2 C2 =4 C1")
}

/// The snapshots that keep a line of text between the conflicts: a conflict
/// deleted with the line before it, or alone; the middle conflict resolved;
/// a conflict resolved and a line edited at once.
#[test]
fn snapshots_that_keep_a_line_between_conflicts_reach_every_term() -> TestResult {
    for plan in [
        "C1 =4 C3",
        "=2 C2 =4 C3",
        "C1 =2 C2",
        "C1 =2 +R3 =4 C3",
        "+R1 =2 C2 +FOUR C3",
    ] {
        assert_planned_snapshot_reaches_every_term(5, &[1, 3, 5], plan)?;
    }
    Ok(())
}

/// One snapshot with two changes that meet: over six lines that conflict at 1
/// and 6, the first conflict and line 2 are replaced with `R1` and `TWO`,
/// which rewrites the text up to line 3 in every term, and line 4 is edited
/// to `FOUR`, a change of its own in the same text.
#[test]
fn a_snapshot_of_two_changes_at_once_reaches_every_term() -> TestResult {
    assert_planned_snapshot_reaches_every_term(6, &[1, 6], "+R1 +TWO =3 +FOUR =5 C2")
}

/// `count` blocks in the form of a virtual base's, each with its own hex,
/// that no merge wrote, as a file can quote them.
fn quoted_blocks(count: usize) -> Vec<u8> {
    let block = |n: usize| {
        let hex = format!("{:032x}", 0x1111 * n);
        format!("{OPEN}{hex} 0 quoted\nq\n{SEPARATOR}{hex}\nr\n{CLOSE}{hex}\n")
    };
    (1..=count).map(block).join("").into_bytes()
}

/// K4 over `header` and then the lines 1 to 6, where A and B change line 6 to
/// `x` and `y`: k, the merge of P1 and P2, left conflicted, with line 5
/// edited to `FIVE`, and u, a child of k that holds P2's text with `FIVE`,
/// rebased onto s, which holds P1's. Asserts that the base of k's conflict
/// holds V's text, as VB does and M does not, and that u is a conflict of two
/// sides whose every term holds `FIVE`.
fn assert_header_keeps_a_virtual_base_and_an_edit(header: &[u8]) -> TestResult {
    let text = |five: &str, last: &str| {
        let lines = format!("1\n2\n3\n4\n{five}\n{last}\n");
        Some([header, lines.as_bytes()].concat())
    };
    let TwoBases {
        mut case, p1, p2, ..
    } = two_bases(
        &[("f", text("5", "6"))],
        &[("f", text("5", "x"))],
        &[("f", text("5", "y"))],
        Made::AFirst,
        &[("f", text("5", "x"))],
        &[("f", text("5", "y"))],
    )?;
    let k = case.commit(&[&p1, &p2], &[])?;
    let base = virtual_base_term(&case, &value_at(&k.tree(), "f")?);
    let content = case.read(base.as_ref()).expect("the base is a file");
    assert!(
        text_lines(&content).contains(&"4"),
        "the base holds none of V's text: {}",
        String::from_utf8_lossy(&content)
    );
    let new_value = snapshot_edits(&case, &k, &[("5", "FIVE")])?;
    let k = case.commit_with_values(&[&p1, &p2], vec![(repo_path_buf("f"), new_value)])?;
    let s = case.commit(&[&k], &[("f", text("FIVE", "x"))])?;
    let u = case.commit(&[&k], &[("f", text("FIVE", "y"))])?;
    let rebased = rebase_commit(case.tx.repo_mut(), u, vec![s.id().clone()]).block_on()?;
    let value = value_at(&rebased.tree(), "f")?.simplify();
    assert_eq!(value.num_sides(), 2, "{}", describe(&case, &value));
    assert_every_term_holds(&case, &value, "FIVE");
    Ok(())
}

/// Q5: a file whose shared text quotes five blocks in the form of a virtual
/// base's gets VB all the same, as the reader tries the blocks that the merge
/// wrote before the quoted ones, and a snapshot keeps an edit on every term.
#[test]
fn q5_a_file_that_quotes_five_blocks_gets_a_virtual_base_and_keeps_an_edit() -> TestResult {
    assert_header_keeps_a_virtual_base_and_an_edit(&quoted_blocks(5))
}

/// A file whose shared text starts with a byte copy of a VB that ends with
/// its block, which verifies by itself there, gets VB all the same, and a
/// snapshot keeps an edit on every term.
#[test]
fn a_file_that_starts_with_a_copy_of_a_virtual_base_gets_one_and_keeps_an_edit() -> TestResult {
    let TwoBases {
        mut case, p1, p2, ..
    } = k4_history(Made::AFirst)?;
    let k = case.commit(&[&p1, &p2], &[])?;
    let base = virtual_base_term(&case, &value_at(&k.tree(), "f")?);
    let copy = case.read(base.as_ref()).expect("the base is a file");
    assert!(copy.ends_with(format!("{CLOSE}{}\n", opening_hex(&text_lines(&copy))).as_bytes()));
    assert_header_keeps_a_virtual_base_and_an_edit(&copy)
}

/// A conflict whose base is M although the terms that M names are text
/// files, as a build that could not read its own VB back wrote one: M shows
/// none of V's text, so the snapshot takes the change that the person made
/// to both sides alike, line 5 edited to `FIVE`, in text that both share with
/// V's VB, and makes it to every term of V.
#[test]
fn snapshot_of_an_m_over_text_files_makes_a_shared_change_to_its_terms() -> TestResult {
    let text = |last: &str| lines(&format!("1 2 3 4 5 {last}"));
    let TwoBases {
        mut case, p1, p2, ..
    } = two_bases(
        &[("f", text("6"))],
        &[("f", text("x"))],
        &[("f", text("y"))],
        Made::AFirst,
        &[("f", text("x"))],
        &[("f", text("y"))],
    )?;
    let [x, o, y] = ["x", "6", "y"]
        .map(|last| case.file(f(), &text(last).unwrap(), false, CopyId::placeholder()));
    let names = [&x, &o, &y].map(|term| file_name(term.as_ref()));
    let sides = names.iter().map(|name| format!("{name}\n").into_bytes());
    let sides = sides.collect_vec();
    let sides = sides.iter().map(|side| side.as_slice()).collect_vec();
    let m = file_value(&case, &crafted_virtual_base(&names.join(" "), &sides))?;
    let value = conflict(&[&x, &m, &y]);
    let w = case.commit_with_values(&[&p1, &p2], vec![(repo_path_buf("f"), value)])?;
    let edited = snapshot_text(&case, &w, |working_copy, _| {
        working_copy
            .replace("\n+5\n", "\n+FIVE\n")
            .replace("\n5\n", "\nFIVE\n")
    })?;
    assert_eq!(
        named_term_texts(&case, &edited.simplify()),
        ["x", "6", "y"].map(|last| format!("1\n2\n3\n4\nFIVE\n{last}\n"))
    );
    Ok(())
}

/// VB5j: a VB that a build from before the hunk level was recorded wrote at
/// word level records none. V's terms merge by word at line 1 and conflict at
/// line 3, so that VB holds one block, where the same terms at line level
/// would give two: the snapshot lays VB out by word where it is not laid out
/// by line, so line 4 edited to `FOUR` in k reaches every term of u, a child
/// of k that holds P2's text with `FOUR`, rebased onto s, which holds P1's.
#[test]
fn vb5j_a_word_level_vb_that_records_no_level_is_laid_out_by_word() -> TestResult {
    let mut case = Case::with_settings(&settings_at_hunk_level("word")?);
    let text = |words: &str| {
        Some(format!("{}\n", words.replace(' ', "\n").replace('_', " ")).into_bytes())
    };
    let root = case.root();
    let o = case.commit(&[&root], &[("f", text("a_b_c 2 o3 4"))])?;
    let a = case.commit(&[&o], &[("f", text("A_b_c 2 x 4"))])?;
    let b = case.commit(&[&o], &[("f", text("a_b_C 2 y 4"))])?;
    let p1 = case.commit(&[&a, &b], &[("f", text("A_b_C 2 x 4"))])?;
    let p2 = case.commit(&[&a, &b], &[("f", text("A_b_C 2 y 4"))])?;
    let file = |case: &Case, words: &str| {
        case.file(f(), &text(words).unwrap(), false, CopyId::placeholder())
    };
    let terms = ["A_b_c 2 x 4", "a_b_c 2 o3 4", "a_b_C 2 y 4"].map(|words| file(&case, words));
    let names = terms.each_ref().map(|term| file_name(term.as_ref()));
    let vb = crafted_block(&names.join(" "), b"A b C\n2\n", &[b"x\n", b"y\n"], b"4\n");
    let vb = file_value(&case, &vb)?;
    let sides = ["A_b_C 2 x 4", "A_b_C 2 y 4"].map(|words| file(&case, words));
    let value = conflict(&[&sides[0], &vb, &sides[1]]);
    let k = case.commit_with_values(&[&p1, &p2], vec![(repo_path_buf("f"), value)])?;
    let new_value = snapshot_edits(&case, &k, &[("4", "FOUR")])?;
    let k = case.commit_with_values(&[&p1, &p2], vec![(repo_path_buf("f"), new_value)])?;
    let s = case.commit(&[&k], &[("f", text("A_b_C 2 x FOUR"))])?;
    let u = case.commit(&[&k], &[("f", text("A_b_C 2 y FOUR"))])?;
    let rebased = rebase_commit(case.tx.repo_mut(), u, vec![s.id().clone()]).block_on()?;
    let value = value_at(&rebased.tree(), "f")?.simplify();
    assert!(!value.is_resolved(), "{}", describe(&case, &value));
    assert_every_term_holds(&case, &value, "FOUR");
    Ok(())
}

/// A snapshot of a conflict whose base passes as VB of real terms but is not
/// their merge (its one block holds one side, `s`): the base shows none of
/// V's text as a merge would, so the snapshot takes the changes that every
/// side of the conflict has alike, here a line appended, and makes them to
/// V's terms: the base is their virtual base, and no side holds a marker
/// line.
#[test]
fn snapshot_of_a_virtual_base_that_is_not_its_terms_merge_edits_its_terms() -> TestResult {
    let mut case = Case::new();
    let [x, o, y, s1, s2] = ["x", "o", "y", "s1", "s2"].map(|value| case.text(value));
    let names = format!(
        "{} {} {}",
        file_name(x.as_ref()),
        file_name(o.as_ref()),
        file_name(y.as_ref())
    );
    let base = file_value(&case, &crafted_virtual_base(&names, &[b"s\n"]))?;
    let root = case.root();
    let base_value = conflict(&[&s1, &base, &s2]);
    let w = case.commit_with_values(&[&root], vec![(repo_path_buf("f"), base_value)])?;
    let value = snapshot_edits(&case, &w, &[("", "END")])?.simplify();
    assert_eq!(
        named_term_texts(&case, &value),
        ["x\nEND\n", "o\nEND\n", "y\nEND\n"]
    );
    assert_no_side_marked(&case, &value);
    Ok(())
}

/// A snapshot after which V's edited terms resolve to a value that is not a
/// file: A sets `f` to a symlink and B to an empty file where o has none, so
/// V is `[symlink, -absent, empty]` and the base of P1's and P2's conflict is
/// M. A line appended in the working copy goes into the absent term and the
/// empty file alike, which then cancel, leaving the symlink: the base keeps
/// M, and no side holds a marker line.
#[test]
fn snapshot_after_which_v_resolves_to_a_symlink_keeps_m() -> TestResult {
    let mut case = Case::new();
    let root = case.root();
    let o = case.commit(&[&root], &[("g", line("o"))])?;
    let symlink_id = case.store().write_symlink(f(), "target").block_on()?;
    let symlink = Merge::normal(TreeValue::Symlink(symlink_id));
    let a = case.commit_with_values(&[&o], vec![(repo_path_buf("f"), symlink)])?;
    let b = case.commit(&[&o], &[("f", Some(vec![]))])?;
    let p1 = case.commit(&[&a, &b], &[("f", line("p1"))])?;
    let p2 = case.commit(&[&a, &b], &[("f", line("p2"))])?;
    let w = case.commit(&[&p1, &p2], &[])?;
    let value = value_at(&w.tree(), "f")?.simplify();
    let removes = value.removes().collect_vec();
    let [base] = removes[..] else {
        panic!("expected one base, got {}", describe(&case, &value));
    };
    let TreeValue::File { id, .. } = base.as_ref().expect("a file base") else {
        panic!("expected a file base, got {}", describe(&case, &value));
    };
    let m = read_file(case.store(), f(), id);
    assert!(
        holds_any_marker(&m),
        "expected M, got {}",
        describe(&case, &value)
    );
    let edited = snapshot_edits(&case, &w, &[("", "END")])?.simplify();
    assert_eq!(edited.removes().collect_vec(), [base]);
    assert_no_side_marked(&case, &edited);
    Ok(())
}

/// The three merge bases A, B and C of merge-grid's t3 and t3x histories.
#[derive(Clone, Copy, Debug)]
enum ThreeBases {
    /// A `x`, B `y` and C `w`, children of o.
    T3,
    /// A `x` and B `y`, children of o1, and C `x`, a child of o2; o1 and o2
    /// are children of r.
    T3x,
}

/// Where P1, a merge of three merge bases, is squashed to, and with what.
#[derive(Clone, Copy, Debug)]
enum SquashInto {
    /// d, the merge of P1 and P2.
    D,
    /// g, a child of d that changes nothing, as `jj squash --insert-after d`
    /// makes one.
    ChildOfD,
    /// m, a merge of d and X, a child of the root that writes `h`, that
    /// changes nothing, as `jj squash --insert-after d --insert-after X`
    /// makes one.
    MergeOfDAndX,
    /// A child of m.
    BelowMergeOfDAndX,
    /// c3, below c1 and c2, in a line of children of d, with c1 squashed
    /// into it too.
    C3WithC1,
    /// A child of Q, a merge of e, a child of d, and of X and Y, children of
    /// the root.
    BelowMergeOfThree,
    /// That child, with Q squashed into it too.
    BelowMergeOfThreeWithIt,
    /// A merge of g and h, children of d.
    MergeOfTwoChildrenOfD,
    /// g, a child of d, with only a part of P1 selected: P1's tree with `f`
    /// as P2 holds it.
    ChildOfDWithAPart,
    /// n, a merge of P1 and P2 that changes nothing, onto which d is
    /// rebased, as `jj squash --insert-before d` makes one.
    BeforeD,
    /// d, once the squash into n is done and d is rebased onto it.
    DAfterBeforeD,
    /// c, a child of P1 alone, that writes `g`.
    ChildOfP1,
}

/// The value at `f` of the commit that `into` names once P1 is squashed into
/// it, as `jj squash --from P1 --into <it> -u` does, where P1 and P2, merges
/// of the merge bases of `history`, hold `x` and `y`, and d, their merge,
/// holds `y`. The merge base at `highest` (0 for A, 1 for B, 2 for C) is made
/// again until its commit id is the highest of the three, so that P1's
/// parents are merged with the other two first. With the value, the tree of
/// the squashed commit.
fn squash_p1(
    history: ThreeBases,
    highest: usize,
    into: SquashInto,
) -> TestResult<(Case, MergedTreeValue, MergedTree)> {
    let mut case = Case::new();
    let root = case.root();
    let mut bases = match history {
        ThreeBases::T3 => {
            let o = case.commit(&[&root], &[("f", line("o"))])?;
            vec![
                case.commit(&[&o], &[("f", line("x"))])?,
                case.commit(&[&o], &[("f", line("y"))])?,
                case.commit(&[&o], &[("f", line("w"))])?,
            ]
        }
        ThreeBases::T3x => {
            let r = case.commit(&[&root], &[("f", line("r"))])?;
            let o1 = case.commit(&[&r], &[("f", line("o1"))])?;
            let o2 = case.commit(&[&r], &[("f", line("o2"))])?;
            vec![
                case.commit(&[&o1], &[("f", line("x"))])?,
                case.commit(&[&o1], &[("f", line("y"))])?,
                case.commit(&[&o2], &[("f", line("x"))])?,
            ]
        }
    };
    let mut attempt = 0;
    while bases.iter().any(|base| base.id() > bases[highest].id()) {
        attempt += 1;
        let again = case
            .tx
            .repo_mut()
            .rewrite_commit(&bases[highest])
            .set_description(format!("made again {attempt}"))
            .write_unwrap();
        bases[highest] = again;
    }
    let parents = bases.iter().collect_vec();
    let p1 = case.commit(&parents, &[("f", line("x"))])?;
    let p2 = case.commit(&parents, &[("f", line("y"))])?;
    let d = case.commit(&[&p1, &p2], &[("f", line("y"))])?;
    let whole = |case: &Case, commit: Commit| -> TestResult<CommitWithSelection> {
        Ok(CommitWithSelection {
            parent_tree: commit.parent_tree(case.tx.repo()).block_on()?,
            selected_tree: commit.tree(),
            commit,
        })
    };
    let mut sources = vec![whole(&case, p1.clone())?];
    let destination = match into {
        SquashInto::D => d.clone(),
        SquashInto::ChildOfD => case.commit(&[&d], &[])?,
        SquashInto::MergeOfDAndX | SquashInto::BelowMergeOfDAndX => {
            let x = case.commit(&[&root], &[("h", line("h"))])?;
            let m = case.commit(&[&d, &x], &[])?;
            match into {
                SquashInto::MergeOfDAndX => m,
                _ => case.commit(&[&m], &[("g", line("g"))])?,
            }
        }
        SquashInto::C3WithC1 => {
            let c1 = case.commit(&[&d], &[("g", line("g"))])?;
            let c2 = case.commit(&[&c1], &[("h", line("h"))])?;
            sources.push(whole(&case, c1)?);
            case.commit(&[&c2], &[("k", line("k"))])?
        }
        SquashInto::BelowMergeOfThree | SquashInto::BelowMergeOfThreeWithIt => {
            let e = case.commit(&[&d], &[("g", line("g"))])?;
            let x = case.commit(&[&root], &[("h", line("h"))])?;
            let y = case.commit(&[&root], &[("k", line("k"))])?;
            let q = case.commit(&[&e, &x, &y], &[])?;
            let below = case.commit(&[&q], &[("g", line("g1"))])?;
            if matches!(into, SquashInto::BelowMergeOfThreeWithIt) {
                sources.push(whole(&case, q)?);
            }
            below
        }
        SquashInto::MergeOfTwoChildrenOfD => {
            let g = case.commit(&[&d], &[("g", line("g"))])?;
            let h = case.commit(&[&d], &[("h", line("h"))])?;
            case.commit(&[&g, &h], &[])?
        }
        SquashInto::ChildOfDWithAPart => {
            let mut builder = MergedTreeBuilder::new(p1.tree());
            builder.set_or_remove(repo_path_buf("f"), value_at(&p2.tree(), "f")?);
            sources[0].selected_tree = builder.write_tree().block_on()?;
            case.commit(&[&d], &[("g", line("g"))])?
        }
        SquashInto::BeforeD | SquashInto::DAfterBeforeD => {
            let n = case.commit(&[&p1, &p2], &[])?;
            rebase_commit(case.tx.repo_mut(), d.clone(), vec![n.id().clone()]).block_on()?;
            n
        }
        SquashInto::ChildOfP1 => case.commit(&[&p1], &[("g", line("g"))])?,
    };
    let squashed = squash_commits(case.tx.repo_mut(), &sources, &destination, false)
        .block_on()?
        .expect("a squash");
    let squashed = squashed.commit_builder.write().block_on()?;
    let read = if matches!(into, SquashInto::DAfterBeforeD) {
        case.tx.repo_mut().rebase_descendants().block_on()?;
        let store = case.store().clone();
        let heads = case.tx.repo().view().heads().iter();
        let heads: Vec<Commit> = heads.map(|id| store.get_commit(id)).try_collect()?;
        let is_d = |commit: &Commit| commit.change_id() == d.change_id();
        heads.into_iter().find(is_d).expect("d, rebased")
    } else {
        squashed
    };
    let value = value_at(&read.tree(), "f")?.simplify();
    Ok((case, value, read.tree()))
}

/// `destination` once all of `source` is squashed into it.
fn squash_all_of(case: &mut Case, source: Commit, destination: &Commit) -> TestResult<Commit> {
    let source = CommitWithSelection {
        parent_tree: source.parent_tree(case.tx.repo()).block_on()?,
        selected_tree: source.tree(),
        commit: source,
    };
    let squashed = squash_commits(case.tx.repo_mut(), &[source], destination, false)
        .block_on()?
        .expect("a squash");
    Ok(squashed.commit_builder.write().block_on()?)
}

/// Asserts that P1, a merge of the three merge bases of `history`, squashed
/// into the commit `into` names, leaves it with `x` or `y`, values that d or
/// its parents held, or a conflict, whichever merge base has the highest
/// commit id.
fn assert_squash_of_p1_keeps_a_held_value(history: ThreeBases, into: SquashInto) -> TestResult {
    for highest in 0..3 {
        let (case, value, _) = squash_p1(history, highest, into)?;
        let held = [case.text("x"), case.text("y")];
        assert!(
            value.as_resolved().is_none_or(|value| held.contains(value)),
            "{history:?} into {into:?} with merge base {highest} highest: {}",
            describe(&case, &value)
        );
    }
    Ok(())
}

/// Asserts that P1 squashed into the commit that each of `into` names keeps
/// a held value or a conflict in t3 and in t3x.
fn assert_squashes_of_p1_keep_a_held_value(into: &[SquashInto]) -> TestResult {
    for &into in into {
        assert_squash_of_p1_keeps_a_held_value(ThreeBases::T3, into)?;
        assert_squash_of_p1_keeps_a_held_value(ThreeBases::T3x, into)?;
    }
    Ok(())
}

/// t3: P1 squashed into d gives `y` or a conflict, never the merge bases' own
/// base `o`, which jj 0.45.1 gives.
#[test]
fn t3_squash_of_a_merge_of_three_into_its_child_keeps_a_held_value() -> TestResult {
    assert_squash_of_p1_keeps_a_held_value(ThreeBases::T3, SquashInto::D)
}

/// t3x: P1 squashed into d gives `y` or a conflict, never `o1`, the base of A
/// and B, which the fork's release from 4d332781 gave where B has the highest
/// commit id, so that A and C, both `x`, merge first.
#[test]
fn t3x_squash_of_a_merge_of_three_into_its_child_keeps_a_held_value() -> TestResult {
    assert_squash_of_p1_keeps_a_held_value(ThreeBases::T3x, SquashInto::D)
}

/// t3 and t3x: P1 squashed into a child of d, as `jj squash --insert-after d`
/// and a squash into a grandchild of P1 do, gives `y` or a conflict, never
/// `o` or `o1`: the rebase of the chain from d to the destination and the
/// change go into one merge.
#[test]
fn squash_of_a_merge_of_three_into_a_grandchild_keeps_a_held_value() -> TestResult {
    assert_squashes_of_p1_keep_a_held_value(&[SquashInto::ChildOfD])
}

/// The chain from P1 to the destination goes through a merge of d and a
/// commit that the squash does not rebase: the merge changes as d does, so
/// the one merge is that of a line of commits. `jj squash --insert-after d
/// --insert-after X` makes that merge itself the destination.
#[test]
fn squash_of_a_merge_of_three_through_a_merge_keeps_a_held_value() -> TestResult {
    assert_squashes_of_p1_keep_a_held_value(&[
        SquashInto::MergeOfDAndX,
        SquashInto::BelowMergeOfDAndX,
    ])
}

/// A commit of the chain is squashed with P1: its child takes the place of
/// the commit that the squash abandons, and the commit's own change, taken
/// out and put back, cancels.
#[test]
fn squash_of_a_merge_of_three_with_a_commit_of_the_chain_keeps_a_held_value() -> TestResult {
    assert_squashes_of_p1_keep_a_held_value(&[SquashInto::C3WithC1])
}

/// The chain goes through another merge of three commits, squashed or not.
#[test]
fn squash_of_a_merge_of_three_through_another_keeps_a_held_value() -> TestResult {
    assert_squashes_of_p1_keep_a_held_value(&[
        SquashInto::BelowMergeOfThree,
        SquashInto::BelowMergeOfThreeWithIt,
    ])
}

/// The destination merges two commits that the squash both rebases: the
/// merges of its parents before and after are the terms of the one merge.
#[test]
fn squash_of_a_merge_of_three_into_a_merge_of_its_descendants_keeps_a_held_value() -> TestResult {
    assert_squashes_of_p1_keep_a_held_value(&[SquashInto::MergeOfTwoChildrenOfD])
}

/// Only a part of P1 is selected, so the squash rewrites P1 without it: the
/// selected changes, taken out of P1 and put into the destination, cancel.
#[test]
fn squash_of_a_part_of_a_merge_of_three_keeps_a_held_value() -> TestResult {
    assert_squashes_of_p1_keep_a_held_value(&[SquashInto::ChildOfDWithAPart])
}

/// `jj squash --insert-before d`: the destination is a new merge of P1 and
/// P2, a conflict whose base is a virtual merge base, and the rebase that
/// takes P1 out of its parents cancels that base against d's own. The
/// squash's changes are made against the virtual merge base of P1's parents,
/// so the new commit holds a conflict or a held value, never `o`, which jj
/// 0.45.1 gives in t3, nor `o1`; and so does d, rebased onto it.
#[test]
fn squash_of_a_merge_of_three_into_a_commit_before_its_child_keeps_a_held_value() -> TestResult {
    assert_squashes_of_p1_keep_a_held_value(&[SquashInto::BeforeD, SquashInto::DAfterBeforeD])
}

/// c, a child of P1 alone: the squash abandons P1, so c's parents become P1's
/// own, and the rebase changes c's tree as the merge of those parents
/// changed from P1's tree. That merge is the term, with a virtual merge base,
/// against which P1's changes are made, so the two cancel, and c holds P1's
/// `x`, as jj 0.45.1 gives.
#[test]
fn squash_of_a_merge_of_three_into_a_child_of_it_alone_keeps_its_value() -> TestResult {
    for history in [ThreeBases::T3, ThreeBases::T3x] {
        for highest in 0..3 {
            let (case, value, _) = squash_p1(history, highest, SquashInto::ChildOfP1)?;
            assert_same_value(&case, &value, &resolved(case.text("x").as_ref()));
        }
    }
    Ok(())
}

/// Where the one merge of a squash leaves a conflict, the tree keeps conflict
/// labels: the destination's own, and the one merge's for the terms that it
/// brings.
#[test]
fn squash_of_a_merge_of_three_keeps_conflict_labels() -> TestResult {
    let (case, value, tree) = squash_p1(ThreeBases::T3, 0, SquashInto::BeforeD)?;
    assert!(!value.is_resolved(), "{}", describe(&case, &value));
    assert!(tree.labels().has_labels(), "{tree:?}");
    Ok(())
}

/// A squash of a merge of two commits keeps jj's squash, the change applied
/// to the rebased destination: in K4, P1 squashed into d, which holds `y`,
/// gives `o`, as jj 0.45.1 does: d rebased onto A and B holds `y`, and P1's
/// change against `[x, -o, y]` takes `y` back to `o`.
#[test]
fn k4_squash_of_a_merge_of_two_into_its_child_is_jjs_squash() -> TestResult {
    let TwoBases {
        mut case, p1, p2, ..
    } = k4_history(Made::AFirst)?;
    let d = case.commit(&[&p1, &p2], &[("f", line("y"))])?;
    let squashed = squash_all_of(&mut case, p1, &d)?;
    let value = value_at(&squashed.tree(), "f")?;
    assert_same_value(&case, &value, &resolved(case.text("o").as_ref()));
    Ok(())
}

/// SQ3: with no virtual merge base, a squash of a merge of three commits into
/// its child is jj's squash: S3, a merge of A, B and C that sets line 1 of
/// `f` to `S`, squashed into d3, its child that sets it back to `1`, gives
/// `S`, as jj 0.45.1 gives.
#[test]
fn sq3_squash_of_a_merge_of_three_with_no_virtual_base_is_jjs_squash() -> TestResult {
    let mut case = Case::new();
    let root = case.root();
    let o = case.commit(&[&root], &[("f", lines("1 2 3 4 5"))])?;
    let a = case.commit(&[&o], &[("a", line("a"))])?;
    let b = case.commit(&[&o], &[("b", line("b"))])?;
    let c = case.commit(&[&o], &[("c", line("c"))])?;
    let s3 = case.commit(&[&a, &b, &c], &[("f", lines("S 2 3 4 5"))])?;
    let d3 = case.commit(&[&s3], &[("f", lines("1 2 3 4 5"))])?;
    let squashed = squash_all_of(&mut case, s3, &d3)?;
    let value = value_at(&squashed.tree(), "f")?;
    let expected = case.file(
        f(),
        &lines("S 2 3 4 5").unwrap(),
        false,
        CopyId::placeholder(),
    );
    assert_same_value(&case, &value, &resolved(expected.as_ref()));
    Ok(())
}

/// An edit of a block's side, in the base of a conflict that the person
/// kept, after which V's terms would merge, is not made: the base's lines
/// are VB's, and the conflict's value stays as it was.
#[test]
fn snapshot_of_a_base_edit_after_which_v_would_resolve_keeps_the_value() -> TestResult {
    let TwoBases {
        mut case, p1, p2, ..
    } = six_line_history("x", "y 2 3")?;
    let w = case.commit(&[&p1, &p2], &[])?;
    let before = value_at(&w.tree(), "f")?;
    let value = snapshot_edits(&case, &w, &[("-y", "-x")])?;
    assert_same_value(&case, &value, &before);
    Ok(())
}

/// An edit of a block's side to the text of V's base, after which V's terms
/// would cancel to one side's text, is not made either.
#[test]
fn snapshot_of_a_base_edit_after_which_v_would_cancel_keeps_the_value() -> TestResult {
    let TwoBases {
        mut case, p1, p2, ..
    } = six_line_history("x", "y 2 3")?;
    let w = case.commit(&[&p1, &p2], &[])?;
    let before = value_at(&w.tree(), "f")?;
    let value = snapshot_edits(&case, &w, &[("-y", "-1")])?;
    assert_same_value(&case, &value, &before);
    Ok(())
}

/// VB6 with `x2` changed in the working copy. The merge of P1 and P2 made
/// `x1 Q x2` of the first side of VB's block, so the change is to what that
/// merge made of the block: the lines that the person left for the block go
/// into every term of V, and u, rebased onto s, holds the change in every
/// term.
#[test]
fn vb6_an_edit_of_what_the_merge_made_of_a_block_reaches_every_term() -> TestResult {
    let TwoBases {
        mut case, p1, p2, ..
    } = vb6_history()?;
    let w = case.commit(&[&p1, &p2], &[])?;
    let new_value = snapshot_edits(&case, &w, &[("x2", "EDIT")])?;
    let w = case.commit_with_values(&[&p1, &p2], vec![(repo_path_buf("f"), new_value)])?;
    let s = case.commit(&[&w], &[("f", lines("x1 Q EDIT 2 3 4 5 6 7 8 9 10"))])?;
    let u = case.commit(&[&w], &[("f", lines("x1 Q EDIT W 2 3 4 5 6 7 8 9 10"))])?;
    let rebased = rebase_commit(case.tx.repo_mut(), u, vec![s.id().clone()]).block_on()?;
    let value = value_at(&rebased.tree(), "f")?.simplify();
    assert_every_term_holds(&case, &value, "EDIT");
    assert_no_side_marked(&case, &value);
    Ok(())
}

/// A line that every side of a conflict of binary files holds, changed in
/// each side of the file that jj materialized. V's terms are files that are
/// not text, so the base is M, which shows none of them: the change goes
/// into each term where a line merge of them puts that line.
#[test]
fn snapshot_of_a_shared_edit_under_a_binary_m_edits_every_file_term() -> TestResult {
    let content =
        |value: &str| Some([binary(value).unwrap(), lines("2 3 4 5 6").unwrap()].concat());
    let TwoBases {
        mut case, p1, p2, ..
    } = two_bases(
        &[("f", content("o"))],
        &[("f", content("x"))],
        &[("f", content("y"))],
        Made::AFirst,
        &[("f", content("x"))],
        &[("f", content("y"))],
    )?;
    let w = case.commit(&[&p1, &p2], &[])?;
    // Line 5 of both sides, in one snapshot: an added line of the diff from
    // the base to the first side, and a line of the second side.
    let edited = snapshot_text(&case, &w, |working_copy, _| {
        working_copy
            .replacen("\n+5\n", "\n+FIVE\n", 1)
            .replacen("\n5\n", "\nFIVE\n", 1)
    })?
    .simplify();
    assert_no_side_marked(&case, &edited);
    let removes = edited.removes().collect_vec();
    let [base] = removes[..] else {
        panic!("expected one base, got {}", describe(&case, &edited));
    };
    let base_content = case.read(base.as_ref()).expect("the base is a file");
    let terms = virtual_base_terms(&base_content).expect("the base is VB or M");
    assert_every_term_holds(&case, &terms, "FIVE");
    Ok(())
}

/// A file whose last conflict's end marker has no newline ends every term of
/// that conflict without its last newline, as jj's parse of the markers
/// reads it: the person removed the file's final newline, which a merge tool
/// does as well, a change of every term of V alike. V's terms then hold the
/// person's other change, and end with no newline.
#[test]
fn a_removed_final_newline_reaches_every_term_of_v() -> TestResult {
    let TwoBases {
        mut case, p1, p2, ..
    } = two_bases(
        &[("f", lines("1 2 3 4 5 6"))],
        &[("f", lines("1 2 3 4 5 x"))],
        &[("f", lines("1 2 3 4 5 y"))],
        Made::AFirst,
        &[("f", lines("1 2 3 4 5 x"))],
        &[("f", lines("1 2 3 4 5 y"))],
    )?;
    let w = case.commit(&[&p1, &p2], &[])?;
    let value = snapshot_text(&case, &w, |working_copy, _| {
        let edited = working_copy.replacen("\n3\n", "\nTHREE\n", 1);
        edited
            .strip_suffix('\n')
            .expect("a final newline")
            .to_owned()
    })?;
    assert_eq!(
        named_term_texts(&case, &value.simplify())
            .into_iter()
            .sorted()
            .collect_vec(),
        [
            "1\n2\nTHREE\n4\n5\n6",
            "1\n2\nTHREE\n4\n5\nx",
            "1\n2\nTHREE\n4\n5\ny"
        ]
    );
    Ok(())
}

/// md (A and P1 delete `f`, so the base of w's conflict is M) with the
/// file's final newline removed: every side ends without one, and so does
/// each term of V that had text. In a child of w, a line appended after the
/// conflict, where jj shows each side with a newline that it lacks, the
/// empty one included, goes into every term of V.
#[test]
fn a_line_appended_after_sides_without_a_final_newline_reaches_every_term() -> TestResult {
    let TwoBases {
        mut case, p1, p2, ..
    } = two_bases(
        &[("f", lines("o 2 3"))],
        &[("f", None)],
        &[("f", lines("y 2 3"))],
        Made::AFirst,
        &[("f", None)],
        &[("f", lines("y 2 3"))],
    )?;
    let w = case.commit(&[&p1, &p2], &[])?;
    let value = snapshot_text(&case, &w, |working_copy, _| {
        working_copy
            .strip_suffix('\n')
            .expect("a final newline")
            .to_owned()
    })?;
    let w = case.commit_with_values(&[&p1, &p2], vec![(repo_path_buf("f"), value)])?;
    let c = case.commit(&[&w], &[])?;
    let value = snapshot_text(&case, &c, |working_copy, _| {
        let newline = if working_copy.ends_with('\n') {
            ""
        } else {
            "\n"
        };
        format!("{working_copy}{newline}EDIT\n")
    })?
    .simplify();
    let removes = value.removes().collect_vec();
    let [base] = removes[..] else {
        panic!("expected one base, got {}", describe(&case, &value));
    };
    let base_content = case.read(base.as_ref()).expect("the base is a file");
    let terms = virtual_base_terms(&base_content).expect("the base is VB or M");
    assert_every_term_holds(&case, &terms, "EDIT");
    Ok(())
}

/// md: A and P1 delete `f`, so V, the conflict of the merge bases, has an
/// absent term, and the base of w's conflict is M. The person pastes M itself
/// at the top of the file, so the deleted side holds M byte for byte. No VB
/// or M can name a V one of whose sides is M, as none would read back: the
/// base stays M, and the paste is in w's sides, the first of which is then
/// M's own file. s keeps w's first side, M's text, and u its second; u,
/// rebased onto s, is s's text, with no new virtual base as its value.
#[test]
fn a_paste_of_m_into_a_deleted_side_keeps_m_as_the_base() -> TestResult {
    let TwoBases {
        mut case, p1, p2, ..
    } = two_bases(
        &[("f", lines("o 2 3"))],
        &[("f", None)],
        &[("f", lines("y 2 3"))],
        Made::AFirst,
        &[("f", None)],
        &[("f", lines("y 2 3"))],
    )?;
    let w = case.commit(&[&p1, &p2], &[])?;
    let before = value_at(&w.tree(), "f")?.simplify();
    let m = case
        .read(before.removes().next().expect("a base").as_ref())
        .expect("M is a file");
    let m = String::from_utf8(m)?;
    let value = snapshot_text(&case, &w, |working_copy, _| format!("{m}{working_copy}"))?;
    for term in &value {
        let content = case.read(term.as_ref()).unwrap_or_default();
        assert!(
            content == m.as_bytes() || virtual_base_terms(&content).is_none(),
            "a term other than M reads as VB or M: {}",
            describe(&case, &value)
        );
    }
    let w = case.commit_with_values(&[&p1, &p2], vec![(repo_path_buf("f"), value)])?;
    let s = case.commit(&[&w], &[("f", Some(m.clone().into_bytes()))])?;
    let u = case.commit(&[&w], &[("f", Some(format!("{m}y\n2\n3\n").into_bytes()))])?;
    let rebased = rebase_commit(case.tx.repo_mut(), u, vec![s.id().clone()]).block_on()?;
    let value = value_at(&rebased.tree(), "f")?.simplify();
    assert_same_value(&case, &value, &value_at(&s.tree(), "f")?);
    Ok(())
}

/// nest: the merge bases of A and B, L1 and L2, are themselves a criss-cross,
/// so a term of V, the conflict of A and B's merge bases, is itself a
/// virtual base. A snapshot with line 5 changed makes the change to the
/// terms that the inner virtual base names, and never to its text: every
/// file that the conflict's base names, at any depth, is a virtual base, or
/// holds the change and no marker line.
#[test]
fn snapshot_of_an_edit_recurses_into_a_term_that_is_a_virtual_base() -> TestResult {
    let mut case = Case::new();
    let root = case.root();
    let text = |first: &str| lines(&format!("{first} 2 3 4 5 6"));
    let o = case.commit(&[&root], &[("f", text("o"))])?;
    let x = case.commit(&[&o], &[("f", text("x"))])?;
    let y = case.commit(&[&o], &[("f", text("y"))])?;
    let l1 = case.commit(&[&x, &y], &[("f", text("x"))])?;
    let l2 = case.commit(&[&x, &y], &[("f", text("y"))])?;
    let a = case.commit(&[&l1, &l2], &[("f", text("a"))])?;
    let b = case.commit(&[&l1, &l2], &[("f", text("b"))])?;
    let p1 = case.commit(&[&a, &b], &[("f", text("a"))])?;
    let p2 = case.commit(&[&a, &b], &[("f", text("b"))])?;
    let w = case.commit(&[&p1, &p2], &[])?;
    let value = snapshot_edits(&case, &w, &[("5", "FIVE")])?.simplify();
    let mut named = value.removes().cloned().collect_vec();
    let mut virtual_bases = 0;
    while let Some(term) = named.pop() {
        let content = case.read(term.as_ref()).expect("a file");
        match virtual_base_terms(&content) {
            Some(terms) => {
                virtual_bases += 1;
                named.extend(terms.iter().cloned());
            }
            None => {
                let text = String::from_utf8_lossy(&content);
                assert!(!holds_any_marker(&content), "a marker line in {text:?}");
                assert!(
                    text_lines(&content).contains(&"FIVE"),
                    "no FIVE in {text:?}"
                );
            }
        }
    }
    assert!(
        virtual_bases >= 2,
        "expected a virtual base among the terms that the base names: {}",
        describe(&case, &value)
    );
    Ok(())
}

/// Q1: a user's file that quotes a virtual base that a merge in this
/// repository wrote merges as any text does: two commits that edit it outside
/// the quote merge clean, with both edits, and so does a rebase of one onto
/// the other.
#[test]
fn user_file_quoting_a_virtual_base_merges_as_text() -> TestResult {
    let Merged { mut case, tree, .. } = case_k4(Made::AFirst, Order::AsNamed)?;
    let vb = virtual_base_term(&case, &value_at(&tree, "f")?);
    let vb = String::from_utf8(case.read(vb.as_ref()).expect("VB is a file"))?;
    let quote = |intro: &str, outro: &str| {
        Some(format!("{intro}\nExample of a jj conflict base:\n{vb}{outro}\n").into_bytes())
    };
    let root = case.root();
    let o = case.commit(&[&root], &[("f", quote("intro", "outro"))])?;
    let a = case.commit(&[&o], &[("f", quote("INTRO", "outro"))])?;
    let b = case.commit(&[&o], &[("f", quote("intro", "OUTRO"))])?;
    let expected = String::from_utf8(quote("INTRO", "OUTRO").unwrap())?;
    let rebased = rebase_commit(case.tx.repo_mut(), b.clone(), vec![a.id().clone()]).block_on()?;
    let merged = Merged::new(case, &[&a, &b])?;
    assert_resolved_text(&merged, &expected)?;
    let merged = Merged {
        tree: rebased.tree(),
        ..merged
    };
    assert_resolved_text(&merged, &expected)
}

/// Q2: a 160 KB user file that quotes a 3,000-line virtual base, with one line
/// of a side changed, merges as text, with both edits: the quote does not pass
/// as a virtual base, and a merge looks for one only among the terms of a
/// conflict, never in merged text. Its time is merge-cases.sh's to check, on a
/// release build.
#[test]
fn large_user_file_quoting_a_virtual_base_merges_as_text() -> TestResult {
    let side = |name: &str| {
        let text = (1..=1500)
            .map(|i| format!("{name}-side-line-{i:04}-of-the-merge-base"))
            .join(" ");
        lines(&text)
    };
    let TwoBases { case, p1, p2, .. } = two_bases(
        &[("f", side("o"))],
        &[("f", side("a"))],
        &[("f", side("b"))],
        Made::AFirst,
        &[("f", side("a"))],
        &[("f", side("b"))],
    )?;
    let Merged { mut case, tree, .. } = Merged::new(case, &[&p1, &p2])?;
    let vb = virtual_base_term(&case, &value_at(&tree, "f")?);
    let vb = String::from_utf8(case.read(vb.as_ref()).expect("VB is a file"))?;
    assert!(vb.lines().count() >= 3000, "VB has fewer than 3,000 lines");
    let quoted = vb.replacen(
        "\na-side-line-0700-of-the-merge-base\n",
        "\nA-side-line-0700-of-the-merge-base\n",
        1,
    );
    assert_ne!(quoted, vb);
    let padding = |from: usize| {
        let mut text = (from..from + 1700)
            .map(|i| format!("padding-line-{i:05}"))
            .join("\n");
        text.push('\n');
        text
    };
    let text = |top: &str, bottom: &str| {
        format!("{top}\n{}{quoted}{}{bottom}\n", padding(0), padding(1700))
    };
    assert!(text("top", "bottom").len() >= 160 * 1024);
    assert!(virtual_base_terms(text("TOP", "BOTTOM").as_bytes()).is_none());
    let root = case.root();
    let o = case.commit(&[&root], &[("f", Some(text("top", "bottom").into()))])?;
    let a = case.commit(&[&o], &[("f", Some(text("TOP", "bottom").into()))])?;
    let b = case.commit(&[&o], &[("f", Some(text("top", "BOTTOM").into()))])?;
    let merged = Merged::new(case, &[&a, &b])?;
    assert_resolved_text(&merged, &text("TOP", "BOTTOM"))
}

/// Q3: VB1a with 3,000-line sides: d, resolved to B's text, rebased onto P1
/// is a conflict of P1's and d's text against o's: the virtual base, which
/// the rebase makes a side, is replaced with V's terms, so no side and no base
/// is its text. Its time is merge-cases.sh's to check, on a release build.
#[test]
fn vb1a_with_long_sides_conflicts() -> TestResult {
    let text = |name: &str| (1..=3000).map(|i| format!("{name}{i}")).join(" ");
    let history = two_bases(
        &[("f", lines(&text("o")))],
        &[("f", lines(&text("x")))],
        &[("f", lines(&text("y")))],
        Made::AFirst,
        &[("f", lines(&text("x")))],
        &[("f", lines(&text("y")))],
    )?;
    let resolved = resolved_merge(history, &[("f", lines(&text("y")))])?;
    let merged = rebase_onto_p1(resolved)?;
    let o_text = String::from_utf8(lines(&text("o")).unwrap())?;
    assert_conflict(&merged, Base::Text(o_text))
}

/// Where a merge leaves virtual bases as sides, it replaces every virtual base
/// at the path, nested ones too, whatever the order of the merge's parents.
/// `outer` is the virtual base of a conflict whose base is `inner`; `inner`
/// and `other` name neither one the other. Each pair is the base of a
/// conflict, as two of its bases, which a merge of its two sides makes sides.
#[test]
fn virtual_bases_expand_whatever_the_order_of_the_parents() -> TestResult {
    let mut case = Case::new();
    let root = case.root();
    let [o, p, q, s1, s2, s3, x, y] =
        ["o", "p", "q", "s1", "s2", "s3", "x", "y"].map(|value| case.text(value));
    let vb = |v: &MergedTreeValue| -> TestResult<Option<TreeValue>> {
        let (side1, side3) = (resolved(s1.as_ref()), resolved(s3.as_ref()));
        Ok(Some(
            virtual_merge_base(case.store(), f(), v, &unlabeled(), &side1, &side3).block_on()?,
        ))
    };
    let inner = vb(&conflict(&[&x, &o, &y]))?;
    let outer = vb(&conflict(&[&p, &inner, &q]))?;
    let other = vb(&conflict(&[&y, &o, &x]))?;
    assert_ne!(inner, other);
    let side1 = case.commit(&[&root], &[("f", line("s1"))])?;
    let side3 = case.commit(&[&root], &[("f", line("s3"))])?;
    let both = conflict(&[&x, &o, &y, &s2, &y, &o, &x]);
    let cases = [
        ("outer and inner", &outer, &inner, conflict(&[&p, &s2, &q])),
        ("inner and outer", &inner, &outer, conflict(&[&p, &s2, &q])),
        ("inner and other", &inner, &other, both.clone()),
        ("other and inner", &other, &inner, both),
    ];
    for (name, first, second, expected) in cases {
        let base_value = conflict(&[&s1, first, &s2, second, &s3]);
        let base = case.commit_with_values(&[&root], vec![(repo_path_buf("f"), base_value)])?;
        for (order, left, right) in [("as named", &side1, &side3), ("swapped", &side3, &side1)] {
            let tree = MergedTree::merge(Merge::from_vec(vec![
                (left.tree(), "left".to_owned()),
                (base.tree(), "base".to_owned()),
                (right.tree(), "right".to_owned()),
            ]))
            .block_on()?;
            let value = value_at(&tree, "f")?;
            assert!(
                same_terms(&value, &expected),
                "{name}, {order}: expected {}, got {}",
                describe(&case, &expected),
                describe(&case, &value)
            );
        }
    }
    Ok(())
}

/// A stated limit, which the fix keeps: m is a conflicted merge of a and b; c1,
/// a child of m, holds `b`, and c2 holds `a`. c1 rebased onto c2 holds `o`, the
/// base of m's conflict, with no conflict, as jj 0.45.1 gives.
#[test]
fn rebase_across_a_conflicted_parent_gives_the_base() -> TestResult {
    let ConflictedBase { mut case, m, .. } = conflicted_base("a", "b")?;
    let c1 = case.commit(&[&m], &[("f", line("b"))])?;
    let c2 = case.commit(&[&m], &[("f", line("a"))])?;
    let rebased =
        rebase_commit(case.tx.repo_mut(), c1.clone(), vec![c2.id().clone()]).block_on()?;
    let merged = Merged {
        case,
        parents: vec![c2, c1],
        tree: rebased.tree(),
    };
    assert_resolved_text(&merged, "o\n")
}

/// S2's history: o holds `o`, A `x` and B `y`; P1, a merge of A and B,
/// resolves `f` to `X`, P2, another, holds `p2`, and d, the merge of P1 and
/// P2, holds `d`.
fn s2_history(p2: &str, d: &str) -> TestResult<(TwoBases, Commit)> {
    let mut history = two_bases(
        &[("f", line("o"))],
        &[("f", line("x"))],
        &[("f", line("y"))],
        Made::AFirst,
        &[("f", line("X"))],
        &[("f", line(p2))],
    )?;
    let d = history
        .case
        .commit(&[&history.p1, &history.p2], &[("f", line(d))])?;
    Ok((history, d))
}

/// S2a, a stated limit, which the fix keeps: the spec's S2 limit, jj 0.45.1's
/// rebase of one commit. d, which resolves to `o` the merge of P1 (`X`) and P2
/// (`y`), rebased onto P1 alone holds `x`, with no conflict, as jj 0.45.1
/// gives. A fix that changes this updates the spec's S2 limit.
#[test]
fn s2a_rebase_onto_one_parent_gives_x() -> TestResult {
    assert_resolved_text(&rebase_onto_p1(s2_history("y", "o")?)?, "x\n")
}

/// S2b, a stated limit, which the fix keeps: the spec's S2 limit, jj 0.45.1's
/// rebase of one commit. d, as in S2a, squashed into P1, as `jj squash --from
/// d --into P1 -u` does, leaves P1 holding `x`, with no conflict, as jj 0.45.1
/// gives. A fix that changes this updates the spec's S2 limit.
#[test]
fn s2b_squash_into_one_parent_gives_x() -> TestResult {
    assert_resolved_text(&squash_into_p1(s2_history("y", "o")?)?, "x\n")
}

/// S2c, a stated limit, which the fix keeps: the spec's S2 limit, jj 0.45.1's
/// rebase of one commit. With P2 and d holding `Y`, d rebased onto P1 alone is
/// a conflict whose sides are `x` and `y`, as jj 0.45.1 gives. A fix that
/// changes this updates the spec's S2 limit.
#[test]
fn s2c_rebase_onto_one_parent_conflicts_x_and_y() -> TestResult {
    let merged = rebase_onto_p1(s2_history("Y", "Y")?)?;
    let value = merged.value()?.simplify();
    let [x, y] = ["x", "y"].map(|value| merged.case.text(value));
    assert_eq!(
        value.adds().counts(),
        [&x, &y].into_iter().counts(),
        "expected a conflict whose sides are x and y, got {}",
        describe(&merged.case, &value)
    );
    Ok(())
}

/// A user's text that holds lines like a virtual base's marker lines, and a
/// whole block of them, with a salt and a term, whose hex no merge wrote.
const USER_MARKER_TEXT: &str = "1\n<<<<<<< jj virtual merge base 0123456789abcdef0123456789abcdef \
                                0 absent\n2\n======= 0123456789abcdef0123456789abcdef\n3\n>>>>>>> \
                                0123456789abcdef0123456789abcdef\n4\n";

/// A file that holds marker lines that no merge wrote merges as any text does:
/// in a merge, in a rebase, and where a rebase cancels every other term of a
/// conflict whose base it is.
#[test]
fn user_marker_lines_merge_as_text() -> TestResult {
    let a_text = USER_MARKER_TEXT.replacen("1\n", "1a\n", 1);
    let b_text = USER_MARKER_TEXT.replacen("4\n", "4b\n", 1);
    let merged_text = a_text.replacen("4\n", "4b\n", 1);

    let mut case = Case::new();
    let root = case.root();
    let o = case.commit(&[&root], &[("f", Some(USER_MARKER_TEXT.into()))])?;
    let a = case.commit(&[&o], &[("f", Some(a_text.clone().into_bytes()))])?;
    let b = case.commit(&[&o], &[("f", Some(b_text.clone().into_bytes()))])?;
    let rebased = rebase_commit(case.tx.repo_mut(), b.clone(), vec![a.id().clone()]).block_on()?;
    let merged = Merged::new(case, &[&a, &b])?;
    assert_resolved_text(&merged, &merged_text)?;
    let merged = Merged {
        tree: rebased.tree(),
        ..merged
    };
    assert_resolved_text(&merged, &merged_text)?;

    let mut case = Case::new();
    let root = case.root();
    let o = case.commit(&[&root], &[("f", Some(USER_MARKER_TEXT.into()))])?;
    let a = case.commit(&[&o], &[("f", line("A"))])?;
    let b = case.commit(&[&o], &[("f", line("B"))])?;
    let m = case.commit(&[&a, &b], &[])?;
    let c1 = case.commit(&[&m], &[("f", line("B"))])?;
    let c2 = case.commit(&[&m], &[("f", line("A"))])?;
    let rebased =
        rebase_commit(case.tx.repo_mut(), c1.clone(), vec![c2.id().clone()]).block_on()?;
    let merged = Merged {
        case,
        parents: vec![c2, c1],
        tree: rebased.tree(),
    };
    assert_resolved_text(&merged, USER_MARKER_TEXT)
}

/// Two conflicts at one path whose V is not a conflict of text files, with
/// different V, get different M: where a later merge meets both, it keeps a
/// conflict rather than cancel one M against the other as unchanged.
#[test]
fn non_text_conflicts_keep_their_own_m() -> TestResult {
    let mut case = Case::new();
    let root = case.root();
    let [o, y, z] = ["o", "y", "z"].map(|value| case.text(value));
    let m_of = |v: &MergedTreeValue, parent2: &MergedTreeValue| {
        virtual_merge_base(
            case.store(),
            f(),
            v,
            &unlabeled(),
            &Merge::absent(),
            parent2,
        )
        .block_on()
    };
    let m1 = Some(m_of(&conflict(&[&None, &o, &y]), &resolved(y.as_ref()))?);
    let m2 = Some(m_of(&conflict(&[&None, &o, &z]), &resolved(z.as_ref()))?);
    assert_ne!(m1, m2);

    // z1 holds `[absent, M1, y]` at `f`; w, its child, holds `[absent, M2, y]`.
    // Rebased onto `o`, w's merge `o + (M1 - y) + (y - M2)` would resolve to
    // `o` if M1 and M2 were one file. M1 is a side, so the merge replaces it
    // and M2 alike with the terms of V that each names, `[absent, o, y]` and
    // `[absent, o, z]`: the result is `o + y - z`.
    let base = case.commit(&[&root], &[("f", line("o"))])?;
    let z1_value = conflict(&[&None, &m1, &y]);
    let z1 = case.commit_with_values(&[&base], vec![(repo_path_buf("f"), z1_value)])?;
    let w_value = conflict(&[&None, &m2, &y]);
    let w = case.commit_with_values(&[&z1], vec![(repo_path_buf("f"), w_value)])?;
    let rebased = rebase_commit(case.tx.repo_mut(), w, vec![base.id().clone()]).block_on()?;
    let value = value_at(&rebased.tree(), "f")?;
    assert_same_value(&case, &value, &conflict(&[&o, &z, &y]));
    Ok(())
}

/// Each level of this history holds three commits that each merge the three
/// commits of the level below, so each step of a merge of merge bases asks for
/// V of the same three commits. The merge builds V once for each set of merge
/// bases, so the levels cost time in proportion to their number, where
/// building V at each step costs twice the time for each level.
#[test]
fn nested_merge_bases_built_once() -> TestResult {
    const NUM_LEVELS: usize = 24;
    let test_repo = TestRepo::init();
    let repo = &test_repo.repo;
    let mut tx = repo.start_transaction();
    let root = tx
        .repo_mut()
        .new_commit(
            vec![repo.store().root_commit_id().clone()],
            create_tree(repo, &[(f(), "o\n")]),
        )
        .set_description("r")
        .write_unwrap();
    let mut level = vec![root];
    for i in 1..=NUM_LEVELS {
        let parent_ids = level.iter().map(|commit| commit.id().clone()).collect_vec();
        level = ["a", "b", "c"]
            .iter()
            .map(|name| {
                tx.repo_mut()
                    .new_commit(
                        parent_ids.clone(),
                        create_tree(repo, &[(f(), &format!("{name}{i}\n"))]),
                    )
                    .set_description(format!("{name}{i}"))
                    .write_unwrap()
            })
            .collect();
    }

    let tops = [level[0].clone(), level[1].clone()];
    let merged = merge_commit_trees(tx.repo(), &tops).block_on()?;
    let value = value_at(&merged, "f")?.simplify();
    for top in &tops {
        let top_value = value_at(&top.tree(), "f")?;
        assert!(
            value.adds().any(|add| add == top_value.first()),
            "the conflict {value:?} does not hold {top_value:?}"
        );
    }
    Ok(())
}

/// The rule's value at `f` for V `base` and the parents' values.
fn rule(
    case: &Case,
    base: &MergedTreeValue,
    parent1: &MergedTreeValue,
    parent2: &MergedTreeValue,
) -> TestResult<MergedTreeValue> {
    Ok(merge_parent_values(case.store(), f(), base, &unlabeled(), parent1, parent2).block_on()?)
}

/// The virtual base of V `base` for the parents' values, and its content.
fn virtual_base(
    case: &Case,
    base: &MergedTreeValue,
    parent1: &MergedTreeValue,
    parent2: &MergedTreeValue,
) -> TestResult<(TreeValue, Vec<u8>)> {
    let value =
        virtual_merge_base(case.store(), f(), base, &unlabeled(), parent1, parent2).block_on()?;
    let content = case
        .read(Some(&value))
        .unwrap_or_else(|| panic!("expected the virtual base to be a file, got {value:?}"));
    Ok((value, content))
}

/// The one remove of the rule's value that holds a marker line.
fn virtual_base_term(case: &Case, value: &MergedTreeValue) -> Option<TreeValue> {
    let simplified = value.simplify();
    let terms = simplified
        .removes()
        .filter(|&term| {
            case.read(term.as_ref())
                .is_some_and(|content| holds_marker(&content))
        })
        .collect_vec();
    let [term] = terms[..] else {
        panic!("expected one virtual base among the removes of {value:?}");
    };
    term.clone()
}

/// Every line of every file term of `value`.
fn term_lines(case: &Case, value: &MergedTreeValue) -> Vec<String> {
    value
        .iter()
        .filter_map(|term| case.read(term.as_ref()))
        .flat_map(|content| {
            String::from_utf8_lossy(&content)
                .lines()
                .map(str::to_owned)
                .collect_vec()
        })
        .collect()
}

/// Both parents hold the same value, resolved or conflicted, where V resolves
/// and where it does not: the merge holds that value.
#[test]
fn rule_parents_equal() -> TestResult {
    let case = Case::new();
    let [a, b, c, d, o] = ["a", "b", "c", "d", "o"].map(|value| case.text(value));
    let (res_c, res_o) = (resolved(c.as_ref()), resolved(o.as_ref()));
    let k = conflict(&[&a, &o, &b]);
    let k2 = conflict(&[&c, &o, &d]);
    assert_same_value(&case, &rule(&case, &res_o, &res_c, &res_c)?, &res_c);
    assert_same_value(&case, &rule(&case, &res_o, &k, &k)?, &k);
    assert_same_value(&case, &rule(&case, &k, &res_c, &res_c)?, &res_c);
    assert_same_value(&case, &rule(&case, &k, &k2, &k2)?, &k2);
    Ok(())
}

/// V resolves: the merge is the three-way merge of the parents against V.
#[test]
fn rule_base_resolves_three_way() -> TestResult {
    let case = Case::new();
    let [a, b, c, o] = ["a", "b", "c", "o"].map(|value| case.text(value));
    let [res_a, res_b, res_c, res_o] = [&a, &b, &c, &o].map(|term| resolved(term.as_ref()));
    assert_same_value(
        &case,
        &rule(&case, &res_o, &res_a, &res_b)?,
        &conflict(&[&a, &o, &b]),
    );
    assert_same_value(&case, &rule(&case, &res_o, &res_a, &res_o)?, &res_a);
    assert_same_value(
        &case,
        &rule(&case, &res_o, &conflict(&[&a, &o, &b]), &res_c)?,
        &conflict(&[&a, &o, &b, &o, &c]),
    );
    Ok(())
}

/// Under `merge.same-change=keep`, two parents that hold the same value merge
/// three-way where V resolves, and the rule applies where V does not.
#[test]
fn rule_keep_same_change() -> TestResult {
    let case = Case::with_settings(&settings_with_same_change(SameChange::Keep));
    let [a, b, c, d, o, r] = ["a", "b", "c", "d", "o", "R"].map(|value| case.text(value));
    let [res_a, res_c, res_d, res_o, res_r] =
        [&a, &c, &d, &o, &r].map(|term| resolved(term.as_ref()));
    assert_same_value(
        &case,
        &rule(&case, &res_o, &res_a, &res_a)?,
        &conflict(&[&a, &o, &a]),
    );
    let v = conflict(&[&a, &o, &b]);
    assert_same_value(&case, &rule(&case, &v, &res_c, &res_c)?, &res_c);
    assert_same_value(&case, &rule(&case, &v, &v, &res_r)?, &res_r);
    let result = rule(&case, &v, &res_c, &res_d)?;
    let base_term = virtual_base_term(&case, &result);
    assert_same_value(&case, &result, &conflict(&[&c, &base_term, &d]));
    Ok(())
}

/// V does not resolve and one parent holds V: the merge holds the other
/// parent's value.
#[test]
fn rule_one_parent_unchanged() -> TestResult {
    let case = Case::new();
    let [a, b, c, d, o, r, x, y] =
        ["a", "b", "c", "d", "o", "R", "x", "y"].map(|value| case.text(value));
    let res_r = resolved(r.as_ref());
    let v = conflict(&[&a, &o, &b]);
    let k2 = conflict(&[&c, &o, &d]);
    assert_same_value(&case, &rule(&case, &v, &v, &res_r)?, &res_r);
    assert_same_value(&case, &rule(&case, &v, &res_r, &v)?, &res_r);
    assert_same_value(&case, &rule(&case, &v, &v, &k2)?, &k2);
    assert_same_value(&case, &rule(&case, &v, &k2, &v)?, &k2);
    assert_same_value(
        &case,
        &rule(&case, &v, &v, &Merge::absent())?,
        &Merge::absent(),
    );
    let v3 = conflict(&[&x, &o, &y, &o, &x]);
    assert_same_value(&case, &rule(&case, &v3, &v3, &res_r)?, &res_r);
    Ok(())
}

/// V does not resolve and both parents hold V: the merge holds V's own
/// conflict.
#[test]
fn rule_both_parents_unchanged() -> TestResult {
    let case = Case::new();
    let [a, b, o] = ["a", "b", "o"].map(|value| case.text(value));
    let v = conflict(&[&a, &o, &b]);
    assert_same_value(&case, &rule(&case, &v, &v, &v)?, &v);
    assert_same_value(&case, &rule(&case, &v, &v, &conflict(&[&b, &o, &a]))?, &v);
    Ok(())
}

/// A parent with V's adds and removes in another order, or with terms that
/// `simplify()` cancels, counts as unchanged.
#[test]
fn rule_equal_in_another_order() -> TestResult {
    let case = Case::new();
    let [a, b, c, o, r, x, y] = ["a", "b", "c", "o", "R", "x", "y"].map(|value| case.text(value));
    let res_r = resolved(r.as_ref());
    let v = conflict(&[&a, &o, &b]);
    let other_order = conflict(&[&b, &o, &a]);
    assert_same_value(&case, &rule(&case, &v, &other_order, &res_r)?, &res_r);
    assert_same_value(&case, &rule(&case, &v, &res_r, &other_order)?, &res_r);
    assert_same_value(
        &case,
        &rule(&case, &v, &conflict(&[&a, &o, &b, &c, &c]), &res_r)?,
        &res_r,
    );
    let v3 = conflict(&[&x, &o, &y, &o, &x]);
    assert_same_value(
        &case,
        &rule(&case, &v3, &conflict(&[&y, &o, &x, &o, &x]), &res_r)?,
        &res_r,
    );
    Ok(())
}

/// V is a conflict of text files and both parents changed it: the merge is
/// `[P1, VB, P2]`, flattened.
#[test]
fn rule_both_changed_text() -> TestResult {
    let case = Case::new();
    let [a, b, c, d, e, o] = ["a", "b", "c", "d", "e", "o"].map(|value| case.text(value));
    let v = conflict(&[&a, &o, &b]);
    let result = rule(&case, &v, &resolved(c.as_ref()), &resolved(d.as_ref()))?;
    assert!(!result.is_resolved(), "expected a conflict, got {result:?}");
    let base_term = virtual_base_term(&case, &result);
    assert_same_value(&case, &result, &conflict(&[&c, &base_term, &d]));
    assert_virtual_base(&case.read(base_term.as_ref()).unwrap(), &["a", "b"]);

    let k2 = conflict(&[&c, &o, &d]);
    let result = rule(&case, &v, &k2, &resolved(e.as_ref()))?;
    let base_term = virtual_base_term(&case, &result);
    assert_same_value(&case, &result, &conflict(&[&c, &o, &d, &base_term, &e]));
    Ok(())
}

/// V is not a conflict of text files and both parents changed it: the merge
/// is `[P1, M, P2]`, where M is one block that names each term of V, and whose
/// marker lines no line of either parent equals.
#[test]
fn rule_conflict_at_non_text() -> TestResult {
    let case = Case::new();
    let [o, x, y] = ["o", "x", "y"].map(|value| case.text(value));
    let [binary_o, binary_x, binary_y] = ["o", "x", "y"]
        .map(|value| case.file(f(), &binary(value).unwrap(), false, CopyId::placeholder()));
    let symlink = Some(TreeValue::Symlink(
        case.store().write_symlink(f(), "x").block_on()?,
    ));
    let submodule = Some(TreeValue::GitSubmodule(CommitId::new(vec![1; 20])));
    let directory = case.directory("x\n");
    let other_directory = case.directory("y\n");
    let cases = [
        ("a deletion", conflict(&[&None, &o, &y]), None, y.clone()),
        (
            "binary files",
            conflict(&[&binary_x, &binary_o, &binary_y]),
            binary_o.clone(),
            binary_y.clone(),
        ),
        (
            "a symlink",
            conflict(&[&symlink, &o, &y]),
            x.clone(),
            y.clone(),
        ),
        (
            "a submodule",
            conflict(&[&submodule, &o, &y]),
            x.clone(),
            y.clone(),
        ),
        (
            "a file against a directory",
            conflict(&[&directory, &None, &x]),
            y.clone(),
            other_directory.clone(),
        ),
    ];
    for (name, v, p1, p2) in cases {
        let (p1, p2) = (Merge::resolved(p1), Merge::resolved(p2));
        let result = rule(&case, &v, &p1, &p2)?;
        assert!(
            !result.is_resolved(),
            "{name}: expected a conflict, got {result:?}"
        );
        let simplified = result.simplify();
        for parent in [&p1, &p2] {
            assert!(
                simplified.adds().any(|add| add == parent.first()),
                "{name}: the conflict {simplified:?} does not hold {parent:?}"
            );
        }
        let removes = simplified.removes().collect_vec();
        let [remove] = removes[..] else {
            panic!("{name}: expected one remove, got {simplified:?}");
        };
        let content = case
            .read(remove.as_ref())
            .unwrap_or_else(|| panic!("{name}: expected M to be a file, got {remove:?}"));
        assert_marker_file(&content);
        let lines = text_lines(&content);
        let hex = opening_hex(&lines);
        let separator = format!("{SEPARATOR}{hex}");
        let is_marker = |line: &str| {
            [OPEN, SEPARATOR, CLOSE]
                .iter()
                .any(|prefix| marker_hex(line, prefix).is_some())
        };
        let num_names = lines.iter().filter(|&&line| !is_marker(line)).count();
        assert_eq!(
            num_names,
            v.simplify().iter().len(),
            "{name}: M does not name each term of V: {lines:?}"
        );
        let parent_lines = [term_lines(&case, &p1), term_lines(&case, &p2)].concat();
        for marker in [lines[0], lines[lines.len() - 1], separator.as_str()] {
            assert!(
                !parent_lines.iter().any(|line| line == marker),
                "{name}: M's marker line {marker:?} is a line of a parent"
            );
        }
    }
    Ok(())
}

/// At a hunk where V's terms merge, VB holds the merged text: E1's VB holds
/// `9a` where line 9 was.
#[test]
fn virtual_base_keeps_merged_hunks() -> TestResult {
    let case = Case::new();
    let file = |words: &str| case.file(f(), &lines(words).unwrap(), false, CopyId::placeholder());
    let v = conflict(&[
        &file("x 2 3 4 5 6 7 8 9a 10"),
        &file("1 2 3 4 5 6 7 8 9 10"),
        &file("y 2 3 4 5 6 7 8 9 10"),
    ]);
    let p1 = Merge::resolved(file("z 2 3 4 5 6 7 8 9 10"));
    let p2 = Merge::resolved(file("z 2 3 4 5 6 7 8 9a 10"));
    let (_, content) = virtual_base(&case, &v, &p1, &p2)?;
    let lines = text_lines(&content);
    let hex = opening_hex(&lines);
    let open = assert_names_terms(&content, &v);
    let expected = [
        open.to_owned(),
        "x".to_owned(),
        format!("{SEPARATOR}{hex}"),
        "y".to_owned(),
        format!("{CLOSE}{hex}"),
    ]
    .into_iter()
    .chain("2 3 4 5 6 7 8 9a 10".split(' ').map(str::to_owned))
    .collect_vec();
    assert_eq!(lines, expected);
    Ok(())
}

/// At a hunk where V's terms do not merge, VB holds V's sides, in V's order,
/// between marker lines; a hunk with three sides gets two separators.
#[test]
fn virtual_base_sides_between_markers() -> TestResult {
    let case = Case::new();
    let file = |words: &str| case.file(f(), &lines(words).unwrap(), false, CopyId::placeholder());

    let v = conflict(&[
        &file("x1 x2 2 3 4 5 6 7 8 9 10"),
        &file("1 2 3 4 5 6 7 8 9 10"),
        &file("y1 2 3 4 5 6 7 8 9 10"),
    ]);
    let p1 = Merge::resolved(file("x1 x2 2 3 4 5 6 7 8 9 10"));
    let p2 = Merge::resolved(file("x1 Q x2 2 3 4 5 6 7 8 9 10"));
    let (_, content) = virtual_base(&case, &v, &p1, &p2)?;
    let lines = text_lines(&content);
    let hex = opening_hex(&lines);
    let open = assert_names_terms(&content, &v);
    let expected = [
        open.to_owned(),
        "x1".to_owned(),
        "x2".to_owned(),
        format!("{SEPARATOR}{hex}"),
        "y1".to_owned(),
        format!("{CLOSE}{hex}"),
    ]
    .into_iter()
    .chain("2 3 4 5 6 7 8 9 10".split(' ').map(str::to_owned))
    .collect_vec();
    assert_eq!(lines, expected);

    let v = conflict(&[
        &file("x 2 3"),
        &file("1 2 3"),
        &file("y 2 3"),
        &file("1 2 3"),
        &file("z 2 3"),
    ]);
    let p1 = Merge::resolved(file("p 2 3"));
    let p2 = Merge::resolved(file("q 2 3"));
    let (_, content) = virtual_base(&case, &v, &p1, &p2)?;
    let lines = text_lines(&content);
    let hex = opening_hex(&lines);
    let open = assert_names_terms(&content, &v);
    let expected = [
        open.to_owned(),
        "x".to_owned(),
        format!("{SEPARATOR}{hex}"),
        "y".to_owned(),
        format!("{SEPARATOR}{hex}"),
        "z".to_owned(),
        format!("{CLOSE}{hex}"),
        "2".to_owned(),
        "3".to_owned(),
    ];
    assert_eq!(lines, expected);
    Ok(())
}

/// Where a line of any term of either parent equals a marker line, the salt
/// moves, so no marker line of VB or M is a line of a parent.
#[test]
fn virtual_base_markers_in_no_parent_line() -> TestResult {
    let case = Case::new();
    let file = |text: &str| case.file(f(), text.as_bytes(), false, CopyId::placeholder());
    let [c, d, e, o, x, y] = ["c", "d", "e", "o", "x", "y"].map(|value| case.text(value));
    let marker_lines = |content: &[u8]| {
        text_lines(content)
            .into_iter()
            .filter(|line| {
                [OPEN, SEPARATOR, CLOSE]
                    .iter()
                    .any(|prefix| marker_hex(line, prefix).is_some())
            })
            .map(str::to_owned)
            .collect_vec()
    };
    let assert_in_no_parent_line = |content: &[u8], parents: [&MergedTreeValue; 2]| {
        let parent_lines = parents
            .iter()
            .flat_map(|parent| term_lines(&case, parent))
            .collect_vec();
        for marker in marker_lines(content) {
            assert!(
                !parent_lines.contains(&marker),
                "the marker line {marker:?} is a line of a parent"
            );
        }
    };

    let v = conflict(&[&x, &o, &y]);
    let (_, salt_0) = virtual_base(&case, &v, &resolved(c.as_ref()), &resolved(d.as_ref()))?;
    let salt_0_lines = marker_lines(&salt_0);
    assert_eq!(
        salt_0_lines.len(),
        3,
        "expected three marker lines in {salt_0_lines:?}"
    );

    // P1 holds VB's opening marker line at salt 0.
    let p1 = Merge::resolved(file(&format!("c\n{}\n", salt_0_lines[0])));
    let p2 = resolved(d.as_ref());
    let (_, content) = virtual_base(&case, &v, &p1, &p2)?;
    assert_ne!(marker_lines(&content), salt_0_lines);
    assert_in_no_parent_line(&content, [&p1, &p2]);

    // A remove of P2, itself a conflict, holds VB's separator line at salt 0.
    let p1 = resolved(c.as_ref());
    let p2 = conflict(&[&d, &file(&format!("o\n{}\n", salt_0_lines[1])), &e]);
    let (_, content) = virtual_base(&case, &v, &p1, &p2)?;
    assert_ne!(marker_lines(&content), salt_0_lines);
    assert_in_no_parent_line(&content, [&p1, &p2]);

    // M: P2 holds M's line at salt 0.
    let v = conflict(&[&None, &o, &y]);
    let (_, m_salt_0) = virtual_base(&case, &v, &Merge::absent(), &resolved(y.as_ref()))?;
    assert_marker_file(&m_salt_0);
    let m_line = text_lines(&m_salt_0)[0].to_owned();
    let p2 = Merge::resolved(file(&format!("y\n{m_line}\n")));
    let (_, content) = virtual_base(&case, &v, &Merge::absent(), &p2)?;
    assert_marker_file(&content);
    assert_ne!(text_lines(&content)[0], m_line);
    assert_in_no_parent_line(&content, [&Merge::absent(), &p2]);
    Ok(())
}

/// V is a conflict of text files where each add is a regular file with no NUL
/// byte in its first 8,000 bytes and each remove is such a file or absent;
/// anything else gives M.
#[test]
fn virtual_base_text_test() -> TestResult {
    let case = Case::new();
    let [c, d, o, x, y] = ["c", "d", "o", "x", "y"].map(|value| case.text(value));
    let (p1, p2) = (resolved(c.as_ref()), resolved(d.as_ref()));
    let file = |content: &[u8]| case.file(f(), content, false, CopyId::placeholder());
    // `x`, a newline, then `a` up to the NUL byte at `index`.
    let nul_at = |index: usize| {
        let mut content = b"x\n".to_vec();
        content.resize(index, b'a');
        content.extend_from_slice(b"\0\n");
        file(&content)
    };
    let late_nul = nul_at(8000);
    let early_nul = nul_at(7999);
    let binary_o = file(b"o\0bin\n");
    let executable_x = case.file(f(), b"x\n", true, CopyId::placeholder());
    let symlink = Some(TreeValue::Symlink(
        case.store().write_symlink(f(), "x").block_on()?,
    ));
    let submodule = Some(TreeValue::GitSubmodule(CommitId::new(vec![1; 20])));
    let directory = case.directory("x\n");

    let text = [
        ("regular files", conflict(&[&x, &o, &y])),
        ("an executable file", conflict(&[&executable_x, &o, &y])),
        (
            "a NUL byte after 8,000 bytes",
            conflict(&[&late_nul, &o, &y]),
        ),
        ("an absent remove", conflict(&[&x, &None, &y])),
    ];
    for (name, v) in text {
        let (_, content) = virtual_base(&case, &v, &p1, &p2)?;
        assert!(
            is_virtual_base(&content) && !is_marker_file(&content),
            "{name}: expected a virtual base, got {:?}",
            String::from_utf8_lossy(&content)
        );
    }

    let not_text = [
        (
            "a NUL byte in the first 8,000 bytes",
            conflict(&[&early_nul, &o, &y]),
        ),
        ("a binary remove", conflict(&[&x, &binary_o, &y])),
        ("a symlink among the adds", conflict(&[&symlink, &o, &y])),
        ("a symlink as a remove", conflict(&[&x, &symlink, &y])),
        ("a deletion among the adds", conflict(&[&None, &o, &y])),
        (
            "a submodule among the adds",
            conflict(&[&submodule, &o, &y]),
        ),
        (
            "a directory among the adds",
            conflict(&[&directory, &None, &x]),
        ),
    ];
    for (name, v) in not_text {
        let (_, content) = virtual_base(&case, &v, &p1, &p2)?;
        assert!(
            is_marker_file(&content),
            "{name}: expected M, got {:?}",
            String::from_utf8_lossy(&content)
        );
    }
    Ok(())
}

/// An add/add V, whose remove is absent, is a conflict of text files: its
/// remove counts as an empty base, and VB holds both sides between markers.
#[test]
fn virtual_base_add_add() -> TestResult {
    let case = Case::new();
    let [c, d, x, y] = ["c", "d", "x", "y"].map(|value| case.text(value));
    let v = conflict(&[&x, &None, &y]);
    let (_, content) = virtual_base(&case, &v, &resolved(c.as_ref()), &resolved(d.as_ref()))?;
    let lines = text_lines(&content);
    let hex = opening_hex(&lines);
    let open = assert_names_terms(&content, &v);
    assert_eq!(
        lines,
        [
            open.to_owned(),
            "x".to_owned(),
            format!("{SEPARATOR}{hex}"),
            "y".to_owned(),
            format!("{CLOSE}{hex}"),
        ]
    );
    Ok(())
}

/// A stated limit: where V is unresolved only because of its terms'
/// executable bits, their contents merge, so VB holds no block, and no merge
/// can tell that it is VB.
#[test]
fn virtual_base_of_an_executable_bit_conflict_holds_no_block() -> TestResult {
    let case = Case::new();
    let [c, d, x] = ["c", "d", "x"].map(|value| case.text(value));
    let x_executable = case.file(f(), b"x\n", true, CopyId::placeholder());
    let v = conflict(&[&x_executable, &None, &x]);
    let (_, content) = virtual_base(&case, &v, &resolved(c.as_ref()), &resolved(d.as_ref()))?;
    assert_eq!(content, b"x\n");
    assert_eq!(virtual_base_terms(&content), None);
    Ok(())
}

/// VB and M do not depend on the order of the parents.
#[test]
fn virtual_base_independent_of_parent_order() -> TestResult {
    let case = Case::new();
    let [c, d, o, x, y] = ["c", "d", "o", "x", "y"].map(|value| case.text(value));
    let v = conflict(&[&x, &o, &y]);
    let (p1, p2) = (resolved(c.as_ref()), resolved(d.as_ref()));
    let (forward, content) = virtual_base(&case, &v, &p1, &p2)?;
    assert_eq!(virtual_base(&case, &v, &p2, &p1)?.0, forward);

    // With a parent that moves the salt.
    let open = text_lines(&content)[0].to_owned();
    let p1 = Merge::resolved(case.file(
        f(),
        format!("c\n{open}\n").as_bytes(),
        false,
        CopyId::placeholder(),
    ));
    assert_eq!(
        virtual_base(&case, &v, &p1, &p2)?.0,
        virtual_base(&case, &v, &p2, &p1)?.0
    );

    let v = conflict(&[&None, &o, &y]);
    let p1 = Merge::absent();
    let p2 = resolved(y.as_ref());
    assert_eq!(
        virtual_base(&case, &v, &p1, &p2)?.0,
        virtual_base(&case, &v, &p2, &p1)?.0
    );
    Ok(())
}

/// VB's executable bit is V's executable bits merged, or `false` where they do
/// not merge; its copy id is V's copy ids merged, or the placeholder.
#[test]
fn virtual_base_executable_and_copy_id() -> TestResult {
    let case = Case::new();
    let [c, d] = ["c", "d"].map(|value| case.text(value));
    let (p1, p2) = (resolved(c.as_ref()), resolved(d.as_ref()));
    let file = |text: &str, executable: bool, copy_id: &CopyId| {
        case.file(f(), &line(text).unwrap(), executable, copy_id.clone())
    };
    let placeholder = CopyId::placeholder();
    let attributes = |v: &MergedTreeValue| -> TestResult<(bool, CopyId)> {
        let (value, _) = virtual_base(&case, v, &p1, &p2)?;
        let TreeValue::File {
            executable,
            copy_id,
            ..
        } = value
        else {
            panic!("expected a file, got {value:?}");
        };
        Ok((executable, copy_id))
    };

    let v = conflict(&[
        &file("x", true, &placeholder),
        &file("o", true, &placeholder),
        &file("y", true, &placeholder),
    ]);
    assert_eq!(attributes(&v)?, (true, placeholder.clone()));
    let v = conflict(&[
        &file("x", true, &placeholder),
        &file("o", false, &placeholder),
        &file("y", false, &placeholder),
    ]);
    assert!(attributes(&v)?.0);
    let v = conflict(&[
        &file("x", true, &placeholder),
        &None,
        &file("y", false, &placeholder),
    ]);
    assert!(!attributes(&v)?.0);

    let copy_id = CopyId::new(b"copy".to_vec());
    let v = conflict(&[
        &file("x", false, &copy_id),
        &file("o", false, &copy_id),
        &file("y", false, &copy_id),
    ]);
    assert_eq!(attributes(&v)?.1, copy_id);
    let v = conflict(&[
        &file("x", false, &CopyId::new(b"copy-x".to_vec())),
        &file("o", false, &CopyId::new(b"copy-o".to_vec())),
        &file("y", false, &CopyId::new(b"copy-y".to_vec())),
    ]);
    assert_eq!(attributes(&v)?.1, placeholder);
    Ok(())
}

/// VB and M show that a merge wrote them, and name V's terms: the verifier
/// reads V's terms from the files that `virtual_merge_base` writes, and from no
/// text in which any byte differs (text around a block, a side, the text
/// outside the blocks, the salt), a block does not close, or no block is.
#[test]
fn virtual_base_verifies() -> TestResult {
    let case = Case::new();
    let [c, d, o, x, y] = ["c", "d", "o", "x", "y"].map(|value| case.text(value));
    let (p1, p2) = (resolved(c.as_ref()), resolved(d.as_ref()));
    let v = conflict(&[&x, &o, &y]);
    let (_, vb) = virtual_base(&case, &v, &p1, &p2)?;
    assert_names_terms(&vb, &v);
    let v_m = conflict(&[&None, &o, &y]);
    let (_, m) = virtual_base(&case, &v_m, &p1, &p2)?;
    assert_names_terms(&m, &v_m);

    // The path is not hashed: the same V at another path gives the same text.
    let g = repo_path("g");
    let [c_g, d_g, o_g, x_g, y_g] = ["c", "d", "o", "x", "y"]
        .map(|value| case.file(g, &line(value).unwrap(), false, CopyId::placeholder()));
    let (p1_g, p2_g) = (resolved(c_g.as_ref()), resolved(d_g.as_ref()));
    let v_g = conflict(&[&x_g, &o_g, &y_g]);
    let other = virtual_merge_base(case.store(), g, &v_g, &unlabeled(), &p1_g, &p2_g).block_on()?;
    let TreeValue::File { id, .. } = other else {
        panic!("expected the virtual base to be a file, got {other:?}");
    };
    assert_eq!(read_file(case.store(), g, &id), vb);

    let text = String::from_utf8(vb)?;
    let file = |words: &str| case.file(f(), &lines(words).unwrap(), false, CopyId::placeholder());
    let v_with_text = conflict(&[&file("x 2 3"), &file("1 2 3"), &file("y 2 3")]);
    let (_, with_text) = virtual_base(&case, &v_with_text, &p1, &p2)?;
    let with_text = String::from_utf8(with_text)?;
    assert_names_terms(with_text.as_bytes(), &v_with_text);
    let changes = [
        ("text around it", format!("before\n{text}after\n")),
        ("a changed side", text.replace("\nx\n", "\nz\n")),
        (
            "changed text after the block",
            with_text.replace("\n2\n", "\ntwo\n"),
        ),
        ("a changed salt", text.replacen(" 0 file:", " 1 file:", 1)),
        (
            "no closing line",
            text[..text.rfind(CLOSE).unwrap()].to_owned(),
        ),
        ("a user's marker lines", USER_MARKER_TEXT.to_owned()),
        ("no block", "no block\n".to_owned()),
    ];
    for (name, changed) in changes {
        assert!(
            virtual_base_terms(changed.as_bytes()).is_none(),
            "{name}: expected the verifier to refuse {changed:?}"
        );
    }
    Ok(())
}

/// A braid of criss-cross merges deeper than a merge that recursed on the
/// machine stack could go: `a<i>` and `b<i>` both merge `a<i-1>` and `b<i-1>`
/// and hold different text at `f`. The test runs itself again as a child
/// process with a small stack.
#[test]
fn braid_deeper_than_the_stack() -> TestResult {
    const CHILD_ENV: &str = "JJ_LIB_TEST_MERGE_RULES_BRAID_CHILD";
    const LEVELS_ENV: &str = "JJ_LIB_TEST_MERGE_RULES_BRAID_LEVELS";
    const CHILD_STACK_SIZE: usize = 512 * 1024;
    const DEFAULT_NUM_LEVELS: usize = 1000;

    let num_levels = std::env::var(LEVELS_ENV)
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(DEFAULT_NUM_LEVELS);
    if std::env::var_os(CHILD_ENV).is_some() {
        return run_braid(num_levels);
    }

    let output = Command::new(std::env::current_exe()?)
        .arg("--exact")
        .arg("test_merge_rules::braid_deeper_than_the_stack")
        .arg("--nocapture")
        .env(CHILD_ENV, "1")
        .env(LEVELS_ENV, num_levels.to_string())
        .env("RUST_MIN_STACK", CHILD_STACK_SIZE.to_string())
        .output()?;
    assert!(
        output.status.success(),
        "a braid of {num_levels} levels failed with a {CHILD_STACK_SIZE} byte child stack: \
         {}\nstdout:\n{}\nstderr:\n{}",
        output.status,
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr),
    );
    Ok(())
}

/// The top two commits of a braid of `num_levels` criss-cross merges over a
/// commit that holds `o` at `f`: `a<i>` and `b<i>` both merge `a<i-1>` and
/// `b<i-1>`, and hold `a<i>` and `b<i>`.
fn braid(tx: &mut Transaction, num_levels: usize) -> (Commit, Commit) {
    let repo = tx.base_repo().clone();
    let root = tx
        .repo_mut()
        .new_commit(
            vec![repo.store().root_commit_id().clone()],
            create_tree(&repo, &[(f(), "o\n")]),
        )
        .set_description("r")
        .write_unwrap();
    let (mut a, mut b) = (root.clone(), root);
    for i in 1..=num_levels {
        let (parents_a, parents_b) = if a.id() == b.id() {
            (vec![a.id().clone()], vec![b.id().clone()])
        } else {
            (
                vec![a.id().clone(), b.id().clone()],
                vec![b.id().clone(), a.id().clone()],
            )
        };
        let next_a = tx
            .repo_mut()
            .new_commit(parents_a, create_tree(&repo, &[(f(), &format!("a{i}\n"))]))
            .set_description(format!("a{i}"))
            .write_unwrap();
        let next_b = tx
            .repo_mut()
            .new_commit(parents_b, create_tree(&repo, &[(f(), &format!("b{i}\n"))]))
            .set_description(format!("b{i}"))
            .write_unwrap();
        (a, b) = (next_a, next_b);
    }
    (a, b)
}

fn run_braid(num_levels: usize) -> TestResult {
    let test_repo = TestRepo::init();
    let mut tx = test_repo.repo.start_transaction();
    let (a, b) = braid(&mut tx, num_levels);
    let merged = merge_commit_trees(tx.repo(), &[a.clone(), b.clone()]).block_on()?;
    let value = value_at(&merged, "f")?;
    let simplified = value.simplify();
    assert_eq!(
        simplified.num_sides(),
        2,
        "expected a 2-sided conflict at f, got {value:?}"
    );
    for top in [&a, &b] {
        let top_value = value_at(&top.tree(), "f")?;
        assert!(
            simplified.adds().any(|add| add == top_value.first()),
            "the conflict {simplified:?} does not hold {top_value:?}"
        );
    }

    let parent_tree = a.parent_tree(tx.repo()).block_on()?;
    if num_levels > 1 {
        assert!(!value_at(&parent_tree, "f")?.is_resolved());
    }
    Ok(())
}

/// A rewrite of the top merge of a braid of 8,000 levels, whose virtual base
/// names the virtual base of the level below, and so on down, replaces each of
/// them, fewer than a merge replaces at one path: d, a child of the merge of
/// the top two commits that sets `c`, rebased onto one of them, completes with
/// `c` among its sides and no marker line in any term.
#[test]
fn rewrite_of_a_braid_top_merge_completes() -> TestResult {
    const NUM_LEVELS: usize = 8000;
    let mut case = Case::new();
    let (a, b) = braid(&mut case.tx, NUM_LEVELS);
    let m = case.commit(&[&a, &b], &[])?;
    let d = case.commit(&[&m], &[("f", line("c"))])?;
    let c_value = value_at(&d.tree(), "f")?;
    let rebased = rebase_commit(case.tx.repo_mut(), d, vec![a.id().clone()]).block_on()?;
    let value = value_at(&rebased.tree(), "f")?;
    assert!(
        value.adds().any(|add| add == c_value.first()),
        "expected `c` among the sides at f"
    );
    let marked = value
        .iter()
        .filter(|term| {
            case.read(term.as_ref())
                .is_some_and(|content| holds_any_marker(&content))
        })
        .count();
    assert_eq!(marked, 0, "{marked} terms hold a marker line");
    Ok(())
}
