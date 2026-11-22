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
//! of V's terms, so the merge is a conflict. VB and M are only ever terms of a
//! conflict, also after a later rebase, abandon or squash (VB1). Merge bases,
//! and the parents of a merge of three or more commits, are merged in the
//! order of their commit ids.
//!
//! The case tests are named for the histories they build: K (the merge bases'
//! own merge is a conflict), D and MD (a merge base deletes the path), T (three
//! merge bases), H, E and X (text that git's recursive merge merges line by
//! line) and BIN (binary files). Each builds its history commit by commit, as
//! `jj new <parents>` followed by file writes makes it, in the order in which
//! a user would make the commits, so that the index lists the merge bases in
//! that order.

use std::process::Command;
use std::sync::Arc;

use itertools::Itertools as _;
use jj_lib::backend::CommitId;
use jj_lib::backend::CopyId;
use jj_lib::backend::MergedTreeValue;
use jj_lib::backend::MergedTreeValueExt as _;
use jj_lib::backend::TreeValue;
use jj_lib::commit::Commit;
use jj_lib::merge::Merge;
use jj_lib::merge::SameChange;
use jj_lib::merge_rule::is_virtual_base_content;
use jj_lib::merge_rule::merge_parent_values;
use jj_lib::merge_rule::virtual_merge_base;
use jj_lib::merged_tree::MergedTree;
use jj_lib::merged_tree_builder::MergedTreeBuilder;
use jj_lib::repo::Repo as _;
use jj_lib::repo_path::RepoPath;
use jj_lib::repo_path::RepoPathBuf;
use jj_lib::rewrite::RebaseOptions;
use jj_lib::rewrite::merge_commit_trees;
use jj_lib::rewrite::merge_commit_trees_no_resolve;
use jj_lib::rewrite::rebase_commit;
use jj_lib::settings::UserSettings;
use jj_lib::store::Store;
use jj_lib::transaction::Transaction;
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

/// The opening marker line of a hunk of the virtual base, before its hex.
const OPEN: &str = "<<<<<<< jj virtual merge base ";
/// The marker line between two sides of a hunk of the virtual base.
const SEPARATOR: &str = "======= ";
/// The closing marker line of a hunk of the virtual base.
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

/// The hex of `line` if it is a marker line that starts with `prefix`.
fn marker_hex<'a>(line: &'a str, prefix: &str) -> Option<&'a str> {
    let hex = line.strip_prefix(prefix)?;
    (hex.len() == 32 && hex.bytes().all(|byte| byte.is_ascii_hexdigit())).then_some(hex)
}

/// The hex of the opening marker line that starts `lines`.
#[track_caller]
fn opening_hex<'a>(lines: &[&'a str]) -> &'a str {
    marker_hex(lines[0], OPEN)
        .unwrap_or_else(|| panic!("expected an opening marker line first, got {lines:?}"))
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
    let names_a_term = |line: &str| {
        line == "absent"
            || ["file ", "symlink ", "tree ", "submodule "]
                .iter()
                .any(|kind| line.starts_with(kind))
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
        && is_virtual_base_content(content)
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
        && is_virtual_base_content(content)
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
/// value there, and whose one remove, after `simplify()`, is `base`.
fn assert_conflict(merged: &Merged, base: Base) -> TestResult {
    let simplified = conflict_at_f(merged)?;
    assert_parents_among_adds(merged, &simplified)?;
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
    let mut history = k4_history(Made::AFirst)?;
    let d = history.case.commit(&[&history.p1, &history.p2], d_writes)?;
    Ok((history, d))
}

/// VB1a: d rebased onto P1, as `jj rebase -r d -d P1` rebases it. The result's
/// conflict must hold P1's `x` and d's `y`.
fn case_vb1a(d_writes: &[Write]) -> TestResult<Merged> {
    let (TwoBases { mut case, p1, .. }, d) = resolved_k4_merge(d_writes)?;
    let rebased = rebase_commit(case.tx.repo_mut(), d.clone(), vec![p1.id().clone()]).block_on()?;
    Ok(Merged {
        case,
        parents: vec![p1, d],
        tree: rebased.tree(),
    })
}

/// VB1b: d rebased onto P2's parents once P2 is abandoned, as `jj abandon P2`
/// rebases it.
fn case_vb1b() -> TestResult<Merged> {
    let (history, d) = resolved_k4_merge(&[("f", line("y"))])?;
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

/// VB1c: P1's tree once d is squashed into it, as `jj squash --from d --into
/// P1` computes it: P1's tree with d's change from d's parent tree.
fn case_vb1c() -> TestResult<Merged> {
    let (TwoBases { case, p1, .. }, d) = resolved_k4_merge(&[("f", line("y"))])?;
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
/// every term of `[x, x - VB + y, y]` but VB; the path stays a conflict of P1's
/// `x` and d's `y` against VB.
#[test]
fn vb1a_rebase_onto_one_parent() -> TestResult {
    assert_conflict(&case_vb1a(&[("f", line("y"))])?, Base::Virtual(&["x", "y"]))
}

/// VB1a where d also changes another file, so that the rebase cancels the
/// terms of the trees in part and VB stands alone only at `f`.
#[test]
fn vb1a_rebase_onto_one_parent_with_another_change() -> TestResult {
    let merged = case_vb1a(&[("f", line("y")), ("h", line("h"))])?;
    assert_conflict(&merged, Base::Virtual(&["x", "y"]))?;
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
    assert_conflict(&case_vb1b()?, Base::Virtual(&["x", "y"]))
}

/// VB1c: d, K4's merge resolved to `y`, squashed into P1.
#[test]
fn vb1c_squash_into_a_parent() -> TestResult {
    assert_conflict(&case_vb1c()?, Base::Virtual(&["x", "y"]))
}

/// c and q, children of a commit that holds K4's conflict, resolve it to `x`
/// and to `y`; c rebased onto q holds a conflict of the two against VB.
#[test]
fn vb1_sibling_rebase() -> TestResult {
    assert_conflict(&case_vb1_sibling()?, Base::Virtual(&["x", "y"]))
}

/// The sibling rebase of D1's conflict: a conflict of `y` and the deletion
/// against M.
#[test]
fn vb1_sibling_rebase_d1() -> TestResult {
    assert_conflict(&case_vb1_sibling_d1()?, Base::Marker)
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

/// A user's text that holds lines like a virtual base's marker lines, and a
/// whole block of them with a hex that no merge wrote.
const USER_MARKER_TEXT: &str =
    "1\n<<<<<<< jj virtual merge base 0123456789abcdef0123456789abcdef\n2\n======= \
     0123456789abcdef0123456789abcdef\n3\n>>>>>>> 0123456789abcdef0123456789abcdef\n4\n";

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
        virtual_merge_base(case.store(), f(), v, &Merge::absent(), parent2).block_on()
    };
    let m1 = Some(m_of(&conflict(&[&None, &o, &y]), &resolved(y.as_ref()))?);
    let m2 = Some(m_of(&conflict(&[&None, &o, &z]), &resolved(z.as_ref()))?);
    assert_ne!(m1, m2);

    // z1 holds `[absent, M1, y]` at `f`; w, its child, holds `[absent, M2, y]`.
    // Rebased onto `o`, w's merge `o + (M1 - y) + (y - M2)` would resolve to
    // `o` if M1 and M2 were one file.
    let base = case.commit(&[&root], &[("f", line("o"))])?;
    let z1_value = conflict(&[&None, &m1, &y]);
    let z1 = case.commit_with_values(&[&base], vec![(repo_path_buf("f"), z1_value)])?;
    let w_value = conflict(&[&None, &m2, &y]);
    let w = case.commit_with_values(&[&z1], vec![(repo_path_buf("f"), w_value)])?;
    let rebased = rebase_commit(case.tx.repo_mut(), w, vec![base.id().clone()]).block_on()?;
    let value = value_at(&rebased.tree(), "f")?.simplify();
    assert!(
        value.adds().any(|add| *add == m1) && value.removes().any(|remove| *remove == m2),
        "expected a conflict that holds M1 as a side and M2 as its base, got {}",
        describe(&case, &value)
    );
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
    Ok(merge_parent_values(case.store(), f(), base, parent1, parent2).block_on()?)
}

/// The virtual base of V `base` for the parents' values, and its content.
fn virtual_base(
    case: &Case,
    base: &MergedTreeValue,
    parent1: &MergedTreeValue,
    parent2: &MergedTreeValue,
) -> TestResult<(TreeValue, Vec<u8>)> {
    let value = virtual_merge_base(case.store(), f(), base, parent1, parent2).block_on()?;
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
    let expected = [
        format!("{OPEN}{hex}"),
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
    let expected = [
        format!("{OPEN}{hex}"),
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
    let expected = [
        format!("{OPEN}{hex}"),
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
    assert_eq!(
        lines,
        [
            format!("{OPEN}{hex}"),
            "x".to_owned(),
            format!("{SEPARATOR}{hex}"),
            "y".to_owned(),
            format!("{CLOSE}{hex}"),
        ]
    );
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

/// VB and M show that a merge wrote them: the verifier accepts the files that
/// `virtual_merge_base` writes, wherever they sit and with any text around
/// their blocks, and no text in which a block's hex does not match the sides
/// it encloses, a block does not close, or no block is.
#[test]
fn virtual_base_verifies() -> TestResult {
    let case = Case::new();
    let [c, d, o, x, y] = ["c", "d", "o", "x", "y"].map(|value| case.text(value));
    let (p1, p2) = (resolved(c.as_ref()), resolved(d.as_ref()));
    let v = conflict(&[&x, &o, &y]);
    let (_, vb) = virtual_base(&case, &v, &p1, &p2)?;
    let (_, m) = virtual_base(&case, &conflict(&[&None, &o, &y]), &p1, &p2)?;
    for content in [&vb, &m] {
        assert!(
            is_virtual_base_content(content),
            "expected the verifier to accept {:?}",
            String::from_utf8_lossy(content)
        );
    }

    // The path is not hashed: the same V at another path gives the same text.
    let g = repo_path("g");
    let [c_g, d_g, o_g, x_g, y_g] = ["c", "d", "o", "x", "y"]
        .map(|value| case.file(g, &line(value).unwrap(), false, CopyId::placeholder()));
    let (p1_g, p2_g) = (resolved(c_g.as_ref()), resolved(d_g.as_ref()));
    let v_g = conflict(&[&x_g, &o_g, &y_g]);
    let other = virtual_merge_base(case.store(), g, &v_g, &p1_g, &p2_g).block_on()?;
    let TreeValue::File { id, .. } = other else {
        panic!("expected the virtual base to be a file, got {other:?}");
    };
    assert_eq!(read_file(case.store(), g, &id), vb);

    let text = String::from_utf8(vb)?;
    let surrounded = format!("before\n{text}after\n");
    assert!(is_virtual_base_content(surrounded.as_bytes()));
    let changed_side = text.replace("\nx\n", "\nz\n");
    assert_ne!(changed_side, text);
    assert!(!is_virtual_base_content(changed_side.as_bytes()));
    let unclosed = &text[..text.rfind(CLOSE).unwrap()];
    assert!(!is_virtual_base_content(unclosed.as_bytes()));
    assert!(!is_virtual_base_content(USER_MARKER_TEXT.as_bytes()));
    assert!(!is_virtual_base_content(b"no block\n"));
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

fn run_braid(num_levels: usize) -> TestResult {
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
            .new_commit(parents_a, create_tree(repo, &[(f(), &format!("a{i}\n"))]))
            .set_description(format!("a{i}"))
            .write_unwrap();
        let next_b = tx
            .repo_mut()
            .new_commit(parents_b, create_tree(repo, &[(f(), &format!("b{i}\n"))]))
            .set_description(format!("b{i}"))
            .write_unwrap();
        (a, b) = (next_a, next_b);
    }

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
