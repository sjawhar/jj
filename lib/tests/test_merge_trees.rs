// Copyright 2020 The Jujutsu Authors
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

use std::iter;
use std::sync::Arc;

use itertools::Itertools as _;
use jj_lib::backend::CopyId;
use jj_lib::backend::TreeValue;
use jj_lib::commit::Commit;
use jj_lib::merge::Merge;
use jj_lib::merge::SameChange;
use jj_lib::merged_tree::MergedTree;
use jj_lib::merged_tree_builder::MergedTreeBuilder;
use jj_lib::repo::ReadonlyRepo;
use jj_lib::repo::Repo as _;
use jj_lib::repo_path::RepoPath;
use jj_lib::repo_path::RepoPathBuf;
use jj_lib::rewrite::merge_commit_trees;
use jj_lib::rewrite::merge_commit_trees_no_resolve;
use jj_lib::rewrite::rebase_commit;
use pollster::FutureExt as _;
use test_case::test_case;
use testutils::CommitBuilderExt as _;
use testutils::TestRepo;
use testutils::TestResult;
use testutils::assert_tree_eq;
use testutils::create_tree;
use testutils::repo_path;
use testutils::repo_path_buf;
use testutils::settings_with_same_change;
use testutils::test_backend::TestBackend;

#[test]
fn test_simplify_conflict_after_resolving_parent() -> TestResult {
    let test_repo = TestRepo::init();
    let repo = &test_repo.repo;

    // Set up a repo like this:
    // D
    // | C
    // | B
    // |/
    // A
    //
    // Commit A has a file with 3 lines. B and D make conflicting changes to the
    // first line. C changes the third line. We then rebase B and C onto D,
    // which creates a conflict. We resolve the conflict in the first line and
    // rebase C2 (the rebased C) onto the resolved conflict. C3 should not have
    // a conflict since it changed an unrelated line.
    let path = repo_path("dir/file");
    let mut tx = repo.start_transaction();
    let tree_a = create_tree(repo, &[(path, "abc\ndef\nghi\n")]);
    let commit_a = tx
        .repo_mut()
        .new_commit(vec![repo.store().root_commit_id().clone()], tree_a)
        .write_unwrap();
    let tree_b = create_tree(repo, &[(path, "Abc\ndef\nghi\n")]);
    let commit_b = tx
        .repo_mut()
        .new_commit(vec![commit_a.id().clone()], tree_b)
        .write_unwrap();
    let tree_c = create_tree(repo, &[(path, "Abc\ndef\nGhi\n")]);
    let commit_c = tx
        .repo_mut()
        .new_commit(vec![commit_b.id().clone()], tree_c)
        .write_unwrap();
    let tree_d = create_tree(repo, &[(path, "abC\ndef\nghi\n")]);
    let commit_d = tx
        .repo_mut()
        .new_commit(vec![commit_a.id().clone()], tree_d)
        .write_unwrap();

    let commit_b2 =
        rebase_commit(tx.repo_mut(), commit_b, vec![commit_d.id().clone()]).block_on()?;
    let commit_c2 =
        rebase_commit(tx.repo_mut(), commit_c, vec![commit_b2.id().clone()]).block_on()?;

    // Test the setup: Both B and C should have conflicts.
    let tree_b2 = commit_b2.tree();
    let tree_c2 = commit_b2.tree();
    assert!(!tree_b2.path_value(path).block_on()?.is_resolved());
    assert!(!tree_c2.path_value(path).block_on()?.is_resolved());

    // Create the resolved B and rebase C on top.
    let tree_b3 = create_tree(repo, &[(path, "AbC\ndef\nghi\n")]);
    let commit_b3 = tx
        .repo_mut()
        .rewrite_commit(&commit_b2)
        .set_tree(tree_b3)
        .write_unwrap();
    let commit_c3 =
        rebase_commit(tx.repo_mut(), commit_c2, vec![commit_b3.id().clone()]).block_on()?;
    tx.repo_mut().rebase_descendants().block_on()?;
    let repo = tx.commit("test").block_on()?;

    // The conflict should now be resolved.
    let tree_c2 = commit_c3.tree();
    let resolved_value = tree_c2.path_value(path).block_on()?;
    match resolved_value.into_resolved() {
        Ok(Some(TreeValue::File {
            id,
            executable: false,
            copy_id: _,
        })) => {
            assert_eq!(
                testutils::read_file(repo.store(), path, &id),
                b"AbC\ndef\nGhi\n"
            );
        }
        other => {
            panic!("unexpected value: {other:#?}");
        }
    }
    Ok(())
}

// TODO: Add tests for simplification of multi-way conflicts. Both the content
// and the executable bit need testing.

#[test_case(SameChange::Keep)]
#[test_case(SameChange::Accept)]
fn test_rebase_linearize_lossy_merge(same_change: SameChange) -> TestResult {
    let settings = settings_with_same_change(same_change);
    let test_repo = TestRepo::init_with_settings(&settings);
    let repo = &test_repo.repo;

    // Test this rebase:
    // D    foo=2          D' foo=1 or 2
    // |\                  |
    // | C  foo=2          |
    // | |           =>    B  foo=2
    // B |  foo=2          |
    // |/                  |
    // A    foo=1          A  foo=1
    //
    // Since both B and C changed "1" to "2" but only one "2" remains in D, it
    // effectively discarded a change from "1" to "2". With `SameChange::Keep`,
    // D' is therefore "1". However, with `SameChange::Accept`, `jj show D` etc.
    // currently don't tell the user about the discarded change, so it's
    // surprising that the change in commit D is interpreted that way.
    let path = repo_path("foo");
    let mut tx = repo.start_transaction();
    let repo_mut = tx.repo_mut();
    let tree_1 = create_tree(repo, &[(path, "1")]);
    let tree_2 = create_tree(repo, &[(path, "2")]);
    let commit_a = repo_mut
        .new_commit(vec![repo.store().root_commit_id().clone()], tree_1.clone())
        .write_unwrap();
    let commit_b = repo_mut
        .new_commit(vec![commit_a.id().clone()], tree_2.clone())
        .write_unwrap();
    let commit_c = repo_mut
        .new_commit(vec![commit_a.id().clone()], tree_2.clone())
        .write_unwrap();
    let commit_d = repo_mut
        .new_commit(
            vec![commit_b.id().clone(), commit_c.id().clone()],
            tree_2.clone(),
        )
        .write_unwrap();

    match same_change {
        SameChange::Keep => assert!(!commit_d.is_empty(repo_mut).block_on()?),
        SameChange::Accept => assert!(commit_d.is_empty(repo_mut).block_on()?),
    }

    let commit_d2 = rebase_commit(repo_mut, commit_d, vec![commit_b.id().clone()]).block_on()?;

    match same_change {
        SameChange::Keep => assert_tree_eq!(commit_d2.tree(), tree_1),
        SameChange::Accept => assert_tree_eq!(commit_d2.tree(), tree_2),
    }
    Ok(())
}

#[test_case(SameChange::Keep)]
#[test_case(SameChange::Accept)]
fn test_rebase_on_lossy_merge(same_change: SameChange) -> TestResult {
    let settings = settings_with_same_change(same_change);
    let test_repo = TestRepo::init_with_settings(&settings);
    let repo = &test_repo.repo;

    // Test this rebase:
    // D    foo=2          D'   foo=3 or 2+(3-1) (conflict)
    // |\                  |\
    // | C  foo=2          | C' foo=3
    // | |           =>    | |
    // B |  foo=2          B |  foo=2
    // |/                  |/
    // A    foo=1          A    foo=1
    //
    // Commit D effectively discarded a change from "1" to "2", so one
    // reasonable result in D' is "3". That's the result with
    // `SameChange::Keep`. However, with `SameChange::Accept`, we resolve the
    // auto-merged parents to just "2" before the rebase in order to be
    // consistent with `jj show D` and other commands for inspecting the commit,
    // so we instead get a conflict after the rebase.
    let path = repo_path("foo");
    let mut tx = repo.start_transaction();
    let repo_mut = tx.repo_mut();
    let tree_1 = create_tree(repo, &[(path, "1")]);
    let tree_2 = create_tree(repo, &[(path, "2")]);
    let tree_3 = create_tree(repo, &[(path, "3")]);
    let commit_a = repo_mut
        .new_commit(vec![repo.store().root_commit_id().clone()], tree_1.clone())
        .write_unwrap();
    let commit_b = repo_mut
        .new_commit(vec![commit_a.id().clone()], tree_2.clone())
        .write_unwrap();
    let commit_c = repo_mut
        .new_commit(vec![commit_a.id().clone()], tree_2.clone())
        .write_unwrap();
    let commit_d = repo_mut
        .new_commit(
            vec![commit_b.id().clone(), commit_c.id().clone()],
            tree_2.clone(),
        )
        .write_unwrap();

    match same_change {
        SameChange::Keep => assert!(!commit_d.is_empty(repo_mut).block_on()?),
        SameChange::Accept => assert!(commit_d.is_empty(repo_mut).block_on()?),
    }

    let commit_c2 = repo_mut
        .new_commit(vec![commit_a.id().clone()], tree_3.clone())
        .write_unwrap();
    let commit_d2 = rebase_commit(
        repo_mut,
        commit_d,
        vec![commit_b.id().clone(), commit_c2.id().clone()],
    )
    .block_on()?;

    match same_change {
        SameChange::Keep => assert_tree_eq!(commit_d2.tree(), tree_3),
        SameChange::Accept => {
            let expected_tree_id = Merge::from_vec(vec![
                tree_2.into_tree_ids(),
                tree_1.into_tree_ids(),
                tree_3.into_tree_ids(),
            ])
            .flatten();
            assert_eq!(*commit_d2.tree_ids(), expected_tree_id);
        }
    }
    Ok(())
}

/// A merge must not silently drop content that both sides carry.
///
/// Two bugfixes independently add the same dependency line, and are merged
/// together twice -- once to base feature X on, once to base feature Y on:
///
/// ```text
///        base          "packages" WITHOUT "bun"
///        /  \
///       A    B         both independently ADD "bun", plus one change of their own
///       |\  /|
///       | \/ |
///       | /\ |
///  AB_for_X  AB_for_Y  two *separate* clean merges of A and B; both keep "bun"
///        \    /
///          XY          merging them must not lose "bun"
/// ```
///
/// `AB_for_X` and `AB_for_Y` have two best common ancestors, `A` and `B`, and
/// both of those carry "bun". Merging the bases but flattening the result into
/// the term list unresolved gives
/// `adds [AB_for_X(bun), base(no bun), AB_for_Y(bun)],
/// removes [A(bun), B(bun)]`, so "bun" cancels to net 0 while the no-bun
/// version wins with +1: the line is resolved away with no conflict marker,
/// and -- because a merge commit's diff is taken against its auto-merged
/// parents -- with no diff either. Merging and resolving the bases into one
/// tree first keeps it.
#[test]
fn test_merge_does_not_drop_content_carried_by_both_sides() -> TestResult {
    let test_repo = TestRepo::init();
    let repo = &test_repo.repo;
    let path = repo_path("devbox.json");

    // Two well-separated regions, so that "packages" and "scripts" fall in
    // different diff hunks.
    let file = |bun: bool, env: &str, script: &str| {
        format!(
            "packages:\n  php\n{}  nginx\n\nenv:\n  PORT=8081\n{}\nscripts:\n  lint\n{}",
            if bun { "  bun\n" } else { "" },
            env,
            script,
        )
    };

    let mut tx = repo.start_transaction();
    let tree_base = create_tree(repo, &[(path, &file(false, "", ""))]);
    let commit_base = tx
        .repo_mut()
        .new_commit(vec![repo.store().root_commit_id().clone()], tree_base)
        .write_unwrap();
    let tree_a = create_tree(repo, &[(path, &file(true, "  LOOKAROUND=1\n", ""))]);
    let commit_a = tx
        .repo_mut()
        .new_commit(vec![commit_base.id().clone()], tree_a)
        .write_unwrap();
    let tree_b = create_tree(repo, &[(path, &file(true, "", "  less\n"))]);
    let commit_b = tx
        .repo_mut()
        .new_commit(vec![commit_base.id().clone()], tree_b)
        .write_unwrap();

    // AB_for_X and AB_for_Y are two independent, cleanly-resolved merges of A
    // and B. Both keep "bun": A and B made the same change to that region.
    let parents = vec![commit_a.id().clone(), commit_b.id().clone()];
    let merged_tree =
        merge_commit_trees(tx.repo(), &[commit_a.clone(), commit_b.clone()]).block_on()?;
    assert!(
        merged_tree.path_value(path).block_on()?.is_resolved(),
        "the test setup requires the two merges to be conflict-free"
    );
    let commit_ab_for_x = tx
        .repo_mut()
        .new_commit(parents.clone(), merged_tree.clone())
        .write_unwrap();
    let commit_ab_for_y = tx
        .repo_mut()
        .new_commit(parents, merged_tree)
        .write_unwrap();
    let repo = tx.commit("test").block_on()?;

    let tree = merge_commit_trees(repo.as_ref(), &[commit_ab_for_x, commit_ab_for_y]).block_on()?;
    let value = tree.path_value(path).block_on()?;
    let Ok(Some(TreeValue::File { id, .. })) = value.clone().into_resolved() else {
        panic!("expected a cleanly resolved file, got: {value:#?}");
    };
    let content = String::from_utf8(testutils::read_file(repo.store(), path, &id)).unwrap();
    assert_eq!(content, file(true, "  LOOKAROUND=1\n", "  less\n"));

    Ok(())
}

/// The reproduction from jj-vcs/jj#6369: `a` and `b` make the *same* change,
/// so `c` and `d` resolve cleanly to it under `SameChange::Accept`, yet the
/// merge of `c` and `d` used to go back to the base's content.
#[test]
fn test_merge_of_merges_with_same_change_keeps_the_change() -> TestResult {
    let test_repo = TestRepo::init();
    let repo = &test_repo.repo;
    let path = repo_path("file.txt");

    let mut tx = repo.start_transaction();
    let tree_base = create_tree(repo, &[(path, "original\n")]);
    let commit_base = tx
        .repo_mut()
        .new_commit(vec![repo.store().root_commit_id().clone()], tree_base)
        .write_unwrap();
    let tree_modified = create_tree(repo, &[(path, "modified\n")]);
    let commit_a = tx
        .repo_mut()
        .new_commit(vec![commit_base.id().clone()], tree_modified.clone())
        .write_unwrap();
    let commit_b = tx
        .repo_mut()
        .new_commit(vec![commit_base.id().clone()], tree_modified)
        .write_unwrap();

    let parents = vec![commit_a.id().clone(), commit_b.id().clone()];
    let merged_tree = merge_commit_trees(tx.repo(), &[commit_a, commit_b]).block_on()?;
    let commit_c = tx
        .repo_mut()
        .new_commit(parents.clone(), merged_tree.clone())
        .write_unwrap();
    let commit_d = tx
        .repo_mut()
        .new_commit(parents, merged_tree)
        .write_unwrap();
    let repo = tx.commit("test").block_on()?;

    let tree = merge_commit_trees(repo.as_ref(), &[commit_c, commit_d]).block_on()?;
    let value = tree.path_value(path).block_on()?;
    let Ok(Some(TreeValue::File { id, .. })) = value.clone().into_resolved() else {
        panic!("expected a cleanly resolved file, got: {value:#?}");
    };
    assert_eq!(testutils::read_file(repo.store(), path, &id), b"modified\n");

    Ok(())
}

/// A conflicted merge base must not let content revert to the base's own
/// version.
///
/// ```text
///    base        "original"
///     / \
///    a   b       "A" and "B" -- a genuine conflict
///     \ /
///      m         conflicted merge of a and b
///     / \
///   r1   r2      two *different* manual resolutions of m
/// ```
///
/// `r1` and `r2` have a single merge base, `m`, whose tree is conflicted.
/// Flattening that conflict into the term list puts `a` and `b` on the remove
/// side, where they cancel `r1` and `r2`, leaving "original" -- a version
/// *neither side chose* -- to win, with no conflict at all.
///
/// Both `r1` and `r2` changed the base's conflict, so they are merged against
/// a virtual merge base that holds the conflict's sides between marker lines,
/// and the merge is a conflict between "A" and "B".
#[test]
fn test_merge_of_two_resolutions_does_not_revert_to_the_base() -> TestResult {
    let test_repo = TestRepo::init();
    let repo = &test_repo.repo;
    let path = repo_path("file");

    let mut tx = repo.start_transaction();
    let tree_base = create_tree(repo, &[(path, "original\n")]);
    let commit_base = tx
        .repo_mut()
        .new_commit(vec![repo.store().root_commit_id().clone()], tree_base)
        .write_unwrap();
    let tree_a = create_tree(repo, &[(path, "A\n")]);
    let commit_a = tx
        .repo_mut()
        .new_commit(vec![commit_base.id().clone()], tree_a.clone())
        .write_unwrap();
    let tree_b = create_tree(repo, &[(path, "B\n")]);
    let commit_b = tx
        .repo_mut()
        .new_commit(vec![commit_base.id().clone()], tree_b.clone())
        .write_unwrap();

    let tree_m = merge_commit_trees(tx.repo(), &[commit_a.clone(), commit_b.clone()]).block_on()?;
    assert!(
        !tree_m.path_value(path).block_on()?.is_resolved(),
        "the test setup requires m to be conflicted"
    );
    let commit_m = tx
        .repo_mut()
        .new_commit(vec![commit_a.id().clone(), commit_b.id().clone()], tree_m)
        .write_unwrap();
    // Two different manual resolutions of the same conflict.
    let commit_r1 = tx
        .repo_mut()
        .new_commit(vec![commit_m.id().clone()], tree_a)
        .write_unwrap();
    let commit_r2 = tx
        .repo_mut()
        .new_commit(vec![commit_m.id().clone()], tree_b)
        .write_unwrap();
    let repo = tx.commit("test").block_on()?;

    let tree = merge_commit_trees(repo.as_ref(), &[commit_r1, commit_r2]).block_on()?;
    let value = tree.path_value(path).block_on()?.simplify();
    assert!(!value.is_resolved(), "expected a conflict, got: {value:#?}");
    let adds: Vec<Vec<u8>> = value
        .adds()
        .map(|add| match add {
            Some(TreeValue::File { id, .. }) => testutils::read_file(repo.store(), path, id),
            other => panic!("expected a file among the sides, got: {other:?}"),
        })
        .collect();
    assert_eq!(adds, [b"A\n".to_vec(), b"B\n".to_vec()]);

    Ok(())
}

/// Runs `merge` and returns its result with the number of files it read from
/// the test backend of `repo`, from caches cleared first.
fn with_file_reads<T>(repo: &Arc<ReadonlyRepo>, merge: impl FnOnce() -> T) -> (T, usize) {
    let store = repo.store();
    store.clear_caches();
    let backend = store.backend_impl::<TestBackend>().unwrap();
    let before = backend.file_reads();
    let result = merge();
    (result, backend.file_reads() - before)
}

/// The value of `commit`'s tree at `path`, which is resolved.
fn resolved_value(commit: &Commit, path: &RepoPath) -> TestResult<Option<TreeValue>> {
    let value = commit.tree().path_value(path).block_on()?;
    Ok(value.into_resolved().expect("a resolved value"))
}

/// A tree that holds `base_text` at each of `files`, with `changes` written
/// over it.
fn tree_with(
    repo: &Arc<ReadonlyRepo>,
    files: &[RepoPathBuf],
    base_text: &str,
    changes: &[(&RepoPath, &str)],
) -> MergedTree {
    let mut contents = files
        .iter()
        .map(|path| (path.as_ref(), base_text))
        .collect_vec();
    for &(path, text) in changes {
        match contents.iter_mut().find(|(file, _)| *file == path) {
            Some(entry) => entry.1 = text,
            None => contents.push((path, text)),
        }
    }
    create_tree(repo, &contents)
}

/// A merge of many commits where two of them conflict at each of four files
/// and every other commit adds a file of its own, as the merge of the branches
/// of a fork can: each conflict is merged, and its sides read, once, when the
/// second commit that changes its file comes in. A merge that resolved again,
/// at each later commit, everything the commits before it left read every
/// conflict's sides again at each step, so the merge of n commits did work in
/// n squared, and an import of such merges, which merges each one's parents to
/// report its new conflicts, did not finish.
#[test]
fn test_merge_of_many_commits_merges_each_conflict_once() -> TestResult {
    let test_repo = TestRepo::init();
    let repo = &test_repo.repo;
    let conflicted = (0..4)
        .map(|i| repo_path_buf(format!("dir/conflicted{i}")))
        .collect_vec();
    let base_text = "base\n2\n3\n";

    let mut tx = repo.start_transaction();
    let base = tx
        .repo_mut()
        .new_commit(
            vec![repo.store().root_commit_id().clone()],
            tree_with(repo, &conflicted, base_text, &[]),
        )
        .write_unwrap();
    let mut sides = vec![];
    let mut parents = vec![];
    for path in &conflicted {
        let [a, b] = ["a\n2\n3\n", "b\n2\n3\n"].map(|text| {
            tx.repo_mut()
                .new_commit(
                    vec![base.id().clone()],
                    tree_with(repo, &conflicted, base_text, &[(path.as_ref(), text)]),
                )
                .write_unwrap()
        });
        sides.push((a.clone(), b.clone()));
        parents.extend([a, b]);
    }
    let own = (0..40)
        .map(|i| repo_path_buf(format!("own/{i}")))
        .collect_vec();
    for path in &own {
        let text = format!("{}\n", path.as_internal_file_string());
        let commit = tx
            .repo_mut()
            .new_commit(
                vec![base.id().clone()],
                tree_with(
                    repo,
                    &conflicted,
                    base_text,
                    &[(path.as_ref(), text.as_str())],
                ),
            )
            .write_unwrap();
        parents.push(commit);
    }
    let repo = tx.commit("test").block_on()?;

    // A merge of three files reads each of them once. A debug build merges a
    // resolved tree again to check it, and the store keeps what a merge of
    // files gave, so that check reads none.
    let bound = 3 * conflicted.len();
    let (unresolved, reads) = with_file_reads(&repo, || {
        merge_commit_trees_no_resolve(repo.as_ref(), &parents).block_on()
    });
    let unresolved = unresolved?;
    assert!(
        reads <= bound,
        "the merge of {} commits read {reads} files, more than {bound}",
        parents.len()
    );
    let (resolved, reads) = with_file_reads(&repo, || {
        merge_commit_trees(repo.as_ref(), &parents).block_on()
    });
    let resolved = resolved?;
    assert!(
        reads <= bound,
        "the resolved merge of {} commits read {reads} files, more than {bound}",
        parents.len()
    );

    for tree in [&unresolved, &resolved] {
        for (path, (a, b)) in iter::zip(&conflicted, &sides) {
            let expected = Merge::from_vec(vec![
                resolved_value(a, path)?,
                resolved_value(&base, path)?,
                resolved_value(b, path)?,
            ]);
            let value = tree.path_value(path).block_on()?;
            assert!(
                value.same_terms_as(&expected),
                "expected {path:?} to be the conflict of its two sides, got: {value:#?}"
            );
        }
        for (path, commit) in iter::zip(&own, &parents[2 * conflicted.len()..]) {
            let value = tree.path_value(path).block_on()?;
            assert_eq!(value, Merge::resolved(resolved_value(commit, path)?));
        }
    }
    Ok(())
}

/// A merge of many commits that descend from a commit with a conflict, as the
/// commits that an import makes visible can: the conflict is every commit's
/// and the merge base's, so after the first two commits no commit changes it,
/// and the merge reads its sides only for the first merge.
#[test]
fn test_merge_of_commits_above_a_conflict_reads_it_once() -> TestResult {
    let test_repo = TestRepo::init();
    let repo = &test_repo.repo;
    let path = repo_path("dir/conflicted");

    let mut tx = repo.start_transaction();
    let base = tx
        .repo_mut()
        .new_commit(
            vec![repo.store().root_commit_id().clone()],
            create_tree(repo, &[(path, "base\n2\n3\n")]),
        )
        .write_unwrap();
    let [a, b] = ["a\n2\n3\n", "b\n2\n3\n"].map(|content| {
        tx.repo_mut()
            .new_commit(
                vec![base.id().clone()],
                create_tree(repo, &[(path, content)]),
            )
            .write_unwrap()
    });
    let conflict_tree = merge_commit_trees(tx.repo(), &[a.clone(), b.clone()]).block_on()?;
    assert!(
        !conflict_tree.path_value(path).block_on()?.is_resolved(),
        "the test setup requires the merge of a and b to conflict"
    );
    let conflict = tx
        .repo_mut()
        .new_commit(vec![a.id().clone(), b.id().clone()], conflict_tree.clone())
        .write_unwrap();
    let mut children = vec![];
    for i in 0..40 {
        let own = repo_path_buf(format!("own/{i}"));
        let file = TreeValue::File {
            id: testutils::write_file(repo.store(), &own, &format!("{i}\n")),
            executable: false,
            copy_id: CopyId::placeholder(),
        };
        let mut builder = MergedTreeBuilder::new(conflict_tree.clone());
        builder.set_or_remove(own, Merge::normal(file));
        let tree = builder.write_tree().block_on()?;
        let child = tx
            .repo_mut()
            .new_commit(vec![conflict.id().clone()], tree)
            .write_unwrap();
        children.push(child);
    }
    let repo = tx.commit("test").block_on()?;

    let bound = 3;
    let (merged, reads) = with_file_reads(&repo, || {
        merge_commit_trees_no_resolve(repo.as_ref(), &children).block_on()
    });
    let merged = merged?;
    assert!(
        reads <= bound,
        "the merge of {} commits read {reads} files, more than {bound}",
        children.len()
    );
    let value = merged.path_value(path).block_on()?;
    let expected = conflict_tree.path_value(path).block_on()?;
    assert!(
        value.same_terms_as(&expected),
        "expected the merge base's conflict, got: {value:#?}"
    );
    Ok(())
}

/// A merge of many commits where four of them change one file differently, so
/// that the file's conflict has seven terms, and every other commit writes a
/// file of its own. Each later commit leaves the conflict as it is, so its
/// value takes a base and a side that are equal, which cancel; the terms that
/// are left are the terms the conflict had, which the merge can give in
/// another order. A merge that read the file's terms again whenever their
/// order moved merged them again at most of the commits, where the order
/// decides nothing: the terms are the same.
#[test]
fn test_merge_of_many_commits_merges_a_conflict_of_four_sides_once() -> TestResult {
    let test_repo = TestRepo::init();
    let repo = &test_repo.repo;
    let path = repo_path("dir/conflicted");
    let files = [path.to_owned()];
    let base_text = "base\n2\n3\n";

    let mut tx = repo.start_transaction();
    let base = tx
        .repo_mut()
        .new_commit(
            vec![repo.store().root_commit_id().clone()],
            tree_with(repo, &files, base_text, &[]),
        )
        .write_unwrap();
    let mut parents = ["a", "b", "c", "d"]
        .iter()
        .map(|side| {
            let text = format!("{side}\n2\n3\n");
            tx.repo_mut()
                .new_commit(
                    vec![base.id().clone()],
                    tree_with(repo, &files, base_text, &[(path, text.as_str())]),
                )
                .write_unwrap()
        })
        .collect_vec();
    for i in 0..40 {
        let own = repo_path_buf(format!("own/{i}"));
        let text = format!("{i}\n");
        let commit = tx
            .repo_mut()
            .new_commit(
                vec![base.id().clone()],
                tree_with(repo, &files, base_text, &[(own.as_ref(), text.as_str())]),
            )
            .write_unwrap();
        parents.push(commit);
    }
    let repo = tx.commit("test").block_on()?;

    // Each of the three commits that find the file already changed merges it
    // against the merge of the commits before it, which is what a merge of
    // many commits does: 3, then 5, then 7 terms, each read once. No later
    // commit reads any of them again, where a merge that read them whenever
    // their order moved read them at most of the commits.
    let bound = 15;
    let (merged, reads) = with_file_reads(&repo, || {
        merge_commit_trees(repo.as_ref(), &parents).block_on()
    });
    let merged = merged?;
    assert!(
        reads <= bound,
        "the merge of {} commits read {reads} files, more than {bound}",
        parents.len()
    );
    let value = merged.path_value(path).block_on()?;
    assert_eq!(
        value.num_sides(),
        4,
        "expected the conflict of the four sides, got: {value:#?}"
    );
    Ok(())
}
