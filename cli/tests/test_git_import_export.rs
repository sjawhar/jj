// Copyright 2022 The Jujutsu Authors
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

use std::path::Path;

use itertools::Itertools as _;
use jj_lib::backend::CommitId;
use testutils::TestResult;
use testutils::git;

use crate::common::CommandOutput;
use crate::common::TestEnvironment;
use crate::common::TestWorkDir;

#[test]
fn test_git_export_basic() {
    let test_env = TestEnvironment::default();
    test_env.run_jj_in(".", ["git", "init", "repo"]).success();
    let work_dir = test_env.work_dir("repo");

    work_dir
        .run_jj(["bookmark", "create", "-r@", "main"])
        .success();

    let output = work_dir.run_jj(["git", "export", "--no-integrate-operation"]);
    insta::assert_snapshot!(output, @"
    ------- stderr -------
    Error: --no-integrate-operation is not respected
    [EOF]
    [exit status: 2]
    ");

    let output = work_dir.run_jj(["git", "export"]);
    insta::assert_snapshot!(output, @"");
    insta::assert_snapshot!(work_dir.run_jj(["log", "-rmain@git"]), @"
    @  qpvuntsm test.user@example.com 2001-02-03 08:05:07 main e8849ae1
    │  (empty) (no description set)
    ~
    [EOF]
    ");
}

#[test]
fn test_resolution_of_git_tracking_bookmarks() -> TestResult {
    let test_env = TestEnvironment::default();
    test_env.run_jj_in(".", ["git", "init", "repo"]).success();
    let work_dir = test_env.work_dir("repo");
    work_dir
        .run_jj(["bookmark", "create", "-r@", "main"])
        .success();
    work_dir
        .run_jj(["describe", "-r", "main", "-m", "old_message"])
        .success();

    // Create local-git tracking bookmark
    let output = work_dir.run_jj(["git", "export"]);
    insta::assert_snapshot!(output, @"");
    // Move the local bookmark somewhere else
    work_dir
        .run_jj(["describe", "-r", "main", "-m", "new_message"])
        .success();
    insta::assert_snapshot!(get_bookmark_output(&work_dir), @"
    main: qpvuntsm 384a1421 (empty) new_message
      @git (ahead by 1 commits, behind by 1 commits): qpvuntsm/1 a7f9930b (hidden) (empty) old_message
    [EOF]
    ");

    // Test that we can address both revisions
    let query = |expr| {
        let template = r#"commit_id ++ " " ++ description"#;
        work_dir.run_jj(["log", "-r", expr, "-T", template, "--no-graph"])
    };
    insta::assert_snapshot!(query("main"), @"
    384a14213707d776d0517f65cdcf954d07d88c40 new_message
    [EOF]
    ");
    insta::assert_snapshot!(query("main@git"), @"
    a7f9930bb6d54ba39e6c254135b9bfe32041fea4 old_message
    [EOF]
    ");
    insta::assert_snapshot!(query(r#"remote_bookmarks("main", "git")"#), @"
    a7f9930bb6d54ba39e6c254135b9bfe32041fea4 old_message
    [EOF]
    ");
    Ok(())
}

#[test]
fn test_git_export_conflicting_git_refs() -> TestResult {
    let test_env = TestEnvironment::default();
    test_env.run_jj_in(".", ["git", "init", "repo"]).success();
    let work_dir = test_env.work_dir("repo");

    work_dir
        .run_jj(["bookmark", "create", "-r@", "main"])
        .success();
    work_dir
        .run_jj(["bookmark", "create", "-r@", "main/sub"])
        .success();
    let output = work_dir.run_jj(["git", "export"]);
    insta::with_settings!({filters => vec![("Failed to set: .*", "Failed to set: ...")]}, {
        insta::assert_snapshot!(output, @r#"
        ------- stderr -------
        Warning: Failed to export some bookmarks:
          main/sub@git: Failed to set: ...
        Hint: Git doesn't allow a branch/tag name that looks like a parent directory of
        another (e.g. `foo` and `foo/bar`). Try to rename the bookmarks/tags that failed
        to export or their "parent" bookmarks/tags.
        [EOF]
        "#);
    });
    Ok(())
}

#[test]
fn test_git_export_undo() -> TestResult {
    let test_env = TestEnvironment::default();
    test_env.run_jj_in(".", ["git", "init", "repo"]).success();
    let work_dir = test_env.work_dir("repo");
    let git_repo = git::open(work_dir.root().join(".jj/repo/store/git"));

    work_dir
        .run_jj(["bookmark", "create", "-r@", "a"])
        .success();
    insta::assert_snapshot!(get_bookmark_output(&work_dir), @"
    a: qpvuntsm e8849ae1 (empty) (no description set)
    [EOF]
    ");
    let output = work_dir.run_jj(["git", "export"]);
    insta::assert_snapshot!(output, @"");
    insta::assert_snapshot!(work_dir.run_jj(["log", "-ra@git"]), @"
    @  qpvuntsm test.user@example.com 2001-02-03 08:05:07 a e8849ae1
    │  (empty) (no description set)
    ~
    [EOF]
    ");

    // Exported refs won't be removed by undoing the export, but the git-tracking
    // bookmark is. This is the same as remote-tracking bookmarks.
    let output = work_dir.run_jj(["undo"]);
    insta::assert_snapshot!(output, @"
    ------- stderr -------
    Undid operation: 29447db431a3 (2001-02-03 08:05:10) export git refs
    Restored to operation: 5142cb3d7c38 (2001-02-03 08:05:08) create bookmark a pointing to commit e8849ae12c709f2321908879bc724fdb2ab8a781
    [EOF]
    ");
    insta::assert_debug_snapshot!(get_git_repo_refs(&git_repo), @r#"
    [
        (
            "refs/heads/a",
            CommitId(
                "e8849ae12c709f2321908879bc724fdb2ab8a781",
            ),
        ),
    ]
    "#);
    insta::assert_snapshot!(work_dir.run_jj(["log", "-ra@git"]), @"
    ------- stderr -------
    Error: Revision `a@git` doesn't exist
    Hint: Did you mean `a`?
    [EOF]
    [exit status: 1]
    ");

    // This would re-export bookmark "a" and create git-tracking bookmark.
    let output = work_dir.run_jj(["git", "export"]);
    insta::assert_snapshot!(output, @"");
    insta::assert_snapshot!(work_dir.run_jj(["log", "-ra@git"]), @"
    @  qpvuntsm test.user@example.com 2001-02-03 08:05:07 a e8849ae1
    │  (empty) (no description set)
    ~
    [EOF]
    ");
    Ok(())
}

#[test]
fn test_git_import_undo() -> TestResult {
    let test_env = TestEnvironment::default();
    test_env.run_jj_in(".", ["git", "init", "repo"]).success();
    let work_dir = test_env.work_dir("repo");
    let git_repo = git::open(work_dir.root().join(".jj/repo/store/git"));

    // Create bookmark "a" in git repo
    let commit_id = work_dir
        .run_jj(&["log", "-Tcommit_id", "--no-graph", "-r@"])
        .success()
        .stdout
        .into_raw();
    let commit_id = gix::ObjectId::from_hex(commit_id.as_bytes()).unwrap();
    git_repo.reference(
        "refs/heads/a",
        commit_id,
        gix::refs::transaction::PreviousValue::Any,
        "",
    )?;

    // Initial state we will return to after `undo`. There are no bookmarks.
    insta::assert_snapshot!(get_bookmark_output(&work_dir), @"");
    let base_operation_id = work_dir.current_operation_id();

    let output = work_dir.run_jj(["git", "import"]);
    insta::assert_snapshot!(output, @"
    ------- stderr -------
    bookmark: a@git [new] tracked
    [EOF]
    ");
    insta::assert_snapshot!(get_bookmark_output(&work_dir), @"
    a: qpvuntsm e8849ae1 (empty) (no description set)
      @git: qpvuntsm e8849ae1 (empty) (no description set)
    [EOF]
    ");

    // "git import" can be undone by default.
    let output = work_dir.run_jj(["op", "restore", &base_operation_id]);
    insta::assert_snapshot!(output, @"
    ------- stderr -------
    Restored to operation: f63ee16f9553 (2001-02-03 08:05:07) add workspace 'default'
    [EOF]
    ");
    insta::assert_snapshot!(get_bookmark_output(&work_dir), @"");
    // Try "git import" again, which should re-import the bookmark "a".
    let output = work_dir.run_jj(["git", "import"]);
    insta::assert_snapshot!(output, @"
    ------- stderr -------
    bookmark: a@git [new] tracked
    [EOF]
    ");
    insta::assert_snapshot!(get_bookmark_output(&work_dir), @"
    a: qpvuntsm e8849ae1 (empty) (no description set)
      @git: qpvuntsm e8849ae1 (empty) (no description set)
    [EOF]
    ");
    Ok(())
}

/// `jj git import` reports the conflicted merge commits it brings in without
/// merging their parents, which is the only way to tell whether a merge commit
/// is empty: for a merge of many commits that takes seconds, and an import can
/// bring in many such commits. So, with the default summary template, the
/// report lists every conflicted merge commit the import brings in, and says of
/// none of them that it is empty.
#[test]
fn test_git_import_reports_merges_without_merging_their_parents() -> TestResult {
    let test_env = TestEnvironment::default();
    set_up_conflicted_merges_to_import(&test_env);
    let dest_dir = test_env.work_dir("dest");

    let trace_path = test_env.env_root().join("import-trace.json");
    let output =
        dest_dir.run_jj_with(|cmd| cmd.env("JJ_TRACE", &trace_path).args(["git", "import"]));
    assert_eq!(
        count_parent_merges(&trace_path),
        0,
        "the import merged the parents of a commit"
    );
    // Both merges are listed, and neither is said to be empty; the empty
    // commit with one parent is left out.
    insta::assert_snapshot!(output, @"
    ------- stderr -------
    bookmark: child@source       [new] untracked
    bookmark: empty-merge@source [new] untracked
    bookmark: left@source        [new] untracked
    bookmark: merge@source       [new] untracked
    bookmark: right@source       [new] untracked
    New conflicts appeared in 2 commits:
      vruxwmqv 3bf0127d merge@source | (conflict) merge
      royxmykx 1f8e87d2 empty-merge@source | (conflict) empty merge
    Hint: To resolve the conflicts, start by creating a commit on top of
    one of the first conflicted commits:
      jj new vruxwmqv
      jj new royxmykx
    Then use `jj resolve`, or edit the conflict markers in the file directly.
    Once the conflicts are resolved, you can inspect the result with `jj diff`.
    Then run `jj squash` to move the resolution into the conflicted commit.
    [EOF]
    ");

    // The control: `jj log` of the empty merge, whose summary says that it is
    // empty, merges the commit's parents, and the trace shows it.
    let trace_path = test_env.env_root().join("log-trace.json");
    let output = dest_dir.run_jj_with(|cmd| {
        cmd.env("JJ_TRACE", &trace_path)
            .args(["log", "--no-graph", "-r", "empty-merge@source"])
    });
    assert_ne!(
        count_parent_merges(&trace_path),
        0,
        "the trace of `jj log` shows no merge of a commit's parents"
    );
    insta::assert_snapshot!(output, @"
    royxmykx test.user@example.com 2001-02-03 08:05:12 empty-merge@source 1f8e87d2 (conflict)
    (empty) empty merge
    [EOF]
    ");
    Ok(())
}

/// A summary template the user configured is used as it is in the report of an
/// import too: a template that asks whether a commit is empty gets the answer,
/// and pays the merge of a merge commit's parents.
#[test]
fn test_git_import_reports_with_a_configured_summary() {
    let test_env = TestEnvironment::default();
    set_up_conflicted_merges_to_import(&test_env);
    test_env.add_config(
        r#"templates.commit_summary = 'description.first_line() ++ " empty=" ++ if(empty, "yes", "no")'"#,
    );
    let output = test_env.work_dir("dest").run_jj(["git", "import"]);
    let stderr = output.success().stderr.into_raw();
    assert!(stderr.contains("\n  empty merge empty=yes\n"), "{stderr}");
    assert!(stderr.contains("\n  merge empty=no\n"), "{stderr}");
}

/// Makes, in the repository "source", two sides that conflict, two conflicted
/// merges of them (one empty, one that adds a file) and an empty commit on top
/// of the second, each with a bookmark, and has Git fetch them into the Git
/// repository of the repository "dest", whose `jj git import` brings them in.
fn set_up_conflicted_merges_to_import(test_env: &TestEnvironment) {
    test_env.run_jj_in(".", ["git", "init", "source"]).success();
    let source_dir = test_env.work_dir("source");
    for side in ["left", "right"] {
        source_dir.run_jj(["new", "root()", "-m", side]).success();
        source_dir.write_file("file", format!("{side}\n"));
        source_dir
            .run_jj(["bookmark", "create", "-r@", side])
            .success();
    }
    // The commit on top of the second merge has one parent, whose tree tells
    // whether it is empty, so the report still leaves it out.
    source_dir
        .run_jj(["new", "left", "right", "-m", "empty merge"])
        .success();
    source_dir
        .run_jj(["bookmark", "create", "-r@", "empty-merge"])
        .success();
    source_dir
        .run_jj(["new", "left", "right", "-m", "merge"])
        .success();
    source_dir.write_file("other", "other\n");
    source_dir
        .run_jj(["bookmark", "create", "-r@", "merge"])
        .success();
    source_dir.run_jj(["new", "-m", "child"]).success();
    source_dir
        .run_jj(["bookmark", "create", "-r@", "child"])
        .success();
    source_dir.run_jj(["git", "export"]).success();

    let dest_git_path = test_env.env_root().join("dest-git");
    git::init(&dest_git_path);
    test_env
        .run_jj_in(".", ["git", "init", "--git-repo=dest-git", "dest"])
        .success();
    let source_git_path = source_dir.root().join(".jj/repo/store/git");
    git::add_remote(&dest_git_path, "source", source_git_path.to_str().unwrap());
    git::fetch(&dest_git_path, "source");
}

/// The number of events in the trace `JJ_TRACE` wrote at `path` of the spans
/// of `merge_commit_trees` and its variants, which `Commit::is_empty()` and the
/// revset `empty()` call to tell whether a merge commit is empty.
fn count_parent_merges(path: &Path) -> usize {
    std::fs::read_to_string(path)
        .unwrap()
        .matches(r#""name":"merge_commit_trees"#)
        .count()
}

#[test]
fn test_git_import_move_export_with_default_undo() -> TestResult {
    let test_env = TestEnvironment::default();
    test_env.run_jj_in(".", ["git", "init", "repo"]).success();
    let work_dir = test_env.work_dir("repo");
    let git_repo = git::open(work_dir.root().join(".jj/repo/store/git"));

    // Create bookmark "a" in git repo
    let commit_id = work_dir
        .run_jj(&["log", "-Tcommit_id", "--no-graph", "-r@"])
        .success()
        .stdout
        .into_raw();
    let commit_id = gix::ObjectId::from_hex(commit_id.as_bytes()).unwrap();
    git_repo.reference(
        "refs/heads/a",
        commit_id,
        gix::refs::transaction::PreviousValue::Any,
        "",
    )?;

    // Initial state we will try to return to after `op restore`. There are no
    // bookmarks.
    insta::assert_snapshot!(get_bookmark_output(&work_dir), @"");
    let base_operation_id = work_dir.current_operation_id();

    let output = work_dir.run_jj(["git", "import"]);
    insta::assert_snapshot!(output, @"
    ------- stderr -------
    bookmark: a@git [new] tracked
    [EOF]
    ");
    insta::assert_snapshot!(get_bookmark_output(&work_dir), @"
    a: qpvuntsm e8849ae1 (empty) (no description set)
      @git: qpvuntsm e8849ae1 (empty) (no description set)
    [EOF]
    ");

    // Move bookmark "a" and export to git repo
    work_dir.run_jj(["new"]).success();
    work_dir
        .run_jj(["bookmark", "set", "a", "--to=@"])
        .success();
    insta::assert_snapshot!(get_bookmark_output(&work_dir), @"
    a: royxmykx e7d0d5fd (empty) (no description set)
      @git (behind by 1 commits): qpvuntsm e8849ae1 (empty) (no description set)
    [EOF]
    ");
    let output = work_dir.run_jj(["git", "export"]);
    insta::assert_snapshot!(output, @"");
    insta::assert_snapshot!(get_bookmark_output(&work_dir), @"
    a: royxmykx e7d0d5fd (empty) (no description set)
      @git: royxmykx e7d0d5fd (empty) (no description set)
    [EOF]
    ");

    // "git import" can be undone with the default `restore` behavior, as shown in
    // the previous test. However, "git export" can't: the bookmarks in the git
    // repo stay where they were.
    let output = work_dir.run_jj(["op", "restore", &base_operation_id]);
    insta::assert_snapshot!(output, @"
    ------- stderr -------
    Restored to operation: f63ee16f9553 (2001-02-03 08:05:07) add workspace 'default'
    Working copy  (@) now at: qpvuntsm e8849ae1 (empty) (no description set)
    Parent commit (@-)      : zzzzzzzz 00000000 (empty) (no description set)
    [EOF]
    ");
    insta::assert_snapshot!(get_bookmark_output(&work_dir), @"");
    insta::assert_debug_snapshot!(get_git_repo_refs(&git_repo), @r#"
    [
        (
            "refs/heads/a",
            CommitId(
                "e7d0d5fdaf96051d0dacec1e74d9413d64a15822",
            ),
        ),
    ]
    "#);

    // The last bookmark "a" state is imported from git. No idea what's the most
    // intuitive result here.
    let output = work_dir.run_jj(["git", "import"]);
    insta::assert_snapshot!(output, @"
    ------- stderr -------
    bookmark: a@git [new] tracked
    [EOF]
    ");
    insta::assert_snapshot!(get_bookmark_output(&work_dir), @"
    a: royxmykx e7d0d5fd (empty) (no description set)
      @git: royxmykx e7d0d5fd (empty) (no description set)
    [EOF]
    ");
    Ok(())
}

#[test]
fn test_git_import_export_stats_color() -> TestResult {
    let test_env = TestEnvironment::default();
    test_env.run_jj_in(".", ["git", "init", "repo"]).success();
    let work_dir = test_env.work_dir("repo");
    let git_repo = git::open(work_dir.root().join(".jj/repo/store/git"));

    work_dir.run_jj(["bookmark", "set", "-r@", "foo"]).success();
    work_dir
        .run_jj(["bookmark", "set", "-r@", "'un:exportable'"])
        .success();
    work_dir.run_jj(["new", "--no-edit", "root()"]).success();
    let other_commit_id = work_dir
        .run_jj(&["log", "-Tcommit_id", "--no-graph", "-rvisible_heads() ~ @"])
        .success()
        .stdout
        .into_raw();

    let output = work_dir
        .run_jj(["git", "export", "--color=always"])
        .success();
    insta::assert_snapshot!(output, @r#"
    ------- stderr -------
    [1m[38;5;3mWarning: [39mFailed to export some bookmarks:[0m
      [38;5;5m"un:exportable"@git[39m: Failed to set: The ref name or path is not a valid ref name: Reference name contains invalid byte: ":": Validation
    [1m[38;5;6mHint: [0m[39mGit doesn't allow a branch/tag name that looks like a parent directory of[39m
    [39manother (e.g. `foo` and `foo/bar`). Try to rename the bookmarks/tags that failed[39m
    [39mto export or their "parent" bookmarks/tags.[39m
    [EOF]
    "#);

    let other_commit_id = gix::ObjectId::from_hex(other_commit_id.as_bytes()).unwrap();
    for name in ["refs/heads/foo", "refs/heads/bar", "refs/tags/baz"] {
        git_repo.reference(
            name,
            other_commit_id,
            gix::refs::transaction::PreviousValue::Any,
            "",
        )?;
    }

    let output = work_dir
        .run_jj(["git", "import", "--color=always"])
        .success();
    insta::assert_snapshot!(output, @"
    ------- stderr -------
    bookmark: [38;5;5mbar@git[39m [new] tracked
    bookmark: [38;5;5mfoo@git[39m [updated] tracked
    tag: [38;5;5mbaz@git[39m [new] tracked
    [EOF]
    ");
    Ok(())
}

#[must_use]
fn get_bookmark_output(work_dir: &TestWorkDir) -> CommandOutput {
    work_dir.run_jj(["bookmark", "list", "--all-remotes"])
}

fn get_git_repo_refs(git_repo: &gix::Repository) -> Vec<(bstr::BString, CommitId)> {
    let mut refs: Vec<_> = git_repo
        .references()
        .unwrap()
        .all()
        .unwrap()
        .filter_ok(|git_ref| {
            matches!(
                git_ref.name().category(),
                Some(
                    gix::reference::Category::Tag
                        | gix::reference::Category::LocalBranch
                        | gix::reference::Category::RemoteBranch
                ),
            )
        })
        .filter_map_ok(|mut git_ref| {
            let full_name = git_ref.name().as_bstr().to_owned();
            let git_commit = git_ref.peel_to_commit().ok()?;
            let commit_id = CommitId::from_bytes(git_commit.id().as_bytes());
            Some((full_name, commit_id))
        })
        .try_collect()
        .unwrap();
    refs.sort();
    refs
}
