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

mod command_output;
mod config_schema_defaults;
mod test_environment;

pub use self::command_output::CommandOutput;
pub use self::config_schema_defaults::default_config_from_schema;
pub use self::test_environment::TestEnvironment;
pub use self::test_environment::TestWorkDir;

pub fn fake_bisector_path() -> String {
    let path = assert_cmd::cargo::cargo_bin!("fake-bisector");
    assert!(path.is_file());
    path.as_os_str().to_str().unwrap().to_owned()
}

pub fn fake_editor_path() -> String {
    let path = assert_cmd::cargo::cargo_bin!("fake-editor");
    assert!(path.is_file());
    path.as_os_str().to_str().unwrap().to_owned()
}

pub fn fake_diff_editor_path() -> String {
    let path = assert_cmd::cargo::cargo_bin!("fake-diff-editor");
    assert!(path.is_file());
    path.as_os_str().to_str().unwrap().to_owned()
}

/// Forcibly enable interactive prompt.
pub fn force_interactive(cmd: &mut assert_cmd::Command) -> &mut assert_cmd::Command {
    cmd.env("JJ_INTERACTIVE", "1")
}

/// Coerces the value type to serialize it as TOML.
pub fn to_toml_value(value: impl Into<toml_edit::Value>) -> toml_edit::Value {
    value.into()
}

pub fn create_commit(work_dir: &TestWorkDir, name: &str, parents: &[&str]) {
    create_commit_with_files(work_dir, name, parents, &[(name, &format!("{name}\n"))]);
}

pub fn create_commit_with_files(
    work_dir: &TestWorkDir,
    name: &str,
    parents: &[&str],
    files: &[(&str, &str)],
) {
    let parents = match parents {
        [] => &["root()"],
        parents => parents,
    };
    work_dir
        .run_jj_with(|cmd| cmd.args(["new", "-m", name]).args(parents))
        .success();
    for (name, content) in files {
        work_dir.write_file(name, content);
    }
    work_dir
        .run_jj(["bookmark", "create", "-r@", name])
        .success();
}

/// Makes a criss-cross history: "a" and "b" change "f" from "o" to "x" and to
/// "y", and "p1" and "p2", two merges of "a" and "b", keep "x" and "y". The
/// merge of "p1" and "p2" is a conflict whose base is the virtual merge base of
/// "a" and "b".
pub fn create_criss_cross_commits(work_dir: &TestWorkDir) {
    create_criss_cross_commits_with(work_dir, "o\n", "x\n", "y\n");
}

/// Makes a criss-cross history: "o" holds `o` at "f", "a" and "b", children of
/// "o", change it to `a` and to `b`, and "p1" and "p2", two merges of "a" and
/// "b", keep `a` and `b`.
pub fn create_criss_cross_commits_with(work_dir: &TestWorkDir, o: &str, a: &str, b: &str) {
    create_commit_with_files(work_dir, "o", &[], &[("f", o)]);
    create_commit_with_files(work_dir, "a", &["o"], &[("f", a)]);
    create_commit_with_files(work_dir, "b", &["o"], &[("f", b)]);
    create_commit_with_files(work_dir, "p1", &["a", "b"], &[("f", a)]);
    create_commit_with_files(work_dir, "p2", &["a", "b"], &[("f", b)]);
}

/// The six lines of "f" in the histories whose merge bases change some of
/// them.
pub const SIX_LINES: &str = "1\n2\n3\n4\n5\n6\n";
/// "a"'s text in the block history: line 1 of `SIX_LINES` changed to "a".
pub const A_LINES: &str = "a\n2\n3\n4\n5\n6\n";
/// "b"'s text in the block history: lines 1 to 3 of `SIX_LINES` changed.
pub const B_LINES: &str = "b1\nb2\nb3\n4\n5\n6\n";
/// `B_LINES` with "b2" changed to "B2".
pub const B2_LINES: &str = "b1\nB2\nb3\n4\n5\n6\n";

/// Makes the block history: a criss-cross history in which "a" changes line 1
/// of "f" and "b" lines 1 to 3, and "d", a merge of "p1" and "p2" that
/// resolves "f" to "b"'s text with "b2" changed to "B2", inside the hunk where
/// "a" and "b" disagree.
pub fn create_b2_merge(work_dir: &TestWorkDir) {
    create_criss_cross_commits_with(work_dir, SIX_LINES, A_LINES, B_LINES);
    create_commit_with_files(work_dir, "d", &["p1", "p2"], &[("f", B2_LINES)]);
}

/// Asserts that no file of `revision`, resolved or conflicted, shows a marker
/// line of a virtual merge base, whole or cut: where a merge leaves a virtual
/// merge base as a side, it merges the terms of the conflict it stands for
/// instead, and a snapshot never writes its marker lines into another term,
/// so no term of the result holds them.
pub fn assert_no_virtual_merge_base_text(work_dir: &TestWorkDir, revision: &str) {
    let files = work_dir.run_jj(["file", "list", "-r", revision]).success();
    for path in files.stdout.raw().lines() {
        let content = work_dir
            .run_jj(["file", "show", "-r", revision, path])
            .success();
        assert!(
            !content
                .stdout
                .raw()
                .lines()
                .any(is_virtual_merge_base_marker),
            "{path} in {revision} holds a virtual merge base's text: {}",
            content.stdout.raw()
        );
    }
}

/// Whether `line`, as jj shows a file, after the `-`, `+` or ` ` of a diff if
/// any, is a marker line of a virtual merge base: an opening line, or a
/// separator or closing line with its 32 hex digits.
fn is_virtual_merge_base_marker(line: &str) -> bool {
    let line = line.strip_prefix(['-', '+', ' ']).unwrap_or(line);
    line.starts_with("<<<<<<< jj virtual merge base")
        || ["======= ", ">>>>>>> "].iter().any(|prefix| {
            line.strip_prefix(prefix).is_some_and(|hex| {
                hex.len() == 32 && hex.bytes().all(|byte| byte.is_ascii_hexdigit())
            })
        })
}
