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

//! The merge of many commits, against what it gave before the fold carried
//! the merge from one commit to the next without writing its trees.
//!
//! Each case is built from its seed alone, so the same seed gives the same
//! history at any commit of this repository, and the merge of its commits is
//! written as one line: the merge's tree ids and labels, then every path it
//! leaves unresolved with that path's terms. `test_merge_folds.given` holds
//! the line that each seed gave before the change, at 8053690b, read by
//! running `cargo test -p jj-lib --test runner -- test_merge_folds::print_given
//! --ignored --nocapture` there with `CARGO_PROFILE_DEV_DEBUG_ASSERTIONS=false`
//! and keeping what follows `GIVEN <seed> | ` on each line it prints. The
//! debug assertions are off because there a debug build's check that a
//! resolved merge resolves no further fails at one seed, which a release does
//! not check. A fold that answers differently at any seed fails here, and the
//! merge rules' own cases (`lib/tests/test_merge_rules.rs`) say which answer is
//! right.

use std::fmt::Write as _;
use std::panic;
use std::sync::Arc;

use itertools::Itertools as _;
use jj_lib::backend::TreeValue;
use jj_lib::commit::Commit;
use jj_lib::config::ConfigLayer;
use jj_lib::config::ConfigSource;
use jj_lib::merge::Merge;
use jj_lib::merged_tree::MergedTree;
use jj_lib::object_id::ObjectId as _;
use jj_lib::repo::ReadonlyRepo;
use jj_lib::repo::Repo as _;
use jj_lib::rewrite::merge_commit_trees;
use jj_lib::rewrite::merge_commit_trees_no_resolve;
use jj_lib::rewrite::steps_writing_trees;
use pollster::FutureExt as _;
use testutils::CommitBuilderExt as _;
use testutils::TestRepo;
use testutils::TestResult;
use testutils::base_user_config;
use testutils::create_tree;
use testutils::repo_path_buf;
use testutils::user_settings_from_config;

/// The number of seeds. Each is a history of its own.
const SEEDS: u64 = 200;

/// A small generator, so that a case depends on its seed and on nothing else.
struct Rand(u64);

impl Rand {
    fn new(seed: u64) -> Self {
        // An odd state, so that the generator never sticks at zero.
        Self(seed.wrapping_mul(6364136223846793005).wrapping_add(1))
    }

    /// The next number below `bound`.
    fn below(&mut self, bound: usize) -> usize {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 >> 11) as usize % bound
    }
}

/// The commits of the case `seed`: between three and sixteen, each a child of
/// one of a few commits above the root, so that some pairs of them have one
/// merge base and some have two or three, and some of them are merges of their
/// own. Each writes a few of six files, taking its text from a small set, so
/// that some commits make the same change at a path (terms that cancel), some
/// make different changes at the same path (a conflict that stands), and some
/// write a file nobody else writes. Now and then a commit holds a file where
/// others hold a directory, and now and then a commit's own tree is a
/// conflict.
fn case(repo: &TestRepo, seed: u64) -> TestResult<(Arc<ReadonlyRepo>, Vec<Commit>)> {
    let mut rand = Rand::new(seed);
    let files = (0..6)
        .map(|i| repo_path_buf(format!("dir{}/f{i}", i % 2)))
        .collect_vec();
    let dir1 = repo_path_buf("dir1");
    let texts = [
        "o\n1\n2\n",
        "a\n1\n2\n",
        "b\n1\n2\n",
        "o\n1\nx\n",
        "o\n1\ny\n",
    ];
    let tree_of = |rand: &mut Rand| {
        let count = 1 + rand.below(3);
        let mut contents = (0..count)
            .map(|_| {
                let file = &files[rand.below(files.len())];
                (file.as_ref(), texts[rand.below(texts.len())])
            })
            // A later write of the same file wins, as a tree holds one value.
            .unique_by(|(file, _)| *file)
            .collect_vec();
        if rand.below(8) == 0 {
            contents.retain(|(file, _)| !file.starts_with(&dir1));
            contents.push((dir1.as_ref(), texts[rand.below(texts.len())]));
        }
        create_tree(&repo.repo, &contents)
    };
    let mut tx = repo.repo.start_transaction();
    let root = repo.repo.store().root_commit_id().clone();
    // Two or three commits above the root, which the rest branch from, so that
    // a pair of them can have more than one merge base.
    let bases = (0..2 + seed as usize % 2)
        .map(|_| {
            let tree = tree_of(&mut rand);
            tx.repo_mut()
                .new_commit(vec![root.clone()], tree)
                .write_unwrap()
        })
        .collect_vec();
    let mut made = bases.clone();
    let mut parents = vec![];
    for _ in 0..3 + rand.below(14) {
        // A commit of one parent, or, now and then, a merge of two of the
        // commits made so far.
        let own = if made.len() > 1 && rand.below(4) == 0 {
            let a = rand.below(made.len());
            let b = (a + 1 + rand.below(made.len() - 1)) % made.len();
            vec![made[a].id().clone(), made[b].id().clone()]
        } else {
            vec![made[rand.below(made.len())].id().clone()]
        };
        let tree = if rand.below(6) == 0 {
            let terms =
                ["side 1", "base", "side 2"].map(|label| (tree_of(&mut rand), label.to_owned()));
            MergedTree::merge_no_resolve(Merge::from_vec(terms.to_vec()))
        } else {
            tree_of(&mut rand)
        };
        let commit = tx.repo_mut().new_commit(own, tree).write_unwrap();
        made.push(commit.clone());
        parents.push(commit);
    }
    let repo = tx.commit("test").block_on()?;
    Ok((repo, parents))
}

/// One line for a merge: its tree ids and labels, then each path it leaves
/// unresolved with that path's terms.
fn line(tree: &MergedTree) -> TestResult<String> {
    let mut out = tree.tree_ids().iter().map(|id| id.hex()).join(",");
    write!(out, " {:?}", tree.labels()).unwrap();
    for (path, value) in tree.conflicts() {
        let value = value?;
        write!(out, " {}=[", path.as_internal_file_string()).unwrap();
        let terms = value
            .iter()
            .map(|term| match term {
                None => "-".to_owned(),
                Some(TreeValue::File { id, executable, .. }) => {
                    format!("f{}{}", id.hex(), if *executable { "x" } else { "" })
                }
                Some(TreeValue::Tree(id)) => format!("t{}", id.hex()),
                Some(other) => format!("{other:?}"),
            })
            .join(",");
        write!(out, "{terms}]").unwrap();
    }
    Ok(out)
}

/// The two lines of a seed: the merge of its commits, and the same merge left
/// unresolved, each as [`line`] writes it or as the error it gave. One seed in
/// four merges under `merge.same-change=keep`. Commits take a fixed time, so
/// that their ids, which order the merge, depend on the seed alone.
fn lines(seed: u64) -> TestResult<(String, String)> {
    let same_change = if seed % 4 == 3 { "keep" } else { "accept" };
    let mut config = base_user_config();
    let text = format!(
        "merge.same-change = \"{same_change}\"\ndebug.commit-timestamp = \
         \"2001-02-03T04:05:06+07:00\"\n"
    );
    config.add_layer(ConfigLayer::parse(ConfigSource::User, &text)?);
    let test_repo = TestRepo::init_with_settings(&user_settings_from_config(config));
    let (repo, parents) = case(&test_repo, seed)?;
    let written = |tree: Result<MergedTree, _>| match tree {
        Ok(tree) => line(&tree),
        Err(err) => Ok(format!("error: {err}")),
    };
    let resolved = written(merge_commit_trees(repo.as_ref(), &parents).block_on())?;
    let unresolved = written(merge_commit_trees_no_resolve(repo.as_ref(), &parents).block_on())?;
    Ok((resolved, unresolved))
}

/// Prints what each seed gives, for `GIVEN`, or that it panicked. Ignored: it
/// is how `GIVEN` is made, not a check of its own.
#[test]
#[ignore]
fn print_given() -> TestResult<()> {
    for seed in 0..SEEDS {
        match panic::catch_unwind(|| lines(seed)) {
            Ok(lines) => {
                let (resolved, unresolved) = lines?;
                println!("GIVEN {seed} | {resolved} | {unresolved}");
            }
            Err(_) => println!("PANICKED {seed}"),
        }
    }
    Ok(())
}

/// Every seed gives what it gave before.
#[test]
fn test_fold_gives_what_it_gave() -> TestResult<()> {
    let given = GIVEN.lines().collect_vec();
    assert_eq!(
        given.len() as u64,
        SEEDS,
        "the table holds {} lines for {SEEDS} seeds; read it again with `print_given`",
        given.len()
    );
    for (seed, given) in given.into_iter().enumerate() {
        let (resolved, unresolved) = lines(seed as u64)?;
        assert_eq!(
            format!("{resolved} | {unresolved}"),
            given,
            "the merge of the commits of seed {seed} is not the merge it was"
        );
    }
    Ok(())
}

/// What each seed gave before the fold was changed, one line for each, read as
/// this file's documentation says.
const GIVEN: &str = include_str!("test_merge_folds.given");

/// The fold carries the merge from one commit to the next, without writing
/// the trees of its terms, at all but a few of the steps of the seeds' merges.
/// A merge whose result is right but whose carried state is not falls back to
/// writing the trees, which this sees and the lines above cannot: the written
/// step gives the same result.
#[test]
fn test_fold_carries_its_steps() -> TestResult<()> {
    let before = steps_writing_trees();
    for seed in 0..SEEDS {
        lines(seed)?;
    }
    let written = steps_writing_trees() - before;
    // Read from a run of these seeds. A step is written where a path of the
    // merge holds a directory in one commit and a file in another, where a
    // merge base's conflict is replaced by the terms it stands for, or where
    // the merge's own terms would lose a pair.
    let bound = WRITTEN_STEPS;
    assert!(
        written <= bound,
        "the merges of {SEEDS} seeds wrote the trees of {written} steps, more than {bound}"
    );
    Ok(())
}

/// The number of steps of the seeds' merges that write the trees of their
/// terms, read as this file's documentation says the table is read.
const WRITTEN_STEPS: usize = 2040;
