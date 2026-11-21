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

//! The tests that need no gix-pack; the vendored gix-pack's own tests read
//! packs through this crate.

use std::env;
use std::ffi::OsStr;
use std::fs;
use std::fs::File;
use std::os::unix::ffi::OsStrExt as _;
use std::os::unix::fs::MetadataExt as _;
use std::path::Path;
use std::path::PathBuf;
use std::process::Command;

use crate::CHUNK_SIZE;
use crate::Mmap;
use crate::registry::parse_limit;
use crate::testing;

/// The file that a child test reads.
const CHILD_FILE: &str = "CHUNKED_MMAP_TEST_FILE";

/// Runs the ignored test `name` of this binary in a process of its own, with
/// `JJ_PACK_MAPPED_LIMIT` set to `limit` or unset, and returns what it
/// printed.
fn run_child(name: &str, file: &Path, limit: Option<&str>) -> String {
    let mut command = Command::new(env::current_exe().unwrap());
    command
        .args([
            "--exact",
            name,
            "--ignored",
            "--nocapture",
            "--test-threads=1",
        ])
        .env(CHILD_FILE, file);
    match limit {
        Some(limit) => command.env("JJ_PACK_MAPPED_LIMIT", limit),
        None => command.env_remove("JJ_PACK_MAPPED_LIMIT"),
    };
    let output = command.output().unwrap();
    let stdout = String::from_utf8_lossy(&output.stdout).into_owned();
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        output.status.success() && stdout.contains("1 passed"),
        "{name} failed:\n{stdout}\n{stderr}"
    );
    stdout
}

fn child_file() -> PathBuf {
    PathBuf::from(env::var_os(CHILD_FILE).expect("the parent test names the file"))
}

/// `len` bytes that differ for each `seed` and do not compress (SplitMix64).
fn random_bytes(seed: u64, len: usize) -> Vec<u8> {
    let mut state = seed;
    let mut bytes = Vec::with_capacity(len.next_multiple_of(8));
    while bytes.len() < len {
        state = state.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = state;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        bytes.extend_from_slice(&(z ^ (z >> 31)).to_le_bytes());
    }
    bytes.truncate(len);
    bytes
}

#[test]
fn budget_from_env() {
    for (value, budget) in [
        (None, 536_870_912),
        (Some("16777216"), 16 << 20),
        (Some("16m"), 16 << 20),
        (Some("512m"), 512 << 20),
        (Some("1g"), 1 << 30),
        (Some("0"), 0),
    ] {
        assert_eq!(parse_limit(value.map(OsStr::new)), Ok(budget), "{value:?}");
    }
    for value in [
        OsStr::new(""),
        OsStr::new("m"),
        OsStr::new("16k"),
        OsStr::new("16M"),
        OsStr::new("1G"),
        OsStr::new("-1"),
        OsStr::new(" 16m"),
        OsStr::new("16m "),
        OsStr::new("1.5g"),
        OsStr::new("99999999999999999999g"),
        OsStr::from_bytes(b"16\xffm"),
    ] {
        let error = parse_limit(Some(value)).unwrap_err();
        assert!(
            error.starts_with("JJ_PACK_MAPPED_LIMIT="),
            "{value:?}: {error}"
        );
    }

    // The same through the environment of a process of its own: unset, a
    // budget, and a value that every open refuses.
    let dir = tempfile::tempdir().unwrap();
    let file = dir.path().join("pack");
    fs::write(&file, b"pack").unwrap();
    for (limit, printed) in [
        (None, "budget 536870912\n"),
        (Some("16m"), "budget 16777216\n"),
        (
            Some("16k"),
            "refused: JJ_PACK_MAPPED_LIMIT=\"16k\" is not a number of bytes",
        ),
    ] {
        let stdout = run_child("tests::budget_from_env_child", &file, limit);
        assert!(stdout.contains(printed), "{limit:?}: {stdout}");
    }
}

#[test]
#[ignore = "run by budget_from_env in a process of its own"]
fn budget_from_env_child() {
    match Mmap::open(&child_file()) {
        Ok(_) => println!("budget {}", testing::limit()),
        Err(error) => println!("refused: {error}"),
    }
}

#[test]
fn rename_between_opens_reads_first_file() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("pack");
    let other = dir.path().join("other");
    let len = 3 * CHUNK_SIZE + 4321;
    let first = random_bytes(1, len);
    fs::write(&path, &first).unwrap();
    fs::write(&other, random_bytes(2, len)).unwrap();

    testing::set_between_opens_hook({
        let path = path.clone();
        move || fs::rename(&other, &path).unwrap()
    });
    let map = Mmap::open(&path).unwrap();
    assert!(
        fs::read(&path).unwrap() != first,
        "the path names the other file now"
    );
    for start in (0..len).step_by(CHUNK_SIZE) {
        let range = start..(start + CHUNK_SIZE).min(len);
        assert!(
            map.slice(range.clone()) == &first[range],
            "the chunk at {start} reads another file"
        );
    }
}

#[test]
fn rlimit_as_leaves_no_mapping() {
    let dir = tempfile::tempdir().unwrap();
    let file = dir.path().join("pack");
    File::create(&file)
        .unwrap()
        .set_len(u64::try_from(8 * CHUNK_SIZE + 4321).unwrap())
        .unwrap();
    let stdout = run_child("tests::rlimit_as_leaves_no_mapping_child", &file, None);
    assert!(
        stdout.contains("both opens failed and left no mapping"),
        "{stdout}"
    );
}

#[test]
#[ignore = "run by rlimit_as_leaves_no_mapping in a process of its own, since RLIMIT_AS is the \
            process's"]
fn rlimit_as_leaves_no_mapping_child() {
    let file = child_file();
    let metadata = fs::metadata(&file).unwrap();
    let len = usize::try_from(metadata.len()).unwrap();

    // The reservation cannot be made: the address space left is half the file.
    let before = address_space_bytes();
    let saved = soft_address_space_limit();
    set_soft_address_space_limit(before + len / 2);
    let result = Mmap::open(&file);
    set_soft_address_space_limit(saved);
    let error = result.expect_err("the reservation fits under RLIMIT_AS");
    assert_eq!(error.raw_os_error(), Some(libc::ENOMEM), "{error}");
    assert_no_mapping_of(metadata.ino(), before, len);

    // A chunk cannot be mapped: once the reservation and two chunks are, the
    // limit drops below the address space that the process holds.
    let before = address_space_bytes();
    let saved = soft_address_space_limit();
    testing::set_before_chunk_hook(Some(Box::new(|chunk: usize| {
        if chunk == 2 {
            set_soft_address_space_limit(testing::page_size());
        }
    })));
    let result = Mmap::open(&file);
    set_soft_address_space_limit(saved);
    testing::set_before_chunk_hook(None);
    let error = result.expect_err("the third chunk maps under RLIMIT_AS");
    assert_eq!(error.raw_os_error(), Some(libc::ENOMEM), "{error}");
    assert_no_mapping_of(metadata.ino(), before, len);

    println!("both opens failed and left no mapping");
}

/// Fails if /proc/self/maps holds a mapping of the file with `inode`, or the
/// address space grew by half of the file's `len` since it was `before`.
fn assert_no_mapping_of(inode: u64, before: usize, len: usize) {
    let inode = inode.to_string();
    let maps = fs::read_to_string("/proc/self/maps").unwrap();
    let of_file: Vec<&str> = maps
        .lines()
        .filter(|line| line.split_whitespace().nth(4) == Some(inode.as_str()))
        .collect();
    assert!(
        of_file.is_empty(),
        "mappings of the file are left:\n{}",
        of_file.join("\n")
    );
    let after = address_space_bytes();
    assert!(
        after < before + len / 2,
        "the address space grew from {before} to {after} bytes"
    );
}

/// The process's address space, `VmSize` of /proc/self/status.
fn address_space_bytes() -> usize {
    let status = fs::read_to_string("/proc/self/status").unwrap();
    let kib = status
        .lines()
        .find_map(|line| line.strip_prefix("VmSize:"))
        .and_then(|value| value.trim().strip_suffix("kB"))
        .unwrap()
        .trim();
    kib.parse::<usize>().unwrap() * 1024
}

fn soft_address_space_limit() -> libc::rlim_t {
    let mut limit = libc::rlimit {
        rlim_cur: 0,
        rlim_max: 0,
    };
    // SAFETY: getrlimit writes the limit it reads into `limit`.
    assert_eq!(
        unsafe { libc::getrlimit(libc::RLIMIT_AS, &raw mut limit) },
        0
    );
    limit.rlim_cur
}

/// Sets the soft `RLIMIT_AS` to `bytes`, which allocates nothing, so it can
/// run while the limit is below the address space the process holds.
fn set_soft_address_space_limit(bytes: impl TryInto<libc::rlim_t>) {
    let mut limit = libc::rlimit {
        rlim_cur: 0,
        rlim_max: 0,
    };
    // SAFETY: getrlimit writes the limit it reads into `limit`.
    assert_eq!(
        unsafe { libc::getrlimit(libc::RLIMIT_AS, &raw mut limit) },
        0
    );
    limit.rlim_cur = bytes
        .try_into()
        .unwrap_or(libc::RLIM_INFINITY)
        .min(limit.rlim_max);
    // SAFETY: setrlimit reads the limit from `limit`.
    assert_eq!(
        unsafe { libc::setrlimit(libc::RLIMIT_AS, &raw const limit) },
        0
    );
}
