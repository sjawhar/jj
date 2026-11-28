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

#![expect(missing_docs)]

use std::any::Any;
use std::borrow::Cow;
use std::cmp::Ordering;
use std::collections::BTreeMap;
use std::collections::HashMap;
use std::collections::HashSet;
use std::error::Error;
use std::ffi::OsStr;
use std::ffi::OsString;
use std::fs;
use std::fs::DirEntry;
use std::fs::File;
use std::fs::Metadata;
use std::fs::OpenOptions;
use std::io;
use std::io::Read as _;
use std::io::Write;
use std::iter;
use std::mem;
use std::num::NonZeroUsize;
use std::ops::Range;
#[cfg(unix)]
use std::os::unix::fs::OpenOptionsExt as _;
#[cfg(unix)]
use std::os::unix::fs::PermissionsExt as _;
use std::panic;
use std::panic::AssertUnwindSafe;
use std::path::Path;
use std::path::PathBuf;
use std::slice;
use std::sync::Arc;
use std::sync::Condvar;
use std::sync::Mutex;
use std::sync::OnceLock;
use std::sync::PoisonError;
use std::sync::atomic::AtomicBool;
use std::sync::atomic::AtomicU64;
use std::sync::atomic::AtomicUsize;
use std::sync::atomic::Ordering as AtomicOrdering;
use std::sync::mpsc::Receiver;
use std::sync::mpsc::Sender;
use std::sync::mpsc::channel;
use std::thread;
use std::time::SystemTime;

use async_trait::async_trait;
use either::Either;
use futures::AsyncRead;
use futures::AsyncReadExt as _;
use futures::StreamExt as _;
use futures::io::AllowStdIo;
use itertools::EitherOrBoth;
use itertools::Itertools as _;
use once_cell::unsync::OnceCell;
use pollster::FutureExt as _;
use prost::Message as _;
use rayon::iter::IntoParallelIterator as _;
use rayon::prelude::IndexedParallelIterator as _;
use rayon::prelude::ParallelIterator as _;
use tempfile::NamedTempFile;
use thiserror::Error;
use tracing::instrument;
use tracing::trace_span;

use crate::backend::BackendError;
use crate::backend::BackendResult;
use crate::backend::CopyId;
use crate::backend::FileId;
use crate::backend::MergedTreeValue;
use crate::backend::MergedTreeValueExt as _;
use crate::backend::MillisSinceEpoch;
use crate::backend::SymlinkId;
use crate::backend::TreeId;
use crate::backend::TreeValue;
use crate::commit::Commit;
use crate::config::ConfigGetError;
use crate::config::ConfigGetResultExt as _;
use crate::conflict_labels::ConflictLabels;
use crate::conflicts;
use crate::conflicts::ConflictMarkerStyle;
use crate::conflicts::ConflictMaterializeOptions;
use crate::conflicts::MIN_CONFLICT_MARKER_LEN;
use crate::conflicts::MaterializedTreeValue;
use crate::conflicts::choose_materialized_conflict_marker_len;
use crate::conflicts::materialize_merge_result_to_bytes;
use crate::conflicts::materialize_tree_value;
pub use crate::eol::EolConversionMode;
use crate::eol::TargetEolStrategy;
use crate::file_util::FileIdentity;
use crate::file_util::check_symlink_support;
use crate::file_util::persist_temp_file;
use crate::file_util::symlink_file;
pub use crate::filter::FilterSettings;
use crate::filter::FilterStrategy;
use crate::fsmonitor::FsmonitorSettings;
#[cfg(feature = "watchman")]
use crate::fsmonitor::WatchmanConfig;
#[cfg(feature = "watchman")]
use crate::fsmonitor::watchman;
use crate::gitattributes::DiskFileLoader;
use crate::gitattributes::GitAttributes;
use crate::gitattributes::TreeFileLoader;
use crate::gitignore::GitIgnoreFile;
use crate::lock::FileLock;
use crate::matchers::DifferenceMatcher;
use crate::matchers::EverythingMatcher;
use crate::matchers::FilesMatcher;
use crate::matchers::IntersectionMatcher;
use crate::matchers::Matcher;
use crate::matchers::PrefixMatcher;
use crate::matchers::UnionMatcher;
use crate::merge::Merge;
use crate::merge::MergeBuilder;
use crate::merge::SameChange;
use crate::merged_tree::MergedTree;
use crate::merged_tree::TreeDiffEntry;
use crate::merged_tree_builder::MergedTreeBuilder;
use crate::object_id::ObjectId as _;
use crate::op_store::OperationId;
use crate::ref_name::WorkspaceName;
use crate::ref_name::WorkspaceNameBuf;
use crate::repo_path::RepoPath;
use crate::repo_path::RepoPathBuf;
use crate::repo_path::RepoPathComponent;
use crate::settings::UserSettings;
use crate::store::Store;
use crate::working_copy::CheckoutError;
use crate::working_copy::CheckoutStats;
use crate::working_copy::FilterIgnoreReason;
use crate::working_copy::LockedWorkingCopy;
use crate::working_copy::ResetError;
use crate::working_copy::SnapshotError;
use crate::working_copy::SnapshotOptions;
use crate::working_copy::SnapshotProgress;
use crate::working_copy::SnapshotStats;
use crate::working_copy::UntrackedReason;
use crate::working_copy::WorkingCopy;
use crate::working_copy::WorkingCopyFactory;
use crate::working_copy::WorkingCopyStateError;

fn symlink_target_convert_to_store(path: &Path) -> Option<Cow<'_, str>> {
    let path = path.to_str()?;
    if std::path::MAIN_SEPARATOR == '/' {
        Some(Cow::Borrowed(path))
    } else {
        // When storing the symlink target on Windows, convert "\" to "/", so that the
        // symlink remains valid on Unix.
        //
        // Note that we don't use std::path to handle the conversion, because it
        // performs poorly with Windows verbatim paths like \\?\Global\C:\file.txt.
        Some(Cow::Owned(path.replace(std::path::MAIN_SEPARATOR_STR, "/")))
    }
}

fn symlink_target_convert_to_disk(path: &str) -> PathBuf {
    let path = if std::path::MAIN_SEPARATOR == '/' {
        Cow::Borrowed(path)
    } else {
        // Use the main separator to reformat the input path to avoid creating a broken
        // symlink with the incorrect separator "/".
        //
        // See https://github.com/jj-vcs/jj/issues/6934 for the relevant bug.
        Cow::Owned(path.replace('/', std::path::MAIN_SEPARATOR_STR))
    };
    PathBuf::from(path.as_ref())
}

/// How to propagate executable bit changes in file metadata to/from the repo.
///
/// On Windows, executable bits are always ignored, but on Unix they are
/// respected by default, but may be ignored by user settings or if we find
/// that the filesystem of the working copy doesn't support executable bits.
#[derive(Clone, Copy, Debug)]
enum ExecChangePolicy {
    Ignore,
    #[cfg_attr(windows, expect(dead_code))]
    Respect,
}

/// The executable bit change setting as exposed to the user.
#[derive(Clone, Copy, Debug, Default, serde::Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum ExecChangeSetting {
    Ignore,
    Respect,
    #[default]
    Auto,
}

impl ExecChangePolicy {
    /// Get the executable bit policy based on user settings and executable bit
    /// support in the working copy's state path.
    ///
    /// On Unix we check whether executable bits are supported in the working
    /// copy to determine respect/ignorance, but we default to respect.
    #[cfg_attr(windows, expect(unused_variables))]
    fn new(exec_change_setting: ExecChangeSetting, state_path: &Path) -> Self {
        #[cfg(windows)]
        return Self::Ignore;
        #[cfg(unix)]
        return match exec_change_setting {
            ExecChangeSetting::Ignore => Self::Ignore,
            ExecChangeSetting::Respect => Self::Respect,
            ExecChangeSetting::Auto => {
                match crate::file_util::check_executable_bit_support(state_path) {
                    Ok(false) => Self::Ignore,
                    Ok(true) => Self::Respect,
                    Err(err) => {
                        tracing::warn!(?err, "Error when checking for executable bit support");
                        Self::Respect
                    }
                }
            }
        };
    }
}

/// On-disk state of file executable as cached in the file states. This does
/// *not* necessarily equal the `executable` field of [`TreeValue::File`]: the
/// two are allowed to diverge if and only if we're ignoring executable bit
/// changes.
///
/// This will only ever be true on Windows if the repo is also being accessed
/// from a Unix version of jj, such as when accessed from WSL.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ExecBit(bool);

impl ExecBit {
    /// Get the executable bit for a tree value to write to the repo store.
    ///
    /// If we're ignoring the executable bit, then we fallback to the previous
    /// in-repo executable bit if present.
    fn for_tree_value(
        self,
        exec_policy: ExecChangePolicy,
        prev_in_repo: impl FnOnce() -> Option<bool>,
    ) -> bool {
        match exec_policy {
            ExecChangePolicy::Ignore => prev_in_repo().unwrap_or(false),
            ExecChangePolicy::Respect => self.0,
        }
    }

    /// Set the on-disk executable bit to be written based on the in-repo bit or
    /// the previous on-disk executable bit.
    ///
    /// On Windows, we return `false` because when we later write files, we
    /// always create them anew, and the executable bit will be `false` even if
    /// shared with a Unix machine.
    ///
    /// `prev_on_disk` is a closure because it is somewhat expensive and is only
    /// used if ignoring the executable bit on Unix.
    fn new_from_repo(
        in_repo: bool,
        exec_policy: ExecChangePolicy,
        prev_on_disk: impl FnOnce() -> Option<Self>,
    ) -> Self {
        match exec_policy {
            _ if cfg!(windows) => Self(false),
            ExecChangePolicy::Ignore => prev_on_disk().unwrap_or(Self(false)),
            ExecChangePolicy::Respect => Self(in_repo),
        }
    }

    /// Load the on-disk executable bit from file metadata.
    #[cfg_attr(windows, expect(unused_variables))]
    fn new_from_disk(metadata: &Metadata) -> Self {
        #[cfg(unix)]
        return Self(metadata.permissions().mode() & 0o111 != 0);
        #[cfg(windows)]
        return Self(false);
    }

    /// The mode that a checkout creates a new file with on Unix: executable
    /// (0o755) or not (0o644), as far as the process's umask allows. Every
    /// file is written by creating it new, so this is the file's whole mode.
    /// It is also the mode that the checkout sets on an executable file that
    /// the umask left with no executable bit.
    #[cfg(unix)]
    fn new_file_mode(self) -> u32 {
        if self.0 { 0o755 } else { 0o644 }
    }
}

#[derive(Debug, PartialEq, Eq, Clone)]
pub enum FileType {
    Normal { exec_bit: ExecBit },
    Symlink,
    GitSubmodule,
}

#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub struct MaterializedConflictData {
    pub conflict_marker_len: u32,
}

#[derive(Debug, PartialEq, Eq, Clone)]
pub struct FileState {
    pub file_type: FileType,
    pub mtime: MillisSinceEpoch,
    pub size: u64,
    pub materialized_conflict_data: Option<MaterializedConflictData>,
    /* TODO: What else do we need here? Git stores a lot of fields.
     * TODO: Could possibly handle case-insensitive file systems keeping an
     *       Option<PathBuf> with the actual path here. */
}

impl FileState {
    /// Check whether a file state appears clean compared to a previous file
    /// state, ignoring materialized conflict data.
    pub fn is_clean(&self, old_file_state: &Self) -> bool {
        self.file_type == old_file_state.file_type
            && self.mtime == old_file_state.mtime
            && self.size == old_file_state.size
    }

    /// Indicates that a file exists in the tree but that it needs to be
    /// re-stat'ed on the next snapshot.
    fn placeholder() -> Self {
        Self {
            file_type: FileType::Normal {
                exec_bit: ExecBit(false),
            },
            mtime: MillisSinceEpoch(0),
            size: 0,
            materialized_conflict_data: None,
        }
    }

    fn for_file(
        exec_bit: ExecBit,
        size: u64,
        metadata: &Metadata,
    ) -> Result<Self, MtimeOutOfRange> {
        Ok(Self {
            file_type: FileType::Normal { exec_bit },
            mtime: mtime_from_metadata(metadata)?,
            size,
            materialized_conflict_data: None,
        })
    }

    fn for_symlink(metadata: &Metadata) -> Result<Self, MtimeOutOfRange> {
        // When using fscrypt, the reported size is not the content size. So if
        // we were to record the content size here (like we do for regular files), we
        // would end up thinking the file has changed every time we snapshot.
        Ok(Self {
            file_type: FileType::Symlink,
            mtime: mtime_from_metadata(metadata)?,
            size: metadata.len(),
            materialized_conflict_data: None,
        })
    }

    fn for_gitsubmodule() -> Self {
        Self {
            file_type: FileType::GitSubmodule,
            mtime: MillisSinceEpoch(0),
            size: 0,
            materialized_conflict_data: None,
        }
    }
}

/// Owned map of path to file states, backed by proto data.
#[derive(Clone, Debug)]
struct FileStatesMap {
    data: Vec<crate::protos::local_working_copy::FileStateEntry>,
}

impl FileStatesMap {
    fn new() -> Self {
        Self { data: Vec::new() }
    }

    fn from_proto(
        mut data: Vec<crate::protos::local_working_copy::FileStateEntry>,
        is_sorted: bool,
    ) -> Self {
        if !is_sorted {
            data.sort_unstable_by(|entry1, entry2| {
                let path1 = RepoPath::from_internal_string(&entry1.path).unwrap();
                let path2 = RepoPath::from_internal_string(&entry2.path).unwrap();
                path1.cmp(path2)
            });
        }
        debug_assert!(is_file_state_entries_proto_unique_and_sorted(&data));
        Self { data }
    }

    /// Merges changed and deleted entries into this map. The changed entries
    /// must be sorted by path.
    fn merge_in(
        &mut self,
        changed_file_states: Vec<(RepoPathBuf, FileState)>,
        deleted_files: &HashSet<RepoPathBuf>,
    ) {
        if changed_file_states.is_empty() && deleted_files.is_empty() {
            return;
        }
        debug_assert!(
            changed_file_states.is_sorted_by(|(path1, _), (path2, _)| path1 < path2),
            "changed_file_states must be sorted and have no duplicates"
        );
        self.data = itertools::merge_join_by(
            mem::take(&mut self.data),
            changed_file_states,
            |old_entry, (changed_path, _)| {
                RepoPath::from_internal_string(&old_entry.path)
                    .unwrap()
                    .cmp(changed_path)
            },
        )
        .filter_map(|diff| match diff {
            EitherOrBoth::Both(_, (path, state)) | EitherOrBoth::Right((path, state)) => {
                debug_assert!(!deleted_files.contains(&path));
                Some(file_state_entry_to_proto(path, &state))
            }
            EitherOrBoth::Left(entry) => {
                let present =
                    !deleted_files.contains(RepoPath::from_internal_string(&entry.path).unwrap());
                present.then_some(entry)
            }
        })
        .collect();
    }

    fn clear(&mut self) {
        self.data.clear();
    }

    /// Returns read-only map containing all file states.
    fn all(&self) -> FileStates<'_> {
        FileStates::from_sorted(&self.data)
    }
}

/// Read-only map of path to file states, possibly filtered by path prefix.
#[derive(Clone, Copy, Debug)]
pub struct FileStates<'a> {
    data: &'a [crate::protos::local_working_copy::FileStateEntry],
}

impl<'a> FileStates<'a> {
    fn from_sorted(data: &'a [crate::protos::local_working_copy::FileStateEntry]) -> Self {
        debug_assert!(is_file_state_entries_proto_unique_and_sorted(data));
        Self { data }
    }

    /// Returns file states under the given directory path.
    pub fn prefixed(&self, base: &RepoPath) -> Self {
        let range = self.prefixed_range(base);
        Self::from_sorted(&self.data[range])
    }

    /// Faster version of `prefixed("<dir>/<base>")`. Requires that all entries
    /// share the same prefix `dir`.
    fn prefixed_at(&self, dir: &RepoPath, base: &RepoPathComponent) -> Self {
        let range = self.prefixed_range_at(dir, base);
        Self::from_sorted(&self.data[range])
    }

    /// Returns true if this contains no entries.
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Returns true if the given `path` exists.
    pub fn contains_path(&self, path: &RepoPath) -> bool {
        self.exact_position(path).is_some()
    }

    /// Returns file state for the given `path`.
    pub fn get(&self, path: &RepoPath) -> Option<FileState> {
        let pos = self.exact_position(path)?;
        let (_, state) = file_state_entry_from_proto(&self.data[pos]);
        Some(state)
    }

    /// Returns the executable bit state if `path` is a normal file.
    pub fn get_exec_bit(&self, path: &RepoPath) -> Option<ExecBit> {
        match self.get(path)?.file_type {
            FileType::Normal { exec_bit } => Some(exec_bit),
            FileType::Symlink | FileType::GitSubmodule => None,
        }
    }

    /// Faster version of `get("<dir>/<name>")`. Requires that all entries share
    /// the same prefix `dir`.
    fn get_at(&self, dir: &RepoPath, name: &RepoPathComponent) -> Option<FileState> {
        let pos = self.exact_position_at(dir, name)?;
        let (_, state) = file_state_entry_from_proto(&self.data[pos]);
        Some(state)
    }

    fn exact_position(&self, path: &RepoPath) -> Option<usize> {
        self.data
            .binary_search_by(|entry| {
                RepoPath::from_internal_string(&entry.path)
                    .unwrap()
                    .cmp(path)
            })
            .ok()
    }

    fn exact_position_at(&self, dir: &RepoPath, name: &RepoPathComponent) -> Option<usize> {
        debug_assert!(self.paths().all(|path| path.starts_with(dir)));
        let slash_len = usize::from(!dir.is_root());
        let prefix_len = dir.as_internal_file_string().len() + slash_len;
        self.data
            .binary_search_by(|entry| {
                let tail = entry.path.get(prefix_len..).unwrap_or("");
                match tail.split_once('/') {
                    // "<name>/*" > "<name>"
                    Some((pre, _)) => pre.cmp(name.as_internal_str()).then(Ordering::Greater),
                    None => tail.cmp(name.as_internal_str()),
                }
            })
            .ok()
    }

    fn prefixed_range(&self, base: &RepoPath) -> Range<usize> {
        let start = self
            .data
            .partition_point(|entry| RepoPath::from_internal_string(&entry.path).unwrap() < base);
        let len = self.data[start..].partition_point(|entry| {
            RepoPath::from_internal_string(&entry.path)
                .unwrap()
                .starts_with(base)
        });
        start..(start + len)
    }

    fn prefixed_range_at(&self, dir: &RepoPath, base: &RepoPathComponent) -> Range<usize> {
        debug_assert!(self.paths().all(|path| path.starts_with(dir)));
        let slash_len = usize::from(!dir.is_root());
        let prefix_len = dir.as_internal_file_string().len() + slash_len;
        let start = self.data.partition_point(|entry| {
            let tail = entry.path.get(prefix_len..).unwrap_or("");
            let entry_name = tail.split_once('/').map_or(tail, |(name, _)| name);
            entry_name < base.as_internal_str()
        });
        let len = self.data[start..].partition_point(|entry| {
            let tail = entry.path.get(prefix_len..).unwrap_or("");
            let entry_name = tail.split_once('/').map_or(tail, |(name, _)| name);
            entry_name == base.as_internal_str()
        });
        start..(start + len)
    }

    /// Iterates file state entries sorted by path.
    pub fn iter(&self) -> FileStatesIter<'a> {
        self.data.iter().map(file_state_entry_from_proto)
    }

    /// Iterates sorted file paths.
    pub fn paths(&self) -> impl ExactSizeIterator<Item = &'a RepoPath> + use<'a> {
        self.data
            .iter()
            .map(|entry| RepoPath::from_internal_string(&entry.path).unwrap())
    }
}

type FileStatesIter<'a> = iter::Map<
    slice::Iter<'a, crate::protos::local_working_copy::FileStateEntry>,
    fn(&crate::protos::local_working_copy::FileStateEntry) -> (&RepoPath, FileState),
>;

impl<'a> IntoIterator for FileStates<'a> {
    type Item = (&'a RepoPath, FileState);
    type IntoIter = FileStatesIter<'a>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

fn file_state_from_proto(proto: &crate::protos::local_working_copy::FileState) -> FileState {
    let file_type = match proto.file_type() {
        crate::protos::local_working_copy::FileType::Normal => FileType::Normal {
            exec_bit: ExecBit(false),
        },
        // On Windows, `FileType::Executable` can exist if the repo is being
        // shared with a Unix version of jj, such as when accessed from WSL.
        crate::protos::local_working_copy::FileType::Executable => FileType::Normal {
            exec_bit: ExecBit(true),
        },
        crate::protos::local_working_copy::FileType::Symlink => FileType::Symlink,
        #[expect(deprecated)]
        crate::protos::local_working_copy::FileType::Conflict => FileType::Normal {
            exec_bit: ExecBit(false),
        },
        crate::protos::local_working_copy::FileType::GitSubmodule => FileType::GitSubmodule,
    };
    FileState {
        file_type,
        mtime: MillisSinceEpoch(proto.mtime_millis_since_epoch),
        size: proto.size,
        materialized_conflict_data: proto.materialized_conflict_data.as_ref().map(|data| {
            MaterializedConflictData {
                conflict_marker_len: data.conflict_marker_len,
            }
        }),
    }
}

fn file_state_to_proto(file_state: &FileState) -> crate::protos::local_working_copy::FileState {
    let mut proto = crate::protos::local_working_copy::FileState::default();
    let file_type = match &file_state.file_type {
        FileType::Normal { exec_bit } => {
            if exec_bit.0 {
                crate::protos::local_working_copy::FileType::Executable
            } else {
                crate::protos::local_working_copy::FileType::Normal
            }
        }
        FileType::Symlink => crate::protos::local_working_copy::FileType::Symlink,
        FileType::GitSubmodule => crate::protos::local_working_copy::FileType::GitSubmodule,
    };
    proto.file_type = file_type as i32;
    proto.mtime_millis_since_epoch = file_state.mtime.0;
    proto.size = file_state.size;
    proto.materialized_conflict_data = file_state.materialized_conflict_data.map(|data| {
        crate::protos::local_working_copy::MaterializedConflictData {
            conflict_marker_len: data.conflict_marker_len,
        }
    });
    proto
}

fn file_state_entry_from_proto(
    proto: &crate::protos::local_working_copy::FileStateEntry,
) -> (&RepoPath, FileState) {
    let path = RepoPath::from_internal_string(&proto.path).unwrap();
    (path, file_state_from_proto(proto.state.as_ref().unwrap()))
}

fn file_state_entry_to_proto(
    path: RepoPathBuf,
    state: &FileState,
) -> crate::protos::local_working_copy::FileStateEntry {
    crate::protos::local_working_copy::FileStateEntry {
        path: path.into_internal_string(),
        state: Some(file_state_to_proto(state)),
    }
}

fn is_file_state_entries_proto_unique_and_sorted(
    data: &[crate::protos::local_working_copy::FileStateEntry],
) -> bool {
    data.iter()
        .map(|entry| RepoPath::from_internal_string(&entry.path).unwrap())
        .is_sorted_by(|path1, path2| path1 < path2)
}

fn sparse_patterns_from_proto(
    proto: Option<&crate::protos::local_working_copy::SparsePatterns>,
) -> Vec<RepoPathBuf> {
    let mut sparse_patterns = vec![];
    if let Some(proto_sparse_patterns) = proto {
        for prefix in &proto_sparse_patterns.prefixes {
            sparse_patterns.push(RepoPathBuf::from_internal_string(prefix).unwrap());
        }
    } else {
        // For compatibility with old working copies.
        // TODO: Delete this is late 2022 or so.
        sparse_patterns.push(RepoPathBuf::root());
    }
    sparse_patterns
}

/// Creates intermediate directories from the `working_copy_path` to the
/// `repo_path` parent. Returns disk path for the `repo_path` file.
///
/// If an intermediate directory exists and if it is a file or symlink, this
/// function returns `Ok(None)` to signal that the path should be skipped.
/// The `working_copy_path` directory may be a symlink.
///
/// If an existing or newly-created sub directory points to ".git" or ".jj",
/// this function returns an error.
///
/// Note that this does not prevent TOCTOU bugs caused by concurrent checkouts.
/// Another process may remove the directory created by this function and put a
/// symlink there.
fn create_parent_dirs(
    working_copy_path: &Path,
    repo_path: &RepoPath,
) -> Result<Option<PathBuf>, CheckoutError> {
    let (parent_path, basename) = repo_path.split().expect("repo path shouldn't be root");
    let mut dir_path = working_copy_path.to_owned();
    for c in parent_path.components() {
        // Ensure that the name is a normal entry of the current dir_path.
        let name = c.to_fs_name().map_err(|err| err.with_path(repo_path))?;
        dir_path.push(name);
        // A directory named ".git" or ".jj" can be temporarily created. It
        // might trick workspace path discovery, but is harmless so long as the
        // directory is empty.
        let (new_dir_created, is_dir) = match fs::create_dir(&dir_path) {
            Ok(()) => (true, true), // New directory
            Err(err) => match dir_path.symlink_metadata() {
                Ok(m) => (false, m.is_dir()), // Existing file or directory
                Err(_) => {
                    return Err(CheckoutError::Other {
                        message: format!(
                            "Failed to create parent directories for {}",
                            repo_path.to_fs_path_unchecked(working_copy_path).display(),
                        ),
                        err: err.into(),
                    });
                }
            },
        };
        // Invalid component (e.g. "..") should have been rejected.
        // The current dir_path should be an entry of dir_path.parent().
        if could_alias_reserved_name(OsStr::new(name)) {
            reject_reserved_existing_path(&dir_path).inspect_err(|_| {
                if new_dir_created {
                    fs::remove_dir(&dir_path).ok();
                }
            })?;
        }
        if !is_dir {
            return Ok(None); // Skip existing file or symlink
        }
    }

    let mut file_path = dir_path;
    file_path.push(
        basename
            .to_fs_name()
            .map_err(|err| err.with_path(repo_path))?,
    );
    Ok(Some(file_path))
}

/// Removes existing file named `disk_path` if any. Returns `Ok(true)` if the
/// file was there and got removed, meaning that new file can be safely created.
///
/// If the existing file points to ".git" or ".jj", this function returns an
/// error.
fn remove_old_file(disk_path: &Path) -> Result<bool, CheckoutError> {
    if disk_path.file_name().is_none_or(could_alias_reserved_name) {
        reject_reserved_existing_path(disk_path)?;
    }
    match fs::remove_file(disk_path) {
        Ok(()) => Ok(true),
        Err(err) if err.kind() == io::ErrorKind::NotFound => Ok(false),
        // Not checking io::ErrorKind::IsADirectory because unlink() of a
        // directory fails with EPERM on macOS (as allowed by POSIX.)
        Err(_) if disk_path.symlink_metadata().is_ok_and(|m| m.is_dir()) => Ok(false),
        Err(err) => Err(CheckoutError::Other {
            message: format!("Failed to remove file {}", disk_path.display()),
            err: err.into(),
        }),
    }
}

/// Removes existing submodule directory named `disk_path` if any. Returns
/// `Ok(true)` if the directory was there and got removed, meaning that new file
/// can be safely created.
///
/// The directory will not be removed if it is not empty, as it could contain
/// untracked or modified files. This is in line with Git's behavior.
fn remove_old_submodule_dir(disk_path: &Path) -> Result<bool, CheckoutError> {
    match fs::remove_dir(disk_path) {
        Ok(()) => Ok(true),
        Err(err) if err.kind() == io::ErrorKind::NotFound => Ok(false),
        Err(err) if err.kind() == io::ErrorKind::DirectoryNotEmpty => Ok(false),
        Err(err) => Err(CheckoutError::Other {
            message: format!(
                "Failed to remove submodule directory {}",
                disk_path.display()
            ),
            err: err.into(),
        }),
    }
}

/// Creates the file `disk_path` that a checkout writes new: once, without
/// following a symlink, and on Unix with the mode of `exec_bit` as far as the
/// process's umask allows.
///
/// If an entry is already at `disk_path`, returns `Ok(None)` to signal that the
/// path should be skipped, unless `old_file_removed` says that this checkout
/// has just removed the file there: then it is an error.
///
/// If the new file is the entry that the name ".git" or ".jj" of its directory
/// points to (a name alias on a case-insensitive file system, for example),
/// git/jj CLI could be tricked to load configuration from an
/// attacker-controlled location. So the file is removed and an error returned
/// before anything is written to it. Only a name that
/// [`could_alias_reserved_name()`] is checked so.
fn create_new_file(
    disk_path: &Path,
    exec_bit: ExecBit,
    old_file_removed: bool,
    dir_cache: &mut DirCache,
) -> Result<Option<File>, CheckoutError> {
    let open_error = || format!("Failed to open file {} for writing", disk_path.display());
    let file = match dir_cache.create_new(disk_path, exec_bit) {
        Ok(file) => file,
        Err(err) if old_file_removed => {
            return Err(CheckoutError::Other {
                message: open_error(),
                err: err.into(),
            });
        }
        Err(err) => {
            reject_existing_entry(disk_path, err, open_error)?;
            return Ok(None);
        }
    };
    if disk_path.file_name().is_none_or(could_alias_reserved_name) {
        reject_reserved_existing_file(&file, disk_path).inspect_err(|_| {
            // We keep the error from `reject_reserved_existing_file`
            fs::remove_file(disk_path).ok();
        })?;
    }
    Ok(Some(file))
}

/// What a checkout thread keeps open to create new files relative to a
/// directory (`openat()`), which looks up one name instead of every component
/// of the path: the working copy root, and the deepest directory below it that
/// the thread last created a file in (at most two open descriptors at rest,
/// and at most three while walking down).
///
/// The directories below the root are opened one name at a time from the root
/// down (or from the kept directory when descending into it), and none through
/// a symlink: each is opened with `O_NOFOLLOW` (and not with `O_DIRECTORY`,
/// with which some systems follow a symlink to a directory despite
/// `O_NOFOLLOW`), checked to be a directory with `fstat()`, and each parent is
/// dropped as soon as its child is open. So regular files are never created
/// through a symlink below the root (except under an unreadable directory,
/// where every component is checked with `symlink_metadata()` first), even if
/// a directory the coordinating thread created was replaced by a symlink
/// since. The root itself may be a symlink.
///
/// A directory may still be removed and made again while it is kept (when a
/// removed path is the same entry on a case-insensitive file system, say). So
/// a create that fails drops the kept directory, and a create that finds a
/// kept directory gone (`ENOENT`) walks again from the root, once.
struct DirCache {
    #[cfg(unix)]
    working_copy_path: PathBuf,
    /// The working copy root, once a file was created.
    #[cfg(unix)]
    root: Option<rustix::fd::OwnedFd>,
    /// The deepest directory below the root that is kept, with its path
    /// relative to the root.
    #[cfg(unix)]
    dir: Option<(PathBuf, rustix::fd::OwnedFd)>,
}

impl DirCache {
    #[cfg_attr(windows, expect(unused_variables))]
    fn new(working_copy_path: &Path) -> Self {
        Self {
            #[cfg(unix)]
            working_copy_path: working_copy_path.to_owned(),
            #[cfg(unix)]
            root: None,
            #[cfg(unix)]
            dir: None,
        }
    }

    /// Creates `disk_path`, which is in the working copy, new (`O_CREAT |
    /// O_EXCL`, which does not follow a symlink there) for writing, with the
    /// mode of `exec_bit`.
    fn create_new(&mut self, disk_path: &Path, exec_bit: ExecBit) -> io::Result<File> {
        #[cfg(unix)]
        return self.create_new_in_dir(disk_path, exec_bit);
        #[cfg(windows)]
        return create_new_by_path(disk_path, exec_bit);
    }

    #[cfg(unix)]
    fn create_new_in_dir(&mut self, disk_path: &Path, exec_bit: ExecBit) -> io::Result<File> {
        use rustix::fs::Mode;
        use rustix::fs::OFlags;
        use rustix::fs::RawMode;
        use rustix::io::Errno;
        let (Some(name), Some(dir)) = (
            disk_path.file_name(),
            disk_path
                .parent()
                .and_then(|parent| parent.strip_prefix(&self.working_copy_path).ok()),
        ) else {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!("{} is not in the working copy", disk_path.display()),
            ));
        };
        let mode = Mode::from_raw_mode(exec_bit.new_file_mode() as RawMode);
        loop {
            let reused = self
                .dir
                .as_ref()
                .is_some_and(|(kept_path, _)| dir.starts_with(kept_path));
            let created = match self.open_dir(dir) {
                Ok(dir_fd) => rustix::fs::openat(
                    dir_fd,
                    name,
                    OFlags::WRONLY | OFlags::CREATE | OFlags::EXCL | OFlags::CLOEXEC,
                    mode,
                ),
                // A directory with write and search permission but no read
                // permission (mode 0o300, say) can't be opened, but a file can
                // be created in it by path.
                Err(Errno::ACCESS) => {
                    self.dir = None;
                    return self.create_new_in_unreadable_dir(disk_path, dir, exec_bit);
                }
                Err(err) => Err(err),
            };
            match created {
                Ok(file) => return Ok(File::from(file)),
                Err(err) => {
                    self.dir = None;
                    // A kept directory is gone if it was removed since: walk
                    // again from the root, once.
                    if !(reused && err == Errno::NOENT) {
                        return Err(err.into());
                    }
                }
            }
        }
    }

    /// Opens the working copy root if it is not open, and walks to `dir`, a
    /// path relative to the root, one component at a time without following a
    /// symlink. Drops each intermediate descriptor as soon as its child is
    /// open, keeping at most the root and `dir`.
    #[cfg(unix)]
    fn open_dir(&mut self, dir: &Path) -> rustix::io::Result<&rustix::fd::OwnedFd> {
        use rustix::fs::FileType;
        use rustix::fs::Mode;
        use rustix::fs::OFlags;
        use rustix::io::Errno;
        if self.root.is_none() {
            self.root = Some(rustix::fs::open(
                self.working_copy_path.as_path(),
                OFlags::RDONLY | OFlags::DIRECTORY | OFlags::CLOEXEC,
                Mode::empty(),
            )?);
        }
        if dir.as_os_str().is_empty() {
            self.dir = None;
            return Ok(self.root.as_ref().unwrap());
        }
        if let Some((kept_path, _)) = &self.dir
            && kept_path == dir
        {
            return Ok(&self.dir.as_ref().unwrap().1);
        }
        let (mut cur, tail) = match self.dir.take() {
            Some((kept_path, kept_fd)) => match dir.strip_prefix(&kept_path) {
                Ok(tail) => (Some(kept_fd), tail),
                Err(_) => (None, dir),
            },
            None => (None, dir),
        };
        let root = self.root.as_ref().unwrap();
        for name in tail {
            let parent = cur.as_ref().unwrap_or(root);
            // `O_NONBLOCK`, so that the open of a FIFO there does not wait for
            // a writer before the check of the entry's type refuses it.
            let fd = rustix::fs::openat(
                parent,
                name,
                OFlags::RDONLY | OFlags::NOFOLLOW | OFlags::NONBLOCK | OFlags::CLOEXEC,
                Mode::empty(),
            )?;
            if FileType::from_raw_mode(rustix::fs::fstat(&fd)?.st_mode) != FileType::Directory {
                return Err(Errno::NOTDIR);
            }
            cur = Some(fd);
        }
        let fd = cur.expect("non-empty relative directory has at least one component");
        Ok(&self.dir.insert((dir.to_owned(), fd)).1)
    }

    /// Creates `disk_path` new by its path, in the directory `dir` (relative
    /// to the working copy root) that can't be opened: only if each directory
    /// from the root down to it is a directory and not a symlink.
    #[cfg(unix)]
    fn create_new_in_unreadable_dir(
        &self,
        disk_path: &Path,
        dir: &Path,
        exec_bit: ExecBit,
    ) -> io::Result<File> {
        let mut dir_path = self.working_copy_path.clone();
        for name in dir {
            dir_path.push(name);
            if !dir_path.symlink_metadata()?.is_dir() {
                return Err(rustix::io::Errno::NOTDIR.into());
            }
        }
        create_new_by_path(disk_path, exec_bit)
    }
}

/// Creates `disk_path` new by its path, without following a symlink there,
/// with the mode of `exec_bit` on Unix.
#[cfg_attr(windows, expect(unused_variables))]
fn create_new_by_path(disk_path: &Path, exec_bit: ExecBit) -> io::Result<File> {
    let mut options = OpenOptions::new();
    // Don't overwrite un-ignored file. Don't follow symlink.
    options.write(true).create_new(true);
    #[cfg(unix)]
    options.mode(exec_bit.new_file_mode());
    options.open(disk_path)
}

/// Handles the error `err` of a checkout's create of a new entry at
/// `disk_path`: returns `Ok(())` if an entry is already there, so that the
/// path is skipped, and an error if the entry may point to a ".git" or ".jj"
/// entry. If no entry is there, returns `err` itself with the message that
/// `message` makes.
fn reject_existing_entry(
    disk_path: &Path,
    err: io::Error,
    message: impl FnOnce() -> String,
) -> Result<(), CheckoutError> {
    // Workaround for "Access is denied. (os error 5)" error on Windows, which
    // an entry that is already there may cause instead of `AlreadyExists`.
    if err.kind() == io::ErrorKind::AlreadyExists {
        reject_reserved_existing_path(disk_path)
    } else if has_existing_entry(disk_path)? {
        Ok(())
    } else {
        Err(CheckoutError::Other {
            message: message(),
            err: err.into(),
        })
    }
}

/// Whether an entry is at `disk_path`, which the checkout then leaves as it
/// is. Returns an error if the entry may point to a ".git" or ".jj" entry.
fn has_existing_entry(disk_path: &Path) -> Result<bool, CheckoutError> {
    match disk_path.symlink_metadata() {
        Ok(_) => {
            reject_reserved_existing_path(disk_path)?;
            Ok(true)
        }
        Err(err) if err.kind() == io::ErrorKind::NotFound => Ok(false),
        Err(err) => Err(CheckoutError::Other {
            message: format!("Failed to stat {}", disk_path.display()),
            err: err.into(),
        }),
    }
}

/// Whether the entry name `name` could point to the ".git" or ".jj" entry of
/// its directory on some file system, as git's own name rules see it: a name
/// that is ".git" or ".jj" in any ASCII case and with trailing dots or spaces
/// (Windows), and any name that [`has_special_alias_char()`] is true for. Any
/// other name never does, so the checks of the entry's identity against ".git"
/// and ".jj" are run only for these names.
fn could_alias_reserved_name(name: &OsStr) -> bool {
    let Some(name) = name.to_str() else {
        return true;
    };
    if has_special_alias_char(name) {
        return true;
    }
    let trimmed = name.trim_end_matches(['.', ' ']);
    RESERVED_DIR_NAMES
        .iter()
        .any(|reserved| trimmed.eq_ignore_ascii_case(reserved))
}

/// Whether the entry name `name` holds a character by which some file system
/// may take it for an entry of another name: `~` (8.3 short names), `:` (NTFS
/// streams), `\`, or any character that is not ASCII (HFS+ ignores some code
/// points, case-insensitive file systems fold more than ASCII, and some file
/// systems take two Unicode normalizations of a name for one entry).
fn has_special_alias_char(name: &str) -> bool {
    !name.is_ascii() || name.contains(['~', ':', '\\'])
}

const RESERVED_DIR_NAMES: &[&str] = &[".git", ".jj"];

fn file_identity_from_symlink_path(disk_path: &Path) -> io::Result<Option<FileIdentity>> {
    match FileIdentity::from_symlink_path(disk_path) {
        Ok(identity) => Ok(Some(identity)),
        Err(err) if err.kind() == io::ErrorKind::NotFound => Ok(None),
        Err(err) => Err(err),
    }
}

/// Wrapper for [`reject_reserved_existing_file_identity`] which avoids a
/// syscall by converting the provided `file` to a `FileIdentity` via its
/// file descriptor.
///
/// See [`reject_reserved_existing_file_identity`] for more info.
fn reject_reserved_existing_file(file: &File, disk_path: &Path) -> Result<(), CheckoutError> {
    // Note: since the file is open, we don't expect that it's possible for
    // `io::ErrorKind::NotFound` to be a possible error returned here.
    let file_identity = FileIdentity::from_file(file).map_err(|err| CheckoutError::Other {
        message: format!("Failed to validate path {}", disk_path.display()),
        err: err.into(),
    })?;

    reject_reserved_existing_file_identity(file_identity, disk_path)
}

/// Wrapper for [`reject_reserved_existing_file_identity`] which converts
/// the provided `disk_path` to a `FileIdentity`.
///
/// See [`reject_reserved_existing_file_identity`] for more info.
///
/// # Remarks
///
/// On Windows, this incurs an additional syscall cost to open and close the
/// file `HANDLE` for `disk_path`. On Unix, `lstat()` is used.
fn reject_reserved_existing_path(disk_path: &Path) -> Result<(), CheckoutError> {
    let Some(disk_identity) =
        file_identity_from_symlink_path(disk_path).map_err(|err| CheckoutError::Other {
            message: format!("Failed to validate path {}", disk_path.display()),
            err: err.into(),
        })?
    else {
        // If the existing disk_path pointed to the reserved path, we would have
        // gotten an identity back. Since we got nothing, the file does not exist
        // and cannot be a reserved path name.
        return Ok(());
    };

    reject_reserved_existing_file_identity(disk_identity, disk_path)
}

/// Suppose the `disk_path` exists, checks if the last component points to
/// ".git" or ".jj" in the same parent directory.
///
/// `disk_identity` is expected to be an identity of the file described by
/// `disk_path`.
///
/// # Remarks
///
/// On Windows, this incurs a syscall cost to open and close a file `HANDLE` for
/// each filename in `RESERVED_DIR_NAMES`. On Unix, `lstat()` is used.
fn reject_reserved_existing_file_identity(
    disk_identity: FileIdentity,
    disk_path: &Path,
) -> Result<(), CheckoutError> {
    let parent_dir_path = disk_path.parent().expect("content path shouldn't be root");
    for name in RESERVED_DIR_NAMES {
        let reserved_path = parent_dir_path.join(name);

        let Some(reserved_identity) =
            file_identity_from_symlink_path(&reserved_path).map_err(|err| {
                CheckoutError::Other {
                    message: format!("Failed to validate path {}", disk_path.display()),
                    err: err.into(),
                }
            })?
        else {
            // If the existing disk_path pointed to the reserved path, we would have
            // gotten an identity back. Since we got nothing, the file does not exist
            // and cannot be a reserved path name.
            continue;
        };

        if disk_identity == reserved_identity {
            return Err(CheckoutError::ReservedPathComponent {
                path: disk_path.to_owned(),
                name,
            });
        }
    }

    Ok(())
}

#[derive(Debug, Error)]
#[error("Out-of-range file modification time")]
struct MtimeOutOfRange;

fn mtime_from_metadata(metadata: &Metadata) -> Result<MillisSinceEpoch, MtimeOutOfRange> {
    let time = metadata
        .modified()
        .expect("File mtime not supported on this platform?");
    system_time_to_millis(time).ok_or(MtimeOutOfRange)
}

fn system_time_to_millis(time: SystemTime) -> Option<MillisSinceEpoch> {
    let millis = match time.duration_since(SystemTime::UNIX_EPOCH) {
        Ok(duration) => i64::try_from(duration.as_millis()).ok()?,
        Err(err) => -i64::try_from(err.duration().as_millis()).ok()?,
    };
    Some(MillisSinceEpoch(millis))
}

/// Create a new [`FileState`] from metadata.
fn file_state(metadata: &Metadata) -> Result<Option<FileState>, MtimeOutOfRange> {
    let metadata_file_type = metadata.file_type();
    let file_type = if metadata_file_type.is_dir() {
        None
    } else if metadata_file_type.is_symlink() {
        Some(FileType::Symlink)
    } else if metadata_file_type.is_file() {
        let exec_bit = ExecBit::new_from_disk(metadata);
        Some(FileType::Normal { exec_bit })
    } else {
        None
    };
    if let Some(file_type) = file_type {
        Ok(Some(FileState {
            file_type,
            mtime: mtime_from_metadata(metadata)?,
            size: metadata.len(),
            materialized_conflict_data: None,
        }))
    } else {
        Ok(None)
    }
}

struct FsmonitorMatcher {
    matcher: Option<Box<dyn Matcher>>,
    watchman_clock: Option<crate::protos::local_working_copy::WatchmanClock>,
    should_update_watchman_clock: bool,
}

/// Settings specific to the tree state of the [`LocalWorkingCopy`] backend.
#[derive(Clone, Debug)]
pub struct TreeStateSettings {
    /// Conflict marker style to use when materializing files or when checking
    /// changed files.
    pub conflict_marker_style: ConflictMarkerStyle,
    /// Configuring auto-converting CRLF line endings into LF when you add a
    /// file to the backend, and vice versa when it checks out code onto your
    /// filesystem.
    pub eol_conversion_mode: EolConversionMode,
    /// Whether to ignore changes to the executable bit for files on Unix.
    pub exec_change_setting: ExecChangeSetting,
    /// The fsmonitor (e.g. Watchman) to use, if any.
    pub fsmonitor_settings: FsmonitorSettings,
    pub filter_settings: FilterSettings,
    /// The number of threads that a checkout reads and writes files on.
    pub checkout_threads: NonZeroUsize,
}

impl TreeStateSettings {
    /// Create [`TreeStateSettings`] from [`UserSettings`].
    pub fn try_from_user_settings(user_settings: &UserSettings) -> Result<Self, ConfigGetError> {
        Ok(Self {
            conflict_marker_style: user_settings.get("ui.conflict-marker-style")?,
            eol_conversion_mode: EolConversionMode::try_from_settings(user_settings)?,
            exec_change_setting: user_settings.get("working-copy.exec-bit-change")?,
            fsmonitor_settings: FsmonitorSettings::from_settings(user_settings)?,
            filter_settings: FilterSettings::try_from_settings(user_settings)?,
            checkout_threads: checkout_threads_from_settings(user_settings)?,
        })
    }
}

/// The number of threads that a checkout reads and writes files on, as
/// `working-copy.checkout-threads` sets it, or [`default_checkout_threads()`]
/// if it is unset.
pub fn checkout_threads_from_settings(
    user_settings: &UserSettings,
) -> Result<NonZeroUsize, ConfigGetError> {
    Ok(user_settings
        .get("working-copy.checkout-threads")
        .optional()?
        .unwrap_or_else(default_checkout_threads))
}

/// The number of threads that a checkout reads and writes files on when
/// `working-copy.checkout-threads` is unset: the number of CPUs that the
/// process may use, or 1 if that is unknown.
pub fn default_checkout_threads() -> NonZeroUsize {
    thread::available_parallelism().unwrap_or(NonZeroUsize::MIN)
}

pub struct TreeState {
    store: Arc<Store>,
    working_copy_path: PathBuf,
    state_path: PathBuf,
    tree: MergedTree,
    file_states: FileStatesMap,
    // Currently only path prefixes
    sparse_patterns: Vec<RepoPathBuf>,
    own_mtime: MillisSinceEpoch,
    symlink_support: bool,

    /// The most recent clock value returned by Watchman. Will only be set if
    /// the repo is configured to use the Watchman filesystem monitor and
    /// Watchman has been queried at least once.
    watchman_clock: Option<crate::protos::local_working_copy::WatchmanClock>,

    conflict_marker_style: ConflictMarkerStyle,
    exec_policy: ExecChangePolicy,
    fsmonitor_settings: FsmonitorSettings,
    target_eol_strategy: TargetEolStrategy,
    filter_strategy: FilterStrategy,
    /// Whether a checkout converts file contents for the working copy: by EOL
    /// conversion, or by a smudge filter that the gitattributes may name. If
    /// not, the contents go to disk as the store has them.
    update_converts_contents: bool,
    checkout_threads: NonZeroUsize,
    /// The bound on the bytes of file contents that a checkout holds read and
    /// not yet written, unless one file is larger.
    max_bytes_in_flight: u64,
    last_checkout_measures: CheckoutMeasures,
}

#[derive(Debug, Error)]
pub enum TreeStateError {
    #[error("Reading tree state from {path}")]
    ReadTreeState { path: PathBuf, source: io::Error },
    #[error("Decoding tree state from {path}")]
    DecodeTreeState {
        path: PathBuf,
        source: prost::DecodeError,
    },
    #[error("Writing tree state to temporary file {path}")]
    WriteTreeState { path: PathBuf, source: io::Error },
    #[error("Persisting tree state to file {path}")]
    PersistTreeState { path: PathBuf, source: io::Error },
    #[error("Filesystem monitor error")]
    Fsmonitor(#[source] Box<dyn Error + Send + Sync>),
}

impl TreeState {
    pub fn working_copy_path(&self) -> &Path {
        &self.working_copy_path
    }

    pub fn current_tree(&self) -> &MergedTree {
        &self.tree
    }

    pub fn file_states(&self) -> FileStates<'_> {
        self.file_states.all()
    }

    pub fn sparse_patterns(&self) -> &Vec<RepoPathBuf> {
        &self.sparse_patterns
    }

    fn sparse_matcher(&self) -> Box<dyn Matcher> {
        Box::new(PrefixMatcher::new(&self.sparse_patterns))
    }

    pub fn init(
        store: Arc<Store>,
        working_copy_path: PathBuf,
        state_path: PathBuf,
        tree_state_settings: &TreeStateSettings,
    ) -> Result<Self, TreeStateError> {
        let mut wc = Self::empty(store, working_copy_path, state_path, tree_state_settings);
        wc.save()?;
        Ok(wc)
    }

    /// Like `init` but does not persist the initial empty tree state to
    /// disk. Use when the caller will save state itself only after a
    /// successful operation (e.g. to use `tree_state` file absence as a
    /// dirty marker).
    pub fn init_without_saving(
        store: Arc<Store>,
        working_copy_path: PathBuf,
        state_path: PathBuf,
        tree_state_settings: &TreeStateSettings,
    ) -> Self {
        Self::empty(store, working_copy_path, state_path, tree_state_settings)
    }

    fn empty(
        store: Arc<Store>,
        working_copy_path: PathBuf,
        state_path: PathBuf,
        TreeStateSettings {
            conflict_marker_style,
            eol_conversion_mode,
            exec_change_setting,
            fsmonitor_settings,
            filter_settings,
            checkout_threads,
        }: &TreeStateSettings,
    ) -> Self {
        let exec_policy = ExecChangePolicy::new(*exec_change_setting, &state_path);
        Self {
            store: store.clone(),
            working_copy_path: working_copy_path.clone(),
            state_path,
            tree: store.empty_merged_tree(),
            file_states: FileStatesMap::new(),
            sparse_patterns: vec![RepoPathBuf::root()],
            own_mtime: MillisSinceEpoch(0),
            symlink_support: check_symlink_support().unwrap_or(false),
            watchman_clock: None,
            conflict_marker_style: *conflict_marker_style,
            exec_policy,
            fsmonitor_settings: fsmonitor_settings.clone(),
            target_eol_strategy: TargetEolStrategy::new(*eol_conversion_mode),
            filter_strategy: FilterStrategy::new(working_copy_path, filter_settings.clone()),
            update_converts_contents: *eol_conversion_mode == EolConversionMode::InputOutput
                || filter_settings.enabled,
            checkout_threads: *checkout_threads,
            max_bytes_in_flight: DEFAULT_MAX_BYTES_IN_FLIGHT,
            last_checkout_measures: CheckoutMeasures::default(),
        }
    }

    pub fn load(
        store: Arc<Store>,
        working_copy_path: PathBuf,
        state_path: PathBuf,
        tree_state_settings: &TreeStateSettings,
    ) -> Result<Self, TreeStateError> {
        let tree_state_path = state_path.join("tree_state");
        let file = match File::open(&tree_state_path) {
            Err(err) if err.kind() == io::ErrorKind::NotFound => {
                return Self::init(store, working_copy_path, state_path, tree_state_settings);
            }
            Err(err) => {
                return Err(TreeStateError::ReadTreeState {
                    path: tree_state_path,
                    source: err,
                });
            }
            Ok(file) => file,
        };

        let mut wc = Self::empty(store, working_copy_path, state_path, tree_state_settings);
        wc.read(&tree_state_path, file)?;
        Ok(wc)
    }

    fn update_own_mtime(&mut self) {
        if let Ok(metadata) = self.state_path.join("tree_state").symlink_metadata()
            && let Ok(mtime) = mtime_from_metadata(&metadata)
        {
            self.own_mtime = mtime;
        } else {
            self.own_mtime = MillisSinceEpoch(0);
        }
    }

    fn read(&mut self, tree_state_path: &Path, mut file: File) -> Result<(), TreeStateError> {
        self.update_own_mtime();
        let mut buf = Vec::new();
        file.read_to_end(&mut buf)
            .map_err(|err| TreeStateError::ReadTreeState {
                path: tree_state_path.to_owned(),
                source: err,
            })?;
        let proto = crate::protos::local_working_copy::TreeState::decode(&*buf).map_err(|err| {
            TreeStateError::DecodeTreeState {
                path: tree_state_path.to_owned(),
                source: err,
            }
        })?;
        #[expect(deprecated)]
        if proto.tree_ids.is_empty() {
            self.tree = MergedTree::resolved(
                self.store.clone(),
                TreeId::new(proto.legacy_tree_id.clone()),
            );
        } else {
            let tree_ids_builder: MergeBuilder<TreeId> = proto
                .tree_ids
                .iter()
                .map(|id| TreeId::new(id.clone()))
                .collect();
            self.tree = MergedTree::new(
                self.store.clone(),
                tree_ids_builder.build(),
                ConflictLabels::from_vec(proto.conflict_labels),
            );
        }
        self.file_states =
            FileStatesMap::from_proto(proto.file_states, proto.is_file_states_sorted);
        self.sparse_patterns = sparse_patterns_from_proto(proto.sparse_patterns.as_ref());
        self.watchman_clock = proto.watchman_clock;
        Ok(())
    }

    #[expect(clippy::assigning_clones, clippy::field_reassign_with_default)]
    pub fn save(&mut self) -> Result<(), TreeStateError> {
        let mut proto: crate::protos::local_working_copy::TreeState = Default::default();
        proto.tree_ids = self
            .tree
            .tree_ids()
            .iter()
            .map(|id| id.to_bytes())
            .collect();
        proto.conflict_labels = self.tree.labels().as_slice().to_owned();
        proto.file_states = self.file_states.data.clone();
        // `FileStatesMap` is guaranteed to be sorted.
        proto.is_file_states_sorted = true;
        let mut sparse_patterns = crate::protos::local_working_copy::SparsePatterns::default();
        for path in &self.sparse_patterns {
            sparse_patterns
                .prefixes
                .push(path.as_internal_file_string().to_owned());
        }
        proto.sparse_patterns = Some(sparse_patterns);
        proto.watchman_clock = self.watchman_clock.clone();

        let wrap_write_err = |source| TreeStateError::WriteTreeState {
            path: self.state_path.clone(),
            source,
        };
        let mut temp_file = NamedTempFile::new_in(&self.state_path).map_err(wrap_write_err)?;
        temp_file
            .as_file_mut()
            .write_all(&proto.encode_to_vec())
            .map_err(wrap_write_err)?;
        // update own write time while we before we rename it, so we know
        // there is no unknown data in it
        self.update_own_mtime();
        // TODO: Retry if persisting fails (it will on Windows if the file happened to
        // be open for read).
        let target_path = self.state_path.join("tree_state");
        persist_temp_file(temp_file, &target_path).map_err(|source| {
            TreeStateError::PersistTreeState {
                path: target_path.clone(),
                source,
            }
        })?;
        Ok(())
    }

    fn reset_watchman(&mut self) {
        self.watchman_clock.take();
    }

    #[cfg(feature = "watchman")]
    #[instrument(skip(self))]
    pub async fn query_watchman(
        &self,
        config: &WatchmanConfig,
    ) -> Result<(watchman::Clock, Option<Vec<PathBuf>>), TreeStateError> {
        let previous_clock = self.watchman_clock.clone().map(watchman::Clock::from);

        let tokio_fn = async || {
            let fsmonitor = watchman::Fsmonitor::init(&self.working_copy_path, config)
                .await
                .map_err(|err| TreeStateError::Fsmonitor(Box::new(err)))?;
            fsmonitor
                .query_changed_files(previous_clock)
                .await
                .map_err(|err| TreeStateError::Fsmonitor(Box::new(err)))
        };

        match tokio::runtime::Handle::try_current() {
            Ok(_handle) => tokio_fn().await,
            Err(_) => {
                let runtime = tokio::runtime::Builder::new_current_thread()
                    .enable_all()
                    .build()
                    .map_err(|err| TreeStateError::Fsmonitor(Box::new(err)))?;
                runtime.block_on(tokio_fn())
            }
        }
    }

    #[cfg(feature = "watchman")]
    #[instrument(skip(self))]
    pub async fn is_watchman_trigger_registered(
        &self,
        config: &WatchmanConfig,
    ) -> Result<bool, TreeStateError> {
        let tokio_fn = async || {
            let fsmonitor = watchman::Fsmonitor::init(&self.working_copy_path, config)
                .await
                .map_err(|err| TreeStateError::Fsmonitor(Box::new(err)))?;
            fsmonitor
                .is_trigger_registered()
                .await
                .map_err(|err| TreeStateError::Fsmonitor(Box::new(err)))
        };

        match tokio::runtime::Handle::try_current() {
            Ok(_handle) => tokio_fn().await,
            Err(_) => {
                let runtime = tokio::runtime::Builder::new_current_thread()
                    .enable_all()
                    .build()
                    .map_err(|err| TreeStateError::Fsmonitor(Box::new(err)))?;
                runtime.block_on(tokio_fn())
            }
        }
    }
}

/// Functions to snapshot local-disk files to the store.
impl TreeState {
    /// Look for changes to the working copy. If there are any changes, create
    /// a new tree from it.
    #[instrument(skip_all)]
    pub async fn snapshot(
        &mut self,
        options: &SnapshotOptions<'_>,
    ) -> Result<(bool, SnapshotStats), SnapshotError> {
        let SnapshotOptions {
            base_ignores,
            progress,
            start_tracking_matcher,
            force_tracking_matcher,
            max_new_file_size,
        } = options;

        let sparse_matcher = self.sparse_matcher();

        let mut is_dirty = false;
        let FsmonitorMatcher {
            matcher: fsmonitor_matcher,
            watchman_clock,
            should_update_watchman_clock,
        } = self
            .make_fsmonitor_matcher(&self.fsmonitor_settings)
            .await?;
        let fsmonitor_matcher = match fsmonitor_matcher.as_ref() {
            None => &EverythingMatcher,
            Some(fsmonitor_matcher) => fsmonitor_matcher.as_ref(),
        };

        let matcher = IntersectionMatcher::new(
            sparse_matcher.as_ref(),
            UnionMatcher::new(fsmonitor_matcher, force_tracking_matcher),
        );
        if matcher.visit(RepoPath::root()).is_nothing() {
            // No need to load the current tree, set up channels, etc.
            if should_update_watchman_clock {
                is_dirty |= self.watchman_clock != watchman_clock;
                self.watchman_clock = watchman_clock;
            }
            return Ok((is_dirty, SnapshotStats::default()));
        }

        let (tree_entries_tx, tree_entries_rx) = channel();
        let (file_states_tx, file_states_rx) = channel();
        let (untracked_paths_tx, untracked_paths_rx) = channel();
        let (invalid_utf8_paths_tx, invalid_utf8_paths_rx) = channel();
        let (unconverted_files_tx, unconverted_files_rx) = channel();
        let (deleted_files_tx, deleted_files_rx) = channel();
        let git_attributes = GitAttributes::new(vec![
            Arc::new(DiskFileLoader::new(self.working_copy_path.clone())),
            Arc::new(TreeFileLoader::new(self.tree.clone())),
        ]);

        trace_span!("traverse filesystem").in_scope(|| -> Result<(), SnapshotError> {
            let snapshotter = FileSnapshotter {
                tree_state: self,
                current_tree: &self.tree,
                matcher: &matcher,
                sparse_matcher: sparse_matcher.as_ref(),
                start_tracking_matcher,
                force_tracking_matcher,
                // Move tx sides so they'll be dropped at the end of the scope.
                tree_entries_tx,
                file_states_tx,
                untracked_paths_tx,
                invalid_utf8_paths_tx,
                unconverted_files_tx,
                deleted_files_tx,
                error: OnceLock::new(),
                progress: *progress,
                max_new_file_size: *max_new_file_size,
                git_attributes: &git_attributes,
            };
            let directory_to_visit = DirectoryToVisit {
                dir: RepoPathBuf::root(),
                disk_dir: self.working_copy_path.clone(),
                git_ignore: base_ignores.clone(),
                file_states: self.file_states.all(),
            };
            // Here we use scope as a queue of per-directory jobs.
            rayon::scope(|scope| {
                snapshotter.spawn_ok(scope, |scope| {
                    snapshotter.visit_directory(directory_to_visit, scope)
                });
            });
            snapshotter.into_result()
        })?;

        let stats = SnapshotStats {
            untracked_paths: untracked_paths_rx.into_iter().collect(),
            invalid_utf8_paths: invalid_utf8_paths_rx.into_iter().collect(),
            unconverted_paths: unconverted_files_rx.into_iter().collect(),
        };
        let mut tree_builder = MergedTreeBuilder::new(self.tree.clone());
        trace_span!("process tree entries").in_scope(|| {
            for (path, tree_values) in &tree_entries_rx {
                tree_builder.set_or_remove(path, tree_values);
            }
        });
        let deleted_files = trace_span!("process deleted tree entries").in_scope(|| {
            let deleted_files = HashSet::from_iter(deleted_files_rx);
            is_dirty |= !deleted_files.is_empty();
            for file in &deleted_files {
                tree_builder.set_or_remove(file.clone(), Merge::absent());
            }
            deleted_files
        });
        trace_span!("process file states").in_scope(|| {
            let changed_file_states = file_states_rx
                .iter()
                .sorted_unstable_by(|(path1, _), (path2, _)| path1.cmp(path2))
                .collect_vec();
            is_dirty |= !changed_file_states.is_empty();
            self.file_states
                .merge_in(changed_file_states, &deleted_files);
        });
        trace_span!("write tree")
            .in_scope(async || -> Result<(), BackendError> {
                let new_tree = tree_builder.write_tree().await?;
                is_dirty |= new_tree.tree_ids_and_labels() != self.tree.tree_ids_and_labels();
                self.tree = new_tree.clone();
                Ok(())
            })
            .await?;
        if cfg!(debug_assertions) {
            let tree_paths: HashSet<_> = self
                .tree
                .entries_matching(sparse_matcher.as_ref())
                .filter_map(|(path, result)| result.is_ok().then_some(path))
                .collect();
            let file_states = self.file_states.all();
            let state_paths: HashSet<_> = file_states.paths().map(|path| path.to_owned()).collect();
            assert_eq!(state_paths, tree_paths);
        }
        // Since untracked paths aren't cached in the tree state, we'll need to
        // rescan the working directory changes to report or track them later.
        // TODO: store untracked paths and update watchman_clock?
        // A failed monitor query has no clock and causes a full scan. Clear the
        // old clock after that scan even if it found untracked paths.
        if should_update_watchman_clock {
            if (stats.untracked_paths.is_empty() && stats.invalid_utf8_paths.is_empty())
                || watchman_clock.is_none()
            {
                is_dirty |= self.watchman_clock != watchman_clock;
                self.watchman_clock = watchman_clock;
            } else {
                tracing::info!("not updating watchman clock because there are untracked files");
            }
        }
        Ok((is_dirty, stats))
    }

    #[instrument(skip_all)]
    async fn make_fsmonitor_matcher(
        &self,
        fsmonitor_settings: &FsmonitorSettings,
    ) -> Result<FsmonitorMatcher, SnapshotError> {
        let (watchman_clock, changed_files) = match fsmonitor_settings {
            FsmonitorSettings::None => (None, None),
            FsmonitorSettings::Test { changed_files } => (None, Some(changed_files.clone())),
            #[cfg(feature = "watchman")]
            FsmonitorSettings::Watchman(config) => match self.query_watchman(config).await {
                Ok((watchman_clock, changed_files)) => (Some(watchman_clock.into()), changed_files),
                Err(err) => {
                    tracing::warn!(?err, "Failed to query filesystem monitor");
                    (None, None)
                }
            },
            #[cfg(not(feature = "watchman"))]
            FsmonitorSettings::Watchman(_) => {
                return Err(SnapshotError::Other {
                    message: "Failed to query the filesystem monitor".to_string(),
                    err: "Cannot query Watchman because jj was not compiled with the `watchman` \
                          feature (consider disabling `fsmonitor.backend`)"
                        .into(),
                });
            }
        };
        // Do not advance the clock when the monitor reports no changed paths.
        // This avoids rewriting the tree state just to save the new clock.
        // Reusing the older clock may make a later query do more work, but it
        // cannot omit changes.
        let should_update_watchman_clock = changed_files
            .as_ref()
            .is_none_or(|changed_files| !changed_files.is_empty());
        let matcher: Option<Box<dyn Matcher>> = match changed_files {
            None => None,
            Some(changed_files) => {
                let (repo_paths, gitignore_prefixes) = trace_span!("processing fsmonitor paths")
                    .in_scope(|| {
                        let repo_paths = changed_files
                            .iter()
                            .filter_map(|path| RepoPathBuf::from_relative_path(path).ok())
                            .collect_vec();
                        // .gitignore changes require rescanning parent directories to pick up newly
                        // unignored files.
                        let gitignore_prefixes = repo_paths
                            .iter()
                            .filter_map(|repo_path| {
                                let (parent, basename) = repo_path.split()?;
                                (basename.as_internal_str() == ".gitignore")
                                    .then(|| parent.to_owned())
                            })
                            .collect_vec();
                        (repo_paths, gitignore_prefixes)
                    });

                let matcher: Box<dyn Matcher> = if gitignore_prefixes.is_empty() {
                    Box::new(FilesMatcher::new(repo_paths))
                } else {
                    Box::new(UnionMatcher::new(
                        FilesMatcher::new(repo_paths),
                        PrefixMatcher::new(gitignore_prefixes),
                    ))
                };

                Some(matcher)
            }
        };
        Ok(FsmonitorMatcher {
            matcher,
            watchman_clock,
            should_update_watchman_clock,
        })
    }
}

struct DirectoryToVisit<'a> {
    dir: RepoPathBuf,
    disk_dir: PathBuf,
    git_ignore: Arc<GitIgnoreFile>,
    file_states: FileStates<'a>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum PresentDirEntryKind {
    Dir,
    File,
}

#[derive(Clone, Debug)]
struct PresentDirEntries {
    dirs: HashSet<String>,
    files: HashSet<String>,
}

/// Helper to scan local-disk directories and files in parallel.
struct FileSnapshotter<'a> {
    tree_state: &'a TreeState,
    current_tree: &'a MergedTree,
    matcher: &'a dyn Matcher,
    sparse_matcher: &'a dyn Matcher,
    start_tracking_matcher: &'a dyn Matcher,
    force_tracking_matcher: &'a dyn Matcher,
    tree_entries_tx: Sender<(RepoPathBuf, MergedTreeValue)>,
    file_states_tx: Sender<(RepoPathBuf, FileState)>,
    untracked_paths_tx: Sender<(RepoPathBuf, UntrackedReason)>,
    invalid_utf8_paths_tx: Sender<(RepoPathBuf, OsString)>,
    unconverted_files_tx: Sender<(RepoPathBuf, FilterIgnoreReason)>,
    deleted_files_tx: Sender<RepoPathBuf>,
    error: OnceLock<SnapshotError>,
    progress: Option<&'a SnapshotProgress<'a>>,
    max_new_file_size: u64,
    git_attributes: &'a GitAttributes,
}

impl FileSnapshotter<'_> {
    fn spawn_ok<'scope, F>(&'scope self, scope: &rayon::Scope<'scope>, body: F)
    where
        F: FnOnce(&rayon::Scope<'scope>) -> Result<(), SnapshotError> + Send + 'scope,
    {
        scope.spawn(|scope| {
            if self.error.get().is_some() {
                return;
            }
            match body(scope) {
                Ok(()) => {}
                Err(err) => self.error.set(err).unwrap_or(()),
            }
        });
    }

    /// Extracts the result of the snapshot.
    fn into_result(self) -> Result<(), SnapshotError> {
        match self.error.into_inner() {
            Some(err) => Err(err),
            None => Ok(()),
        }
    }

    /// Visits the directory entries, spawns jobs to recurse into sub
    /// directories.
    fn visit_directory<'scope>(
        &'scope self,
        directory_to_visit: DirectoryToVisit<'scope>,
        scope: &rayon::Scope<'scope>,
    ) -> Result<(), SnapshotError> {
        let DirectoryToVisit {
            dir,
            disk_dir,
            mut git_ignore,
            file_states,
        } = directory_to_visit;

        let git_ignore_file_name = RepoPathComponent::new(".gitignore").expect("static string");
        if let Some(git_ignore_contents) = self
            .read_gitignore(&disk_dir, &dir, git_ignore_file_name)
            .block_on()?
        {
            let git_ignore_disk_path = disk_dir.join(git_ignore_file_name.as_internal_str());
            git_ignore = git_ignore.chain(&dir, &git_ignore_disk_path, &git_ignore_contents)?;
        }
        let dir_entries: Vec<_> = disk_dir
            .read_dir()
            .and_then(|entries| entries.try_collect())
            .map_err(|err| SnapshotError::Other {
                message: format!("Failed to read directory {}", disk_dir.display()),
                err: err.into(),
            })?;
        let (dirs, files) = dir_entries
            .into_par_iter()
            // Don't split into too many small jobs. For a small directory,
            // sequential scan should be fast enough.
            .with_min_len(100)
            .filter_map(|entry| {
                self.process_dir_entry(&dir, &git_ignore, file_states, &entry, scope)
                    .block_on()
                    .transpose()
            })
            .map(|item| match item {
                Ok((PresentDirEntryKind::Dir, name)) => Ok(Either::Left(name)),
                Ok((PresentDirEntryKind::File, name)) => Ok(Either::Right(name)),
                Err(err) => Err(err),
            })
            .collect::<Result<_, _>>()?;
        let present_entries = PresentDirEntries { dirs, files };
        self.emit_deleted_files(&dir, file_states, &present_entries);
        Ok(())
    }

    /// Loads the ignore file named `file_name` and returns its contents if it
    /// exists, supports sparse working copies.
    ///
    /// * `disk_dir` is the directory on disk containing the ignore file
    /// * `repo_dir` is the corresponding repo directory used to acquire the
    ///   fallback ignore file if it is excluded by the sparse patterns.
    /// * `file_name` is the name of the ignore file (e.g. `.gitignore`)
    ///
    /// If the file isn't materialized on disk because it is excluded by the sparse
    /// patterns, it is read from the repository directory instead to make sure
    /// that ignored files are not tracked.
    ///
    /// Function also applies special handling for symlinks, `.gitignore` files are only
    /// processed if they are regular files, symlinks and other filesystem entities are
    /// explicitly ignored.
    async fn read_gitignore(
        &self,
        disk_dir: &Path,
        repo_dir: &RepoPath,
        file_name: &RepoPathComponent,
    ) -> Result<Option<Vec<u8>>, SnapshotError> {
        let ignore_disk_path = disk_dir.join(file_name.as_internal_str());
        match ignore_disk_path.symlink_metadata() {
            Ok(symlink_metadata) => {
                // TODO: If the path is excluded by the sparse patterns, the potential on-disk .gitignore file
                // in that dir will never be snapshotted. Should we prefer the tree version in that case?
                // Currently we match Git, which prefers on-disk content.
                return symlink_metadata
                    .is_file()
                    .then(|| {
                        fs::read(&ignore_disk_path).map_err(|err| SnapshotError::Other {
                            message: format!(
                                "Failed to read ignore patterns from file {}",
                                ignore_disk_path.display()
                            ),
                            err: Box::new(err),
                        })
                    })
                    .transpose();
            }
            Err(err) if err.kind() == io::ErrorKind::NotFound => {
                // File doesn't exist, try to read from the repo dir.
            }
            Err(err) => {
                return Err(SnapshotError::Other {
                    message: format!(
                        "Error accessing ignore patterns file {}",
                        ignore_disk_path.display()
                    ),
                    err: Box::new(err),
                });
            }
        }

        let ignore_fallback_repo_path = repo_dir.join(file_name);
        if self.sparse_matcher.matches(&ignore_fallback_repo_path) {
            // The file should have been materialized, so its absence means it was deleted from the working copy.
            return Ok(None);
        }

        let tree_values = self
            .current_tree
            .path_value(&ignore_fallback_repo_path)
            .await?;
        let file_id = match tree_values.as_normal() {
            Some(TreeValue::File { id, .. }) => id,
            None
            | Some(TreeValue::Symlink(_) | TreeValue::GitSubmodule(_) | TreeValue::Tree(_)) => {
                return Ok(None);
            }
        };
        let mut buf = vec![];
        let mut reader = self
            .store()
            .read_file(&ignore_fallback_repo_path, file_id)
            .await?;
        reader
            .read_to_end(&mut buf)
            .await
            .map_err(|err| BackendError::ReadFile {
                path: ignore_fallback_repo_path,
                id: file_id.clone(),
                source: err.into(),
            })?;
        Ok(Some(buf))
    }

    async fn process_dir_entry<'scope>(
        &'scope self,
        dir: &RepoPath,
        git_ignore: &Arc<GitIgnoreFile>,
        file_states: FileStates<'scope>,
        entry: &DirEntry,
        scope: &rayon::Scope<'scope>,
    ) -> Result<Option<(PresentDirEntryKind, String)>, SnapshotError> {
        let file_type = entry.file_type().unwrap();
        let file_name = entry.file_name();
        let name_string = match file_name.into_string() {
            Ok(name_string) => name_string,
            Err(name) => {
                // A path that isn't valid UTF-8 can't be represented as a
                // RepoPath, so it can never be tracked. Skip it instead of
                // failing the whole snapshot, and let the caller report it.
                self.invalid_utf8_paths_tx.send((dir.to_owned(), name)).ok();
                return Ok(None);
            }
        };

        if RESERVED_DIR_NAMES.contains(&name_string.as_str()) {
            return Ok(None);
        }
        let name = RepoPathComponent::new(&name_string).unwrap();
        let path = dir.join(name);
        let maybe_current_file_state = file_states.get_at(dir, name);
        if let Some(file_state) = &maybe_current_file_state
            && file_state.file_type == FileType::GitSubmodule
        {
            return Ok(None);
        }

        if file_type.is_dir() {
            let file_states = file_states.prefixed_at(dir, name);
            // If a submodule was added in commit C, and a user decides to run
            // `jj new <something before C>` from after C, then the submodule
            // files stick around but it is no longer seen as a submodule.
            // We need to ensure that it is not tracked as if it was added to
            // the main repo.
            // See https://github.com/jj-vcs/jj/issues/4349.
            // To solve this, we ignore all nested repos entirely.
            let disk_dir = entry.path();
            for &name in RESERVED_DIR_NAMES {
                if disk_dir.join(name).symlink_metadata().is_ok() {
                    return Ok(None);
                }
            }

            if git_ignore.matches_dir(&path)
                && self.force_tracking_matcher.visit(&path).is_nothing()
            {
                // If the whole directory is ignored by .gitignore, visit only
                // paths we're already tracking. This is because .gitignore in
                // ignored directory must be ignored. It's also more efficient.
                // start_tracking_matcher is NOT tested here because we need to
                // scan directory entries to report untracked paths.
                self.spawn_ok(scope, move |_| {
                    self.visit_tracked_files(file_states).block_on()
                });
            } else if !self.matcher.visit(&path).is_nothing() {
                let directory_to_visit = DirectoryToVisit {
                    dir: path,
                    disk_dir,
                    git_ignore: git_ignore.clone(),
                    file_states,
                };
                self.spawn_ok(scope, |scope| {
                    self.visit_directory(directory_to_visit, scope)
                });
            }
            // Whether or not the directory path matches, any child file entries
            // shouldn't be touched within the current recursion step.
            Ok(Some((PresentDirEntryKind::Dir, name_string)))
        } else if self.matcher.matches(&path) {
            if let Some(progress) = self.progress {
                progress(&path);
            }
            if maybe_current_file_state.is_none()
                && (git_ignore.matches_file(&path) && !self.force_tracking_matcher.matches(&path))
            {
                // If it wasn't already tracked and it matches
                // the ignored paths, then ignore it.
                Ok(None)
            } else if maybe_current_file_state.is_none()
                && !self.start_tracking_matcher.matches(&path)
            {
                // Leave the file untracked
                self.untracked_paths_tx
                    .send((path, UntrackedReason::FileNotAutoTracked))
                    .ok();
                Ok(None)
            } else {
                let metadata = entry.metadata().map_err(|err| SnapshotError::Other {
                    message: format!("Failed to stat file {}", entry.path().display()),
                    err: err.into(),
                })?;
                if maybe_current_file_state.is_none()
                    && (metadata.len() > self.max_new_file_size
                        && !self.force_tracking_matcher.matches(&path))
                {
                    // Leave the large file untracked
                    let reason = UntrackedReason::FileTooLarge {
                        size: metadata.len(),
                        max_size: self.max_new_file_size,
                    };
                    self.untracked_paths_tx.send((path, reason)).ok();
                    Ok(None)
                } else if let Some(new_file_state) = file_state(&metadata)
                    .map_err(|err| snapshot_error_for_mtime_out_of_range(err, &entry.path()))?
                {
                    self.process_present_file(
                        path,
                        &entry.path(),
                        maybe_current_file_state.as_ref(),
                        new_file_state,
                    )
                    .await?;
                    Ok(Some((PresentDirEntryKind::File, name_string)))
                } else {
                    // Special file is not considered present
                    Ok(None)
                }
            }
        } else {
            Ok(None)
        }
    }

    /// Visits only paths we're already tracking.
    async fn visit_tracked_files(&self, file_states: FileStates<'_>) -> Result<(), SnapshotError> {
        for (tracked_path, current_file_state) in file_states {
            if current_file_state.file_type == FileType::GitSubmodule {
                continue;
            }
            if !self.matcher.matches(tracked_path) {
                continue;
            }
            let disk_path = tracked_path.to_fs_path(&self.tree_state.working_copy_path)?;
            let metadata = match disk_path.symlink_metadata() {
                Ok(metadata) => Some(metadata),
                Err(err) if err.kind() == io::ErrorKind::NotFound => None,
                Err(err) => {
                    return Err(SnapshotError::Other {
                        message: format!("Failed to stat file {}", disk_path.display()),
                        err: err.into(),
                    });
                }
            };
            if let Some(metadata) = &metadata
                && let Some(new_file_state) = file_state(metadata)
                    .map_err(|err| snapshot_error_for_mtime_out_of_range(err, &disk_path))?
            {
                self.process_present_file(
                    tracked_path.to_owned(),
                    &disk_path,
                    Some(&current_file_state),
                    new_file_state,
                )
                .await?;
            } else {
                self.deleted_files_tx.send(tracked_path.to_owned()).ok();
            }
        }
        Ok(())
    }

    async fn process_present_file(
        &self,
        path: RepoPathBuf,
        disk_path: &Path,
        maybe_current_file_state: Option<&FileState>,
        mut new_file_state: FileState,
    ) -> Result<(), SnapshotError> {
        let update = self
            .get_updated_tree_value(&path, disk_path, maybe_current_file_state, &new_file_state)
            .await?;
        // Preserve materialized conflict data for normal, non-resolved files
        if matches!(new_file_state.file_type, FileType::Normal { .. })
            && !update.as_ref().is_some_and(|update| update.is_resolved())
        {
            new_file_state.materialized_conflict_data =
                maybe_current_file_state.and_then(|state| state.materialized_conflict_data);
        }
        if let Some(tree_value) = update {
            self.tree_entries_tx.send((path.clone(), tree_value)).ok();
        }
        if Some(&new_file_state) != maybe_current_file_state {
            self.file_states_tx.send((path, new_file_state)).ok();
        }
        Ok(())
    }

    /// Emits file paths that don't exist in the `present_entries`.
    fn emit_deleted_files(
        &self,
        dir: &RepoPath,
        file_states: FileStates<'_>,
        present_entries: &PresentDirEntries,
    ) {
        let file_state_chunks = file_states.iter().chunk_by(|(path, _state)| {
            // Extract <name> from <dir>, <dir>/<name>, or <dir>/<name>/**.
            // (file_states may contain <dir> file on file->dir transition.)
            debug_assert!(path.starts_with(dir));
            let slash = usize::from(!dir.is_root());
            let len = dir.as_internal_file_string().len() + slash;
            let tail = path.as_internal_file_string().get(len..).unwrap_or("");
            match tail.split_once('/') {
                Some((name, _)) => (PresentDirEntryKind::Dir, name),
                None => (PresentDirEntryKind::File, tail),
            }
        });
        file_state_chunks
            .into_iter()
            .filter(|&((kind, name), _)| match kind {
                PresentDirEntryKind::Dir => !present_entries.dirs.contains(name),
                PresentDirEntryKind::File => !present_entries.files.contains(name),
            })
            .flat_map(|(_, chunk)| chunk)
            // Whether or not the entry exists, submodule should be ignored
            .filter(|(_, state)| state.file_type != FileType::GitSubmodule)
            .filter(|(path, _)| self.matcher.matches(path))
            .try_for_each(|(path, _)| self.deleted_files_tx.send(path.to_owned()))
            .ok();
    }

    async fn get_updated_tree_value(
        &self,
        repo_path: &RepoPath,
        disk_path: &Path,
        maybe_current_file_state: Option<&FileState>,
        new_file_state: &FileState,
    ) -> Result<Option<MergedTreeValue>, SnapshotError> {
        let clean = match maybe_current_file_state {
            None => {
                // untracked
                false
            }
            Some(current_file_state) => {
                // If the file's mtime was set at the same time as this state file's own mtime,
                // then we don't know if the file was modified before or after this state file.
                new_file_state.is_clean(current_file_state)
                    && current_file_state.mtime < self.tree_state.own_mtime
            }
        };
        if clean {
            Ok(None)
        } else {
            let current_tree_values = self.current_tree.path_value(repo_path).await?;
            let new_file_type = if !self.tree_state.symlink_support {
                let mut new_file_type = new_file_state.file_type.clone();
                if matches!(new_file_type, FileType::Normal { .. })
                    && matches!(current_tree_values.as_normal(), Some(TreeValue::Symlink(_)))
                {
                    new_file_type = FileType::Symlink;
                }
                new_file_type
            } else {
                new_file_state.file_type.clone()
            };
            let new_tree_values = match new_file_type {
                FileType::Normal { exec_bit } => {
                    self.write_path_to_store(
                        repo_path,
                        disk_path,
                        &current_tree_values,
                        exec_bit,
                        maybe_current_file_state.and_then(|state| state.materialized_conflict_data),
                    )
                    .await?
                }
                FileType::Symlink => {
                    let id = self.write_symlink_to_store(repo_path, disk_path).await?;
                    Merge::normal(TreeValue::Symlink(id))
                }
                FileType::GitSubmodule => panic!("git submodule cannot be written to store"),
            };
            if new_tree_values != current_tree_values {
                Ok(Some(new_tree_values))
            } else {
                Ok(None)
            }
        }
    }

    fn store(&self) -> &Store {
        &self.tree_state.store
    }

    async fn write_path_to_store(
        &self,
        repo_path: &RepoPath,
        disk_path: &Path,
        current_tree_values: &MergedTreeValue,
        exec_bit: ExecBit,
        materialized_conflict_data: Option<MaterializedConflictData>,
    ) -> Result<MergedTreeValue, SnapshotError> {
        if let Some(current_tree_value) = current_tree_values.as_resolved() {
            let id = self.write_file_to_store(repo_path, disk_path).await?;
            // On Windows, we preserve the executable bit from the current tree.
            let executable = exec_bit.for_tree_value(self.tree_state.exec_policy, || {
                if let Some(TreeValue::File {
                    id: _,
                    executable,
                    copy_id: _,
                }) = current_tree_value
                {
                    Some(*executable)
                } else {
                    None
                }
            });
            // Preserve the copy id from the current tree
            let copy_id = {
                if let Some(TreeValue::File {
                    id: _,
                    executable: _,
                    copy_id,
                }) = current_tree_value
                {
                    copy_id.clone()
                } else {
                    CopyId::placeholder()
                }
            };
            Ok(Merge::normal(TreeValue::File {
                id,
                executable,
                copy_id,
            }))
        } else if let Some(old_file_ids) = current_tree_values.to_file_merge() {
            // Safe to unwrap because the copy id exists exactly on the file variant
            let copy_id_merge = current_tree_values.to_copy_id_merge().unwrap();
            let copy_id = copy_id_merge
                .resolve_trivial(SameChange::Accept)
                .cloned()
                .flatten()
                .unwrap_or_else(CopyId::placeholder);
            let mut contents = vec![];
            let file = File::open(disk_path).map_err(|err| SnapshotError::Other {
                message: format!("Failed to open file {}", disk_path.display()),
                err: err.into(),
            })?;
            let (file, ignore_reason) = self
                .tree_state
                .filter_strategy
                .convert_to_store(AllowStdIo::new(file), repo_path, self.git_attributes)
                .await
                .map_err(|err| SnapshotError::Other {
                    message: "Failed to use the filter to convert the contents.".to_string(),
                    err,
                })?;
            if let Some(reason @ FilterIgnoreReason::FilterCommandFailed { .. }) = ignore_reason
                && self
                    .unconverted_files_tx
                    .send((repo_path.to_owned(), reason))
                    .is_err()
            {
                tracing::trace!(
                    "Failed to send the unconverted file information back on snapshot."
                );
            }
            self.tree_state
                .target_eol_strategy
                .convert_eol_for_snapshot(file)
                .await
                .map_err(|err| SnapshotError::Other {
                    message: "Failed to convert the EOL".to_string(),
                    err: err.into(),
                })?
                .read_to_end(&mut contents)
                .await
                .map_err(|err| SnapshotError::Other {
                    message: "Failed to read the EOL converted contents".to_string(),
                    err: err.into(),
                })?;
            // If the file contained a conflict before and is a normal file on
            // disk, we try to parse any conflict markers in the file into a
            // conflict.
            let new_file_ids = conflicts::update_from_content(
                &old_file_ids,
                self.store(),
                repo_path,
                &contents,
                materialized_conflict_data.map_or(MIN_CONFLICT_MARKER_LEN, |data| {
                    data.conflict_marker_len as usize
                }),
            )
            .await?;
            match new_file_ids.into_resolved() {
                Ok(file_id) => {
                    // On Windows, we preserve the executable bit from the merged trees.
                    let executable = exec_bit.for_tree_value(self.tree_state.exec_policy, || {
                        current_tree_values
                            .to_executable_merge()
                            .as_ref()
                            .and_then(conflicts::resolve_file_executable)
                    });
                    Ok(Merge::normal(TreeValue::File {
                        id: file_id.unwrap(),
                        executable,
                        copy_id,
                    }))
                }
                Err(new_file_ids) => {
                    if new_file_ids != old_file_ids {
                        Ok(current_tree_values.with_new_file_ids(&new_file_ids))
                    } else {
                        Ok(current_tree_values.clone())
                    }
                }
            }
        } else {
            Ok(current_tree_values.clone())
        }
    }

    async fn write_file_to_store(
        &self,
        path: &RepoPath,
        disk_path: &Path,
    ) -> Result<FileId, SnapshotError> {
        let file = File::open(disk_path).map_err(|err| SnapshotError::Other {
            message: format!("Failed to open file {}", disk_path.display()),
            err: err.into(),
        })?;
        let (file, ignore_reason) = self
            .tree_state
            .filter_strategy
            .convert_to_store(AllowStdIo::new(file), path, self.git_attributes)
            .await
            .map_err(|err| SnapshotError::Other {
                message: "Failed to use the filter to convert the contents.".to_string(),
                err,
            })?;
        if let Some(reason @ FilterIgnoreReason::FilterCommandFailed { .. }) = ignore_reason
            && self
                .unconverted_files_tx
                .send((path.to_owned(), reason))
                .is_err()
        {
            tracing::trace!("Failed to send the unconverted file information back on snapshot.");
        }
        let mut contents = self
            .tree_state
            .target_eol_strategy
            .convert_eol_for_snapshot(file)
            .await
            .map_err(|err| SnapshotError::Other {
                message: "Failed to convert the EOL".to_string(),
                err: err.into(),
            })?;
        Ok(self.store().write_file(path, &mut contents).await?)
    }

    async fn write_symlink_to_store(
        &self,
        path: &RepoPath,
        disk_path: &Path,
    ) -> Result<SymlinkId, SnapshotError> {
        if self.tree_state.symlink_support {
            let target = disk_path.read_link().map_err(|err| SnapshotError::Other {
                message: format!("Failed to read symlink {}", disk_path.display()),
                err: err.into(),
            })?;
            let str_target = symlink_target_convert_to_store(&target).ok_or_else(|| {
                SnapshotError::InvalidUtf8SymlinkTarget {
                    path: disk_path.to_path_buf(),
                }
            })?;
            Ok(self.store().write_symlink(path, &str_target).await?)
        } else {
            let target = fs::read(disk_path).map_err(|err| SnapshotError::Other {
                message: format!("Failed to read file {}", disk_path.display()),
                err: err.into(),
            })?;
            let string_target =
                String::from_utf8(target).map_err(|_| SnapshotError::InvalidUtf8SymlinkTarget {
                    path: disk_path.to_path_buf(),
                })?;
            Ok(self.store().write_symlink(path, &string_target).await?)
        }
    }
}

fn snapshot_error_for_mtime_out_of_range(err: MtimeOutOfRange, path: &Path) -> SnapshotError {
    SnapshotError::Other {
        message: format!("Failed to process file metadata {}", path.display()),
        err: err.into(),
    }
}

/// Functions to update local-disk files from the store.
impl TreeState {
    pub fn check_out(&mut self, new_tree: &MergedTree) -> Result<CheckoutStats, CheckoutError> {
        let old_tree = self.tree.clone();
        let stats = self.update(&old_tree, new_tree, self.sparse_matcher().as_ref())?;
        self.tree = new_tree.clone();
        Ok(stats)
    }

    pub fn set_sparse_patterns(
        &mut self,
        sparse_patterns: Vec<RepoPathBuf>,
    ) -> Result<CheckoutStats, CheckoutError> {
        let tree = self.tree.clone();
        let old_matcher = PrefixMatcher::new(&self.sparse_patterns);
        let new_matcher = PrefixMatcher::new(&sparse_patterns);
        let added_matcher = DifferenceMatcher::new(&new_matcher, &old_matcher);
        let removed_matcher = DifferenceMatcher::new(&old_matcher, &new_matcher);
        let empty_tree = self.store.empty_merged_tree();
        let added_stats = self.update(&empty_tree, &tree, &added_matcher)?;
        let removed_stats = self.update(&tree, &empty_tree, &removed_matcher)?;
        self.sparse_patterns = sparse_patterns;
        assert_eq!(added_stats.updated_files, 0);
        assert_eq!(added_stats.removed_files, 0);
        assert_eq!(removed_stats.updated_files, 0);
        assert_eq!(removed_stats.added_files, 0);
        assert_eq!(removed_stats.skipped_files, 0);
        Ok(CheckoutStats {
            updated_files: 0,
            added_files: added_stats.added_files,
            removed_files: removed_stats.removed_files,
            skipped_files: added_stats.skipped_files,
            unconverted_paths: BTreeMap::new(),
        })
    }

    /// Sets the bound on the bytes of file contents that a checkout holds read
    /// and not yet written, in place of the default of 256 MiB. For tests.
    #[doc(hidden)]
    pub fn set_max_bytes_in_flight(&mut self, bytes: u64) {
        self.max_bytes_in_flight = bytes;
    }

    /// What the last checkout measured of its threads and its bytes in flight.
    /// For tests.
    #[doc(hidden)]
    pub fn last_checkout_measures(&self) -> CheckoutMeasures {
        self.last_checkout_measures
    }

    /// Writes the changes from `old_tree` to `new_tree` that `matcher` matches
    /// to the working copy.
    ///
    /// This thread walks the diff in path order ([`CheckoutCoordinator`]): it
    /// removes old files and directories, creates directories and submodule
    /// directories, and queues each file, symlink and conflict to write. Up to
    /// `checkout_threads` worker threads read the queued contents from the
    /// store and write them ([`CheckoutContext::run_worker()`]). The file
    /// states are merged in the order of the diff.
    fn update(
        &mut self,
        old_tree: &MergedTree,
        new_tree: &MergedTree,
        matcher: &dyn Matcher,
    ) -> Result<CheckoutStats, CheckoutError> {
        let git_attributes = GitAttributes::new(vec![
            Arc::new(TreeFileLoader::new(new_tree.clone())),
            Arc::new(TreeFileLoader::new(old_tree.clone())),
        ]);

        // If a conflicted file didn't change between the two trees, but the conflict
        // labels did, we still need to re-materialize it in the working copy. We don't
        // need to do this if the conflicts have different numbers of sides though since
        // these conflicts are considered different, so they will be materialized by
        // `MergedTree::diff_stream_for_file_system` already.
        let conflicts_to_rematerialize: HashMap<RepoPathBuf, MergedTreeValue> =
            if old_tree.tree_ids().num_sides() == new_tree.tree_ids().num_sides()
                && old_tree.labels() != new_tree.labels()
            {
                // TODO: it might be better to use an async stream here and merge it with the
                // other diff stream, but it could be difficult since the diff stream is not
                // sorted in the same order as the conflicts iterator.
                new_tree
                    .conflicts_matching(matcher)
                    .map(|(path, value)| value.map(|value| (path, value)))
                    .try_collect()?
            } else {
                HashMap::new()
            };

        let output = {
            let (result_sender, result_receiver) = channel();
            let context = CheckoutContext {
                tree_state: &*self,
                labels: new_tree.labels(),
                git_attributes: &git_attributes,
                bytes_in_flight: BytesInFlight::new(self.max_bytes_in_flight),
                aborted: AtomicBool::new(false),
                first_error: Mutex::new(None),
                writers: AtomicUsize::new(0),
            };
            thread::scope(|scope| {
                let mut coordinator =
                    CheckoutCoordinator::new(&context, scope, result_sender, result_receiver);
                let walked =
                    coordinator.walk(old_tree, new_tree, matcher, conflicts_to_rematerialize);
                coordinator.finish(walked)
            })
        }?;
        let CheckoutOutput {
            stats,
            mut changed_file_states,
            deleted_files,
            sort,
            measures,
        } = output;
        if sort {
            // We need to re-sort the changed file states since we may have inserted a
            // conflicted file out of order.
            changed_file_states.sort_unstable_by(|(path1, _), (path2, _)| path1.cmp(path2));
        }

        self.file_states
            .merge_in(changed_file_states, &deleted_files);
        if stats.skipped_files > 0 {
            // Skipped files are recorded as placeholders. Make the next
            // snapshot scan everything since the filesystem monitor may not
            // report them as changed.
            self.watchman_clock = None;
        }
        self.last_checkout_measures = measures;
        Ok(stats)
    }

    pub async fn reset(&mut self, new_tree: &MergedTree) -> Result<(), ResetError> {
        let matcher = self.sparse_matcher();
        let mut changed_file_states = Vec::new();
        let mut deleted_files = HashSet::new();
        let mut diff_stream = self
            .tree
            .diff_stream_for_file_system(new_tree, matcher.as_ref());
        while let Some(TreeDiffEntry { path, values }) = diff_stream.next().await {
            let after = values?.after;
            if after.is_absent() {
                deleted_files.insert(path);
            } else {
                let file_type = match after.into_resolved() {
                    Ok(value) => match value.unwrap() {
                        TreeValue::File {
                            id: _,
                            executable,
                            copy_id: _,
                        } => {
                            let get_prev_exec = || self.file_states().get_exec_bit(&path);
                            let exec_bit =
                                ExecBit::new_from_repo(executable, self.exec_policy, get_prev_exec);
                            FileType::Normal { exec_bit }
                        }
                        TreeValue::Symlink(_id) => FileType::Symlink,
                        TreeValue::GitSubmodule(_id) => {
                            eprintln!("ignoring git submodule at {path:?}");
                            FileType::GitSubmodule
                        }
                        TreeValue::Tree(_id) => {
                            panic!("unexpected tree entry in diff at {path:?}");
                        }
                    },
                    Err(_values) => {
                        // TODO: Try to set the executable bit based on the conflict
                        FileType::Normal {
                            exec_bit: ExecBit(false),
                        }
                    }
                };
                let file_state = FileState {
                    file_type,
                    mtime: MillisSinceEpoch(0),
                    size: 0,
                    materialized_conflict_data: None,
                };
                changed_file_states.push((path, file_state));
            }
        }
        if !changed_file_states.is_empty() || !deleted_files.is_empty() {
            // The file states were updated without looking at the files on
            // disk. Make the next snapshot scan everything since the
            // filesystem monitor may not report them as changed.
            self.watchman_clock = None;
        }
        self.file_states
            .merge_in(changed_file_states, &deleted_files);
        self.tree = new_tree.clone();
        Ok(())
    }

    pub async fn recover(&mut self, new_tree: &MergedTree) -> Result<(), ResetError> {
        self.file_states.clear();
        // The file states are dropped without looking at the files on disk,
        // even if the new tree is empty. Make the next snapshot scan
        // everything.
        self.watchman_clock = None;
        self.tree = self.store.empty_merged_tree();
        self.reset(new_tree).await
    }
}

/// The bound on the bytes of file contents that a checkout holds read and not
/// yet written, unless one file is larger.
const DEFAULT_MAX_BYTES_IN_FLIGHT: u64 = 256 << 20;

/// The most bytes that a checkout writes to a new file with one `write` call:
/// a file whose contents are no larger goes in with one call.
const MAX_CONTENTS_WRITE: usize = 8 << 20;

/// The size that a checkout worker's buffer for contents starts at.
const MIN_CONTENTS_BUF: usize = 64 << 10;

/// The most jobs that a checkout's coordinating thread hands to the worker
/// threads at once. A checkout's first batches are smaller, so that a small
/// checkout spreads over the workers too.
const MAX_JOB_BATCH: usize = 64;

/// The jobs that a checkout's coordinating thread keeps queued or running for
/// each worker thread before it waits for some to finish.
const QUEUED_JOBS_PER_THREAD: usize = 4 * MAX_JOB_BATCH;

/// What a checkout measured of its threads and of its bytes in flight, for
/// tests ([`TreeState::last_checkout_measures()`]).
#[doc(hidden)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct CheckoutMeasures {
    /// The threads that read and wrote the checkout's files: the worker
    /// threads that the checkout started, or 1 where the calling thread did
    /// that work itself.
    pub threads: usize,
    /// The threads that wrote at least one file, symlink or conflict.
    pub writers: usize,
    /// The most bytes of file contents that were read and not yet written at
    /// one time.
    pub peak_bytes_in_flight: u64,
}

/// The bytes of file contents that a checkout's threads have read and not yet
/// written, under a bound. A thread reserves the bytes of a file before it
/// reads it and gives them back once the file is written. A reservation is
/// granted once its bytes fit under the bound or, for a file larger than the
/// bound, once nothing else is in flight. So the bytes in flight never pass
/// the bound or the largest file, whichever is larger.
///
/// While every reservation fits, the count is one atomic and no thread takes
/// a lock. A reservation that does not fit waits in a queue, and while the
/// queue is not empty every new reservation joins it, so that they are granted
/// in the order they came and a large file is not kept waiting by smaller ones
/// that come after it.
struct BytesInFlight {
    limit: u64,
    in_flight: AtomicU64,
    peak: AtomicU64,
    /// The reservations in the queue.
    waiters: AtomicUsize,
    /// Set when the checkout stops: the reservations that wait give up.
    closed: AtomicBool,
    queue: Mutex<ReservationQueue>,
    changed: Condvar,
}

struct ReservationQueue {
    /// The ticket that the next reservation to join the queue takes.
    next_ticket: u64,
    /// The ticket of the reservation that is granted next.
    next_grant: u64,
}

impl BytesInFlight {
    fn new(limit: u64) -> Self {
        Self {
            limit,
            in_flight: AtomicU64::new(0),
            peak: AtomicU64::new(0),
            waiters: AtomicUsize::new(0),
            closed: AtomicBool::new(false),
            queue: Mutex::new(ReservationQueue {
                next_ticket: 0,
                next_grant: 0,
            }),
            changed: Condvar::new(),
        }
    }

    fn lock_queue(&self) -> std::sync::MutexGuard<'_, ReservationQueue> {
        self.queue.lock().unwrap_or_else(PoisonError::into_inner)
    }

    /// Adds `bytes` to the bytes in flight if they fit under the bound, or if
    /// nothing else is in flight. Returns whether it did.
    fn try_add(&self, bytes: u64) -> bool {
        let mut in_flight = self.in_flight.load(AtomicOrdering::SeqCst);
        loop {
            let new_in_flight = in_flight.saturating_add(bytes);
            if in_flight != 0 && new_in_flight > self.limit {
                return false;
            }
            match self.in_flight.compare_exchange_weak(
                in_flight,
                new_in_flight,
                AtomicOrdering::SeqCst,
                AtomicOrdering::SeqCst,
            ) {
                Ok(_) => {
                    self.peak.fetch_max(new_in_flight, AtomicOrdering::Relaxed);
                    return true;
                }
                Err(actual) => in_flight = actual,
            }
        }
    }

    /// Waits until `bytes` may be in flight and reserves them. Returns `None`
    /// if the checkout stopped meanwhile.
    fn reserve(&self, bytes: u64) -> Option<BytesReservation<'_>> {
        if self.closed.load(AtomicOrdering::SeqCst) {
            return None;
        }
        let reservation = || BytesReservation {
            bytes_in_flight: self,
            bytes,
        };
        if self.waiters.load(AtomicOrdering::SeqCst) == 0 && self.try_add(bytes) {
            return Some(reservation());
        }
        let mut queue = self.lock_queue();
        // Counted before the bytes in flight are read again, and under the
        // lock: a release that this reservation misses sees the count, and
        // notifies once this thread waits.
        self.waiters.fetch_add(1, AtomicOrdering::SeqCst);
        let ticket = queue.next_ticket;
        queue.next_ticket += 1;
        let reserved = loop {
            if self.closed.load(AtomicOrdering::SeqCst) {
                break false;
            }
            if queue.next_grant == ticket && self.try_add(bytes) {
                queue.next_grant += 1;
                break true;
            }
            queue = self
                .changed
                .wait(queue)
                .unwrap_or_else(PoisonError::into_inner);
        };
        self.waiters.fetch_sub(1, AtomicOrdering::SeqCst);
        drop(queue);
        // The next reservation in the queue may fit as well.
        self.changed.notify_all();
        reserved.then(reservation)
    }

    fn release(&self, bytes: u64) {
        self.in_flight.fetch_sub(bytes, AtomicOrdering::SeqCst);
        if self.waiters.load(AtomicOrdering::SeqCst) != 0 {
            // A waiting reservation holds the lock from its look at the bytes
            // in flight until it waits, so it is notified after either.
            drop(self.lock_queue());
            self.changed.notify_all();
        }
    }

    /// Makes every reservation that waits, and every later one, give up.
    fn close(&self) {
        self.closed.store(true, AtomicOrdering::SeqCst);
        drop(self.lock_queue());
        self.changed.notify_all();
    }

    fn peak(&self) -> u64 {
        self.peak.load(AtomicOrdering::Relaxed)
    }
}

/// Bytes reserved in [`BytesInFlight`], given back when this drops.
struct BytesReservation<'a> {
    bytes_in_flight: &'a BytesInFlight,
    bytes: u64,
}

impl Drop for BytesReservation<'_> {
    fn drop(&mut self) {
        self.bytes_in_flight.release(self.bytes);
    }
}

/// A file, symlink or conflict that a checkout's coordinating thread queues
/// for a worker thread to read and write.
struct CheckoutJob {
    /// The job's slot in [`CheckoutCoordinator::slots`].
    index: usize,
    /// The key of the path's name ([`folded_name_key()`]), which the
    /// coordinating thread keeps while the job is with the workers.
    name_key: u64,
    path: RepoPathBuf,
    disk_path: PathBuf,
    /// The path's value in the new tree.
    value: MergedTreeValue,
    /// Whether the checkout has just removed the old file at the path.
    old_file_removed: bool,
}

/// What a worker sends back for each job that it takes.
struct CheckoutResult {
    index: usize,
    name_key: u64,
    /// `None` if the job failed, or did not run because the checkout stopped.
    outcome: Option<JobOutcome>,
}

struct JobOutcome {
    path: RepoPathBuf,
    outcome: CheckoutOutcome,
    /// The reason why the smudge filter left the contents as they were, if
    /// its command failed.
    unconverted: Option<FilterIgnoreReason>,
}

/// What a checkout did with one path.
enum CheckoutOutcome {
    /// The path was written, and has this state.
    Written(FileState),
    /// An entry was already at the path, or its parent is not a directory: the
    /// path is left as it is, with a placeholder state.
    Skipped,
    /// The store refused to read the contents, and no entry was at the path:
    /// the path at this disk path is treated as removed.
    AccessDenied(PathBuf),
}

enum CheckoutSlot {
    /// Queued for a worker, whose result has not come back.
    Queued,
    Done(RepoPathBuf, CheckoutOutcome),
}

struct CheckoutOutput {
    stats: CheckoutStats,
    changed_file_states: Vec<(RepoPathBuf, FileState)>,
    deleted_files: HashSet<RepoPathBuf>,
    /// Whether `changed_file_states` is out of path order.
    sort: bool,
    measures: CheckoutMeasures,
}

/// What one thread that reads and writes a checkout's files keeps from one
/// file to the next.
struct CheckoutWorker {
    /// The buffer that file contents go through.
    buf: Vec<u8>,
    dir_cache: DirCache,
}

impl CheckoutWorker {
    fn new(working_copy_path: &Path) -> Self {
        Self {
            buf: Vec::new(),
            dir_cache: DirCache::new(working_copy_path),
        }
    }
}

/// The results of one batch of jobs, with the index of the worker that sends
/// them.
type BatchResults = (usize, Vec<CheckoutResult>);

/// What the threads of one checkout share.
struct CheckoutContext<'a> {
    tree_state: &'a TreeState,
    /// The conflict labels of the new tree.
    labels: &'a ConflictLabels,
    git_attributes: &'a GitAttributes,
    bytes_in_flight: BytesInFlight,
    aborted: AtomicBool,
    first_error: Mutex<Option<CheckoutError>>,
    /// The workers that wrote at least one path.
    writers: AtomicUsize,
}

impl CheckoutContext<'_> {
    fn is_aborted(&self) -> bool {
        self.aborted.load(AtomicOrdering::Relaxed)
    }

    /// Stops the checkout: the coordinating thread queues nothing more, and
    /// the workers skip the jobs left in the queue.
    fn abort(&self) {
        self.aborted.store(true, AtomicOrdering::Relaxed);
        self.bytes_in_flight.close();
    }

    /// Keeps `err` if it is the checkout's first error, and stops the checkout.
    fn fail(&self, err: CheckoutError) {
        self.first_error
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .get_or_insert(err);
        self.abort();
    }

    /// The loop of one worker thread: it takes batches of jobs from its queue
    /// until the queue closes, and sends the results of each batch. A job
    /// that panics fails the checkout as an error does, and the worker goes
    /// on to answer for the jobs left in its queue.
    fn run_worker(
        &self,
        worker_index: usize,
        jobs: Receiver<Vec<CheckoutJob>>,
        results: &Sender<BatchResults>,
    ) {
        let mut worker = CheckoutWorker::new(self.tree_state.working_copy_path());
        let mut wrote = false;
        for batch in jobs {
            let mut batch_results = Vec::with_capacity(batch.len());
            for job in batch {
                let (index, name_key) = (job.index, job.name_key);
                let outcome = if self.is_aborted() {
                    None
                } else {
                    let checked_out = panic::catch_unwind(AssertUnwindSafe(|| {
                        self.check_out_job(job, &mut worker).block_on()
                    }));
                    match checked_out {
                        Ok(Ok(outcome)) => outcome,
                        Ok(Err(err)) => {
                            self.fail(err);
                            None
                        }
                        Err(payload) => {
                            self.fail(CheckoutError::Other {
                                message: "A checkout thread panicked".to_owned(),
                                err: panic_message(payload.as_ref()).into(),
                            });
                            worker = CheckoutWorker::new(self.tree_state.working_copy_path());
                            None
                        }
                    }
                };
                wrote |= matches!(
                    outcome,
                    Some(JobOutcome {
                        outcome: CheckoutOutcome::Written(_),
                        ..
                    })
                );
                batch_results.push(CheckoutResult {
                    index,
                    name_key,
                    outcome,
                });
            }
            if results.send((worker_index, batch_results)).is_err() {
                break;
            }
        }
        if wrote {
            self.writers.fetch_add(1, AtomicOrdering::Relaxed);
        }
    }

    /// Reads the contents of one queued path from the store and writes them.
    /// Returns `None` if the checkout stopped while the job waited.
    async fn check_out_job(
        &self,
        job: CheckoutJob,
        worker: &mut CheckoutWorker,
    ) -> Result<Option<JobOutcome>, CheckoutError> {
        let CheckoutJob {
            index: _,
            name_key: _,
            path,
            disk_path,
            value,
            old_file_removed,
        } = job;
        let tree_state = self.tree_state;
        let store = &tree_state.store;
        // Reserve the bytes that the read brings into memory before reading. A
        // backend that cannot tell them reserves the whole bound, so that its
        // files are read one at a time.
        let size = contents_size(store, &path, &value).await?;
        let reserved_bytes = size.unwrap_or(self.bytes_in_flight.limit);
        let _reservation = if reserved_bytes == 0 {
            None
        } else {
            let Some(reservation) = self.bytes_in_flight.reserve(reserved_bytes) else {
                return Ok(None);
            };
            Some(reservation)
        };
        let size_hint = size.unwrap_or(0);
        // We get the previous executable bit from the file states and not
        // the tree value because only the file states store the on-disk
        // executable bit.
        let get_prev_exec = || tree_state.file_states().get_exec_bit(&path);
        let (outcome, unconverted) = match materialize_tree_value(store, &path, value, self.labels)
            .await?
        {
            MaterializedTreeValue::AccessDenied(_) => {
                // As for a path that the checkout writes, an entry that is
                // already at the path is left as it is.
                if !old_file_removed && has_existing_entry(&disk_path)? {
                    (CheckoutOutcome::Skipped, None)
                } else {
                    (CheckoutOutcome::AccessDenied(disk_path), None)
                }
            }
            MaterializedTreeValue::File(file) => {
                let exec_bit =
                    ExecBit::new_from_repo(file.executable, tree_state.exec_policy, get_prev_exec);
                self.write_file(
                    &path,
                    &disk_path,
                    file.reader,
                    exec_bit,
                    true,
                    old_file_removed,
                    size_hint,
                    worker,
                )
                .await?
            }
            MaterializedTreeValue::Symlink { id: _, target } => {
                if tree_state.symlink_support {
                    let outcome = write_symlink(
                        &disk_path,
                        &target,
                        old_file_removed,
                        &mut worker.dir_cache,
                    )?;
                    (outcome, None)
                } else {
                    // The fake symlink file shouldn't be executable.
                    self.write_file(
                        &path,
                        &disk_path,
                        target.as_bytes(),
                        ExecBit(false),
                        false,
                        old_file_removed,
                        size_hint,
                        worker,
                    )
                    .await?
                }
            }
            MaterializedTreeValue::FileConflict(file) => {
                let conflict_marker_len = choose_materialized_conflict_marker_len(&file.contents);
                let options = ConflictMaterializeOptions {
                    marker_style: tree_state.conflict_marker_style,
                    marker_len: Some(conflict_marker_len),
                    merge: store.merge_options().clone(),
                };
                let exec_bit = ExecBit::new_from_repo(
                    file.executable.unwrap_or(false),
                    tree_state.exec_policy,
                    get_prev_exec,
                );
                let contents =
                    materialize_merge_result_to_bytes(&file.contents, &file.labels, &options);
                let (mut outcome, unconverted) = self
                    .write_conflict(
                        &path,
                        &disk_path,
                        &contents,
                        exec_bit,
                        old_file_removed,
                        worker,
                    )
                    .await?;
                if let CheckoutOutcome::Written(file_state) = &mut outcome {
                    file_state.materialized_conflict_data = Some(MaterializedConflictData {
                        conflict_marker_len: conflict_marker_len.try_into().unwrap_or(u32::MAX),
                    });
                }
                (outcome, unconverted)
            }
            MaterializedTreeValue::OtherConflict { id, labels } => {
                // Unless all terms are regular files, we can't do much
                // better than trying to describe the merge.
                let contents = id.describe(&labels);
                // Since this is a dummy file, it shouldn't be executable.
                self.write_conflict(
                    &path,
                    &disk_path,
                    contents.as_bytes(),
                    ExecBit(false),
                    old_file_removed,
                    worker,
                )
                .await?
            }
            MaterializedTreeValue::Absent
            | MaterializedTreeValue::GitSubmodule(_)
            | MaterializedTreeValue::Tree(_) => {
                unreachable!("the coordinating thread checks out {path:?}")
            }
        };
        Ok(Some(JobOutcome {
            path,
            outcome,
            unconverted,
        }))
    }

    /// Writes a file that the checkout creates new, with `contents` converted
    /// for the working copy if `apply_conversion`. `size_hint` is the size
    /// that the contents are expected to have.
    #[expect(clippy::too_many_arguments)]
    async fn write_file(
        &self,
        repo_path: &RepoPath,
        disk_path: &Path,
        contents: impl AsyncRead + Send + Unpin,
        exec_bit: ExecBit,
        apply_conversion: bool,
        old_file_removed: bool,
        size_hint: u64,
        worker: &mut CheckoutWorker,
    ) -> Result<(CheckoutOutcome, Option<FilterIgnoreReason>), CheckoutError> {
        let CheckoutWorker { buf, dir_cache } = worker;
        let Some(mut file) = create_new_file(disk_path, exec_bit, old_file_removed, dir_cache)?
        else {
            return Ok((CheckoutOutcome::Skipped, None));
        };
        let mut unconverted = None;
        let written = if apply_conversion && self.tree_state.update_converts_contents {
            let contents = self
                .tree_state
                .target_eol_strategy
                .convert_eol_for_update(contents)
                .await
                .map_err(|err| CheckoutError::Other {
                    message: "Failed to convert the EOL for the content".to_string(),
                    err: err.into(),
                })?;
            let (contents, reason) = self.convert_to_working_copy(contents, repo_path).await?;
            unconverted = reason;
            write_contents(contents, size_hint, buf, &mut file).await
        } else {
            // Neither the EOL conversion nor a filter is on: the contents go
            // to the file as they are.
            write_contents(contents, size_hint, buf, &mut file).await
        };
        let size = written.map_err(|err| CheckoutError::Other {
            message: format!(
                "Failed to write the content to the file {}",
                disk_path.display()
            ),
            err: err.into(),
        })?;
        let file_state = new_file_state(&file, exec_bit, size, disk_path)?;
        Ok((CheckoutOutcome::Written(file_state), unconverted))
    }

    /// Writes the file of a conflict that the checkout creates new. The
    /// contents are converted before the file is created, so that a conversion
    /// that fails leaves no file.
    async fn write_conflict(
        &self,
        repo_path: &RepoPath,
        disk_path: &Path,
        contents: &[u8],
        exec_bit: ExecBit,
        old_file_removed: bool,
        worker: &mut CheckoutWorker,
    ) -> Result<(CheckoutOutcome, Option<FilterIgnoreReason>), CheckoutError> {
        let CheckoutWorker { buf, dir_cache } = worker;
        let write_error = |err: io::Error| CheckoutError::Other {
            message: format!("Failed to write conflict to file {}", disk_path.display()),
            err: err.into(),
        };
        if !self.tree_state.update_converts_contents {
            let Some(mut file) = create_new_file(disk_path, exec_bit, old_file_removed, dir_cache)?
            else {
                return Ok((CheckoutOutcome::Skipped, None));
            };
            file.write_all(contents).map_err(write_error)?;
            let file_state = new_file_state(&file, exec_bit, contents.len() as u64, disk_path)?;
            return Ok((CheckoutOutcome::Written(file_state), None));
        }
        let size_hint = contents.len() as u64;
        let contents = self
            .tree_state
            .target_eol_strategy
            .convert_eol_for_update(contents)
            .await
            .map_err(|err| CheckoutError::Other {
                message: "Failed to convert the EOL when writing a merge conflict".to_string(),
                err: err.into(),
            })?;
        let (contents, unconverted) = self.convert_to_working_copy(contents, repo_path).await?;
        let Some(mut file) = create_new_file(disk_path, exec_bit, old_file_removed, dir_cache)?
        else {
            return Ok((CheckoutOutcome::Skipped, None));
        };
        let size = write_contents(contents, size_hint, buf, &mut file)
            .await
            .map_err(write_error)?;
        let file_state = new_file_state(&file, exec_bit, size, disk_path)?;
        Ok((CheckoutOutcome::Written(file_state), unconverted))
    }

    /// Runs the smudge filter that the gitattributes name for `repo_path` on
    /// `contents`. Returns the reason why the contents are left as they were if
    /// the filter command failed.
    async fn convert_to_working_copy<'c>(
        &self,
        contents: Box<dyn AsyncRead + Send + Unpin + 'c>,
        repo_path: &RepoPath,
    ) -> Result<
        (
            Box<dyn AsyncRead + Send + Unpin + 'c>,
            Option<FilterIgnoreReason>,
        ),
        CheckoutError,
    > {
        let (contents, ignore_reason) = self
            .tree_state
            .filter_strategy
            .convert_to_working_copy(contents, repo_path, self.git_attributes)
            .await
            .map_err(|err| CheckoutError::Other {
                message: "Failed to use the filter to convert the contents.".to_string(),
                err,
            })?;
        let unconverted = ignore_reason
            .filter(|reason| matches!(reason, FilterIgnoreReason::FilterCommandFailed { .. }));
        Ok((contents, unconverted))
    }
}

/// The message of a panic, from what it was raised with.
fn panic_message(payload: &(dyn Any + Send)) -> String {
    if let Some(message) = payload.downcast_ref::<&str>() {
        (*message).to_owned()
    } else if let Some(message) = payload.downcast_ref::<String>() {
        message.clone()
    } else {
        "unknown panic".to_owned()
    }
}

/// One worker thread's queue of jobs, as the coordinating thread sees it.
struct WorkerQueue {
    /// The queue's sender; the worker ends once this is dropped.
    jobs: Sender<Vec<CheckoutJob>>,
    /// How many jobs were sent to the worker that it has sent no result for.
    outstanding: usize,
}

/// The coordinating thread of one checkout. It walks the diff in path order,
/// and for each path:
///
/// - creates the parent directories, never through a symlink, as
///   [`create_parent_dirs()`] does;
/// - removes the old file or submodule directory;
/// - for a path that the new tree removes, removes the parent directories
///   that this leaves empty;
/// - creates the directory of a submodule;
/// - and queues a file, symlink or conflict for the workers, which read its
///   contents from the store and write it. With one thread, this thread does
///   that itself, and starts no worker.
///
/// Workers create no directories, and remove nothing but a file they just
/// created and must refuse. Where this thread's work on a path could meet a
/// worker's on an earlier one, it first waits for the workers to finish what
/// is queued, so that the outcome is the one of a checkout one path at a time:
///
/// - before it removes the directories that a removed path leaves empty, if a
///   queued path may be about to be created in one of them;
/// - before it removes the directory of an old submodule, in which a queued
///   path may be about to be created by a name that differs by case;
/// - before a path whose name, or the name of a directory to create for it,
///   differs from a queued path's by ASCII case only, which a case-insensitive
///   file system takes for the same entry;
/// - before and after a path that a file system may take for a path that
///   differs from it by more than ASCII case ([`path_must_check_out_alone()`]):
///   one with a name that is not ASCII (by Unicode case folding or
///   normalization), that ends with a dot or a space (which Windows strips),
///   or that holds `~` (an 8.3 short name), `:` or `\`;
/// - before a path below the path queued last, which only a conflicted tree
///   has.
struct CheckoutCoordinator<'scope, 'env> {
    context: &'scope CheckoutContext<'scope>,
    scope: &'scope thread::Scope<'scope, 'env>,
    /// The workers' queues: empty before the first job is queued, and once
    /// the checkout is finished.
    workers: Vec<WorkerQueue>,
    /// The worker that the next batch goes to if no worker has less to do.
    next_worker: usize,
    /// The sender that each worker gets a clone of; `None` once the workers
    /// are started, so that `results` ends when the last worker does.
    result_sender: Option<Sender<BatchResults>>,
    results: Receiver<BatchResults>,
    threads: usize,
    /// With one thread, what this thread keeps between the files that it
    /// reads and writes itself.
    inline_worker: Option<CheckoutWorker>,
    inline_wrote: bool,
    spawned: usize,
    /// The jobs to send to a worker with the next batch.
    pending: Vec<CheckoutJob>,
    /// How many jobs were sent to the workers.
    sent: usize,
    /// How many jobs the workers sent a result for.
    received: usize,
    /// For each key of a name ([`folded_name_key()`]), how many jobs with it
    /// are pending or with the workers.
    in_flight_names: HashMap<u64, u32>,
    /// What happened to each changed path, in the order of the diff.
    slots: Vec<CheckoutSlot>,
    deleted_files: HashSet<RepoPathBuf>,
    stats: CheckoutStats,
    sort: bool,
    /// The parent of the last path whose parent directories were created.
    prev_created_path: RepoPathBuf,
    /// The path queued last for the workers.
    last_queued: RepoPathBuf,
}

impl<'scope, 'env> CheckoutCoordinator<'scope, 'env> {
    fn new(
        context: &'scope CheckoutContext<'scope>,
        scope: &'scope thread::Scope<'scope, 'env>,
        result_sender: Sender<BatchResults>,
        results: Receiver<BatchResults>,
    ) -> Self {
        let threads = context.tree_state.checkout_threads.get();
        Self {
            context,
            scope,
            workers: Vec::new(),
            next_worker: 0,
            result_sender: Some(result_sender),
            results,
            threads,
            inline_worker: (threads == 1)
                .then(|| CheckoutWorker::new(context.tree_state.working_copy_path())),
            inline_wrote: false,
            spawned: 0,
            pending: Vec::new(),
            sent: 0,
            received: 0,
            in_flight_names: HashMap::new(),
            slots: Vec::new(),
            deleted_files: HashSet::new(),
            // TODO: maybe it's better not include the skipped counts in the "intended"
            // counts
            stats: CheckoutStats {
                updated_files: 0,
                added_files: 0,
                removed_files: 0,
                skipped_files: 0,
                unconverted_paths: BTreeMap::new(),
            },
            sort: false,
            prev_created_path: RepoPathBuf::root(),
            last_queued: RepoPathBuf::root(),
        }
    }

    /// Walks the diff from `old_tree` to `new_tree`, then the conflicts to
    /// rematerialize. Stops at the first error, its own or a worker's.
    fn walk(
        &mut self,
        old_tree: &MergedTree,
        new_tree: &MergedTree,
        matcher: &dyn Matcher,
        mut conflicts_to_rematerialize: HashMap<RepoPathBuf, MergedTreeValue>,
    ) -> Result<(), CheckoutError> {
        let mut diff_stream = old_tree.diff_stream_for_file_system(new_tree, matcher);
        while let Some(TreeDiffEntry { path, values }) = diff_stream.next().block_on() {
            if self.context.is_aborted() {
                return Ok(());
            }
            let diff = values?;
            conflicts_to_rematerialize.remove(&path);
            self.check_out_entry(path, diff.before, diff.after)?;
        }
        if !conflicts_to_rematerialize.is_empty() {
            self.sort = true;
            // These paths come in no order of the diff's: let the workers
            // finish the diff's paths first.
            self.wait_for_workers();
            let conflicts = conflicts_to_rematerialize
                .into_iter()
                .sorted_unstable_by(|(path1, _), (path2, _)| path1.cmp(path2));
            for (path, conflict) in conflicts {
                if self.context.is_aborted() {
                    return Ok(());
                }
                self.check_out_entry(path, conflict.clone(), conflict)?;
            }
        }
        Ok(())
    }

    /// Whether a job is pending or with the workers.
    fn has_in_flight(&self) -> bool {
        !self.pending.is_empty() || self.received < self.sent
    }

    /// Checks out one path whose value is `before` in the old tree and `after`
    /// in the new one.
    fn check_out_entry(
        &mut self,
        path: RepoPathBuf,
        before: MergedTreeValue,
        after: MergedTreeValue,
    ) -> Result<(), CheckoutError> {
        if after.is_absent() {
            self.stats.removed_files += 1;
        } else if before.is_absent() {
            self.stats.added_files += 1;
        } else {
            self.stats.updated_files += 1;
        }

        // Existing Git submodule can be a non-empty directory on disk. We
        // shouldn't attempt to manage it as a tracked path.
        //
        // TODO: It might be better to add general support for paths not
        // tracked by jj than processing submodules specially. For example,
        // paths excluded by .gitignore can be marked as such so that
        // newly-"unignored" paths won't be snapshotted automatically.
        let before_is_submodule = matches!(before.as_normal(), Some(TreeValue::GitSubmodule(_)));
        if before_is_submodule && matches!(after.as_normal(), Some(TreeValue::GitSubmodule(_))) {
            eprintln!("ignoring git submodule at {path:?}");
            // Not updating the file state as if there were no diffs. Leave
            // the state type as FileType::GitSubmodule if it was before.
            return Ok(());
        }

        let working_copy_path = self.context.tree_state.working_copy_path();
        // This path and the previous one we did work for may have a common prefix. We
        // can adjust the "working copy" path to the parent directory which we know
        // is already created. If there is no common prefix, this will by default use
        // RepoPath::root() as the common prefix.
        let (common_prefix, adjusted_diff_file_path) =
            path.split_common_prefix(&self.prev_created_path);

        let mut name_key = 0;
        // A file system may take a path for one that differs from it by more
        // than ASCII case, which `folded_name_key()` does not tell: a path that
        // is not ASCII, by Unicode case folding or by Unicode normalization (as
        // APFS takes the NFC "é" and the NFD "e\u{301}" for one name), and on
        // Windows, a name with a trailing dot or space for the name without it
        // ("foo." for "foo"), and an 8.3 short name ("PROGRA~1") for its long
        // name. So a path that may be such (`path_must_check_out_alone()`) is
        // checked out alone: once the workers have finished every queued path,
        // and before the next path.
        let alone = self.inline_worker.is_none() && path_must_check_out_alone(&path);
        if self.inline_worker.is_none() {
            name_key = folded_name_key(&path);
            if self.has_in_flight() {
                // The directories that this path may have created for it.
                let common_prefix_len = common_prefix.as_internal_file_string().len();
                let mut new_dirs = path
                    .ancestors()
                    .skip(1)
                    .take_while(|dir| dir.as_internal_file_string().len() > common_prefix_len);
                // A worker may still be writing the path queued last. Before
                // walking through that path to one below it (which only a
                // conflicted tree has), and before a path that a
                // case-insensitive file system takes for a queued one, wait
                // for the workers, so that this path finds what they made
                // there, as if the paths were checked out one after the other.
                if alone
                    || path
                        .strip_prefix(&self.last_queued)
                        .is_some_and(|rest| !rest.is_root())
                    || self.in_flight_names.contains_key(&name_key)
                    || new_dirs.any(|dir| self.in_flight_names.contains_key(&folded_name_key(dir)))
                {
                    self.wait_for_workers();
                }
            }
        }

        let disk_path = if adjusted_diff_file_path.is_root() {
            // The path being "root" here implies that the entire path has already been
            // created.
            //
            // e.g we may have have already processed a path like: "foo/bar/baz" and this is
            // our `prev_created_path`.
            //
            // and the current path is:
            // "foo/bar"
            //
            // This results in a common prefix of "foo/bar" with empty string for the
            // remainder since its entire prefix has already been created.
            // This means that we _dont_ need to create its parent dirs
            // either.

            path.to_fs_path(working_copy_path)?
        } else {
            let adjusted_working_copy_path = common_prefix.to_fs_path(working_copy_path)?;

            // Create parent directories no matter if after.is_present(). This
            // ensures that the path never traverses symlinks.
            let Some(disk_path) =
                create_parent_dirs(&adjusted_working_copy_path, adjusted_diff_file_path)?
            else {
                self.slots
                    .push(CheckoutSlot::Done(path, CheckoutOutcome::Skipped));
                return Ok(());
            };

            // Cache this path for the next iteration. This must occur after
            // `create_parent_dirs` to ensure that the path is only set when
            // no symlinks are encountered. Otherwise there could be
            // opportunity for a filesystem write-what-where attack.
            self.prev_created_path = path
                .parent()
                .map(RepoPath::to_owned)
                .expect("diff path has no parent");

            disk_path
        };

        // If the path was present, check reserved path first and delete it.
        // TODO: Check that the file has not changed before overwriting/removing it.
        let old_file_removed = before.is_present()
            && if before_is_submodule {
                // A queued path may be about to be created in the submodule
                // directory, by a name that differs by case (on a
                // case-insensitive file system). If the directory were removed
                // before that, the worker could create the path through what
                // takes the directory's place, such as a symlink that this
                // path is. So wait for the workers first, as `remove_path()`
                // does before it removes a directory.
                self.wait_for_workers();
                remove_old_submodule_dir(&disk_path)?
            } else {
                remove_old_file(&disk_path)?
            };

        if after.is_absent() {
            self.remove_path(path, &disk_path, old_file_removed, before_is_submodule)
        } else if matches!(after.as_normal(), Some(TreeValue::GitSubmodule(_))) {
            self.create_submodule_dir(path, &disk_path, old_file_removed)
        } else if matches!(after.as_normal(), Some(TreeValue::Tree(_))) {
            panic!("unexpected tree entry in diff at {path:?}");
        } else {
            self.queue(CheckoutJob {
                index: self.slots.len(),
                name_key,
                path,
                disk_path,
                value: after,
                old_file_removed,
            })?;
            if alone {
                self.wait_for_workers();
            }
            Ok(())
        }
    }

    /// Finishes the removal of a path that the new tree does not have.
    fn remove_path(
        &mut self,
        path: RepoPathBuf,
        disk_path: &Path,
        old_file_removed: bool,
        before_is_submodule: bool,
    ) -> Result<(), CheckoutError> {
        // An entry left on disk is skipped. Failing to delete un-tracked
        // submodule directory is not an error, as the, possibly untracked,
        // contents would otherwise be lost: go on in case there are parents to
        // be deleted.
        if !old_file_removed && has_existing_entry(disk_path)? && !before_is_submodule {
            self.slots
                .push(CheckoutSlot::Done(path, CheckoutOutcome::Skipped));
            return Ok(());
        }
        // Reset the previous path to avoid scenarios where this path is deleted,
        // then on the next iteration recreation is skipped because of this
        // optimization.
        self.prev_created_path = RepoPathBuf::root();
        // A directory above this path is removed only if it is empty, and a
        // directory that a queued file is still to be created in may look
        // empty. On a case-insensitive file system, or one that takes two
        // Unicode spellings of a name for one entry, that directory can be
        // spelled otherwise in the queued path than in this one, so no
        // comparison of the paths tells which directories those are. So wait
        // for the workers to finish every queued job before any directory is
        // removed. A checkout that removes no path, as into an empty working
        // copy, never waits here.
        self.wait_for_workers();
        remove_empty_parent_dirs(self.context.tree_state.working_copy_path(), disk_path);
        self.deleted_files.insert(path);
        Ok(())
    }

    /// Creates the directory of a submodule that the new tree has. Git
    /// behavior: Create the submodule directory but don't populate/overwrite
    /// the contents.
    fn create_submodule_dir(
        &mut self,
        path: RepoPathBuf,
        disk_path: &Path,
        old_file_removed: bool,
    ) -> Result<(), CheckoutError> {
        if !old_file_removed {
            match fs::create_dir(disk_path) {
                Ok(()) => reject_reserved_existing_path(disk_path).inspect_err(|_| {
                    fs::remove_dir(disk_path).ok();
                })?,
                Err(err) => {
                    reject_existing_entry(disk_path, err, || {
                        format!(
                            "Failed to create submodule directory {}",
                            disk_path.display()
                        )
                    })?;
                    // Failing to materialize submodule, over a directory which
                    // is presumably the submodule before it was added in a
                    // commit, is not an error.
                    if !disk_path.is_dir() {
                        self.slots
                            .push(CheckoutSlot::Done(path, CheckoutOutcome::Skipped));
                        return Ok(());
                    }
                }
            }
        }
        eprintln!("ignoring git submodule at {path:?}");
        if old_file_removed {
            match fs::create_dir(disk_path) {
                Ok(()) => {}
                Err(err) if err.kind() == io::ErrorKind::AlreadyExists => {}
                Err(err) => {
                    eprintln!("warning: failed to create submodule directory {path:?}: {err}");
                }
            }
        }
        let file_state = FileState::for_gitsubmodule();
        self.slots.push(CheckoutSlot::Done(
            path,
            CheckoutOutcome::Written(file_state),
        ));
        Ok(())
    }

    /// Records what a job did with its path.
    fn record(&mut self, index: usize, outcome: JobOutcome) {
        let JobOutcome {
            path,
            outcome,
            unconverted,
        } = outcome;
        if let Some(reason) = unconverted {
            self.stats.unconverted_paths.insert(path.clone(), reason);
        }
        self.slots[index] = CheckoutSlot::Done(path, outcome);
    }

    /// Queues a file, symlink or conflict for the workers, or with one thread
    /// reads and writes it now.
    fn queue(&mut self, job: CheckoutJob) -> Result<(), CheckoutError> {
        self.slots.push(CheckoutSlot::Queued);
        if let Some(worker) = &mut self.inline_worker {
            let index = job.index;
            if let Some(outcome) = self.context.check_out_job(job, worker).block_on()? {
                self.inline_wrote |= matches!(outcome.outcome, CheckoutOutcome::Written(_));
                self.record(index, outcome);
            }
            return Ok(());
        }

        let path: &RepoPath = &job.path;
        path.clone_into(&mut self.last_queued);
        *self.in_flight_names.entry(job.name_key).or_insert(0) += 1;
        if self.workers.is_empty() {
            self.spawn_workers()?;
        }
        self.pending.push(job);
        // The first batches are small, so that each worker gets jobs early.
        if self.pending.len() >= (self.sent / self.threads).clamp(1, MAX_JOB_BATCH) {
            self.flush();
        }
        while self.sent - self.received >= self.threads * QUEUED_JOBS_PER_THREAD
            && self.receive_results()
        {}
        Ok(())
    }

    /// Starts the worker threads, each with a queue of its own.
    fn spawn_workers(&mut self) -> Result<(), CheckoutError> {
        // This thread keeps no sender of results: once every worker has
        // ended, no result is waited for any longer.
        let results = self
            .result_sender
            .take()
            .expect("the workers are started once");
        for worker_index in 0..self.threads {
            let context = self.context;
            let results = results.clone();
            let (jobs, queue) = channel();
            thread::Builder::new()
                .name("jj checkout".to_owned())
                .spawn_scoped(self.scope, move || {
                    context.run_worker(worker_index, queue, &results);
                })
                .map_err(|err| CheckoutError::Other {
                    message: "Failed to start a checkout thread".to_owned(),
                    err: err.into(),
                })?;
            self.workers.push(WorkerQueue {
                jobs,
                outstanding: 0,
            });
            self.spawned += 1;
        }
        Ok(())
    }

    /// Sends the pending jobs as one batch to the worker that has the fewest
    /// jobs to do, in turns among those that have as few.
    fn flush(&mut self) {
        if self.pending.is_empty() {
            return;
        }
        let batch = mem::replace(&mut self.pending, Vec::with_capacity(MAX_JOB_BATCH));
        let batch_len = batch.len();
        let worker_count = self.workers.len();
        let worker_index = (0..worker_count)
            .map(|turn| (self.next_worker + turn) % worker_count)
            .min_by_key(|&worker_index| self.workers[worker_index].outstanding)
            .expect("the workers are started before a job is pending");
        self.next_worker = (worker_index + 1) % worker_count;
        self.sent += batch_len;
        let worker = &mut self.workers[worker_index];
        if worker.jobs.send(batch).is_ok() {
            worker.outstanding += batch_len;
        } else {
            // The worker has ended, which it does not do before its queue is
            // closed. No result comes for these jobs.
            self.received += batch_len;
            self.context.fail(CheckoutError::Other {
                message: "A checkout thread ended early".to_owned(),
                err: "the thread's queue is closed".into(),
            });
        }
    }

    /// Waits for the results of one batch of jobs and records them. Returns
    /// false if no worker is left to send any.
    fn receive_results(&mut self) -> bool {
        let Ok((worker_index, results)) = self.results.recv() else {
            return false;
        };
        if let Some(worker) = self.workers.get_mut(worker_index) {
            worker.outstanding -= results.len();
        }
        for result in results {
            let CheckoutResult {
                index,
                name_key,
                outcome,
            } = result;
            self.received += 1;
            if let Some(count) = self.in_flight_names.get_mut(&name_key) {
                *count -= 1;
                if *count == 0 {
                    self.in_flight_names.remove(&name_key);
                }
            }
            if let Some(outcome) = outcome {
                self.record(index, outcome);
            }
        }
        true
    }

    /// Sends the pending jobs, and waits until the workers have finished every
    /// job.
    fn wait_for_workers(&mut self) {
        self.flush();
        while self.received < self.sent && self.receive_results() {}
    }

    /// Closes the queues, waits for the workers to end, and returns what the
    /// checkout did, or its first error.
    fn finish(
        mut self,
        walked: Result<(), CheckoutError>,
    ) -> Result<CheckoutOutput, CheckoutError> {
        if let Err(err) = walked {
            self.context.fail(err);
        }
        if !self.context.is_aborted() {
            self.flush();
        }
        // Each worker ends once its queue is closed and empty, and `results`
        // ends when the last worker does.
        self.workers.clear();
        self.result_sender = None;
        while self.receive_results() {}
        if let Some(err) = self
            .context
            .first_error
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .take()
        {
            return Err(err);
        }

        let mut changed_file_states = Vec::with_capacity(self.slots.len());
        for slot in mem::take(&mut self.slots) {
            let CheckoutSlot::Done(path, outcome) = slot else {
                unreachable!("every queued job sent its result");
            };
            match outcome {
                CheckoutOutcome::Written(file_state) => {
                    changed_file_states.push((path, file_state));
                }
                CheckoutOutcome::Skipped => {
                    changed_file_states.push((path, FileState::placeholder()));
                    self.stats.skipped_files += 1;
                }
                CheckoutOutcome::AccessDenied(disk_path) => {
                    // The workers are done, so every directory that is empty
                    // now stays so.
                    let working_copy_path = self.context.tree_state.working_copy_path();
                    remove_empty_parent_dirs(working_copy_path, &disk_path);
                    self.deleted_files.insert(path);
                }
            }
        }
        let measures = if self.inline_worker.is_some() {
            CheckoutMeasures {
                threads: 1,
                writers: usize::from(self.inline_wrote),
                peak_bytes_in_flight: self.context.bytes_in_flight.peak(),
            }
        } else {
            CheckoutMeasures {
                threads: self.spawned,
                writers: self.context.writers.load(AtomicOrdering::Relaxed),
                peak_bytes_in_flight: self.context.bytes_in_flight.peak(),
            }
        };
        Ok(CheckoutOutput {
            stats: self.stats,
            changed_file_states,
            deleted_files: self.deleted_files,
            sort: self.sort,
            measures,
        })
    }
}

/// A key of `path` that is the same for two paths that differ by ASCII case
/// only, which a case-insensitive file system takes for one entry. (Two paths
/// that differ otherwise have the same key only by chance.) A file system may
/// also take two paths for one entry that differ by more than ASCII case,
/// although their keys differ, so the coordinating thread checks out such a
/// path alone, whatever its key ([`path_must_check_out_alone()`]).
fn folded_name_key(path: &RepoPath) -> u64 {
    // FNV-1a over the ASCII lower-case bytes.
    let mut key: u64 = 0xcbf2_9ce4_8422_2325;
    for byte in path.as_internal_file_string().bytes() {
        let folded = u64::from(byte.to_ascii_lowercase());
        key = (key ^ folded).wrapping_mul(0x0000_0100_0000_01b3);
    }
    key
}

/// Whether a file system may take the entry name `name` for a name that
/// differs from it by more than ASCII case: one that
/// [`has_special_alias_char()`] is true for, or one that ends with a dot or a
/// space, which Windows strips ("foo." and "foo " are "foo").
fn component_may_alias_beyond_ascii_case(name: &str) -> bool {
    has_special_alias_char(name) || name.ends_with(['.', ' '])
}

/// Whether the coordinating thread checks out `path` alone: if a file system
/// may take the name of the path, or of a directory above it, for a name that
/// differs from it by more than ASCII case, which [`folded_name_key()`] does
/// not tell.
fn path_must_check_out_alone(path: &RepoPath) -> bool {
    path.components()
        .map(RepoPathComponent::as_internal_str)
        .any(component_may_alias_beyond_ascii_case)
}

/// Removes the directories above `disk_path`, a path below the working copy
/// root `working_copy_path`, that its removal left empty, from the deepest one
/// up to the first one that is not empty, stopping before the root.
///
/// No directory is removed through a symlink: each entry from the root down
/// is checked to be a directory and not a symlink (as one that the checkout
/// walked through may have been replaced by a symlink since), and only the
/// directories above the first entry that is not one are removed. The root
/// itself may be a symlink.
fn remove_empty_parent_dirs(working_copy_path: &Path, disk_path: &Path) {
    let Some(dir) = disk_path
        .parent()
        .and_then(|parent| parent.strip_prefix(working_copy_path).ok())
    else {
        return;
    };
    // Walk down from the root while each entry is a directory, which
    // `symlink_metadata()` does not take a symlink to a directory for:
    // `dir_path` ends as the deepest of these `depth` directories.
    let mut dir_path = working_copy_path.to_owned();
    let mut depth = 0;
    for name in dir {
        dir_path.push(name);
        if !fs::symlink_metadata(&dir_path).is_ok_and(|metadata| metadata.is_dir()) {
            dir_path.pop();
            break;
        }
        depth += 1;
    }
    for _ in 0..depth {
        if fs::remove_dir(&dir_path).is_err() {
            break;
        }
        dir_path.pop();
    }
}

/// The bytes that reading `value`'s contents from the store brings into
/// memory, or `None` if the backend cannot tell them without reading. A value
/// that reads no file contents brings 0 (a symlink's target is not counted).
async fn contents_size(
    store: &Store,
    path: &RepoPath,
    value: &MergedTreeValue,
) -> BackendResult<Option<u64>> {
    if let Some(value) = value.as_resolved() {
        return match value {
            Some(TreeValue::File { id, .. }) => file_size(store, path, id).await,
            _ => Ok(Some(0)),
        };
    }
    // A conflict whose terms are not all files is described, not read.
    let Some(ids) = value.to_file_merge() else {
        return Ok(Some(0));
    };
    let mut total: u64 = 0;
    for id in ids.iter().flatten() {
        let Some(size) = file_size(store, path, id).await? else {
            return Ok(None);
        };
        total = total.saturating_add(size);
    }
    Ok(Some(total))
}

async fn file_size(store: &Store, path: &RepoPath, id: &FileId) -> BackendResult<Option<u64>> {
    match store.file_size(path, id).await {
        // The read reports the denial, as `MaterializedTreeValue::AccessDenied`.
        Err(BackendError::ReadAccessDenied { .. }) => Ok(None),
        result => result,
    }
}

/// Writes all of `contents` to `writer`, through `buf`: contents of up to
/// [`MAX_CONTENTS_WRITE`] bytes go in with one `write` call, and larger ones
/// with one call for each piece of that size. `size_hint` is the size that
/// the contents are expected to have, to size `buf` before the first read.
/// Returns the size of the contents.
async fn write_contents<W: Write + ?Sized>(
    contents: impl AsyncRead,
    size_hint: u64,
    buf: &mut Vec<u8>,
    writer: &mut W,
) -> io::Result<u64> {
    let mut contents = std::pin::pin!(contents);
    let wanted = usize::try_from(size_hint)
        .unwrap_or(usize::MAX)
        .clamp(MIN_CONTENTS_BUF, MAX_CONTENTS_WRITE);
    if buf.len() < wanted {
        buf.resize(wanted, 0);
    }
    let mut written: u64 = 0;
    let mut filled = 0;
    loop {
        if filled == buf.len() {
            if buf.len() < MAX_CONTENTS_WRITE {
                // The contents are larger than expected: read more of them
                // before writing.
                buf.resize((buf.len() * 2).min(MAX_CONTENTS_WRITE), 0);
            } else {
                writer.write_all(&buf[..filled])?;
                written += filled as u64;
                filled = 0;
            }
        }
        let read = contents.read(&mut buf[filled..]).await?;
        if read == 0 {
            break;
        }
        filled += read;
    }
    if filled > 0 {
        writer.write_all(&buf[..filled])?;
        written += filled as u64;
    }
    Ok(written)
}

/// Reads the state of a file that the checkout has just written from its
/// handle. That way, we know that the file exists and is of the expected type,
/// and the stat information is most likely accurate, except for other
/// processes modifying the file concurrently (The mtime is set at write time
/// and won't change when we close the file.)
fn new_file_state(
    file: &File,
    exec_bit: ExecBit,
    size: u64,
    disk_path: &Path,
) -> Result<FileState, CheckoutError> {
    #[cfg_attr(windows, expect(unused_mut))]
    let mut metadata = file
        .metadata()
        .map_err(|err| checkout_error_for_stat_error(err, disk_path))?;
    #[cfg(unix)]
    if exec_bit.0 && metadata.permissions().mode() & 0o111 == 0 {
        // The process's umask took every executable bit away from the mode
        // that the file was created with, so the next snapshot would read the
        // file as not executable. Set the mode on the handle, as the checkout
        // did for every file before it created files with their mode.
        file.set_permissions(fs::Permissions::from_mode(exec_bit.new_file_mode()))
            .map_err(|err| CheckoutError::Other {
                message: format!("Failed to set the mode of file {}", disk_path.display()),
                err: err.into(),
            })?;
        metadata = file
            .metadata()
            .map_err(|err| checkout_error_for_stat_error(err, disk_path))?;
    }
    FileState::for_file(exec_bit, size, &metadata)
        .map_err(|err| checkout_error_for_mtime_out_of_range(err, disk_path))
}

/// Creates the symlink `disk_path` to `target` for a checkout. The path is
/// skipped if an entry is already there, unless the checkout has just removed
/// the old file there: then it is an error.
fn write_symlink(
    disk_path: &Path,
    target: &str,
    old_file_removed: bool,
    dir_cache: &mut DirCache,
) -> Result<CheckoutOutcome, CheckoutError> {
    let target = symlink_target_convert_to_disk(target);

    if cfg!(windows) {
        // On Windows, "/" can't be part of valid file name, and "/" is also not a valid
        // separator for the symlink target. See an example of this issue in
        // https://github.com/jj-vcs/jj/issues/6934.
        //
        // We use debug_assert_* instead of assert_* because we want to avoid panic in
        // release build, and we are sure that we shouldn't create invalid symlinks in
        // tests.
        debug_assert_ne!(
            target.as_os_str().to_str().map(|path| path.contains('/')),
            Some(true),
            r#"Expect the symlink target doesn't contain "/", but got invalid symlink target: {}."#,
            target.display()
        );
    }

    // If the new symlink were the entry that the name ".git" or ".jj" of its
    // directory points to, git/jj CLI could be tricked to load configuration
    // from an attacker-controlled location. So a name that could be one is
    // tested with an empty file first: the symlink itself is never there
    // under such a name.
    if disk_path.file_name().is_none_or(could_alias_reserved_name) {
        let Some(probe) = create_new_file(disk_path, ExecBit(false), old_file_removed, dir_cache)?
        else {
            return Ok(CheckoutOutcome::Skipped);
        };
        drop(probe);
        fs::remove_file(disk_path).map_err(|err| CheckoutError::Other {
            message: format!("Failed to remove temporary file {}", disk_path.display()),
            err: err.into(),
        })?;
    }

    // On Windows, this will create a nonfunctional link for directories,
    // but at the moment we don't have enough information in the tree to
    // determine whether the symlink target is a file or a directory.
    let symlink_error = || {
        format!(
            "Failed to create symlink from {} to {}",
            disk_path.display(),
            target.display()
        )
    };
    match symlink_file(&target, disk_path) {
        Ok(()) => {}
        Err(err) if !old_file_removed => {
            reject_existing_entry(disk_path, err, symlink_error)?;
            return Ok(CheckoutOutcome::Skipped);
        }
        Err(err) => {
            return Err(CheckoutError::Other {
                message: symlink_error(),
                err: err.into(),
            });
        }
    }
    let metadata = disk_path
        .symlink_metadata()
        .map_err(|err| checkout_error_for_stat_error(err, disk_path))?;
    let file_state = FileState::for_symlink(&metadata)
        .map_err(|err| checkout_error_for_mtime_out_of_range(err, disk_path))?;
    Ok(CheckoutOutcome::Written(file_state))
}

fn checkout_error_for_stat_error(err: io::Error, path: &Path) -> CheckoutError {
    CheckoutError::Other {
        message: format!("Failed to stat file {}", path.display()),
        err: err.into(),
    }
}

fn checkout_error_for_mtime_out_of_range(err: MtimeOutOfRange, path: &Path) -> CheckoutError {
    CheckoutError::Other {
        message: format!("Failed to process file metadata {}", path.display()),
        err: err.into(),
    }
}

/// Working copy state stored in "checkout" file.
#[derive(Clone, Debug)]
struct CheckoutState {
    operation_id: OperationId,
    workspace_name: WorkspaceNameBuf,
}

impl CheckoutState {
    fn load(state_path: &Path) -> Result<Self, WorkingCopyStateError> {
        let wrap_err = |err| WorkingCopyStateError {
            message: "Failed to read checkout state".to_owned(),
            err,
        };
        let buf = fs::read(state_path.join("checkout")).map_err(|err| wrap_err(err.into()))?;
        let proto = crate::protos::local_working_copy::Checkout::decode(&*buf)
            .map_err(|err| wrap_err(err.into()))?;
        Ok(Self {
            operation_id: OperationId::new(proto.operation_id),
            workspace_name: if proto.workspace_name.is_empty() {
                // For compatibility with old working copies.
                // TODO: Delete in mid 2022 or so
                WorkspaceName::DEFAULT.to_owned()
            } else {
                proto.workspace_name.into()
            },
        })
    }

    #[instrument(skip_all)]
    fn save(&self, state_path: &Path) -> Result<(), WorkingCopyStateError> {
        let wrap_err = |err| WorkingCopyStateError {
            message: "Failed to write checkout state".to_owned(),
            err,
        };
        let proto = crate::protos::local_working_copy::Checkout {
            operation_id: self.operation_id.to_bytes(),
            workspace_name: (*self.workspace_name).into(),
        };
        let mut temp_file =
            NamedTempFile::new_in(state_path).map_err(|err| wrap_err(err.into()))?;
        temp_file
            .as_file_mut()
            .write_all(&proto.encode_to_vec())
            .map_err(|err| wrap_err(err.into()))?;
        // TODO: Retry if persisting fails (it will on Windows if the file happened to
        // be open for read).
        persist_temp_file(temp_file, state_path.join("checkout"))
            .map_err(|err| wrap_err(err.into()))?;
        Ok(())
    }
}

pub struct LocalWorkingCopy {
    store: Arc<Store>,
    working_copy_path: PathBuf,
    state_path: PathBuf,
    checkout_state: CheckoutState,
    tree_state: OnceCell<TreeState>,
    tree_state_settings: TreeStateSettings,
}

#[async_trait(?Send)]
impl WorkingCopy for LocalWorkingCopy {
    fn name(&self) -> &str {
        Self::NAME
    }

    fn workspace_name(&self) -> &WorkspaceName {
        &self.checkout_state.workspace_name
    }

    fn operation_id(&self) -> &OperationId {
        &self.checkout_state.operation_id
    }

    fn tree(&self) -> Result<&MergedTree, WorkingCopyStateError> {
        Ok(self.tree_state()?.current_tree())
    }

    fn sparse_patterns(&self) -> Result<&[RepoPathBuf], WorkingCopyStateError> {
        Ok(self.tree_state()?.sparse_patterns())
    }

    async fn start_mutation(&self) -> Result<Box<dyn LockedWorkingCopy>, WorkingCopyStateError> {
        let lock_path = self.state_path.join("working_copy.lock");
        let lock = FileLock::lock(lock_path).map_err(|err| WorkingCopyStateError {
            message: "Failed to lock working copy".to_owned(),
            err: err.into(),
        })?;

        let wc = Self {
            store: self.store.clone(),
            working_copy_path: self.working_copy_path.clone(),
            state_path: self.state_path.clone(),
            // Re-read the state after taking the lock
            checkout_state: CheckoutState::load(&self.state_path)?,
            // Empty so we re-read the state after taking the lock
            // TODO: It's expensive to reload the whole tree. We should copy it from `self` if it
            // hasn't changed.
            tree_state: OnceCell::new(),
            tree_state_settings: self.tree_state_settings.clone(),
        };
        let old_operation_id = wc.operation_id().clone();
        let old_tree = wc.tree()?.clone();
        Ok(Box::new(LockedLocalWorkingCopy {
            wc,
            old_operation_id,
            old_tree,
            tree_state_dirty: false,
            new_workspace_name: None,
            _lock: lock,
        }))
    }
}

impl LocalWorkingCopy {
    pub const NAME: &str = "local";

    /// Initializes a new working copy at `working_copy_path`. The working
    /// copy's state will be stored in the `state_path` directory. The working
    /// copy will have the empty tree checked out.
    pub fn init(
        store: Arc<Store>,
        working_copy_path: PathBuf,
        state_path: PathBuf,
        operation_id: OperationId,
        workspace_name: WorkspaceNameBuf,
        user_settings: &UserSettings,
    ) -> Result<Self, WorkingCopyStateError> {
        let checkout_state = CheckoutState {
            operation_id,
            workspace_name,
        };
        checkout_state.save(&state_path)?;
        let tree_state_settings = TreeStateSettings::try_from_user_settings(user_settings)
            .map_err(|err| WorkingCopyStateError {
                message: "Failed to read the tree state settings".to_string(),
                err: err.into(),
            })?;
        let tree_state = TreeState::init(
            store.clone(),
            working_copy_path.clone(),
            state_path.clone(),
            &tree_state_settings,
        )
        .map_err(|err| WorkingCopyStateError {
            message: "Failed to initialize working copy state".to_string(),
            err: err.into(),
        })?;
        Ok(Self {
            store,
            working_copy_path,
            state_path,
            checkout_state,
            tree_state: OnceCell::with_value(tree_state),
            tree_state_settings,
        })
    }

    pub fn load(
        store: Arc<Store>,
        working_copy_path: PathBuf,
        state_path: PathBuf,
        user_settings: &UserSettings,
    ) -> Result<Self, WorkingCopyStateError> {
        let checkout_state = CheckoutState::load(&state_path)?;
        let tree_state_settings = TreeStateSettings::try_from_user_settings(user_settings)
            .map_err(|err| WorkingCopyStateError {
                message: "Failed to read the tree state settings".to_string(),
                err: err.into(),
            })?;
        Ok(Self {
            store,
            working_copy_path,
            state_path,
            checkout_state,
            tree_state: OnceCell::new(),
            tree_state_settings,
        })
    }

    pub fn state_path(&self) -> &Path {
        &self.state_path
    }

    #[instrument(skip_all)]
    fn tree_state(&self) -> Result<&TreeState, WorkingCopyStateError> {
        self.tree_state.get_or_try_init(|| {
            TreeState::load(
                self.store.clone(),
                self.working_copy_path.clone(),
                self.state_path.clone(),
                &self.tree_state_settings,
            )
            .map_err(|err| WorkingCopyStateError {
                message: "Failed to read working copy state".to_string(),
                err: err.into(),
            })
        })
    }

    fn tree_state_mut(&mut self) -> Result<&mut TreeState, WorkingCopyStateError> {
        self.tree_state()?; // ensure loaded
        Ok(self.tree_state.get_mut().unwrap())
    }

    pub fn file_states(&self) -> Result<FileStates<'_>, WorkingCopyStateError> {
        Ok(self.tree_state()?.file_states())
    }

    #[cfg(feature = "watchman")]
    pub async fn query_watchman(
        &self,
        config: &WatchmanConfig,
    ) -> Result<(watchman::Clock, Option<Vec<PathBuf>>), WorkingCopyStateError> {
        self.tree_state()?
            .query_watchman(config)
            .await
            .map_err(|err| WorkingCopyStateError {
                message: "Failed to query watchman".to_string(),
                err: err.into(),
            })
    }

    #[cfg(feature = "watchman")]
    pub async fn is_watchman_trigger_registered(
        &self,
        config: &WatchmanConfig,
    ) -> Result<bool, WorkingCopyStateError> {
        self.tree_state()?
            .is_watchman_trigger_registered(config)
            .await
            .map_err(|err| WorkingCopyStateError {
                message: "Failed to query watchman".to_string(),
                err: err.into(),
            })
    }
}

pub struct LocalWorkingCopyFactory {}

impl WorkingCopyFactory for LocalWorkingCopyFactory {
    fn init_working_copy(
        &self,
        store: Arc<Store>,
        working_copy_path: PathBuf,
        state_path: PathBuf,
        operation_id: OperationId,
        workspace_name: WorkspaceNameBuf,
        settings: &UserSettings,
    ) -> Result<Box<dyn WorkingCopy>, WorkingCopyStateError> {
        Ok(Box::new(LocalWorkingCopy::init(
            store,
            working_copy_path,
            state_path,
            operation_id,
            workspace_name,
            settings,
        )?))
    }

    fn load_working_copy(
        &self,
        store: Arc<Store>,
        working_copy_path: PathBuf,
        state_path: PathBuf,
        settings: &UserSettings,
    ) -> Result<Box<dyn WorkingCopy>, WorkingCopyStateError> {
        Ok(Box::new(LocalWorkingCopy::load(
            store,
            working_copy_path,
            state_path,
            settings,
        )?))
    }
}

/// A working copy that's locked on disk. The lock is held until you call
/// `finish()` or `discard()`.
pub struct LockedLocalWorkingCopy {
    wc: LocalWorkingCopy,
    old_operation_id: OperationId,
    old_tree: MergedTree,
    tree_state_dirty: bool,
    new_workspace_name: Option<WorkspaceNameBuf>,
    _lock: FileLock,
}

#[async_trait]
impl LockedWorkingCopy for LockedLocalWorkingCopy {
    fn old_operation_id(&self) -> &OperationId {
        &self.old_operation_id
    }

    fn old_tree(&self) -> &MergedTree {
        &self.old_tree
    }

    async fn snapshot(
        &mut self,
        options: &SnapshotOptions,
    ) -> Result<(MergedTree, SnapshotStats), SnapshotError> {
        let tree_state = self.wc.tree_state_mut()?;
        let (is_dirty, stats) = tree_state.snapshot(options).await?;
        self.tree_state_dirty |= is_dirty;
        Ok((tree_state.current_tree().clone(), stats))
    }

    async fn check_out(&mut self, commit: &Commit) -> Result<CheckoutStats, CheckoutError> {
        // TODO: Write a "pending_checkout" file with the new TreeId so we can
        // continue an interrupted update if we find such a file.
        let new_tree = commit.tree();
        let tree_state = self.wc.tree_state_mut()?;
        if tree_state.tree.tree_ids_and_labels() != new_tree.tree_ids_and_labels() {
            let stats = tree_state.check_out(&new_tree)?;
            self.tree_state_dirty = true;
            Ok(stats)
        } else {
            Ok(CheckoutStats::default())
        }
    }

    fn rename_workspace(&mut self, new_name: WorkspaceNameBuf) {
        self.new_workspace_name = Some(new_name);
    }

    async fn reset(&mut self, commit: &Commit) -> Result<(), ResetError> {
        let new_tree = commit.tree();
        self.wc.tree_state_mut()?.reset(&new_tree).await?;
        self.tree_state_dirty = true;
        Ok(())
    }

    async fn recover(&mut self, commit: &Commit) -> Result<(), ResetError> {
        let new_tree = commit.tree();
        self.wc.tree_state_mut()?.recover(&new_tree).await?;
        self.tree_state_dirty = true;
        Ok(())
    }

    fn sparse_patterns(&self) -> Result<&[RepoPathBuf], WorkingCopyStateError> {
        self.wc.sparse_patterns()
    }

    async fn set_sparse_patterns(
        &mut self,
        new_sparse_patterns: Vec<RepoPathBuf>,
    ) -> Result<CheckoutStats, CheckoutError> {
        // TODO: Write a "pending_checkout" file with new sparse patterns so we can
        // continue an interrupted update if we find such a file.
        let stats = self
            .wc
            .tree_state_mut()?
            .set_sparse_patterns(new_sparse_patterns)?;
        self.tree_state_dirty = true;
        Ok(stats)
    }

    #[instrument(skip_all)]
    async fn finish(
        mut self: Box<Self>,
        operation_id: OperationId,
    ) -> Result<Box<dyn WorkingCopy>, WorkingCopyStateError> {
        assert!(
            self.tree_state_dirty
                || self.old_tree.tree_ids_and_labels() == self.wc.tree()?.tree_ids_and_labels()
        );
        if self.tree_state_dirty {
            self.wc
                .tree_state_mut()?
                .save()
                .map_err(|err| WorkingCopyStateError {
                    message: "Failed to write working copy state".to_string(),
                    err: Box::new(err),
                })?;
        }
        if self.old_operation_id != operation_id || self.new_workspace_name.is_some() {
            self.wc.checkout_state.operation_id = operation_id;
            if let Some(workspace_name) = self.new_workspace_name {
                self.wc.checkout_state.workspace_name = workspace_name;
            }
            self.wc.checkout_state.save(&self.wc.state_path)?;
        }
        // TODO: Clear the "pending_checkout" file here.
        Ok(Box::new(self.wc))
    }
}

impl LockedLocalWorkingCopy {
    pub fn reset_watchman(&mut self) -> Result<(), SnapshotError> {
        self.wc.tree_state_mut()?.reset_watchman();
        self.tree_state_dirty = true;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use maplit::hashset;

    use super::*;

    fn repo_path(value: &str) -> &RepoPath {
        RepoPath::from_internal_string(value).unwrap()
    }

    fn repo_path_component(value: &str) -> &RepoPathComponent {
        RepoPathComponent::new(value).unwrap()
    }

    fn new_state(size: u64) -> FileState {
        FileState {
            file_type: FileType::Normal {
                exec_bit: ExecBit(false),
            },
            mtime: MillisSinceEpoch(0),
            size,
            materialized_conflict_data: None,
        }
    }

    #[test]
    fn test_file_states_merge() {
        let new_static_entry = |path: &'static str, size| (repo_path(path), new_state(size));
        let new_owned_entry = |path: &str, size| (repo_path(path).to_owned(), new_state(size));
        let new_proto_entry = |path: &str, size| {
            file_state_entry_to_proto(repo_path(path).to_owned(), &new_state(size))
        };
        let data = vec![
            new_proto_entry("aa", 0),
            new_proto_entry("b#", 4), // '#' < '/'
            new_proto_entry("b/c", 1),
            new_proto_entry("b/d/e", 2),
            new_proto_entry("b/e", 3),
            new_proto_entry("bc", 5),
        ];
        let mut file_states = FileStatesMap::from_proto(data, false);

        let changed_file_states = vec![
            new_owned_entry("aa", 10),    // change
            new_owned_entry("b/d/f", 11), // add
            new_owned_entry("b/e", 12),   // change
            new_owned_entry("c", 13),     // add
        ];
        let deleted_files = hashset! {
            repo_path("b/c").to_owned(),
            repo_path("b#").to_owned(),
        };
        file_states.merge_in(changed_file_states, &deleted_files);
        assert_eq!(
            file_states.all().iter().collect_vec(),
            vec![
                new_static_entry("aa", 10),
                new_static_entry("b/d/e", 2),
                new_static_entry("b/d/f", 11),
                new_static_entry("b/e", 12),
                new_static_entry("bc", 5),
                new_static_entry("c", 13),
            ],
        );
    }

    #[test]
    fn test_file_states_lookup() {
        let new_proto_entry = |path: &str, size| {
            file_state_entry_to_proto(repo_path(path).to_owned(), &new_state(size))
        };
        let data = vec![
            new_proto_entry("aa", 0),
            new_proto_entry("b/c", 1),
            new_proto_entry("b/d/e", 2),
            new_proto_entry("b/e", 3),
            new_proto_entry("b#", 4), // '#' < '/'
            new_proto_entry("bc", 5),
        ];
        let file_states = FileStates::from_sorted(&data);

        assert_eq!(
            file_states.prefixed(repo_path("")).paths().collect_vec(),
            ["aa", "b/c", "b/d/e", "b/e", "b#", "bc"].map(repo_path)
        );
        assert!(file_states.prefixed(repo_path("a")).is_empty());
        assert_eq!(
            file_states.prefixed(repo_path("aa")).paths().collect_vec(),
            ["aa"].map(repo_path)
        );
        assert_eq!(
            file_states.prefixed(repo_path("b")).paths().collect_vec(),
            ["b/c", "b/d/e", "b/e"].map(repo_path)
        );
        assert_eq!(
            file_states.prefixed(repo_path("b/d")).paths().collect_vec(),
            ["b/d/e"].map(repo_path)
        );
        assert_eq!(
            file_states.prefixed(repo_path("b#")).paths().collect_vec(),
            ["b#"].map(repo_path)
        );
        assert_eq!(
            file_states.prefixed(repo_path("bc")).paths().collect_vec(),
            ["bc"].map(repo_path)
        );
        assert!(file_states.prefixed(repo_path("z")).is_empty());

        assert!(!file_states.contains_path(repo_path("a")));
        assert!(file_states.contains_path(repo_path("aa")));
        assert!(file_states.contains_path(repo_path("b/d/e")));
        assert!(!file_states.contains_path(repo_path("b/d")));
        assert!(file_states.contains_path(repo_path("b#")));
        assert!(file_states.contains_path(repo_path("bc")));
        assert!(!file_states.contains_path(repo_path("z")));

        assert_eq!(file_states.get(repo_path("a")), None);
        assert_eq!(file_states.get(repo_path("aa")), Some(new_state(0)));
        assert_eq!(file_states.get(repo_path("b/d/e")), Some(new_state(2)));
        assert_eq!(file_states.get(repo_path("bc")), Some(new_state(5)));
        assert_eq!(file_states.get(repo_path("z")), None);
    }

    #[test]
    fn test_file_states_lookup_at() {
        let new_proto_entry = |path: &str, size| {
            file_state_entry_to_proto(repo_path(path).to_owned(), &new_state(size))
        };
        let data = vec![
            new_proto_entry("b/c", 0),
            new_proto_entry("b/d/e", 1),
            new_proto_entry("b/d#", 2), // '#' < '/'
            new_proto_entry("b/e", 3),
            new_proto_entry("b#", 4), // '#' < '/'
        ];
        let file_states = FileStates::from_sorted(&data);

        // At root
        assert_eq!(
            file_states.get_at(RepoPath::root(), repo_path_component("b")),
            None
        );
        assert_eq!(
            file_states.get_at(RepoPath::root(), repo_path_component("b#")),
            Some(new_state(4))
        );

        // At prefixed dir
        let prefixed_states = file_states.prefixed_at(RepoPath::root(), repo_path_component("b"));
        assert_eq!(
            prefixed_states.paths().collect_vec(),
            ["b/c", "b/d/e", "b/d#", "b/e"].map(repo_path)
        );
        assert_eq!(
            prefixed_states.get_at(repo_path("b"), repo_path_component("c")),
            Some(new_state(0))
        );
        assert_eq!(
            prefixed_states.get_at(repo_path("b"), repo_path_component("d")),
            None
        );
        assert_eq!(
            prefixed_states.get_at(repo_path("b"), repo_path_component("d#")),
            Some(new_state(2))
        );

        // At nested prefixed dir
        let prefixed_states = prefixed_states.prefixed_at(repo_path("b"), repo_path_component("d"));
        assert_eq!(
            prefixed_states.paths().collect_vec(),
            ["b/d/e"].map(repo_path)
        );
        assert_eq!(
            prefixed_states.get_at(repo_path("b/d"), repo_path_component("e")),
            Some(new_state(1))
        );
        assert_eq!(
            prefixed_states.get_at(repo_path("b/d"), repo_path_component("#")),
            None
        );

        // At prefixed file
        let prefixed_states = file_states.prefixed_at(RepoPath::root(), repo_path_component("b#"));
        assert_eq!(prefixed_states.paths().collect_vec(), ["b#"].map(repo_path));
        assert_eq!(
            prefixed_states.get_at(repo_path("b#"), repo_path_component("#")),
            None
        );
    }

    #[test]
    fn test_system_time_to_millis() {
        let epoch = SystemTime::UNIX_EPOCH;
        assert_eq!(system_time_to_millis(epoch), Some(MillisSinceEpoch(0)));
        if let Some(time) = epoch.checked_add(Duration::from_millis(1)) {
            assert_eq!(system_time_to_millis(time), Some(MillisSinceEpoch(1)));
        }
        if let Some(time) = epoch.checked_sub(Duration::from_millis(1)) {
            assert_eq!(system_time_to_millis(time), Some(MillisSinceEpoch(-1)));
        }
        if let Some(time) = epoch.checked_add(Duration::from_millis(i64::MAX as u64)) {
            assert_eq!(
                system_time_to_millis(time),
                Some(MillisSinceEpoch(i64::MAX))
            );
        }
        if let Some(time) = epoch.checked_sub(Duration::from_millis(i64::MAX as u64)) {
            assert_eq!(
                system_time_to_millis(time),
                Some(MillisSinceEpoch(-i64::MAX))
            );
        }
        if let Some(time) = epoch.checked_sub(Duration::from_millis(i64::MAX as u64 + 1)) {
            // i64::MIN could be returned, but we don't care such old timestamp
            assert_eq!(system_time_to_millis(time), None);
        }
    }

    #[test]
    fn test_new_file_content_goes_in_one_write() {
        struct CountingWriter {
            writes: usize,
            data: Vec<u8>,
        }

        impl Write for CountingWriter {
            fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
                self.writes += 1;
                self.data.extend_from_slice(buf);
                Ok(buf.len())
            }

            fn flush(&mut self) -> io::Result<()> {
                Ok(())
            }
        }

        // Within `MAX_CONTENTS_WRITE`, each size goes in with one write call
        // (0 for an empty file), whether the size hint matches or is 0.
        let mut buf = Vec::new();
        for &(size, expected_writes) in &[
            (0, 0),
            (1, 1),
            (32 << 10, 1),
            (128 << 10, 1),
            (MAX_CONTENTS_WRITE, 1),
            (MAX_CONTENTS_WRITE + 1, 2),
        ] {
            let content: Vec<u8> = (0..size).map(|i| (i % 251) as u8).collect();
            for size_hint in [size as u64, 0] {
                let mut writer = CountingWriter {
                    writes: 0,
                    data: Vec::new(),
                };
                let written = write_contents(content.as_slice(), size_hint, &mut buf, &mut writer)
                    .block_on()
                    .unwrap();
                assert_eq!(written, size as u64);
                assert_eq!(writer.data, content);
                assert_eq!(
                    writer.writes, expected_writes,
                    "size={size}, size_hint={size_hint}"
                );
            }
        }
    }

    #[cfg(unix)]
    #[test]
    fn test_dir_cache_survives_removed_and_remade_dir() {
        let temp_dir = crate::tests::new_temp_dir();
        let dir = temp_dir.path().join("dir");
        let remake_dir = |file: &str| {
            fs::remove_file(dir.join(file)).unwrap();
            fs::remove_dir(&dir).unwrap();
            fs::create_dir(&dir).unwrap();
        };
        fs::create_dir(&dir).unwrap();
        fs::write(dir.join("old"), "").unwrap();
        let mut dir_cache = DirCache::new(temp_dir.path());

        // A create that fails because the file exists keeps no directory, so
        // the next create in the directory, which was removed (as when the
        // removal of the old file left it empty) and made again in between,
        // goes in the new directory.
        let err = dir_cache
            .create_new(&dir.join("old"), ExecBit(false))
            .unwrap_err();
        assert_eq!(err.kind(), io::ErrorKind::AlreadyExists);
        remake_dir("old");
        dir_cache
            .create_new(&dir.join("new"), ExecBit(false))
            .unwrap();
        assert!(dir.join("new").is_file());

        // A directory kept after a create is opened again if it was removed
        // and made again since.
        remake_dir("new");
        dir_cache
            .create_new(&dir.join("newer"), ExecBit(false))
            .unwrap();
        assert!(dir.join("newer").is_file());
    }

    #[cfg(unix)]
    #[test]
    fn test_dir_cache_creates_in_dir_without_read_permission() {
        use std::os::unix::fs::PermissionsExt as _;

        let temp_dir = crate::tests::new_temp_dir();
        let dir = temp_dir.path().join("dir");
        fs::create_dir(&dir).unwrap();
        // Write and search permission, but no read permission.
        fs::set_permissions(&dir, fs::Permissions::from_mode(0o300)).unwrap();
        let readable = fs::read_dir(&dir).is_ok();
        let created = DirCache::new(temp_dir.path()).create_new(&dir.join("file"), ExecBit(false));
        // Let the temporary directory be removed whatever happened.
        fs::set_permissions(&dir, fs::Permissions::from_mode(0o700)).unwrap();
        assert!(!readable, "the directory should not be readable");
        created.unwrap();
        assert!(dir.join("file").is_file());
    }

    #[cfg(unix)]
    #[test]
    fn test_dir_cache_refuses_symlinked_dir() {
        use std::os::unix::fs::PermissionsExt as _;
        use std::os::unix::fs::symlink;

        let temp_dir = crate::tests::new_temp_dir();
        let working_copy_path = temp_dir.path().join("wc");
        let outside = temp_dir.path().join("outside");
        fs::create_dir(&working_copy_path).unwrap();
        fs::create_dir_all(outside.join("sub")).unwrap();
        for dir in ["dir", "swap", "locked"] {
            fs::create_dir(working_copy_path.join(dir)).unwrap();
        }
        for link in ["link", "dir/link", "locked/link"] {
            symlink(&outside, working_copy_path.join(link)).unwrap();
        }
        let mut dir_cache = DirCache::new(&working_copy_path);
        let mut create =
            |path: &str| dir_cache.create_new(&working_copy_path.join(path), ExecBit(false));

        // A symlink as the file's directory, or as a directory above it, is
        // not followed, from the root or from a directory that is kept open.
        for path in ["link/file", "link/sub/file"] {
            assert!(create(path).is_err(), "{path}");
        }
        for path in ["dir/link/file", "dir/link/sub/file"] {
            create("dir/kept").unwrap();
            fs::remove_file(working_copy_path.join("dir/kept")).unwrap();
            assert!(create(path).is_err(), "{path}");
        }

        // A kept directory that is replaced by a symlink is not followed.
        create("swap/file").unwrap();
        fs::remove_file(working_copy_path.join("swap/file")).unwrap();
        fs::remove_dir(working_copy_path.join("swap")).unwrap();
        symlink(&outside, working_copy_path.join("swap")).unwrap();
        assert!(create("swap/file").is_err());

        // Nor is a symlink below a directory that can't be opened.
        let locked = working_copy_path.join("locked");
        fs::set_permissions(&locked, fs::Permissions::from_mode(0o300)).unwrap();
        let created = create("locked/link/file");
        fs::set_permissions(&locked, fs::Permissions::from_mode(0o700)).unwrap();
        assert!(created.is_err());

        let outside_entries = fs::read_dir(&outside)
            .unwrap()
            .map(|entry| entry.unwrap().file_name())
            .collect_vec();
        assert_eq!(outside_entries, ["sub"]);
        assert_eq!(fs::read_dir(outside.join("sub")).unwrap().count(), 0);

        // The working copy root itself may be a symlink.
        let root_link = temp_dir.path().join("root-link");
        symlink(&working_copy_path, &root_link).unwrap();
        DirCache::new(&root_link)
            .create_new(&root_link.join("dir/file"), ExecBit(false))
            .unwrap();
        assert!(working_copy_path.join("dir/file").is_file());
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn test_dir_cache_bounds_open_fds() {
        // How many descriptors of this process are open on `dir` or on an
        // entry below it, which no other test opens.
        fn open_fds_under(dir: &Path) -> usize {
            fs::read_dir("/proc/self/fd")
                .unwrap()
                .filter_map(|entry| fs::read_link(entry.ok()?.path()).ok())
                .filter(|target| target.starts_with(dir))
                .count()
        }

        let temp_dir = crate::tests::new_temp_dir();
        let working_copy_path = temp_dir.path().join("wc");
        let mut deep_dir = working_copy_path.clone();
        for i in 0..20 {
            deep_dir.push(format!("d{i}"));
        }
        fs::create_dir_all(&deep_dir).unwrap();
        let sibling_dir = deep_dir.parent().unwrap().join("sibling");
        fs::create_dir(&sibling_dir).unwrap();
        // The links in "/proc/self/fd" point to canonical paths.
        let canonical_wc = working_copy_path.canonicalize().unwrap();
        assert_eq!(open_fds_under(&canonical_wc), 0);

        // However deep the directory of the file created last, the cache
        // keeps at most the root and that directory open.
        let mut dir_cache = DirCache::new(&working_copy_path);
        dir_cache
            .create_new(&deep_dir.join("file.txt"), ExecBit(false))
            .unwrap();
        let open_fds = open_fds_under(&canonical_wc);
        assert!((1..=2).contains(&open_fds), "open_fds={open_fds}");
        dir_cache
            .create_new(&sibling_dir.join("other.txt"), ExecBit(false))
            .unwrap();
        let open_fds = open_fds_under(&canonical_wc);
        assert!((1..=2).contains(&open_fds), "open_fds={open_fds}");
        assert!(deep_dir.join("file.txt").is_file());
        assert!(sibling_dir.join("other.txt").is_file());

        drop(dir_cache);
        assert_eq!(open_fds_under(&canonical_wc), 0);
    }

    #[cfg(unix)]
    #[test]
    fn test_remove_empty_parent_dirs_refuses_symlink() {
        use std::os::unix::fs::symlink;

        let temp_dir = crate::tests::new_temp_dir();
        let wc = temp_dir.path().join("wc");
        let outside = temp_dir.path().join("outside");
        fs::create_dir(&wc).unwrap();
        fs::create_dir_all(outside.join("b")).unwrap();
        symlink(&outside, wc.join("a")).unwrap();

        // An empty directory is not removed through a symlink in the working
        // copy, nor is the symlink.
        remove_empty_parent_dirs(&wc, &wc.join("a/b/x"));
        assert!(outside.join("b").is_dir());
        assert!(wc.join("a").symlink_metadata().unwrap().is_symlink());

        // Empty directories are removed from the deepest one up, and the
        // working copy root is kept.
        fs::remove_file(wc.join("a")).unwrap();
        fs::create_dir_all(wc.join("real/sub")).unwrap();
        remove_empty_parent_dirs(&wc, &wc.join("real/sub/x"));
        assert!(!wc.join("real").exists());
        assert!(wc.is_dir());
        assert_eq!(fs::read_dir(&wc).unwrap().count(), 0);

        // The working copy root itself may be a symlink.
        let root_link = temp_dir.path().join("root-link");
        symlink(&wc, &root_link).unwrap();
        fs::create_dir_all(wc.join("real/sub")).unwrap();
        remove_empty_parent_dirs(&root_link, &root_link.join("real/sub/x"));
        assert!(!wc.join("real").exists());
        assert!(root_link.symlink_metadata().unwrap().is_symlink());
    }
}
