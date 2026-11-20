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

//! Settings, views of the registry and hooks for tests, here and in the
//! vendored gix-pack: tests need smaller chunks and budgets than real packs
//! do, need to see what a read marked, and need steps between the system calls
//! of an open.
//!
//! The chunk size and the budget are the whole process's; the hooks are the
//! calling thread's.

use std::cell::RefCell;
use std::io;
use std::ops::Range;
use std::os::unix::fs::MetadataExt as _;
use std::path::Path;
use std::sync::atomic::AtomicUsize;
use std::sync::atomic::Ordering::Relaxed;

use crate::CHUNK_SIZE;
use crate::registry;
use crate::registry::Mapping;

static CHUNK_SIZE_SETTING: AtomicUsize = AtomicUsize::new(CHUNK_SIZE);

type BetweenOpensHook = Box<dyn FnOnce()>;
/// A hook that runs before each chunk is mapped, with its index.
pub type BeforeChunkHook = Box<dyn FnMut(usize)>;
/// A hook that runs after each open, with the mapping's addresses.
pub type AfterOpenHook = Box<dyn FnMut(Range<usize>)>;
/// A hook that runs before each mapping leaves the registry, with its length
/// and its marked chunks.
pub type BeforeCloseHook = Box<dyn FnMut(usize, Vec<usize>)>;

thread_local! {
    static BETWEEN_OPENS_HOOK: RefCell<Option<BetweenOpensHook>> = const { RefCell::new(None) };
    static BEFORE_CHUNK_HOOK: RefCell<Option<BeforeChunkHook>> = const { RefCell::new(None) };
    static AFTER_OPEN_HOOK: RefCell<Option<AfterOpenHook>> = const { RefCell::new(None) };
    static BEFORE_CLOSE_HOOK: RefCell<Option<BeforeCloseHook>> = const { RefCell::new(None) };
}

/// The size of the system's pages.
pub fn page_size() -> usize {
    // SAFETY: sysconf has no preconditions.
    usize::try_from(unsafe { libc::sysconf(libc::_SC_PAGESIZE) })
        .expect("the page size is positive")
}

/// The chunk size of the mappings that opens make from now on.
pub fn chunk_size() -> usize {
    CHUNK_SIZE_SETTING.load(Relaxed)
}

/// Sets the chunk size of the mappings that opens make from now on, a
/// multiple of the page size.
pub fn set_chunk_size(size: usize) {
    let page_size = page_size();
    assert!(
        size > 0 && size.is_multiple_of(page_size),
        "chunk size {size} is not a multiple of the page size {page_size}"
    );
    CHUNK_SIZE_SETTING.store(size, Relaxed);
}

/// The budget of marked bytes of the process.
pub fn limit() -> usize {
    registry::limit()
}

/// Sets the budget of marked bytes of the process, in place of
/// `JJ_PACK_MAPPED_LIMIT`'s; marks after this keep to it.
pub fn set_limit(bytes: usize) {
    registry::set_limit_for_tests(bytes);
}

/// The bytes of the chunks marked now, across every mapping of the process.
pub fn marked_bytes() -> usize {
    registry::marked_bytes()
}

/// Runs a release now, as a mark that passes the budget does.
pub fn release() {
    registry::release_now();
}

/// Forgets every mark of the process without releasing a chunk.
pub fn clear_marks() {
    registry::clear_marks();
}

/// The marked chunks of every mapping of the file at `path`, by index, in
/// order.
pub fn marked_chunks(path: &Path) -> io::Result<Vec<usize>> {
    let metadata = path.metadata()?;
    Ok(registry::marked_chunks((metadata.dev(), metadata.ino())))
}

/// How many `madvise` calls of releases have failed in this process.
pub fn failed_releases() -> u64 {
    registry::failed_releases()
}

/// Runs `hook` on this thread's next open of a file larger than one chunk,
/// after the first open of the file and before the second.
pub fn set_between_opens_hook(hook: impl FnOnce() + 'static) {
    BETWEEN_OPENS_HOOK.set(Some(Box::new(hook)));
}

/// Runs `hook` with each chunk's index before the chunk is mapped, in this
/// thread's opens, until it is cleared with `None`.
pub fn set_before_chunk_hook(hook: Option<BeforeChunkHook>) {
    BEFORE_CHUNK_HOOK.set(hook);
}

/// Runs `hook` with the addresses of each mapping that this thread's opens
/// make, until it is cleared with `None`.
pub fn set_after_open_hook(hook: Option<AfterOpenHook>) {
    AFTER_OPEN_HOOK.set(hook);
}

/// Runs `hook` with the length and the marked chunks of each mapping that is
/// dropped on this thread, before it leaves the registry, until it is cleared
/// with `None`.
pub fn set_before_close_hook(hook: Option<BeforeCloseHook>) {
    BEFORE_CLOSE_HOOK.set(hook);
}

pub(crate) fn run_between_opens_hook() {
    if let Some(hook) = BETWEEN_OPENS_HOOK.take() {
        hook();
    }
}

pub(crate) fn run_before_chunk_hook(chunk: usize) {
    BEFORE_CHUNK_HOOK.with_borrow_mut(|hook| {
        if let Some(hook) = hook {
            hook(chunk);
        }
    });
}

pub(crate) fn run_after_open_hook(mapping: &Mapping) {
    AFTER_OPEN_HOOK.with_borrow_mut(|hook| {
        if let Some(hook) = hook {
            let start = mapping.ptr.as_ptr().addr();
            hook(start..start + mapping.len);
        }
    });
}

pub(crate) fn run_before_close_hook(mapping: &Mapping) {
    BEFORE_CLOSE_HOOK.with_borrow_mut(|hook| {
        if let Some(hook) = hook {
            hook(mapping.len, mapping.marked_chunks());
        }
    });
}
