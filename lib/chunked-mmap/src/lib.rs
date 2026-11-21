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

//! Read-only file mappings made of chunks that a process keeps resident up to
//! a budget.
//!
//! gix-pack maps each pack file, pack index and multi-pack-index whole. Under
//! gVisor, the first read of one byte of a mapping counts all of the mapping as
//! the process's memory, so one read of a 12 GB pack costs 12 GB. On Linux,
//! [`Mmap`] maps a file larger than one chunk as adjacent chunks of 8 MiB
//! inside one reserved address range instead, taking the chunks in turn from
//! two open file descriptions of the file. gVisor merges two adjacent mappings
//! only when they come from the same open file description, and Linux only
//! when they come from the same `struct file`, so each chunk stays a mapping of
//! its own and is counted on its own.
//!
//! The bytes are reached only through [`Mmap::slice`] and [`Mmap::read_from`],
//! which mark the chunks that hold the bytes a reader consumed. Once the
//! chunks marked across every mapping of the process pass the budget, every
//! chunk that no read has marked since it was mapped or last released is
//! released with `madvise(MADV_DONTNEED)`, and so are the least recently marked
//! chunks until the marked chunks fit in the budget again. A later read of a
//! released chunk faults it back in from the file. The budget is 512 MiB, or
//! the bytes that the environment variable `JJ_PACK_MAPPED_LIMIT` gives, with
//! an optional suffix `m` (MiB) or `g` (GiB).
//!
//! On other targets, [`Mmap`] is memmap2's one mapping of the whole file, and
//! its reads mark nothing.

#[cfg(target_os = "linux")]
mod chunked;
#[cfg(target_os = "linux")]
mod registry;
#[cfg(target_os = "linux")]
#[doc(hidden)]
pub mod testing;
#[cfg(all(test, target_os = "linux"))]
mod tests;

use std::fmt;
use std::io;
use std::ops::Range;
use std::path::Path;
#[cfg(target_os = "linux")]
use std::sync::Arc;

/// The size of each chunk of a mapping, but its last.
pub const CHUNK_SIZE: usize = 8 << 20;

/// A read-only, private mapping of a whole file. On Linux, it is made of chunks
/// that count against the process's budget of resident chunks as reads mark
/// them.
pub struct Mmap {
    /// `None` for an empty file, which maps nothing.
    #[cfg(target_os = "linux")]
    mapping: Option<Arc<registry::Mapping>>,
    #[cfg(not(target_os = "linux"))]
    map: memmap2::Mmap,
}

impl Mmap {
    /// Maps the whole file at `path`, read-only and private (`PROT_READ`,
    /// `MAP_PRIVATE`), as `memmap2::MmapOptions::map_copy_read_only` does.
    ///
    /// On Linux, a file no larger than one chunk is one mapping. A larger file
    /// is mapped as chunks over one `PROT_NONE` reservation of its length, the
    /// even chunks from the file opened at `path` and the odd chunks from a
    /// second open of `/proc/self/fd/<n>` of the first, which names the same
    /// file even when another file has replaced `path` in the meantime. Both
    /// descriptors are closed once the chunks are mapped. When the reservation
    /// or a chunk cannot be mapped, the whole range is unmapped and the error
    /// is returned.
    ///
    /// The open fails, naming the variable, when `JJ_PACK_MAPPED_LIMIT` holds a
    /// value that is not a number of bytes.
    pub fn open(path: &Path) -> io::Result<Self> {
        #[cfg(target_os = "linux")]
        {
            Ok(Self {
                mapping: chunked::open(path)?,
            })
        }
        #[cfg(not(target_os = "linux"))]
        {
            let file = std::fs::File::open(path)?;
            // SAFETY: as for gix-pack's own mapping, nothing writes into a pack
            // or an index file once it exists.
            let map = unsafe { memmap2::MmapOptions::new().map_copy_read_only(&file)? };
            Ok(Self { map })
        }
    }

    /// The length of the file in bytes.
    pub fn len(&self) -> usize {
        self.bytes().len()
    }

    /// Whether the file is empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Returns the bytes of `range`, marking the chunks that hold them as read
    /// now.
    ///
    /// # Panics
    ///
    /// If `range` is not within the file, as indexing a slice does.
    pub fn slice(&self, range: Range<usize>) -> &[u8] {
        let bytes = &self.bytes()[range.clone()];
        self.mark(range);
        bytes
    }

    /// Lends the bytes from `start` to the end of the file to `read`, which
    /// returns its result and how many bytes from `start` it consumed, and
    /// marks the chunks that hold those as read now.
    ///
    /// # Panics
    ///
    /// If `start` is past the end of the file, as indexing a slice does.
    pub fn read_from<R>(&self, start: usize, read: impl FnOnce(&[u8]) -> (R, usize)) -> R {
        let (result, consumed) = read(&self.bytes()[start..]);
        self.mark(start..start.saturating_add(consumed));
        result
    }

    fn bytes(&self) -> &[u8] {
        #[cfg(target_os = "linux")]
        {
            self.mapping
                .as_deref()
                .map_or(&[], registry::Mapping::bytes)
        }
        #[cfg(not(target_os = "linux"))]
        {
            &self.map
        }
    }

    fn mark(&self, range: Range<usize>) {
        #[cfg(target_os = "linux")]
        if let Some(mapping) = &self.mapping {
            mapping.mark(range);
        }
        #[cfg(not(target_os = "linux"))]
        let _ = range;
    }
}

#[cfg(target_os = "linux")]
impl Drop for Mmap {
    fn drop(&mut self) {
        if let Some(mapping) = self.mapping.take() {
            chunked::close(&mapping);
        }
    }
}

impl fmt::Debug for Mmap {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Mmap")
            .field("ptr", &self.bytes().as_ptr())
            .field("len", &self.len())
            .finish()
    }
}
