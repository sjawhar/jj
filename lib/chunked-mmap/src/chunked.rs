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

//! The system calls that map a file in chunks, and unmap it.

use std::fs::File;
use std::io;
use std::os::fd::AsRawFd as _;
use std::os::unix::fs::MetadataExt as _;
use std::path::Path;
use std::ptr;
use std::ptr::NonNull;
use std::sync::Arc;

// glibc's `off_t` is 32 bits on a 32-bit target, so an offset of 2 GiB or more
// takes `mmap64`, as memmap2 maps; musl's `off_t` is 64 bits on every target.
#[cfg(target_env = "musl")]
use libc::mmap;
#[cfg(not(target_env = "musl"))]
use libc::mmap64 as mmap;
#[cfg(target_env = "musl")]
use libc::off_t;
#[cfg(not(target_env = "musl"))]
use libc::off64_t as off_t;

use crate::registry;
use crate::registry::Mapping;
use crate::testing;

/// Maps the whole file at `path` as [`crate::Mmap::open`] says, and registers
/// the mapping. An empty file maps nothing.
pub(crate) fn open(path: &Path) -> io::Result<Option<Arc<Mapping>>> {
    registry::check_limit()?;
    let first = File::open(path)?;
    let metadata = first.metadata()?;
    let len = usize::try_from(metadata.len())
        .map_err(|_| io::Error::from(io::ErrorKind::FileTooLarge))?;
    if len == 0 {
        return Ok(None);
    }
    let chunk_size = testing::chunk_size();
    let ptr = if len <= chunk_size {
        map_whole(&first, len)?
    } else {
        testing::run_between_opens_hook();
        let second = File::open(format!("/proc/self/fd/{}", first.as_raw_fd()))?;
        map_chunks([&first, &second], len, chunk_size)?
    };
    // Both descriptors close here; each mapping keeps the file open.
    let mapping = registry::register(ptr, len, chunk_size, (metadata.dev(), metadata.ino()));
    testing::run_after_open_hook(&mapping);
    Ok(Some(mapping))
}

/// Unregisters `mapping` under the lock that every release takes, so no
/// release reaches its range once it is unmapped and perhaps mapped again for
/// something else, then unmaps it.
pub(crate) fn close(mapping: &Arc<Mapping>) {
    testing::run_before_close_hook(mapping);
    registry::unregister(mapping);
    unmap(mapping.ptr.as_ptr(), mapping.len);
}

/// Maps all `len` bytes of `file` as one mapping.
fn map_whole(file: &File, len: usize) -> io::Result<NonNull<u8>> {
    // SAFETY: a new private read-only mapping at an address the kernel picks
    // touches no memory that exists already.
    let address = unsafe {
        mmap(
            ptr::null_mut(),
            len,
            libc::PROT_READ,
            libc::MAP_PRIVATE,
            file.as_raw_fd(),
            0,
        )
    };
    if address == libc::MAP_FAILED {
        return Err(io::Error::last_os_error());
    }
    Ok(NonNull::new(address.cast()).expect("mmap gives no null address without MAP_FIXED"))
}

/// Maps all `len` bytes of the file that `files` both name, as chunks of
/// `chunk_size` bytes, taken from `files` in turn, over one reservation.
fn map_chunks(files: [&File; 2], len: usize, chunk_size: usize) -> io::Result<NonNull<u8>> {
    // SAFETY: a new anonymous reservation at an address the kernel picks
    // touches no memory that exists already.
    let base = unsafe {
        mmap(
            ptr::null_mut(),
            len,
            libc::PROT_NONE,
            libc::MAP_PRIVATE | libc::MAP_ANONYMOUS | libc::MAP_NORESERVE,
            -1,
            0,
        )
    };
    if base == libc::MAP_FAILED {
        return Err(io::Error::last_os_error());
    }
    let base = base.cast::<u8>();
    for (chunk, offset) in (0..len).step_by(chunk_size).enumerate() {
        testing::run_before_chunk_hook(chunk);
        let address = base.wrapping_add(offset);
        let chunk_len = chunk_size.min(len - offset);
        if let Err(err) = map_chunk(address, chunk_len, files[chunk % 2], offset) {
            unmap(base, len);
            return Err(err);
        }
    }
    Ok(NonNull::new(base).expect("mmap gives no null address without MAP_FIXED"))
}

/// Maps `len` bytes of `file` from `offset` at `address`, inside a reservation.
fn map_chunk(address: *mut u8, len: usize, file: &File, offset: usize) -> io::Result<()> {
    let offset =
        off_t::try_from(offset).map_err(|_| io::Error::from(io::ErrorKind::FileTooLarge))?;
    // SAFETY: `MAP_FIXED` replaces only pages of the reservation, which the
    // caller owns and nothing has read yet.
    let mapped = unsafe {
        mmap(
            address.cast(),
            len,
            libc::PROT_READ,
            libc::MAP_PRIVATE | libc::MAP_FIXED,
            file.as_raw_fd(),
            offset,
        )
    };
    if mapped == libc::MAP_FAILED {
        return Err(io::Error::last_os_error());
    }
    Ok(())
}

/// Unmaps the `len` bytes from `ptr`, which no reference may borrow any more.
fn unmap(ptr: *mut u8, len: usize) {
    // SAFETY: the caller owns the range and nothing borrows it.
    let result = unsafe { libc::munmap(ptr.cast(), len) };
    debug_assert_eq!(result, 0, "munmap: {}", io::Error::last_os_error());
}
