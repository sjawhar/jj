//! Git stores all of its data as _Objects_, which are data along with a hash over all data. Storing objects efficiently
//! is what git packs are concerned about.
//!
//! Packs consist of [data files][data::File] and [index files][index::File]. The latter can be generated from a data file
//! and make accessing objects within a pack feasible.
//!
//! A [Bundle] conveniently combines a data pack alongside its index to allow [finding][Find] objects or verifying the pack.
//! Objects returned by `.find(…)` are [objects][gix_object::Data] which know their pack location in order to speed up
//! various common operations like creating new packs from existing ones.
//!
//! When traversing all objects in a pack, a _delta tree acceleration structure_ can be built from pack data or an index
//! in order to decompress packs in parallel and without any waste.
//! ## Feature Flags
#![cfg_attr(
    all(doc, feature = "document-features"),
    doc = ::document_features::document_features!()
)]
#![cfg_attr(all(doc, feature = "document-features"), feature(doc_cfg))]
#![deny(missing_docs, unsafe_code)]
// Patched in this tree: gitoxide publishes this crate and jj's tree carries it,
// formatted as gitoxide formats (120 columns), so jj's rustfmt and clippy rules
// do not apply to it. The rustfmt attributes on the `mod` declarations below
// keep rustfmt out of the other files.
#![cfg_attr(rustfmt, rustfmt::skip)]
#![allow(clippy::all, clippy::pedantic)]

use std::{borrow::Cow, ops::Deref, path::Path};

/// The default memory-backed storage for pack data and index files.
pub use memmap2::Mmap as MMap;

/// A byte-oriented backing store for pack data and indices.
pub trait FileData: Deref<Target = [u8]> {}

impl<T> FileData for T where T: Deref<Target = [u8]> {}

///
#[cfg_attr(rustfmt, rustfmt::skip)]
pub mod bundle;
/// A bundle of pack data and the corresponding pack index
pub struct Bundle {
    /// The pack file corresponding to `index`
    pub pack: data::File,
    /// The index file corresponding to `pack`
    pub index: index::File,
}

///
#[cfg_attr(rustfmt, rustfmt::skip)]
pub mod find;

///
#[cfg_attr(rustfmt, rustfmt::skip)]
pub mod cache;
///
#[cfg_attr(rustfmt, rustfmt::skip)]
pub mod data;

#[cfg_attr(rustfmt, rustfmt::skip)]
mod find_traits;
pub use find_traits::{Find, FindExt};

///
#[cfg_attr(rustfmt, rustfmt::skip)]
pub mod index;
///
#[cfg_attr(rustfmt, rustfmt::skip)]
pub mod multi_index;

///
#[cfg_attr(rustfmt, rustfmt::skip)]
pub mod verify;

///
#[cfg(feature = "testing")]
#[cfg_attr(rustfmt, rustfmt::skip)]
pub mod testing;

mod mmap {
    use std::path::Path;

    pub fn read_only(path: &Path) -> std::io::Result<memmap2::Mmap> {
        let file = std::fs::File::open(path)?;
        // SAFETY: we have to take the risk of somebody changing the file underneath. Git never writes into the same file.
        #[expect(unsafe_code)]
        unsafe {
            memmap2::MmapOptions::new().map_copy_read_only(&file)
        }
    }
}

/// Return a display-friendly name for pack- or index-related progress messages.
///
/// Prefer the file name, but fall back to the full path for paths without a terminal component.
fn source_name(path: &Path) -> Cow<'_, str> {
    if path.as_os_str().is_empty() {
        Cow::Borrowed("<memory>")
    } else if let Some(name) = path.file_name() {
        name.to_string_lossy()
    } else {
        path.as_os_str().to_string_lossy()
    }
}

#[inline]
fn read_u32(b: &[u8]) -> u32 {
    u32::from_be_bytes(b.try_into().unwrap())
}

#[inline]
fn read_u64(b: &[u8]) -> u64 {
    u64::from_be_bytes(b.try_into().unwrap())
}

fn exact_vec<T>(capacity: usize) -> Vec<T> {
    let mut v = Vec::new();
    _ = v.try_reserve_exact(capacity);
    v
}

#[inline]
fn fan_is_monotonically_increasing(fan: &[u32]) -> bool {
    !fan.windows(2).any(|window| window[0] > window[1])
}

#[cfg(test)]
mod tests {
    #[test]
    fn exact_capacity_is_only_an_optimization() {
        let result = std::panic::catch_unwind(|| super::exact_vec::<u8>(usize::MAX));
        assert!(result.is_ok(), "failure to preallocate an optimization must not panic");
        assert_eq!(
            result.expect("preallocation does not panic").capacity(),
            0,
            "a failed preallocation leaves the vector empty"
        );
    }
}
