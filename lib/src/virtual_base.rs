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

//! The virtual merge base: the file that stands for V, the unresolved merge of
//! the merge bases of two sides that both changed it, as the base of their
//! merge (<https://github.com/jj-vcs/jj/issues/6369>; see
//! [`crate::merge_rule`]).
//!
//! Where V is a conflict of text files, the file is VB: jj's line merge of V's
//! terms, with V's sides between marker lines where they do not merge, as git
//! builds a virtual merge base. Otherwise it is M: one block whose sides each
//! name a term of V. VB and M are only ever bases of conflicts: where a merge
//! leaves one as a side, [`MergedTree::merge`] replaces it with the terms of V
//! that it names.
//!
//! A block is an opening marker line, `<<<<<<< jj virtual merge base <hex>
//! <salt> <term>...`, its sides, a line `======= <hex>` between two sides, and
//! a closing marker line `>>>>>>> <hex>`. The terms name V's terms after
//! `simplify()`, adds and removes alternating, and every block names the same
//! ones. `<hex>` is 32 hex digits of a hash over a fixed tag, the block's
//! ordinal among the file's blocks, the terms, the text between the previous
//! block (or the start of the file) and this one, the block's sides, the text
//! after the block if it is the last one, and the salt. So the hashes cover
//! every byte of the file: a file that anyone changed is no longer VB or M. The
//! writer takes the lowest salt at which no marker line is a line of either
//! side of the merge, and the reader hashes each block once, with the salt
//! that the block carries.
//!
//! [`MergedTree::merge`]: crate::merged_tree::MergedTree::merge

use std::collections::HashSet;
use std::fmt::Write as _;
use std::iter::zip;
use std::mem;

use blake2::Blake2b512;
use bstr::BString;
use bstr::ByteSlice as _;
use digest::Digest;
use futures::AsyncReadExt as _;
use futures::try_join;
use itertools::Itertools as _;

use crate::backend::BackendError;
use crate::backend::BackendResult;
use crate::backend::CommitId;
use crate::backend::CopyId;
use crate::backend::FileId;
use crate::backend::MergedTreeValue;
use crate::backend::MergedTreeValueExt as _;
use crate::backend::SymlinkId;
use crate::backend::TreeId;
use crate::backend::TreeValue;
use crate::files;
use crate::files::MergeResult;
use crate::files::is_binary_content;
use crate::hex_util;
use crate::merge::Merge;
use crate::merge::SameChange;
use crate::object_id::ObjectId as _;
use crate::repo_path::RepoPath;
use crate::store::Store;

/// The tag that the hash of every block starts with.
const TAG: &str = "jj virtual merge base";
/// The start of the opening marker line of a block.
const OPEN: &str = "<<<<<<< jj virtual merge base ";
/// The start of the marker line between two sides of a block.
const SEPARATOR: &str = "======= ";
/// The start of the closing marker line of a block.
const CLOSE: &str = ">>>>>>> ";

/// Writes to the store the virtual base against which two parents that both
/// changed `base`, V, an unresolved merge at `path`, are merged, and returns it
/// as a file value.
///
/// Where `base` is a conflict of text files (each add a regular file whose
/// first 8,000 bytes hold no NUL byte, each remove such a file or absent, read
/// as an empty file), the file is VB, jj's line merge of `base`'s terms: each
/// hunk that merges keeps its merged text, and each hunk that does not is a
/// block that holds `base`'s sides at that hunk, in `base`'s order. Otherwise
/// the file is M, one block whose sides each name a term of `base`, in
/// `base`'s order. Every block names `base`'s terms, and its marker lines are
/// at the lowest salt at which no line of any term of either parent equals one
/// of them (see the module's documentation). The file is executable where the
/// executable bits of `base` merge to `true`, and keeps the copy id of `base`
/// where its copy ids merge, else the placeholder. The result does not depend
/// on the order of the parents.
pub async fn virtual_merge_base(
    store: &Store,
    path: &RepoPath,
    base: &MergedTreeValue,
    parent1: &MergedTreeValue,
    parent2: &MergedTreeValue,
) -> BackendResult<TreeValue> {
    let base = base.simplify();
    let base_contents = text_file_contents(store, path, &base);
    let contents1 = file_term_contents(store, path, parent1);
    let contents2 = file_term_contents(store, path, parent2);
    let (base_contents, contents1, contents2) = try_join!(base_contents, contents1, contents2)?;
    let parent_lines = marker_like_lines(
        contents1
            .iter()
            .chain(contents2.iter())
            .map(|content| content.as_slice()),
    );
    write_virtual_base(store, path, &base, base_contents, &parent_lines).await
}

/// The terms of V that `content` names, if `content` is VB or M as
/// [`virtual_merge_base`] writes it: it holds at least one block, every block
/// names the same terms, and the hex of each block is the hash of the block, as
/// the writer hashes it, with the salt that the block carries. Text that no
/// merge wrote is neither VB nor M, nor is VB or M with any byte changed.
pub fn virtual_base_terms(content: &[u8]) -> Option<MergedTreeValue> {
    let blocks = blocks(content)?;
    let [first, ..] = &blocks[..] else {
        return None;
    };
    let names = first.terms;
    if blocks.iter().any(|block| block.terms != names) {
        return None;
    }
    let terms = str::from_utf8(names)
        .ok()?
        .split(' ')
        .map(parse_term_name)
        .collect::<Option<Vec<_>>>()?;
    if terms.len().is_multiple_of(2) {
        return None;
    }
    let last = blocks.len() - 1;
    let mut text_start = 0;
    for (ordinal, block) in blocks.iter().enumerate() {
        let before = &content[text_start..block.start];
        let after: &[u8] = if ordinal == last {
            &content[block.end..]
        } else {
            &[]
        };
        let hasher = block_hasher(ordinal, names, before, &block.sides, after);
        if salted_hex(&hasher, block.salt).as_bytes() != block.hex {
            return None;
        }
        text_start = block.end;
    }
    Some(Merge::from_vec(terms))
}

/// The terms of V that `value` stands for, if `value` is VB or M: a file whose
/// content names them (see [`virtual_base_terms`]), each of which the store
/// holds.
pub(crate) async fn stored_virtual_base_terms(
    store: &Store,
    path: &RepoPath,
    value: &TreeValue,
) -> BackendResult<Option<MergedTreeValue>> {
    let TreeValue::File { id, .. } = value else {
        return Ok(None);
    };
    let content = read_file(store, path, id).await?;
    match virtual_base_terms(&content) {
        Some(terms) if holds_terms(store, path, &terms).await? => Ok(Some(terms)),
        _ => Ok(None),
    }
}

/// The file that takes the place of `old`, a base of a conflict at `path`,
/// where a parse of the conflict's materialized markers (a snapshot of the
/// working copy, or a merge tool's output) gave that base the content `new`:
/// if `old` is VB or M, the VB or M of V's terms with the change from `old` to
/// `new` merged into each of them, naming those terms, written to the store.
/// `contents` are the new contents of the conflict's terms, whose lines the
/// marker lines avoid. `None` where `old` is not VB or M, or where the change
/// does not merge cleanly into every file term of V or V has a term that is
/// neither a file nor absent (as where the change lies inside a block): then
/// `new` is a base of the person's own, and a plain file.
///
/// A working copy keeps the marker lines that it was written with after an
/// earlier change rewrote VB, so marker lines that differ from `old`'s, in
/// blocks that hold `old`'s sides, are not part of the change.
pub(crate) async fn rewrite_virtual_base(
    store: &Store,
    path: &RepoPath,
    old: &[u8],
    new: &[u8],
    contents: impl IntoIterator<Item = &[u8]>,
) -> BackendResult<Option<FileId>> {
    let Some(terms) = virtual_base_terms(old) else {
        return Ok(None);
    };
    if !holds_terms(store, path, &terms).await? {
        return Ok(None);
    }
    let new = with_marker_lines_of(old, new);
    if new == old {
        return Ok(Some(store.write_file(path, &mut &*new).await?));
    }
    let new = new.as_slice();
    let mut new_terms = vec![];
    for term in terms {
        let new_term = match term {
            None => None,
            Some(TreeValue::File {
                id,
                executable,
                copy_id,
            }) => {
                let content = read_file(store, path, &id).await?;
                let inputs = Merge::from_vec(vec![new, old, content.as_slice()]);
                let Some(merged) = files::try_merge(&inputs, store.merge_options()) else {
                    return Ok(None);
                };
                let id = store.write_file(path, &mut merged.as_slice()).await?;
                Some(TreeValue::File {
                    id,
                    executable,
                    copy_id,
                })
            }
            Some(_) => return Ok(None),
        };
        new_terms.push(new_term);
    }
    let base = Merge::from_vec(new_terms).simplify();
    if let Some(resolved) = base.as_resolved() {
        return Ok(match resolved {
            Some(TreeValue::File { id, .. }) => Some(id.clone()),
            _ => None,
        });
    }
    let base_contents = text_file_contents(store, path, &base).await?;
    let parent_lines = marker_like_lines(contents);
    match write_virtual_base(store, path, &base, base_contents, &parent_lines).await? {
        TreeValue::File { id, .. } => Ok(Some(id)),
        _ => unreachable!("a virtual base is a file"),
    }
}

/// `new` with the marker lines of each of its blocks replaced by those of the
/// block of `old` with the same ordinal, if `new` has as many blocks as `old`
/// and each holds the sides of `old`'s; otherwise `new` itself.
fn with_marker_lines_of(old: &[u8], new: &[u8]) -> Vec<u8> {
    let (Some(old_blocks), Some(new_blocks)) = (blocks(old), blocks(new)) else {
        return new.to_vec();
    };
    let same_sides = old_blocks.len() == new_blocks.len()
        && zip(&old_blocks, &new_blocks).all(|(old, new)| old.sides == new.sides);
    if !same_sides {
        return new.to_vec();
    }
    let mut content = vec![];
    let mut text_start = 0;
    for (old_block, new_block) in zip(old_blocks, new_blocks) {
        content.extend_from_slice(&new[text_start..new_block.start]);
        content.extend_from_slice(&old[old_block.start..old_block.end]);
        text_start = new_block.end;
    }
    content.extend_from_slice(&new[text_start..]);
    content
}

/// Writes to the store the VB or M of `base`, V after `simplify()`, whose
/// contents are `base_contents` if it is a conflict of text files, with marker
/// lines that are none of `parent_lines`.
async fn write_virtual_base(
    store: &Store,
    path: &RepoPath,
    base: &MergedTreeValue,
    base_contents: Option<Merge<BString>>,
    parent_lines: &HashSet<&[u8]>,
) -> BackendResult<TreeValue> {
    let hunks = match base_contents {
        Some(base_contents) => text_hunks(&base_contents, store),
        None => {
            let names = base
                .iter()
                .map(|term| term_name(term.as_ref()).into_bytes());
            vec![VirtualBaseHunk::Block(names.collect())]
        }
    };
    let content = virtual_base_content(hunks, base, parent_lines);
    let id = store.write_file(path, &mut content.as_slice()).await?;
    // The executable bit and the copy id depend on `base` alone, not on the
    // order of the parents.
    let executable = base
        .to_executable_merge()
        .and_then(|bits| bits.resolve_trivial(SameChange::Accept).copied().flatten())
        .unwrap_or(false);
    let copy_id = base
        .to_copy_id_merge()
        .and_then(|ids| ids.resolve_trivial(SameChange::Accept).cloned().flatten())
        .unwrap_or_else(CopyId::placeholder);
    Ok(TreeValue::File {
        id,
        executable,
        copy_id,
    })
}

/// The lines of `contents` that start like a marker line: only those can
/// equal one.
fn marker_like_lines<'a>(contents: impl IntoIterator<Item = &'a [u8]>) -> HashSet<&'a [u8]> {
    contents
        .into_iter()
        .flat_map(|content| content.lines())
        .filter(|line| {
            [OPEN, SEPARATOR, CLOSE]
                .iter()
                .any(|prefix| line.starts_with(prefix.as_bytes()))
        })
        .collect()
}

/// A hunk of VB or M: text that V's terms merge to, or the sides of a block.
enum VirtualBaseHunk {
    Resolved(Vec<u8>),
    Block(Vec<Vec<u8>>),
}

/// The hunks of the VB of `contents`, the terms of a conflict of text files,
/// merged with the store's merge options.
fn text_hunks(contents: &Merge<BString>, store: &Store) -> Vec<VirtualBaseHunk> {
    match files::merge_hunks(contents, store.merge_options()) {
        MergeResult::Resolved(content) => vec![VirtualBaseHunk::Resolved(content.into())],
        MergeResult::Conflict(hunks) => hunks
            .into_iter()
            .map(|hunk| match hunk.into_resolved() {
                Ok(content) => VirtualBaseHunk::Resolved(content.into()),
                Err(hunk) => {
                    VirtualBaseHunk::Block(hunk.adds().map(|side| side.to_vec()).collect())
                }
            })
            .collect(),
    }
}

/// The content of VB or M of `terms`, V's terms, made of `hunks`: the text of
/// each resolved hunk, and each block's sides, each ending with a newline
/// unless it is empty, between marker lines at the lowest salt at which none
/// is one of `parent_lines`.
fn virtual_base_content(
    hunks: Vec<VirtualBaseHunk>,
    terms: &MergedTreeValue,
    parent_lines: &HashSet<&[u8]>,
) -> Vec<u8> {
    let names = terms.iter().map(|term| term_name(term.as_ref())).join(" ");
    // Each block with the text before it, and the text after the last block.
    let mut blocks: Vec<(Vec<u8>, Vec<Vec<u8>>)> = vec![];
    let mut text = vec![];
    for hunk in hunks {
        match hunk {
            VirtualBaseHunk::Resolved(content) => text.extend(content),
            VirtualBaseHunk::Block(sides) => {
                let mut before = mem::take(&mut text);
                end_line(&mut before);
                let sides = sides
                    .into_iter()
                    .map(|mut side| {
                        end_line(&mut side);
                        side
                    })
                    .collect();
                blocks.push((before, sides));
            }
        }
    }
    let after = text;
    let last = blocks.len().saturating_sub(1);
    let hashers = blocks
        .iter()
        .enumerate()
        .map(|(ordinal, (before, sides))| {
            let after: &[u8] = if ordinal == last { &after } else { &[] };
            block_hasher(ordinal, names.as_bytes(), before, sides, after)
        })
        .collect_vec();
    // Each salt gives other opening lines, so the search ends after at most one
    // salt more than there are `parent_lines`.
    let markers = (0..)
        .map(|salt| {
            hashers
                .iter()
                .map(|hasher| marker_lines(&salted_hex(hasher, salt), salt, &names))
                .collect_vec()
        })
        .find(|markers| {
            markers
                .iter()
                .flatten()
                .all(|line| !parent_lines.contains(line.as_bytes()))
        })
        .expect("some salt gives marker lines that no parent holds");
    let mut content = vec![];
    for ((before, sides), [open, separator, close]) in zip(blocks, markers) {
        content.extend(before);
        push_line(&mut content, &open);
        for (index, side) in sides.into_iter().enumerate() {
            if index > 0 {
                push_line(&mut content, &separator);
            }
            content.extend(side);
        }
        push_line(&mut content, &close);
    }
    content.extend(after);
    content
}

/// Appends a newline to `text` unless it is empty or ends with one.
fn end_line(text: &mut Vec<u8>) {
    if text.last().is_some_and(|&byte| byte != b'\n') {
        text.push(b'\n');
    }
}

/// Appends `line` and a newline to `content`.
fn push_line(content: &mut Vec<u8>, line: &str) {
    content.extend_from_slice(line.as_bytes());
    content.push(b'\n');
}

/// The opening, separator and closing marker lines of a block whose hex is
/// `hex`, at `salt`, that names the terms `names`, without their newlines.
fn marker_lines(hex: &str, salt: u64, names: &str) -> [String; 3] {
    [
        format!("{OPEN}{hex} {salt} {names}"),
        format!("{SEPARATOR}{hex}"),
        format!("{CLOSE}{hex}"),
    ]
}

/// The hash of a block before its salt: the tag, the block's ordinal, the
/// names of the terms, the text before the block, its sides, and the text
/// after it, each with its length first.
fn block_hasher(
    ordinal: usize,
    names: &[u8],
    before: &[u8],
    sides: &[impl AsRef<[u8]>],
    after: &[u8],
) -> Blake2b512 {
    let mut hasher = Blake2b512::new();
    hash_bytes(&mut hasher, TAG.as_bytes());
    Digest::update(&mut hasher, (ordinal as u64).to_le_bytes());
    hash_bytes(&mut hasher, names);
    hash_bytes(&mut hasher, before);
    Digest::update(&mut hasher, (sides.len() as u64).to_le_bytes());
    for side in sides {
        hash_bytes(&mut hasher, side.as_ref());
    }
    hash_bytes(&mut hasher, after);
    hasher
}

/// Hashes `bytes` into `hasher`, with their length first.
fn hash_bytes(hasher: &mut Blake2b512, bytes: &[u8]) {
    Digest::update(hasher, (bytes.len() as u64).to_le_bytes());
    Digest::update(hasher, bytes);
}

/// 32 hex digits of the hash of `hasher` with `salt` hashed last.
fn salted_hex(hasher: &Blake2b512, salt: u64) -> String {
    let mut hasher = hasher.clone();
    Digest::update(&mut hasher, salt.to_le_bytes());
    hex_util::encode_hex(&hasher.finalize()[..16])
}

/// The name of a term of V in an opening marker line and in a side of M:
/// `absent`; `file:<id>`, then `:x` if the file is executable and
/// `:copy-<id>` if its copy id is not the placeholder; `symlink:<id>`;
/// `tree:<id>`; or `submodule:<id>`.
fn term_name(term: Option<&TreeValue>) -> String {
    match term {
        None => "absent".to_owned(),
        Some(TreeValue::File {
            id,
            executable,
            copy_id,
        }) => {
            let mut name = format!("file:{}", id.hex());
            if *executable {
                name.push_str(":x");
            }
            if !copy_id.as_bytes().is_empty() {
                write!(name, ":copy-{}", copy_id.hex()).unwrap();
            }
            name
        }
        Some(TreeValue::Symlink(id)) => format!("symlink:{}", id.hex()),
        Some(TreeValue::Tree(id)) => format!("tree:{}", id.hex()),
        Some(TreeValue::GitSubmodule(id)) => format!("submodule:{}", id.hex()),
    }
}

/// The term that `name` names, if [`term_name`] gives `name` for it.
fn parse_term_name(name: &str) -> Option<Option<TreeValue>> {
    let term = if name == "absent" {
        None
    } else {
        let mut parts = name.split(':');
        let kind = parts.next()?;
        let hex = parts.next()?;
        Some(match kind {
            "file" => {
                let id = FileId::try_from_hex(hex)?;
                let mut executable = false;
                let mut copy_id = CopyId::placeholder();
                for part in parts {
                    if part == "x" {
                        executable = true;
                    } else {
                        copy_id = CopyId::try_from_hex(part.strip_prefix("copy-")?)?;
                    }
                }
                TreeValue::File {
                    id,
                    executable,
                    copy_id,
                }
            }
            "symlink" => TreeValue::Symlink(SymlinkId::try_from_hex(hex)?),
            "tree" => TreeValue::Tree(TreeId::try_from_hex(hex)?),
            "submodule" => TreeValue::GitSubmodule(CommitId::try_from_hex(hex)?),
            _ => return None,
        })
    };
    // Each term has one name, so that the hashes cover the terms themselves.
    (term_name(term.as_ref()) == name).then_some(term)
}

/// A block of VB or M: the offsets of its opening line and of the end of its
/// closing line, the hex, the salt and the names of the terms its opening line
/// carries, and its sides, each with its newlines.
struct Block<'a> {
    start: usize,
    end: usize,
    hex: &'a [u8],
    salt: u64,
    terms: &'a [u8],
    sides: Vec<&'a [u8]>,
}

/// The blocks of `content`, in order, or `None` if a block does not close. A
/// block opens at an opening marker line, and each of its sides ends at the
/// next line that is [`SEPARATOR`] or [`CLOSE`] and the same hex.
fn blocks(content: &[u8]) -> Option<Vec<Block<'_>>> {
    let mut blocks = vec![];
    // The block being read, and the offset at which its current side starts.
    let mut current: Option<(Block<'_>, usize)> = None;
    let mut offset = 0;
    for line in content.lines_with_terminator() {
        let line_start = offset;
        offset += line.len();
        match current.take() {
            None => {
                if let Some((hex, salt, terms)) = parse_opening_line(line) {
                    let block = Block {
                        start: line_start,
                        end: offset,
                        hex,
                        salt,
                        terms,
                        sides: vec![],
                    };
                    current = Some((block, offset));
                }
            }
            Some((mut block, side_start)) => {
                let is_separator = marker_hex(line, SEPARATOR) == Some(block.hex);
                let is_close = marker_hex(line, CLOSE) == Some(block.hex);
                if is_separator || is_close {
                    block.sides.push(&content[side_start..line_start]);
                }
                if is_close {
                    block.end = offset;
                    blocks.push(block);
                } else if is_separator {
                    current = Some((block, offset));
                } else {
                    current = Some((block, side_start));
                }
            }
        }
    }
    current.is_none().then_some(blocks)
}

/// The hex, the salt and the names of the terms of `line` if it is an opening
/// marker line: [`OPEN`], 32 lowercase hex digits, a salt in decimal, the
/// names, and a newline.
fn parse_opening_line(line: &[u8]) -> Option<(&[u8], u64, &[u8])> {
    let rest = line.strip_prefix(OPEN.as_bytes())?.strip_suffix(b"\n")?;
    let (hex, rest) = rest.split_once_str(" ")?;
    let (salt, terms) = rest.split_once_str(" ")?;
    let salt_text = salt.to_str().ok()?;
    let salt: u64 = salt_text.parse().ok()?;
    let canonical = salt.to_string() == salt_text;
    (is_hex(hex) && canonical && !terms.is_empty()).then_some((hex, salt, terms))
}

/// The hex of `line` if `line` is `prefix`, 32 lowercase hex digits and a
/// newline.
fn marker_hex<'a>(line: &'a [u8], prefix: &str) -> Option<&'a [u8]> {
    let hex = line.strip_prefix(prefix.as_bytes())?.strip_suffix(b"\n")?;
    is_hex(hex).then_some(hex)
}

/// Whether `hex` is 32 lowercase hex digits.
fn is_hex(hex: &[u8]) -> bool {
    hex.len() == 32
        && hex
            .iter()
            .all(|&byte| matches!(byte, b'0'..=b'9' | b'a'..=b'f'))
}

/// Whether the store holds each term of `terms`. A submodule's commit is not
/// an object of the store, so it counts as held.
async fn holds_terms(
    store: &Store,
    path: &RepoPath,
    terms: &MergedTreeValue,
) -> BackendResult<bool> {
    for term in terms.iter().flatten() {
        let read = match term {
            TreeValue::File { id, .. } => store.read_file(path, id).await.map(drop),
            TreeValue::Symlink(id) => store.read_symlink(path, id).await.map(drop),
            TreeValue::Tree(id) => store.backend().read_tree(path, id).await.map(drop),
            TreeValue::GitSubmodule(_) => Ok(()),
        };
        match read {
            Ok(()) => {}
            Err(BackendError::ObjectNotFound { .. }) => return Ok(false),
            Err(err) => return Err(err),
        }
    }
    Ok(true)
}

/// The contents of the terms of `base` if `base` is a conflict of text files:
/// each add is a regular file whose first 8,000 bytes hold no NUL byte, which
/// is git's test for a binary file, and each remove is such a file or absent,
/// which reads as an empty file.
async fn text_file_contents(
    store: &Store,
    path: &RepoPath,
    base: &MergedTreeValue,
) -> BackendResult<Option<Merge<BString>>> {
    let Some(file_ids) = base.to_file_merge() else {
        return Ok(None);
    };
    if file_ids.adds().any(Option::is_none) {
        return Ok(None);
    }
    let contents = file_contents(store, path, &file_ids).await?;
    let is_text = contents.iter().all(|content| !is_binary_content(content));
    Ok(is_text.then_some(contents))
}

/// The contents of the file terms of `value`, and an empty content for each
/// other term.
async fn file_term_contents(
    store: &Store,
    path: &RepoPath,
    value: &MergedTreeValue,
) -> BackendResult<Merge<BString>> {
    let file_ids = value.map(|term| match term {
        Some(TreeValue::File { id, .. }) => Some(id.clone()),
        _ => None,
    });
    file_contents(store, path, &file_ids).await
}

/// The contents of the files `file_ids`, and an empty content for each absent
/// term.
async fn file_contents(
    store: &Store,
    path: &RepoPath,
    file_ids: &Merge<Option<FileId>>,
) -> BackendResult<Merge<BString>> {
    file_ids
        .try_map_async(async |id| match id {
            Some(id) => Ok(BString::new(read_file(store, path, id).await?)),
            None => Ok(BString::default()),
        })
        .await
}

/// The content of the file `id` at `path`.
async fn read_file(store: &Store, path: &RepoPath, id: &FileId) -> BackendResult<Vec<u8>> {
    let mut content = vec![];
    store
        .read_file(path, id)
        .await?
        .read_to_end(&mut content)
        .await
        .map_err(|err| BackendError::ReadFile {
            path: path.to_owned(),
            id: id.clone(),
            source: err.into(),
        })?;
    Ok(content)
}
