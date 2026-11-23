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
//! leaves one as a side, [`MergedTree::merge`] replaces every one at that path
//! with the terms of V that it names.
//!
//! A block is an opening marker line, `<<<<<<< jj virtual merge base <hex>
//! <salt> [word ]<term>...`, its sides, a line `======= <hex>` between two
//! sides, and a closing marker line `>>>>>>> <hex>`. `word` records that the
//! merge that wrote VB split V's terms into hunks by word
//! (`merge.hunk-level=word`), and its absence that it split them by line, so
//! that a snapshot lays VB out as it was written. The terms name V's terms
//! after `simplify()`, adds and removes alternating, each with the label of
//! the commit it comes from, and every block names the same ones. `<hex>` is
//! 32 hex digits of a hash over a fixed tag, the block's ordinal among the
//! file's blocks, what the opening line holds after the salt, the text
//! between the previous block (or the start of the file) and this one, the
//! block's sides, the text after the block if it is the last one, and the
//! salt. So the hashes cover every byte of the file: a file that anyone
//! changed is no longer VB or M. The writer takes the lowest salt at which no
//! marker line is a line of either side of the merge, and reads what it wrote
//! back as the reader does, writing M where VB does not read back. The reader
//! takes a line in the form of an opening marker line as the start of a block
//! only where the block closes and its hash checks out; every other line is
//! text, whatever its form. As anyone can write a file whose hashes check
//! out, a file counts as VB or M only where the terms it names are in the
//! store and none of its sides is VB or M, as for every VB and M that a merge
//! writes.
//!
//! [`MergedTree::merge`]: crate::merged_tree::MergedTree::merge

use std::collections::HashMap;
use std::collections::HashSet;
use std::fmt::Write as _;
use std::iter::zip;
use std::mem;
use std::ops::Range;

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
use crate::conflict_labels::ConflictLabels;
use crate::files;
use crate::files::FileMergeHunkLevel;
use crate::files::MergeResult;
use crate::files::is_binary_content;
use crate::hex_util;
use crate::merge::Merge;
use crate::merge::SameChange;
use crate::object_id::ObjectId as _;
use crate::repo_path::RepoPath;
use crate::store::Store;
use crate::tree_merge::MergeOptions;

mod expand;
mod snapshot;

pub(crate) use expand::Expansion;
pub(crate) use expand::NamedTermsCache;
pub(crate) use expand::Replacements;
pub(crate) use expand::Substitute;
pub(crate) use expand::append_pairs;
pub(crate) use expand::replaces_at;
pub(crate) use snapshot::rewrite_virtual_base;

/// The tag that the hash of every block starts with.
const TAG: &str = "jj virtual merge base";
/// The start of the opening marker line of a block.
const OPEN: &str = "<<<<<<< jj virtual merge base ";
/// The start of the marker line between two sides of a block.
const SEPARATOR: &str = "======= ";
/// The start of the closing marker line of a block.
const CLOSE: &str = ">>>>>>> ";
/// What an opening marker line holds after its salt, before the names of the
/// terms, where the merge that wrote VB split V's terms into hunks by word.
const WORD_LEVEL: &str = "word ";

/// Writes to the store the virtual base against which two parents that both
/// changed `base`, V, an unresolved merge at `path` whose terms have the labels
/// `labels`, are merged, and returns it as a file value.
///
/// Where `base` is a conflict of text files (each add a regular file whose
/// first 8,000 bytes hold no NUL byte, each remove such a file or absent, read
/// as an empty file), the file is VB, jj's line merge of `base`'s terms: each
/// hunk that merges keeps its merged text, and each hunk that does not is a
/// block that holds `base`'s sides at that hunk, in `base`'s order. Otherwise
/// the file is M, one block whose sides each name a term of `base`, in
/// `base`'s order. Every block names `base`'s terms, each with its label, so
/// that a merge that replaces the file with those terms labels each with the
/// commit it comes from, and its marker lines are at the lowest salt at which
/// no line of any term of either parent equals one of them (see the module's
/// documentation). The file is executable where the executable bits of `base`
/// merge to `true`, and keeps the copy id of `base` where its copy ids merge,
/// else the placeholder. The result does not depend on the order of the
/// parents.
pub async fn virtual_merge_base(
    store: &Store,
    path: &RepoPath,
    base: &MergedTreeValue,
    labels: &ConflictLabels,
    parent1: &MergedTreeValue,
    parent2: &MergedTreeValue,
) -> BackendResult<TreeValue> {
    let (base, labels) = simplified_with_labels(base, labels);
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
    let id = write_virtual_base(store, path, &base, &labels, base_contents, &parent_lines).await?;
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

/// The terms of V that `content` names, if `content` is VB or M as
/// [`virtual_merge_base`] writes it: it holds at least one block whose hash
/// checks out (see the module's documentation), and every block names the
/// same terms. Text that no merge wrote is neither VB nor M, nor is VB or M
/// with any byte changed.
pub fn virtual_base_terms(content: &[u8]) -> Option<MergedTreeValue> {
    parse_named_terms(content).map(|named| named.terms)
}

/// The terms of V that VB or M names, each with the label of the commit it
/// comes from.
#[derive(Clone, Debug)]
pub(crate) struct NamedTerms {
    pub(crate) terms: MergedTreeValue,
    pub(crate) labels: Merge<String>,
}

/// The terms of V that `content` names, with their labels, if `content` is
/// VB or M (see [`virtual_base_terms`]).
fn parse_named_terms(content: &[u8]) -> Option<NamedTerms> {
    named_terms(&blocks(content)?)
}

/// The terms of V that `blocks`, the blocks of VB or M, name, with their
/// labels, if every block names the same ones and they are an odd number.
fn named_terms(blocks: &[Block<'_>]) -> Option<NamedTerms> {
    let [first, ..] = blocks else {
        return None;
    };
    if blocks.iter().any(|block| block.terms != first.terms) {
        return None;
    }
    let (_, names) = split_hunk_level(first.terms);
    let (terms, labels): (Vec<_>, Vec<_>) = str::from_utf8(names)
        .ok()?
        .split(' ')
        .map(parse_term_name)
        .collect::<Option<Vec<_>>>()?
        .into_iter()
        .unzip();
    if terms.len().is_multiple_of(2) {
        return None;
    }
    Some(NamedTerms {
        terms: Merge::from_vec(terms),
        labels: Merge::from_vec(labels),
    })
}

/// The hunk level at which the merge that wrote VB, whose blocks are
/// `blocks`, split V's terms into hunks.
fn recorded_hunk_level(blocks: &[Block<'_>]) -> FileMergeHunkLevel {
    blocks.first().map_or(FileMergeHunkLevel::Line, |block| {
        split_hunk_level(block.terms).0
    })
}

/// The hunk level that `tail`, what an opening marker line holds after its
/// salt, records, and the names of the terms that follow it: [`WORD_LEVEL`]
/// records word level, and no word line level.
fn split_hunk_level(tail: &[u8]) -> (FileMergeHunkLevel, &[u8]) {
    match tail.strip_prefix(WORD_LEVEL.as_bytes()) {
        Some(names) => (FileMergeHunkLevel::Word, names),
        None => (FileMergeHunkLevel::Line, tail),
    }
}

/// The terms of V that `value` stands for, with their labels, if `value` is
/// VB or M: a file whose content names them (see [`virtual_base_terms`]),
/// and that [`holds_terms`] accepts. Otherwise `value` is a plain file.
pub(crate) async fn stored_virtual_base_terms(
    store: &Store,
    path: &RepoPath,
    value: &TreeValue,
) -> BackendResult<Option<NamedTerms>> {
    let TreeValue::File { id, .. } = value else {
        return Ok(None);
    };
    let content = read_file(store, path, id).await?;
    match parse_named_terms(&content) {
        Some(named) if holds_terms(store, path, &named.terms).await? => Ok(Some(named)),
        _ => Ok(None),
    }
}

/// Writes to the store the VB or M of `base`, V after `simplify()`, each of
/// whose terms has the label of `labels` at its place, whose contents are
/// `base_contents` if it is a conflict of text files, with marker lines that
/// are none of `parent_lines` (see [`virtual_base_file`]).
async fn write_virtual_base(
    store: &Store,
    path: &RepoPath,
    base: &MergedTreeValue,
    labels: &Merge<String>,
    base_contents: Option<Merge<BString>>,
    parent_lines: &HashSet<&[u8]>,
) -> BackendResult<FileId> {
    let options = store.merge_options();
    let content = virtual_base_file(base, labels, base_contents, options, parent_lines);
    store.write_file(path, &mut content.as_slice()).await
}

/// The content of the VB or M of `base`, as [`write_virtual_base`] writes it.
/// VB records `options`' hunk level, at which it merges `base_contents`. VB
/// holds V's text, whose lines can have any form, so VB with a block is read
/// back as any reader will read it, and the content is M, which holds none of
/// V's text, where it does not name `base` and `labels`. VB with no block,
/// for a V whose contents merge, is their merged text.
fn virtual_base_file(
    base: &MergedTreeValue,
    labels: &Merge<String>,
    base_contents: Option<Merge<BString>>,
    options: &MergeOptions,
    parent_lines: &HashSet<&[u8]>,
) -> Vec<u8> {
    let names = zip(base.iter(), labels.iter())
        .map(|(term, label)| term_name(term.as_ref(), label))
        .collect_vec();
    let joined_names = names.join(" ");
    let vb = base_contents
        .map(|base_contents| {
            let tail = match options.hunk_level {
                FileMergeHunkLevel::Line => joined_names.clone(),
                FileMergeHunkLevel::Word => format!("{WORD_LEVEL}{joined_names}"),
            };
            let hunks = text_hunks(&base_contents, options);
            let has_block = hunks
                .iter()
                .any(|hunk| matches!(hunk, VirtualBaseHunk::Block(_)));
            (has_block, virtual_base_content(hunks, &tail, parent_lines))
        })
        .filter(|(has_block, vb)| {
            !has_block
                || parse_named_terms(vb)
                    .is_some_and(|named| named.terms == *base && named.labels == *labels)
        });
    match vb {
        Some((_, vb)) => vb,
        None => {
            let sides = names.iter().map(|name| format!("{name}\n").into_bytes());
            let hunks = vec![VirtualBaseHunk::Block(sides.collect())];
            virtual_base_content(hunks, &joined_names, parent_lines)
        }
    }
}

/// The labels of the terms of `base`, V, from `labels`, after `simplify()`,
/// with `base` simplified too; an empty label for each term where `labels`
/// has none.
fn simplified_with_labels(
    base: &MergedTreeValue,
    labels: &ConflictLabels,
) -> (MergedTreeValue, Merge<String>) {
    let labels = if labels.num_sides() == Some(base.num_sides()) {
        labels.clone()
    } else {
        ConflictLabels::unlabeled()
    };
    let (labels, base) = labels.simplify_with(base);
    let labels = if labels.has_labels() {
        labels.into_merge()
    } else {
        Merge::repeated(String::new(), base.num_sides())
    };
    (base, labels)
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
/// merged with `options`.
fn text_hunks(contents: &Merge<BString>, options: &MergeOptions) -> Vec<VirtualBaseHunk> {
    match files::merge_hunks(contents, options) {
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

/// The content of VB or M whose opening lines hold `names` after the salt
/// (the hunk level of VB and the names of V's terms), made of `hunks`: the
/// text of each resolved hunk, and each block's sides, each ending with a
/// newline unless it is empty, between marker lines at the lowest salt at
/// which none is one of `parent_lines`.
fn virtual_base_content(
    hunks: Vec<VirtualBaseHunk>,
    names: &str,
    parent_lines: &HashSet<&[u8]>,
) -> Vec<u8> {
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
            hasher_with_after(
                block_hasher(ordinal, names.as_bytes(), before, sides),
                after,
            )
        })
        .collect_vec();
    // Each salt gives other opening lines, so the search ends after at most one
    // salt more than there are `parent_lines`.
    let markers = (0..)
        .map(|salt| {
            hashers
                .iter()
                .map(|hasher| marker_lines(&salted_hex(hasher, salt), salt, names))
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

/// The hash of a block before the text after it and its salt: the tag, the
/// block's ordinal, the names of the terms, the text before the block, and its
/// sides, each with its length first.
fn block_hasher(
    ordinal: usize,
    names: &[u8],
    before: &[u8],
    sides: &[impl AsRef<[u8]>],
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
    hasher
}

/// `hasher`, a [`block_hasher`], with `after` hashed next, with its length
/// first: the text after the block if it is the last one, else nothing.
fn hasher_with_after(mut hasher: Blake2b512, after: &[u8]) -> Blake2b512 {
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

/// The name of a term of V, with its label, in an opening marker line and in a
/// side of M: `absent`; `file:<id>`, then `:x` if the file is executable and
/// `:copy-<id>` if its copy id is not the placeholder; `symlink:<id>`;
/// `tree:<id>`; or `submodule:<id>`; then, unless `label` is empty, `@` and
/// the label with each byte other than an ASCII letter, a digit, `-`, `.`,
/// `_`, `~` or `/` written as `%` and two hex digits.
fn term_name(term: Option<&TreeValue>, label: &str) -> String {
    let mut name = match term {
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
    };
    if !label.is_empty() {
        name.push('@');
        for byte in label.bytes() {
            if byte.is_ascii_alphanumeric() || b"-._~/".contains(&byte) {
                name.push(char::from(byte));
            } else {
                write!(name, "%{byte:02X}").unwrap();
            }
        }
    }
    name
}

/// The term that `name` names, and its label, if [`term_name`] gives `name`
/// for them.
fn parse_term_name(name: &str) -> Option<(Option<TreeValue>, String)> {
    let (term_part, label) = match name.split_once('@') {
        Some((term_part, encoded)) => (term_part, decode_label(encoded)?),
        None => (name, String::new()),
    };
    let term = if term_part == "absent" {
        None
    } else {
        let mut parts = term_part.split(':');
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
    // Each term and label has one name, so that the hashes cover them.
    (term_name(term.as_ref(), &label) == name).then_some((term, label))
}

/// The label that `encoded` writes as [`term_name`] writes one.
fn decode_label(encoded: &str) -> Option<String> {
    let mut bytes = vec![];
    let mut rest = encoded.as_bytes();
    while let Some((&byte, tail)) = rest.split_first() {
        if byte == b'%' {
            let hex = tail.get(..2)?;
            bytes.push(u8::from_str_radix(str::from_utf8(hex).ok()?, 16).ok()?);
            rest = &tail[2..];
        } else {
            bytes.push(byte);
            rest = tail;
        }
    }
    String::from_utf8(bytes).ok()
}

/// A block of VB or M: the offsets of its opening line and of the end of its
/// closing line, what its opening line carries after the salt (the hunk level
/// and the names of the terms), and its sides, each with its newlines, and
/// their offsets.
struct Block<'a> {
    start: usize,
    end: usize,
    terms: &'a [u8],
    sides: Vec<&'a [u8]>,
    side_ranges: Vec<Range<usize>>,
}

/// The most bytes that [`blocks`] hashes for opening marker lines whose blocks
/// do not verify, as a multiple of the length of the content.
const MAX_UNVERIFIED_HASHING: usize = 4;

/// The blocks of `content`, in order, as the writer wrote them. An opening
/// marker line opens a block only where the next closing marker line of its
/// hex follows, with no other opening marker line of that hex before it, and
/// the block verifies: its hex is the hash of the block with the salt that
/// its opening line carries, where the block's ordinal counts the blocks
/// before it, the text before it starts where the previous block ends, and
/// the text after it is the rest of the content for the last block and
/// nothing for another. Each side ends at a line that is [`SEPARATOR`] and
/// the block's hex, or at the closing line. Every other line is text,
/// whatever its form. `None` where no block verifies, or where the last one
/// verifies only as a block that another follows; and, as for a file that is
/// neither VB nor M, where checking opening lines whose blocks do not verify
/// would hash more than [`MAX_UNVERIFIED_HASHING`] times the content's
/// length, so that reading any content takes time in proportion to its
/// length.
fn blocks(content: &[u8]) -> Option<Vec<Block<'_>>> {
    let lines = content
        .lines_with_terminator()
        .scan(0, |offset, line| {
            let start = *offset;
            *offset += line.len();
            Some((start, line))
        })
        .collect_vec();
    // The closing line of each opening line's block: the next closing line of
    // its hex, unless another opening line of that hex comes first.
    let mut closes = vec![None; lines.len()];
    let mut next_close: HashMap<&[u8], usize> = HashMap::new();
    for (index, &(_, line)) in lines.iter().enumerate().rev() {
        if let Some(hex) = marker_hex(line, CLOSE) {
            next_close.insert(hex, index);
        } else if let Some((hex, _, _)) = parse_opening_line(line) {
            closes[index] = next_close.remove(hex);
        }
    }
    let mut budget = content.len().saturating_mul(MAX_UNVERIFIED_HASHING);
    let mut blocks = vec![];
    let mut text_start = 0;
    let mut index = 0;
    while index < lines.len() {
        let (start, line) = lines[index];
        let (Some(close), Some((hex, salt, terms))) = (closes[index], parse_opening_line(line))
        else {
            index += 1;
            continue;
        };
        let mut sides = vec![];
        let mut side_ranges = vec![];
        let mut side_start = start + line.len();
        for &(line_start, line) in &lines[index + 1..=close] {
            if [SEPARATOR, CLOSE]
                .iter()
                .any(|prefix| marker_hex(line, prefix) == Some(hex))
            {
                sides.push(&content[side_start..line_start]);
                side_ranges.push(side_start..line_start);
                side_start = line_start + line.len();
            }
        }
        let end = side_start;
        let hasher = block_hasher(blocks.len(), terms, &content[text_start..start], &sides);
        let block = Block {
            start,
            end,
            terms,
            sides,
            side_ranges,
        };
        if salted_hex(&hasher_with_after(hasher.clone(), &[]), salt).as_bytes() == hex {
            blocks.push(block);
            text_start = end;
            index = close + 1;
            continue;
        }
        let after = &content[end..];
        if salted_hex(&hasher_with_after(hasher, after), salt).as_bytes() == hex {
            blocks.push(block);
            return Some(blocks);
        }
        budget = budget.checked_sub(content.len() - text_start)?;
        index += 1;
    }
    (!blocks.is_empty() && text_start == content.len()).then_some(blocks)
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

/// Whether `terms`, the terms of V that a VB or M names, can be V's: the store
/// holds each of them, and no side of V is itself VB or M, which no merge
/// writes as a side. V's bases can be VB or M, once or more. A submodule's
/// commit is not an object of the store, so it counts as held. The side rule
/// is not recursive: a side counts as VB or M where its marker lines verify
/// ([`virtual_base_terms`]), whether or not the terms it names would pass
/// this check. A planted chain of files that each name the one before as a
/// side is thus refused at its first link; one that names the one before as
/// bases is bounded where a merge replaces the files (see `MergedTree::merge`).
async fn holds_terms(
    store: &Store,
    path: &RepoPath,
    terms: &MergedTreeValue,
) -> BackendResult<bool> {
    for (index, term) in terms.iter().enumerate() {
        let read = match term {
            None => continue,
            Some(TreeValue::File { id, .. }) if index.is_multiple_of(2) => {
                match read_file(store, path, id).await {
                    Ok(content) if virtual_base_terms(&content).is_some() => return Ok(false),
                    result => result.map(drop),
                }
            }
            Some(TreeValue::File { id, .. }) => store.read_file(path, id).await.map(drop),
            Some(TreeValue::Symlink(id)) => store.read_symlink(path, id).await.map(drop),
            Some(TreeValue::Tree(id)) => store.backend().read_tree(path, id).await.map(drop),
            Some(TreeValue::GitSubmodule(_)) => Ok(()),
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

#[cfg(test)]
mod tests {
    use super::*;

    /// A line in the form of an opening marker line, whose block never
    /// closes.
    const OPENING_FORM: &str =
        "<<<<<<< jj virtual merge base 0123456789abcdef0123456789abcdef 0 quoted\n";

    /// VB that names `x o y`, made of `hunks`.
    fn vb(hunks: Vec<VirtualBaseHunk>) -> Vec<u8> {
        virtual_base_content(hunks, "x o y", &HashSet::new())
    }

    /// A block with the sides `x` and `y`.
    fn block() -> VirtualBaseHunk {
        VirtualBaseHunk::Block(vec![b"x\n".to_vec(), b"y\n".to_vec()])
    }

    /// The sides of each of `blocks`.
    fn sides<'a>(blocks: &[Block<'a>]) -> Vec<Vec<&'a [u8]>> {
        blocks.iter().map(|block| block.sides.clone()).collect()
    }

    #[test]
    fn an_opening_line_whose_block_never_closes_is_text() {
        let text = format!("1\n{OPENING_FORM}3\n");
        let content = vb(vec![
            VirtualBaseHunk::Resolved(text.into_bytes()),
            block(),
            VirtualBaseHunk::Resolved(b"after\n".to_vec()),
        ]);
        let blocks = blocks(&content).expect("VB reads back");
        assert_eq!(sides(&blocks), [[b"x\n" as &[u8], b"y\n"]]);
    }

    #[test]
    fn a_quoted_block_whose_hash_does_not_check_out_is_text() {
        // VB's own text quotes another VB whole, closing line and all, after
        // a line of text (as the first line of a file, a byte copy of VB is
        // VB) before VB's own block, and after it.
        let quoted = vb(vec![block()]);
        let before = [b"1\n".as_slice(), &quoted].concat();
        let content = vb(vec![
            VirtualBaseHunk::Resolved(before),
            block(),
            VirtualBaseHunk::Resolved(quoted),
        ]);
        let blocks = blocks(&content).expect("VB reads back");
        assert_eq!(sides(&blocks), [[b"x\n" as &[u8], b"y\n"]]);
    }

    #[test]
    fn vb_with_any_byte_changed_is_not_vb() {
        let content = vb(vec![
            block(),
            VirtualBaseHunk::Resolved(b"between\n".to_vec()),
            block(),
            VirtualBaseHunk::Resolved(b"after\n".to_vec()),
        ]);
        assert_eq!(blocks(&content).map(|blocks| blocks.len()), Some(2));
        let changed = content.replace("after", "AFTER");
        assert!(blocks(&changed).is_none());
        let changed = content.replace("between", "BETWEEN");
        assert!(blocks(&changed).is_none());
    }

    #[test]
    fn the_hunk_level_reads_back() {
        let content = virtual_base_content(vec![block()], "word x o y", &HashSet::new());
        let blocks = blocks(&content).expect("VB reads back");
        assert_eq!(recorded_hunk_level(&blocks), FileMergeHunkLevel::Word);
        assert_eq!(split_hunk_level(blocks[0].terms).1, b"x o y");
    }

    #[test]
    fn vb_that_does_not_read_back_is_written_as_m() {
        let file = |id: u8| {
            Some(TreeValue::File {
                id: FileId::new(vec![id]),
                executable: false,
                copy_id: CopyId::placeholder(),
            })
        };
        let base = Merge::from_vec(vec![file(1), file(2), file(3)]);
        let labels = Merge::from_vec(vec!["x".to_owned(), "o".to_owned(), "y".to_owned()]);
        let options = MergeOptions {
            hunk_level: FileMergeHunkLevel::Line,
            same_change: SameChange::Accept,
        };
        let contents = |prefix: &[u8]| {
            let content = |last: &str| BString::from([prefix, last.as_bytes()].concat());
            Merge::from_vec(vec![content("x\n"), content("o\n"), content("y\n")])
        };
        let names = "file:01@x file:02@o file:03@y";
        // VB of terms that differ at their last line, after a first line.
        let written = virtual_base_file(
            &base,
            &labels,
            Some(contents(b"1\n")),
            &options,
            &HashSet::new(),
        );
        let blocks_written = blocks(&written).expect("VB reads back");
        assert_eq!(sides(&blocks_written), [[b"x\n" as &[u8], b"y\n"]]);
        // Terms that start with a byte copy of a VB, which, as the first text
        // of a file, is VB: their VB would not read back, so the file is M.
        let quoted = vb(vec![block()]);
        let written = virtual_base_file(
            &base,
            &labels,
            Some(contents(&quoted)),
            &options,
            &HashSet::new(),
        );
        let blocks_written = blocks(&written).expect("M reads back");
        let m_sides = names
            .split(' ')
            .map(|name| format!("{name}\n"))
            .collect_vec();
        let m_sides = m_sides.iter().map(|side| side.as_bytes()).collect_vec();
        assert_eq!(sides(&blocks_written), [m_sides]);
        assert_eq!(
            named_terms(&blocks_written).map(|named| named.terms),
            Some(base)
        );
    }
}
