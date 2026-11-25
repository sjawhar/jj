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
//! that a snapshot lays VB out as it was written; a block holds the level and
//! the names apart (`split_hunk_level` and `opening_tail`). The terms
//! name V's terms after `simplify()`, adds and removes alternating, each with
//! the label of the commit it comes from, and every block names the same
//! ones. `<hex>` is 32 hex digits of a hash over a fixed tag, the block's
//! ordinal among the file's blocks, what the opening line holds after the
//! salt, the text between the previous block (or the start of the file) and
//! this one, the block's sides, the text after the block if it is the last
//! one, and the salt. So the hashes cover every byte of the file: a file that
//! anyone changed is no longer VB or M.
//!
//! V's text can hold lines in the form of marker lines, a copy of another VB
//! among them. The writer takes the lowest salt, from the one above the salt
//! of every line of V's text in the form of an opening marker line (2^64-1
//! where such a line has that salt, as none is above it), at which no marker
//! line is a line of either side of the merge. The reader takes the lines in
//! the form of an opening marker line whose block closes, groups them by
//! their salt and by what follows it, and tries the groups from the highest
//! salt down: the blocks are those of the first group whose every line opens
//! a block whose hash checks out, each after the one before; every other
//! line is text, whatever its form. So the reader tries the writer's own
//! blocks first, unless V's text quotes blocks at the salt 2^64-1 too, and a
//! file that no merge wrote costs it a bounded amount of hashing
//! (`MAX_UNVERIFIED_HASHING`). The writer reads what it wrote back as the
//! reader does, and writes M where VB does not read back, and where no salt
//! is left at which VB's marker lines are none of the sides' lines. As anyone
//! can write a file whose hashes check out, a file counts as VB or M only
//! where the terms it names are in the store and none of its sides is VB or
//! M, as for every VB and M that a merge writes.
//!
//! The reader, and the finder of the blocks of VB that the base of a
//! conflict holds, which a snapshot reads, are in the module `blocks`.
//!
//! [`MergedTree::merge`]: crate::merged_tree::MergedTree::merge

use std::collections::HashSet;
use std::fmt::Write as _;
use std::iter;
use std::iter::zip;
use std::mem;

use blake2::Blake2b512;
use bstr::BString;
use bstr::ByteSlice as _;
use digest::Digest;
use futures::AsyncReadExt as _;
use futures::try_join;
use itertools::Itertools as _;

use self::blocks::Block;
use self::blocks::blocks;
use self::blocks::parse_opening_line;
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

mod blocks;
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
/// block that holds `base`'s sides at that hunk, in `base`'s order. Otherwise,
/// and where VB would not read back or no salt is left for its marker lines,
/// the file is M, one block whose sides each name a term of `base`, in
/// `base`'s order. Every block names `base`'s terms, each with its label, so
/// that a merge that replaces the file with those terms labels each with the
/// commit it comes from, and its marker lines are at a salt at which no line
/// of any term of either parent equals one of them (see the module's
/// documentation); the merge is refused where no salt gives even M such
/// marker lines. The file is executable where the executable bits of `base`
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
/// labels: every block names the same ones (see [`blocks`]).
fn named_terms(blocks: &[Block<'_>]) -> Option<NamedTerms> {
    parse_names(blocks.first()?.names)
}

/// The terms that `names`, the names in an opening marker line, name, with
/// their labels, if each is a name as [`term_name`] writes one and they are
/// an odd number.
fn parse_names(names: &[u8]) -> Option<NamedTerms> {
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
    blocks
        .first()
        .map_or(FileMergeHunkLevel::Line, |block| block.hunk_level)
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

/// What an opening marker line holds after its salt for a merge that split
/// V's terms into hunks at `hunk_level` and names them `names`: what
/// [`split_hunk_level`] reads back.
fn opening_tail(hunk_level: FileMergeHunkLevel, names: &str) -> String {
    match hunk_level {
        FileMergeHunkLevel::Line => names.to_owned(),
        FileMergeHunkLevel::Word => format!("{WORD_LEVEL}{names}"),
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
/// are none of `parent_lines` (see [`virtual_base_file`]). The merge is
/// refused where no salt gives M such marker lines either.
async fn write_virtual_base(
    store: &Store,
    path: &RepoPath,
    base: &MergedTreeValue,
    labels: &Merge<String>,
    base_contents: Option<Merge<BString>>,
    parent_lines: &HashSet<&[u8]>,
) -> BackendResult<FileId> {
    let options = store.merge_options();
    let Some(content) = virtual_base_file(base, labels, base_contents, options, parent_lines)
    else {
        let path = path.as_internal_file_string();
        return Err(BackendError::Unsupported(format!(
            "Refused to merge: at {path} no salt gives a virtual merge base marker lines that no \
             line of the merge's sides holds; no change was made"
        )));
    };
    store.write_file(path, &mut content.as_slice()).await
}

/// The content of the VB or M of `base`, as [`write_virtual_base`] writes it.
/// VB records `options`' hunk level, at which it merges `base_contents`. VB
/// holds V's text, whose lines can have any form, so VB with a block is read
/// back as any reader will read it, and the content is M, which holds none of
/// V's text, where it does not name `base` and `labels`, and where no salt is
/// left at which VB's marker lines are none of `parent_lines` (see
/// [`Draft::write`]). VB with no block, for a V whose contents merge, is their
/// merged text. `None` where no salt gives M such marker lines either.
fn virtual_base_file(
    base: &MergedTreeValue,
    labels: &Merge<String>,
    base_contents: Option<Merge<BString>>,
    options: &MergeOptions,
    parent_lines: &HashSet<&[u8]>,
) -> Option<Vec<u8>> {
    let names = zip(base.iter(), labels.iter())
        .map(|(term, label)| term_name(term.as_ref(), label))
        .collect_vec();
    let joined_names = names.join(" ");
    let vb = base_contents.and_then(|base_contents| {
        let tail = opening_tail(options.hunk_level, &joined_names);
        let hunks = text_hunks(&base_contents, options);
        let has_block = hunks
            .iter()
            .any(|hunk| matches!(hunk, VirtualBaseHunk::Block(_)));
        let vb = virtual_base_content(hunks, &tail, parent_lines)?;
        let reads_back = !has_block
            || parse_named_terms(&vb)
                .is_some_and(|named| named.terms == *base && named.labels == *labels);
        reads_back.then_some(vb)
    });
    vb.or_else(|| {
        let sides = names.iter().map(|name| format!("{name}\n").into_bytes());
        let hunks = vec![VirtualBaseHunk::Block(sides.collect())];
        let tail = opening_tail(FileMergeHunkLevel::Line, &joined_names);
        virtual_base_content(hunks, &tail, parent_lines)
    })
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

/// The content of VB or M whose opening lines hold `tail` after the salt
/// (see [`opening_tail`]), made of `hunks`: the text of each resolved hunk,
/// and each block's sides, each ending with a newline unless it is empty,
/// between marker lines at the lowest salt, from the one above the salt of
/// every line of that text in the form of an opening marker line (see
/// [`Draft::salt_above_text`]), at which no marker line is one of
/// `parent_lines` (see the module's documentation). `None` where no such salt
/// is left.
fn virtual_base_content(
    hunks: Vec<VirtualBaseHunk>,
    tail: &str,
    parent_lines: &HashSet<&[u8]>,
) -> Option<Vec<u8>> {
    let draft = Draft::new(hunks);
    let first_salt = draft.salt_above_text();
    draft.write(tail, first_salt, parent_lines)
}

/// VB or M before its marker lines are written: each block with the text
/// before it, and the text after the last block.
struct Draft {
    blocks: Vec<(Vec<u8>, Vec<Vec<u8>>)>,
    after: Vec<u8>,
}

impl Draft {
    /// The draft of `hunks`: each side of a block, and the text before a
    /// block, ends with a newline unless it is empty.
    fn new(hunks: Vec<VirtualBaseHunk>) -> Self {
        let mut blocks = vec![];
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
        Self {
            blocks,
            after: text,
        }
    }

    /// The lowest salt that is above the salt of every line of the draft's
    /// text in the form of an opening marker line, so that the reader tries
    /// the blocks written at it before any block that the text quotes; where
    /// such a line has the salt 2^64-1, none is above it, and it is that salt.
    fn salt_above_text(&self) -> u64 {
        self.blocks
            .iter()
            .flat_map(|(before, sides)| iter::once(before).chain(sides))
            .chain([&self.after])
            .flat_map(|text| text.lines_with_terminator())
            .filter_map(parse_opening_line)
            .map(|opening| opening.salt.saturating_add(1))
            .max()
            .unwrap_or(0)
    }

    /// The content, with marker lines that hold `tail` after the salt, at
    /// the lowest salt from `first_salt` up at which none is one of
    /// `parent_lines`; `None` where no salt is left.
    fn write(self, tail: &str, first_salt: u64, parent_lines: &HashSet<&[u8]>) -> Option<Vec<u8>> {
        let Self { blocks, after } = self;
        let last = blocks.len().saturating_sub(1);
        let hashers = blocks
            .iter()
            .enumerate()
            .map(|(ordinal, (before, sides))| {
                let after: &[u8] = if ordinal == last { &after } else { &[] };
                hasher_with_after(block_hasher(ordinal, tail.as_bytes(), before, sides), after)
            })
            .collect_vec();
        // Each salt gives another opening line, and other separator and
        // closing lines unless two salts give a block one hex, so a search
        // from salt 0, as M's is, ends within one salt more than there are
        // `parent_lines`. A search from higher up, as VB's above the salts of
        // V's text, can run out of salts.
        let markers = (first_salt..=u64::MAX)
            .map(|salt| {
                hashers
                    .iter()
                    .map(|hasher| marker_lines(&salted_hex(hasher, salt), salt, tail))
                    .collect_vec()
            })
            .find(|markers| {
                markers
                    .iter()
                    .flatten()
                    .all(|line| !parent_lines.contains(line.as_bytes()))
            })?;
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
        Some(content)
    }
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
/// `hex`, at `salt`, whose opening line holds `tail` after the salt, without
/// their newlines.
fn marker_lines(hex: &str, salt: u64, tail: &str) -> [String; 3] {
    [
        format!("{OPEN}{hex} {salt} {tail}"),
        format!("{SEPARATOR}{hex}"),
        format!("{CLOSE}{hex}"),
    ]
}

/// The hash of a block before the text after it and its salt: the tag, the
/// block's ordinal, what its opening line holds after the salt, the text
/// before the block, and its sides, each with its length first.
fn block_hasher(
    ordinal: usize,
    tail: &[u8],
    before: &[u8],
    sides: &[impl AsRef<[u8]>],
) -> Blake2b512 {
    let mut hasher = Blake2b512::new();
    hash_bytes(&mut hasher, TAG.as_bytes());
    Digest::update(&mut hasher, (ordinal as u64).to_le_bytes());
    hash_bytes(&mut hasher, tail);
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

    /// The sides of each of `blocks`.
    pub(super) fn sides<'a>(blocks: &[Block<'a>]) -> Vec<Vec<&'a [u8]>> {
        blocks.iter().map(|block| block.sides.clone()).collect()
    }

    /// `count` blocks that close and name a term each, a different one for
    /// each block, at `salt`, whose hashes do not check out.
    pub(super) fn quoted_blocks(count: usize, salt: u64) -> Vec<u8> {
        (1..=count)
            .map(|n| {
                let [open, separator, close] =
                    marker_lines(&format!("{n:032x}"), salt, &format!("file:{n:02x}"));
                format!("{open}\nq\n{separator}\nr\n{close}\n")
            })
            .join("")
            .into_bytes()
    }

    /// V's terms: three files, `x`, `o` and `y`, with those labels, and the
    /// merge options of a merge by line.
    fn labeled_terms() -> (MergedTreeValue, Merge<String>, MergeOptions) {
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
        (base, labels, options)
    }

    /// The contents of [`labeled_terms`]: `prefix`, then `x`, `o` and `y`, so
    /// that they conflict at their last line.
    fn labeled_contents(prefix: &[u8]) -> Merge<BString> {
        let content = |last: &str| BString::from([prefix, last.as_bytes()].concat());
        Merge::from_vec(vec![content("x\n"), content("o\n"), content("y\n")])
    }

    /// Asserts that `written` is M that names [`labeled_terms`].
    #[track_caller]
    fn assert_m(written: &[u8]) {
        let (base, _, _) = labeled_terms();
        let blocks_written = blocks(written).expect("M reads back");
        let m_sides = ["file:01@x\n", "file:02@o\n", "file:03@y\n"].map(str::as_bytes);
        assert_eq!(sides(&blocks_written), [m_sides]);
        assert_eq!(
            named_terms(&blocks_written).map(|named| named.terms),
            Some(base)
        );
    }

    #[test]
    fn vb_that_does_not_read_back_is_written_as_m() {
        let (base, labels, options) = labeled_terms();
        let no_lines = HashSet::new();
        // VB of terms that differ at their last line, after text that quotes
        // eight blocks: it reads back.
        let contents = labeled_contents(&quoted_blocks(8, 0));
        let written = virtual_base_file(&base, &labels, Some(contents), &options, &no_lines);
        let written = written.expect("a salt is free");
        let blocks_written = blocks(&written).expect("VB reads back");
        assert_eq!(sides(&blocks_written), [[b"x\n" as &[u8], b"y\n"]]);
        // No salt is above the highest one, so where the text quotes eight
        // blocks at that salt, the reader tries them first, past its bound:
        // VB would not read back, and the file is M.
        let contents = labeled_contents(&quoted_blocks(8, u64::MAX));
        let written = virtual_base_file(&base, &labels, Some(contents), &options, &no_lines);
        assert_m(&written.expect("a salt is free"));
    }

    #[test]
    fn vb_with_no_salt_left_is_written_as_m() {
        // V's text holds a line in the form of an opening marker line at salt
        // 2^64-2, so the writer's first salt is 2^64-1, the last one; or at
        // 2^64-1, and the writer takes that salt too, as none is above it. VB
        // reads back at that salt, but where a parent holds VB's separator
        // line there, no salt is left for VB, and the file is M.
        let (base, labels, options) = labeled_terms();
        for salt in [u64::MAX - 1, u64::MAX] {
            let [opening, _, _] = marker_lines(&"0123456789abcdef".repeat(2), salt, "x");
            let contents = labeled_contents(format!("{opening}\n").as_bytes());
            let no_lines = HashSet::new();
            let written =
                virtual_base_file(&base, &labels, Some(contents.clone()), &options, &no_lines);
            let written = written.expect("a salt is free");
            let blocks_written = blocks(&written).expect("VB reads back");
            assert_eq!(sides(&blocks_written), [[b"x\n" as &[u8], b"y\n"]]);
            let separator = written
                .lines()
                .find(|line| line.starts_with(SEPARATOR.as_bytes()))
                .expect("VB's separator line");
            let parent_lines = HashSet::from([separator]);
            let written =
                virtual_base_file(&base, &labels, Some(contents), &options, &parent_lines);
            assert_m(&written.expect("M's salt 0 is free"));
        }
    }
}
