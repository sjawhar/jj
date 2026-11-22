#!/usr/bin/env python3
# Copyright 2026 The Jujutsu Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Writes a planted chain of files that pass as virtual merge bases.

Usage: vb-chain.py side|base <k> <dir>

Writes the k+1 files g0 .. gk into <dir> and copies gk to <dir>/f, in the
format of lib/src/virtual_base.rs: one block with the side `s`, whose opening
line names the terms by their git blob ids. g0 is the plain text `leaf`.

  side: gi names [g(i-1), absent, g(i-1)], g(i-1) as both sides. No merge
        writes a virtual merge base that names one as a side, so jj reads
        each gi as a plain file.
  base: gi names [g0, g(i-1), g0, g(i-1), g0], g(i-1) as both bases, so that
        replacing gk with the terms it names, and so on down, doubles at each
        level: 2^k - 1 replacements in all.

Commit every file, so that the store holds the terms that each gi names.
"""

import hashlib
import shutil
import struct
import sys
from pathlib import Path

TAG = b"jj virtual merge base"


def hash_bytes(hasher, data):
    hasher.update(struct.pack("<Q", len(data)))
    hasher.update(data)


def git_blob_id(content):
    return hashlib.sha1(b"blob %d\0" % len(content) + content).hexdigest()


def virtual_base(names, sides, salt=0):
    """One block naming `names`, with `sides`, at `salt`."""
    hasher = hashlib.blake2b()
    hash_bytes(hasher, TAG)
    hasher.update(struct.pack("<Q", 0))
    hash_bytes(hasher, names)
    hash_bytes(hasher, b"")
    hasher.update(struct.pack("<Q", len(sides)))
    for side in sides:
        hash_bytes(hasher, side)
    hash_bytes(hasher, b"")
    hasher.update(struct.pack("<Q", salt))
    hex_digits = hasher.digest()[:16].hex().encode()
    content = b"<<<<<<< jj virtual merge base %s %d %s\n" % (hex_digits, salt, names)
    for index, side in enumerate(sides):
        if index > 0:
            content += b"======= %s\n" % hex_digits
        content += side
    content += b">>>>>>> %s\n" % hex_digits
    return content


def main(argv):
    if len(argv) != 4 or argv[1] not in ("side", "base") or not argv[2].isdigit():
        sys.exit("usage: vb-chain.py side|base <k> <dir>")
    kind, depth, directory = argv[1], int(argv[2]), Path(argv[3])
    directory.mkdir(parents=True, exist_ok=True)
    leaf = b"leaf\n"
    leaf_name = b"file:" + git_blob_id(leaf).encode()
    content = leaf
    (directory / "g0").write_bytes(content)
    for index in range(1, depth + 1):
        name = b"file:" + git_blob_id(content).encode()
        if kind == "side":
            names = b" ".join([name, b"absent", name])
        else:
            names = b" ".join([leaf_name, name, leaf_name, name, leaf_name])
        content = virtual_base(names, [b"s\n"])
        (directory / f"g{index}").write_bytes(content)
    shutil.copyfile(directory / f"g{depth}", directory / "f")


if __name__ == "__main__":
    main(sys.argv)
