"""Index persistence in two formats: pickle and compact array('I') binary (Lab 2, M3)."""

import json
import pickle
import struct
import sys
from array import array
from pathlib import Path
from typing import Literal, cast

from findex.index import DocMeta, InvertedIndex, Posting

StorageFormat = Literal["binary", "pickle"]

BINARY_MAGIC = b"FIDX0001"


def _save_pickle(index: InvertedIndex, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as fh:
        pickle.dump(index, fh, protocol=pickle.HIGHEST_PROTOCOL)


def _load_pickle(path: Path) -> InvertedIndex:
    # SECURITY WARNING: pickle.load executes arbitrary Python callables embedded
    # via __reduce__ during unpickling (Remote Code Execution vulnerability).
    # Only ever load pickle index files that you generated locally yourself —
    # NEVER unpickle files downloaded from the internet or received over a network.
    with path.open("rb") as fh:
        obj: object = pickle.load(fh)
    if not isinstance(obj, InvertedIndex):
        raise TypeError(f"Expected InvertedIndex in {path}, got {type(obj).__name__}")
    return obj


def _save_binary(index: InvertedIndex, path: Path) -> None:
    """Save index as a JSON metadata header followed by contiguous uint32 arrays."""
    path.parent.mkdir(parents=True, exist_ok=True)

    terms: list[str] = []
    counts: list[int] = []
    doc_ids: array[int] = array("I")
    tfs: array[int] = array("I")
    positions_flat: array[int] = array("I")

    for term, plist in index.postings.items():
        terms.append(term)
        counts.append(len(plist))
        for p in plist:
            doc_ids.append(p.doc_id)
            tfs.append(p.tf)
            if index.has_positions:
                positions_flat.extend(p.positions)

    docs_table = [
        [meta.doc_id, index.doc_lengths[meta.doc_id], meta.title, meta.path]
        for meta in index.doc_meta.values()
    ]

    header_dict: dict[str, object] = {
        "has_positions": index.has_positions,
        "total_postings": len(doc_ids),
        "total_positions": len(positions_flat),
        "terms": terms,
        "counts": counts,
        "docs": docs_table,
    }
    header_bytes = json.dumps(header_dict, ensure_ascii=False, separators=(",", ":")).encode(
        "utf-8"
    )

    if sys.byteorder != "little":
        doc_ids.byteswap()
        tfs.byteswap()
        positions_flat.byteswap()

    with path.open("wb") as fh:
        fh.write(BINARY_MAGIC)
        fh.write(struct.pack("<Q", len(header_bytes)))
        fh.write(header_bytes)
        fh.write(doc_ids.tobytes())
        fh.write(tfs.tobytes())
        if index.has_positions:
            fh.write(positions_flat.tobytes())


def _load_binary(path: Path) -> InvertedIndex:
    """Load index safely from the compact JSON-header + uint32 array format."""
    with path.open("rb") as fh:
        magic = fh.read(len(BINARY_MAGIC))
        if magic != BINARY_MAGIC:
            raise ValueError(f"Invalid binary index header in {path}")

        (header_len,) = struct.unpack("<Q", fh.read(8))
        raw_header: object = json.loads(fh.read(header_len).decode("utf-8"))
        if not isinstance(raw_header, dict):
            raise ValueError(f"Corrupted index header in {path}")
        header = cast("dict[str, object]", raw_header)

        has_positions = bool(header["has_positions"])
        total_postings = cast("int", header["total_postings"])
        total_positions = cast("int", header["total_positions"])
        terms = cast("list[str]", header["terms"])
        counts = cast("list[int]", header["counts"])
        docs_table = cast("list[list[object]]", header["docs"])

        doc_ids: array[int] = array("I")
        doc_ids.frombytes(fh.read(total_postings * 4))

        tfs: array[int] = array("I")
        tfs.frombytes(fh.read(total_postings * 4))

        positions_flat: array[int] = array("I")
        if has_positions:
            positions_flat.frombytes(fh.read(total_positions * 4))

    if sys.byteorder != "little":
        doc_ids.byteswap()
        tfs.byteswap()
        positions_flat.byteswap()

    postings: dict[str, list[Posting]] = {}
    idx = 0
    pos_idx = 0
    for term, count in zip(terms, counts, strict=True):
        plist: list[Posting] = []
        end = idx + count
        if has_positions:
            for i in range(idx, end):
                tf = tfs[i]
                pos_end = pos_idx + tf
                plist.append(
                    Posting(
                        doc_id=doc_ids[i],
                        tf=tf,
                        positions=tuple(positions_flat[pos_idx:pos_end]),
                    )
                )
                pos_idx = pos_end
        else:
            for i in range(idx, end):
                plist.append(Posting(doc_id=doc_ids[i], tf=tfs[i]))
        postings[term] = plist
        idx = end

    doc_lengths: dict[int, int] = {}
    doc_meta: dict[int, DocMeta] = {}
    for row in docs_table:
        doc_id = cast("int", row[0])
        doc_len = cast("int", row[1])
        title = cast("str", row[2])
        doc_path = cast("str", row[3])
        doc_lengths[doc_id] = doc_len
        doc_meta[doc_id] = DocMeta(doc_id=doc_id, title=title, path=doc_path)

    return InvertedIndex(
        postings=postings,
        doc_lengths=doc_lengths,
        doc_meta=doc_meta,
        has_positions=has_positions,
    )


def save(index: InvertedIndex, path: Path, fmt: StorageFormat = "binary") -> None:
    """Persist an InvertedIndex to disk in the requested format."""
    if fmt == "binary":
        _save_binary(index, path)
    else:
        _save_pickle(index, path)


def load(path: Path, fmt: StorageFormat | None = None) -> InvertedIndex:
    """Load an InvertedIndex from disk, auto-detecting format if fmt is None."""
    if fmt == "binary":
        return _load_binary(path)
    if fmt == "pickle":
        return _load_pickle(path)

    with path.open("rb") as fh:
        prefix = fh.read(len(BINARY_MAGIC))
    if prefix == BINARY_MAGIC:
        return _load_binary(path)
    return _load_pickle(path)
