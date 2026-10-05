"""Strict reading of the pinned IRef-VLA source formats (A2.2a, D74).

Nothing here guesses: a header must match exactly, a row must have one of the allowed widths, a number must be written
as a finite decimal (Python's float() would also take 'nan', 'inf', '1_000' and padded whitespace), a label must
already satisfy the contract's whole-string label rule (D73), and JSON goes through the contract's strict parser
(UTF-8 without a byte-order mark, no repeated keys, no NaN or Infinity). Problems are collected as issues
{file, path, code, message} and raised together as AdapterInputError.
"""
from __future__ import annotations

import csv
import io
import re

from ...contract import validate as contract

NUMBER = re.compile(r"-?(?:0|[1-9][0-9]*)(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?")
DECIMAL_ID = re.compile(r"0|[1-9][0-9]*")
LABEL = re.compile(r"\S(?:.*\S)?")  # the contract's label rule as a whole-string match (D73)


class AdapterInputError(ValueError):
    """Invalid, unsupported or mismatched input, or an existing output folder (CLI exit 2)."""

    def __init__(self, issues):
        self.issues = [dict(i) for i in issues]
        super().__init__("; ".join(f"{i['file']}: {i['path']}: {i['code']}: {i['message']}" for i in self.issues))


class AdapterOutputError(OSError):
    """The output could not be written (CLI exit 3); no output bundle was published."""

    def __init__(self, issues):
        self.issues = [dict(i) for i in issues]
        super().__init__("; ".join(f"{i['file']}: {i['path']}: {i['code']}: {i['message']}" for i in self.issues))


def issue(file, path, code, message):
    return {"file": file, "path": path, "code": code, "message": message}


class Issues:
    """Collects issues for one stage; raise_if_any() stops before a later stage relies on bad data."""

    def __init__(self):
        self.items = []

    def add(self, file, path, code, message):
        self.items.append(issue(file, path, code, message))

    def raise_if_any(self):
        if self.items:
            raise AdapterInputError(self.items)


def read_csv(file, data: bytes, header, widths, issues: Issues):
    """Rows (lists of strings) after the exact header, or None after recording why the shape is wrong."""
    if data.startswith(b"\xef\xbb\xbf"):
        issues.add(file, "byte 0", "E_IREF_SOURCE_SHAPE", "the file starts with a UTF-8 byte-order mark")
        return None
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as e:
        issues.add(file, f"byte {e.start}", "E_IREF_SOURCE_SHAPE", "the file is not valid UTF-8")
        return None
    rows = list(csv.reader(io.StringIO(text, newline="")))
    if not rows:
        issues.add(file, "row 1", "E_IREF_SOURCE_SHAPE", "the file is empty")
        return None
    got = rows[0]
    repeated = sorted({h for h in got if got.count(h) > 1})
    if repeated:
        issues.add(file, "row 1", "E_IREF_SOURCE_SHAPE", f"repeated header names: {', '.join(repeated)}")
        return None
    if got != list(header):
        issues.add(file, "row 1", "E_IREF_SOURCE_SHAPE", f"the header is not the pinned source's {len(header)} columns: "
                                                         f"expected {list(header)!r}, found {got!r}")
        return None
    body = rows[1:]
    for n, row in enumerate(body, start=2):
        if len(row) not in widths:
            issues.add(file, f"row {n}", "E_IREF_SOURCE_SHAPE",
                       f"{len(row)} fields; this source allows {', '.join(map(str, sorted(widths)))}")
    return body if not issues.items else None


def read_json(file, data: bytes, issues: Issues):
    """The parsed JSON value, or None after recording the contract parser's findings as source-shape issues."""
    value, found = contract.parse(file, data)
    for i in found:
        issues.add(file, i.path, "E_IREF_SOURCE_SHAPE", f"{i.code}: {i.message}")
    return None if found else value


def number(text, file, where, issues: Issues):
    if not NUMBER.fullmatch(text):
        issues.add(file, where, "E_IREF_SOURCE_VALUE", f"{text!r} is not a finite decimal number")
        return None
    value = float(text)
    if value != value or value in (float("inf"), float("-inf")):
        issues.add(file, where, "E_IREF_SOURCE_VALUE", f"{text!r} is too large for a double")
        return None
    return value


def decimal_id(text, file, where, issues: Issues):
    if not DECIMAL_ID.fullmatch(text):
        issues.add(file, where, "E_IREF_SOURCE_VALUE",
                   f"{text!r} is not a canonical nonnegative decimal ID (0, or a nonzero digit then digits)")
        return None
    return text


def label(text, file, where, issues: Issues):
    if not LABEL.fullmatch(text):
        issues.add(file, where, "E_IREF_SOURCE_VALUE", f"{text!r} is not a valid label: empty, or with whitespace at "
                                                       "an end or a line break inside (not trimmed or rewritten)")
        return None
    return text
