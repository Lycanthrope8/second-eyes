"""Check records against the scene contract, v1 (A2.1a, D66), with scene format v2 (A2.2a, D74): scenes, command contexts
and category maps.

    python grounding/contract/validate.py FILE.json [FILE.json ...]

Each file holds one record. The checks run in layers: strict parsing (UTF-8 without a byte-order mark, no repeated
keys, no NaN or Infinity, no number too large for a double); the record's JSON Schema in schemas/, chosen by its
record_type and schema_version; the semantic rules in checks.py; and, for records given together, the cross-record
rules: a command against its scene, a scene against its category map, and no two records with the same identity.
A layer runs only if the one before it passed. Prints one line per error or warning and exits 1 if there is any
error. Never writes. The contract: docs/scene-contract.md.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import jsonschema

sys.path.insert(0, str(Path(__file__).resolve().parent))
import checks  # noqa: E402  (checks.py sits next to this file)

from checks import Issue  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
SCHEMAS = {
    ("scene", 1): "scene.v1.json",
    ("scene", 2): "scene.v2.json",
    ("command_context", 1): "command-context.v1.json",
    ("category_map", 1): "category-map.v1.json",
}
KEYWORD_CODES = {
    "required": "E_SCHEMA_REQUIRED",
    "additionalProperties": "E_SCHEMA_ADDITIONAL",
    "type": "E_SCHEMA_TYPE",
    "enum": "E_SCHEMA_ENUM",
    "const": "E_SCHEMA_CONST",
    "pattern": "E_SCHEMA_PATTERN",
    "minItems": "E_SCHEMA_LENGTH", "maxItems": "E_SCHEMA_LENGTH", "minLength": "E_SCHEMA_LENGTH",
    "minimum": "E_SCHEMA_RANGE", "maximum": "E_SCHEMA_RANGE",
    "exclusiveMinimum": "E_SCHEMA_RANGE", "exclusiveMaximum": "E_SCHEMA_RANGE",
    "uniqueItems": "E_SCHEMA_UNIQUE",
}
_validators = {}


class _ParseError(Exception):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


def _no_constant(name: str):
    raise _ParseError("E_NONFINITE", f"{name} is not a JSON number")


def _finite_float(text: str) -> float:
    value = float(text)
    if not math.isfinite(value):
        raise _ParseError("E_NONFINITE", f"{text} is too large for a double")
    return value


def _no_repeats(pairs):
    out = {}
    for key, value in pairs:
        if key in out:
            raise _ParseError("E_PARSE_DUPLICATE_KEY", f"key {key!r} appears twice in one object")
        out[key] = value
    return out


def parse(name: str, data: bytes):
    """Strictly parse one file's bytes. Returns (record, issues); parsing failed if issues isn't empty. A file
    holding JSON null parses to None, which is a record value like any other and fails the record-type check."""
    if data.startswith(b"\xef\xbb\xbf"):
        return None, [Issue(name, "$", "E_PARSE_ENCODING", "starts with a UTF-8 byte-order mark")]
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as e:
        return None, [Issue(name, "$", "E_PARSE_ENCODING", f"not UTF-8 at byte {e.start}")]
    try:
        record = json.loads(text, object_pairs_hook=_no_repeats, parse_constant=_no_constant,
                            parse_float=_finite_float)
    except _ParseError as e:
        return None, [Issue(name, "$", e.code, str(e))]
    except json.JSONDecodeError as e:
        return None, [Issue(name, "$", "E_PARSE_JSON", f"{e.msg} at line {e.lineno}, column {e.colno}")]
    return record, []


def _validator(schema_file: str):
    if schema_file not in _validators:
        schema = json.loads((REPO / "schemas" / schema_file).read_text(encoding="utf-8"))
        cls = jsonschema.validators.validator_for(schema)
        cls.check_schema(schema)
        _validators[schema_file] = cls(schema)
    return _validators[schema_file]


def _json_path(parts) -> str:
    return "$" + "".join(f"[{p}]" if isinstance(p, int) else f".{p}" for p in parts)


def check_record(name: str, record) -> list:
    """Record type, schema and semantic rules for one parsed record."""
    if not isinstance(record, dict):
        return [Issue(name, "$", "E_RECORD_TYPE", "the file's value is not a JSON object")]
    record_type, version = record.get("record_type"), record.get("schema_version")
    # Check the types before the lookup: a list or object can't be part of the key (bool is not an int here).
    if not isinstance(record_type, str) or type(version) is not int or (record_type, version) not in SCHEMAS:
        return [Issue(name, "$", "E_RECORD_TYPE", f"record_type {record_type!r} with schema_version {version!r} "
                                                  f"is not a supported format")]
    schema_file = SCHEMAS[(record_type, version)]
    errors = sorted(_validator(schema_file).iter_errors(record), key=lambda e: [str(p) for p in e.absolute_path])
    if errors:
        return [Issue(name, _json_path(e.absolute_path), KEYWORD_CODES.get(e.validator, "E_SCHEMA_OTHER"), e.message)
                for e in errors]
    return checks.check_record(name, record)


def validate(named_records: list) -> list:
    """All layers after parsing, for a list of (name, record); returns every issue, errors and warnings."""
    issues, passed = [], []
    for name, record in named_records:
        found = check_record(name, record)
        issues.extend(found)
        if not found:
            passed.append((name, record))
    return issues + checks.check_cross(passed)


def validate_paths(paths: list) -> list:
    """Read, parse and validate files; records that fail to parse are left out of the later layers."""
    issues, parsed = [], []
    for path in paths:
        name = str(path)
        record, found = parse(name, Path(path).read_bytes())
        issues.extend(found)
        if not found:
            parsed.append((name, record))
    return issues + validate(parsed)


def main(argv: list) -> int:
    if not argv:
        print(__doc__.strip().splitlines()[2].strip(), file=sys.stderr)
        return 2
    issues = validate_paths(argv)
    for issue in issues:
        print(issue.line())
    errors = sum(1 for i in issues if i.is_error)
    warnings = len(issues) - errors
    bad = len({i.file for i in issues if i.is_error})
    print(f"{len(argv)} file(s): {len(argv) - bad} passed, {bad} with errors; {errors} error(s), {warnings} warning(s)")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
