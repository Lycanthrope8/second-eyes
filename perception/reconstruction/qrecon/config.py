"""Configuration: every tunable value lives in YAML, nothing in code."""
import hashlib
import json
import yaml


def load(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def digest(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()


def file_sha256(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()
