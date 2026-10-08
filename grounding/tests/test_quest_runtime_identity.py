"""A2.5 delivery 1, step 2, part 1 (D99(7)): the ELF and GGUF readers and the runtime identity check.

Run directly (python grounding/tests/test_quest_runtime_identity.py) or as a module. The ARM64 libraries and GGUF files
are built byte by byte here from the formats' definitions; a labelled fake adb serves the headset's files and records
every command. Expectations are written from those definitions, never read from the code under test.
"""
from __future__ import annotations

import hashlib
import importlib
import json
import re
import struct
import sys
import tempfile
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
FAILS, PASSES = [], []
HEADER = (REPO / "native" / "se_llama.h").read_text(encoding="utf-8")
REQUIRED = [m.group(1) for m in (re.match(r"^SE_API\b.*?\b(\w+)\s*\(", x) for x in HEADER.splitlines()) if m]
VERSION_STRING = b"b11277 (eae11d22)"


def check(name, ok, detail=""):
    (PASSES if ok else FAILS).append(name)
    print(("  ok    " if ok else "  FAIL  ") + name + ("" if ok or not detail else f"  [{detail}]"))


def hb(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


# ------------------------------------------------------------------------------------------------ hand-built ELF

GLOBAL, WEAK, LOCAL = 1, 2, 0
FUNC, OBJECT, NOTYPE = 2, 1, 0
DEFAULT, HIDDEN, PROTECTED = 0, 2, 3


def sym(name, bind=GLOBAL, typ=FUNC, vis=DEFAULT, defined=True):
    return (name, bind, typ, vis, defined)


def build_elf(symbols, machine=183, etype=3, ei_class=2, ei_data=1, symtab_shift=0, no_sections=False, extra=VERSION_STRING):
    """A minimal ELF64 shared object: ELF header, one PT_DYNAMIC segment, and the sections null, .dynsym, .dynstr,
    .dynamic and .shstrtab. Addresses equal file offsets. `extra` is appended as read-only data."""
    def align(b, n=8):
        return b + b"\0" * (-len(b) % n)
    dynstr = b"\0"
    offsets = []
    for name, *_ in symbols:
        offsets.append(len(dynstr))
        dynstr += name.encode() + b"\0"
    body = bytearray(b"\0" * 64 + b"\0" * 56)          # ELF header and one program header, filled in below
    dynstr_off = len(body)
    body += align(dynstr)
    dynsym_off = len(body)
    body += b"\0" * 24                                   # the null symbol
    for (name, bind, typ, vis, defined), off in zip(symbols, offsets):
        body += struct.pack("<IBBHQQ", off, (bind << 4) | typ, vis, 3 if defined else 0, 0x1000 + 16 * len(body), 16)
    dynsym_size = 24 * (len(symbols) + 1)
    dyn_off = len(body)
    dyn = [(6, dynsym_off + symtab_shift), (5, dynstr_off), (10, len(dynstr)), (11, 24), (0, 0)]
    body += b"".join(struct.pack("<qQ", t, v) for t, v in dyn)
    dyn_size = 16 * len(dyn)
    shstr = b"\0.dynsym\0.dynstr\0.dynamic\0.shstrtab\0"
    shstr_off = len(body)
    body += align(shstr)
    body += align(extra)
    sh_off = len(body)
    names = {".dynsym": 1, ".dynstr": 9, ".dynamic": 17, ".shstrtab": 26}
    sections = [b"\0" * 64,
                struct.pack("<IIQQQQIIQQ", names[".dynsym"], 11, 2, dynsym_off, dynsym_off, dynsym_size, 2, 1, 8, 24),
                struct.pack("<IIQQQQIIQQ", names[".dynstr"], 3, 2, dynstr_off, dynstr_off, len(dynstr), 0, 0, 1, 0),
                struct.pack("<IIQQQQIIQQ", names[".dynamic"], 6, 3, dyn_off, dyn_off, dyn_size, 2, 0, 8, 16),
                struct.pack("<IIQQQQIIQQ", names[".shstrtab"], 3, 0, 0, shstr_off, len(shstr), 0, 0, 1, 0)]
    body += b"".join(sections)
    ident = b"\x7fELF" + bytes([ei_class, ei_data, 1, 0]) + b"\0" * 8
    hdr = ident + struct.pack("<HHIQQQIHHHHHH", etype, machine, 1, 0, 64, 0 if no_sections else sh_off, 0, 64, 56, 1, 64,
                              len(sections), 4)
    body[0:64] = hdr
    body[64:120] = struct.pack("<IIQQQQQQ", 2, 6, dyn_off, dyn_off, dyn_off, dyn_size, dyn_size, 8)
    return bytes(body)


def good_symbols():
    return [sym(n) for n in REQUIRED] + [sym("se_extra_tool"), sym("malloc", typ=NOTYPE, defined=False),
                                          sym("llama_internal", vis=HIDDEN)]


def check_elf(B):
    print("-- the ELF reader (hand-built ARM64 shared objects)")
    elf = B.elf_dynamic_symbols(build_elf(good_symbols()))
    check("an ELF64 little-endian ARM64 shared object: machine 183, type 3",
          elf["format"] == {"class": "ELF64", "endianness": "little", "machine": 183, "machine_is_aarch64": True, "type": 3,
                            "type_is_shared_object": True})
    s = elf["symbols"]
    check("each symbol keeps its binding, type, visibility and definedness",
          s["se_eval"] == {"bind": "GLOBAL", "type": "FUNC", "visibility": "DEFAULT", "defined": True,
                           "value": s["se_eval"]["value"], "size": 16}
          and s["malloc"]["defined"] is False and s["llama_internal"]["visibility"] == "HIDDEN")
    r = B.usable_exports(s, REQUIRED)
    check(f"all {len(REQUIRED)} header functions are usable; extra exports and imports change nothing",
          all(r[n]["usable"] for n in REQUIRED) and len(REQUIRED) == 23)
    cases = [("missing", lambda xs: [x for x in xs if x[0] != "se_logits"], "se_logits", "not in the dynamic symbol table"),
             ("undefined", lambda xs: [sym(x[0], defined=False) if x[0] == "se_logprob" else x for x in xs], "se_logprob",
              "undefined"),
             ("hidden", lambda xs: [sym(x[0], vis=HIDDEN) if x[0] == "se_eval" else x for x in xs], "se_eval", "visibility HIDDEN"),
             ("local", lambda xs: [sym(x[0], bind=LOCAL) if x[0] == "se_n_ctx" else x for x in xs], "se_n_ctx", "binding LOCAL"),
             ("an object", lambda xs: [sym(x[0], typ=OBJECT) if x[0] == "se_n_vocab" else x for x in xs], "se_n_vocab",
              "type OBJECT")]
    for label, edit, name, reason in cases:
        r = B.usable_exports(B.elf_dynamic_symbols(build_elf(edit(good_symbols())))["symbols"], REQUIRED)
        bad = [n for n in REQUIRED if not r[n]["usable"]]
        check(f"{label}: exactly {name} is reported, with its reason", bad == [name] and any(reason in x for x in r[name]["reasons"]),
              str(r[name]))
    r = B.usable_exports(B.elf_dynamic_symbols(build_elf([sym(n, bind=WEAK if n == "se_score" else GLOBAL,
                                                                vis=PROTECTED if n == "se_piece" else DEFAULT)
                                                            for n in REQUIRED]))["symbols"], REQUIRED)
    check("weak binding and protected visibility are still usable exports", all(r[n]["usable"] for n in REQUIRED))
    f = B.elf_dynamic_symbols(build_elf(good_symbols(), machine=62, etype=2))["format"]
    check("another machine and another file type are read and flagged (x86-64, executable)",
          (f["machine_is_aarch64"], f["type_is_shared_object"]) == (False, False))
    for label, data in (("not an ELF file", b"MZ" + b"\0" * 200),
                        ("ELF32", build_elf(good_symbols(), ei_class=1)),
                        ("big-endian", build_elf(good_symbols(), ei_data=2)),
                        ("truncated", build_elf(good_symbols())[:300]),
                        ("no section header table", build_elf(good_symbols(), no_sections=True)),
                        ("a symbol table the dynamic segment does not name", build_elf(good_symbols(), symtab_shift=24))):
        try:
            B.elf_dynamic_symbols(data)
            check(f"unsupported: {label}", False)
        except B.Unsupported as e:
            check(f"unsupported: {label} ({e})", True)


# ------------------------------------------------------------------------------------------------ hand-built GGUF

def gstr(s):
    b = s.encode()
    return struct.pack("<Q", len(b)) + b


def build_gguf(path, version=3, file_type=7, magic=b"GGUF", tensors=((8, 3), (0, 2)), nested=False, cut=None):
    kvs = [gstr("general.architecture") + struct.pack("<I", 8) + gstr("qwen2"),
           gstr("general.name") + struct.pack("<I", 8) + gstr("fixture model"),
           gstr("tokenizer.ggml.tokens") + struct.pack("<IIQ", 9, 8, 5) + b"".join(gstr(t) for t in "abcde"),
           gstr("qwen2.rope.freq_base") + struct.pack("<If", 6, 1000000.0),
           gstr("tokenizer.ggml.add_bos_token") + struct.pack("<I?", 7, False),
           gstr("general.quantization_version") + struct.pack("<II", 4, 2)]
    if file_type is not None:
        kvs.append(gstr("general.file_type") + struct.pack("<II", 4, file_type))
    if nested:
        kvs.append(gstr("bad.nested") + struct.pack("<IIQ", 9, 9, 1) + struct.pack("<IQ", 4, 0))
    infos, k = b"", 0
    for ggml_type, count in tensors:
        for _ in range(count):
            infos += gstr(f"blk.{k}.weight") + struct.pack("<I", 2) + struct.pack("<QQ", 32, 4) + struct.pack("<IQ", ggml_type, 0)
            k += 1
    data = magic + struct.pack("<IQQ", version, k, len(kvs)) + b"".join(kvs) + infos + b"\0" * 64
    Path(path).write_bytes(data[:cut] if cut else data)
    return data[:cut] if cut else data


def check_gguf(B):
    print("-- the GGUF reader (hand-built files)")
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "m.gguf"
        build_gguf(p)
        g = B.gguf_header(p)
        check("version 3, five tensors, seven metadata entries, qwen2",
              (g["version"], g["tensor_count"], g["metadata_count"], g["architecture"]) == (3, 5, 7, "qwen2"))
        check("general.file_type 7 is MOSTLY_Q8_0; tensor types counted: three Q8_0, two F32",
              (g["file_type"], g["file_type_name"], g["tensor_types"]) == (7, "MOSTLY_Q8_0", {"F32": 2, "Q8_0": 3}))
        build_gguf(p, file_type=None)
        check("no general.file_type: reported as None, not guessed", B.gguf_header(p)["file_type"] is None)
        for label, kw in (("not a GGUF file", {"magic": b"GGML"}), ("GGUF version 1", {"version": 1}),
                          ("truncated", {"cut": 120}), ("a nested array", {"nested": True})):
            build_gguf(p, **kw)
            try:
                B.gguf_header(p)
                check(f"unsupported: {label}", False)
            except B.Unsupported as e:
                check(f"unsupported: {label} ({e})", True)


# ------------------------------------------------------------------------------------------------ the fake headset

PKG = "com.secondeyes.quest"
APP_DIR = "/sdcard/Android/data/com.secondeyes.quest/files"
BASE = "/data/app/~~abc/com.secondeyes.quest-xyz/base.apk"
ALLOWED = {("devices",), ("version",), ("shell", "getprop"), ("shell", "pm"), ("shell", "dumpsys"), ("shell", "ls"), ("pull",)}


class FakeAdb:
    """Labelled test double for adb: serves files and properties, and records every command it is given."""

    def __init__(self, files, devices=("SERIAL1",), apks=(BASE,), raise_on=None):
        self.files, self.devices, self.apks, self.raise_on, self.calls = dict(files), list(devices), list(apks), raise_on, []

    def run(self, *args):
        self.calls.append(args)
        if self.raise_on and args[:2] == self.raise_on:
            raise RuntimeError("simulated failure")
        if args == ("devices",):
            return 0, "List of devices attached\n" + "".join(f"{d}\tdevice\n" for d in self.devices) + "\n", ""
        if args == ("version",):
            return 0, "Android Debug Bridge version 1.0.41 (fake)\n", ""
        if args[0] == "pull":
            if args[1] in self.files:
                Path(args[2]).write_bytes(self.files[args[1]])
                return 0, "1 file pulled\n", ""
            return 1, "", f"adb: error: failed to stat remote object '{args[1]}': No such file or directory\n"
        if args[:2] == ("shell", "getprop"):
            return 0, {"ro.product.model": "Quest 3", "ro.product.manufacturer": "Oculus"}.get(args[2], "fixture-value") + "\n", ""
        if args[:3] == ("shell", "pm", "path"):
            return (0, "".join(f"package:{a}\n" for a in self.apks), "") if self.apks else (1, "", "")
        if args[:3] == ("shell", "dumpsys", "package"):
            return 0, "Packages:\n  versionCode=42 minSdk=32 targetSdk=32\n  versionName=0.1.0\n" \
                      "  firstInstallTime=2026-10-01 10:00:00\n  lastUpdateTime=2026-10-06 18:00:00\n", ""
        if args[:3] == ("shell", "ls", "-l"):
            return (0, f"-rw-r--r-- 1 system system 1 {args[3]}\n", "") if args[3] in self.shell_files() else \
                (1, "", f"ls: {args[3]}: No such file or directory\n")
        return 1, "", "unsupported in the fake"

    def shell_files(self):
        return self.files

    def shell(self, *args):
        return self.run("shell", *args)

    def pull(self, remote, local):
        return self.run("pull", remote, str(local))


def apk(entries):
    import io
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("AndroidManifest.xml", b"fixture")
        for name, data in entries.items():
            z.writestr(name, data)
    return buf.getvalue()


def build_record(lib_hash, llama="b11277 (eae11d22)"):
    return json.dumps({"target": "android", "ndk": {"revision": "27.2.12479018"}, "arm_arch": "armv8.2-a+dotprod+fp16",
                       "files": {"libse_llama.so": {"bytes": 1, "sha256": lib_hash}}, "llama_cpp": llama}).encode()


class World:
    """A fixture repository (header, build record, Unity copy, provenance runs) and a fake headset."""

    def __init__(self, tmp, lib=None, unity=None, record=True, llama="b11277 (eae11d22)", provenance=("r_match",),
                 gguf_kw=None, model_on_device=True, apks=None, extracted=None):
        self.tmp = Path(tmp)
        self.repo = self.tmp / "repo"
        (self.repo / "native" / "out" / "android-arm64").mkdir(parents=True)
        (self.repo / "native" / "se_llama.h").write_text(HEADER, encoding="utf-8")
        self.lib = lib if lib is not None else build_elf(good_symbols())
        if record:
            (self.repo / "native" / "out" / "android-arm64" / "build.json").write_bytes(build_record(hb(self.lib), llama))
        u = self.repo / "quest-app" / "Assets" / "Plugins" / "Android" / "libs" / "arm64-v8a"
        u.mkdir(parents=True)
        (u / "libse_llama.so").write_bytes(unity if unity is not None else self.lib)
        for run in provenance:
            (self.repo / "runs" / run / "raw").mkdir(parents=True)
            h = hb(self.lib) if run == "r_match" else "0" * 64
            (self.repo / "runs" / run / "raw" / "build.json").write_bytes(build_record(h))
        self.gguf = build_gguf(self.tmp / "m.gguf", **(gguf_kw or {}))
        pol = json.loads((REPO / "grounding" / "quest" / "runtime-identity.v1.json").read_text(encoding="utf-8"))
        pol["accepted_model_sha256"] = hb(build_gguf(self.tmp / "accepted.gguf"))
        pol["provenance_runs"] = ["r_match", "r_other", "r_absent"]
        self.policy = self.tmp / "policy.json"
        self.policy.write_text(json.dumps(pol), encoding="utf-8")
        files = {}
        for path, data in (apks if apks is not None else {BASE: apk({"lib/arm64-v8a/libse_llama.so": self.lib})}).items():
            files[path] = data
        if model_on_device:
            files[f"{APP_DIR}/qwen2.5-0.5b-instruct-q8_0.gguf"] = self.gguf
        if extracted is not None:
            files["/data/app/~~abc/com.secondeyes.quest-xyz/lib/arm64/libse_llama.so"] = extracted
        self.files = files
        self.apk_paths = list(apks) if apks is not None else [BASE]

    def run(self, RI, name="out", adb=None, **kw):
        adb = adb if adb is not None else FakeAdb(self.files, apks=self.apk_paths, **kw)
        rec = RI.check_runtime_identity(out=self.tmp / name, adb=adb, repo=self.repo, policy=self.policy, progress=lambda m: None)
        return rec, adb, self.tmp / name


def st(rec, k):
    return rec["checks"][k]["status"]


def check_identity(RI):
    print("-- the runtime identity check (fixture repository, labelled fake adb)")
    with tempfile.TemporaryDirectory() as tmp:
        w = World(tmp)
        rec, adb, out = w.run(RI)
        check("everything agrees: current deployed identity verified, exit 0",
              rec["status"]["current_identity"] == "verified" and rec["status"]["exit_code"] == 0
              and all(st(rec, k) in ("pass", "not_applicable") for k in RI.CURRENT), str(rec["status"]))
        check("continuity linked to the one A1.8c-era record that names the deployed hash; the others listed as they are",
              rec["continuity"]["status"] == "linked" and [r["run"] for r in rec["continuity"]["records"] if r.get("matches_deployed")]
              == ["r_match"] and [r["found"] for r in rec["continuity"]["records"]] == [True, False, False])
        check("the summary keeps the two verdicts apart",
              rec["status"]["line"] == "current deployed identity verified; A1.8c binary continuity linked")
        files = sorted(p.relative_to(out).as_posix() for p in out.rglob("*") if p.is_file())
        check("evidence kept: record, report, build record, deployed library, provenance record, manifest; the accepted model "
              "is not copied", files == ["build.json", "identity.json", "libse_llama.so", "manifest.json",
                                         "provenance/r_match-build.json", "report.md"], str(files))
        man = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
        check("the manifest hashes every other file", all(hb((out / k).read_bytes()) == v for k, v in man["files"].items())
              and sorted(man["files"]) == [f for f in files if f != "manifest.json"])
        ev = rec["checks"]
        check("full hashes recorded: build record, Unity copy, deployed library and model",
              ev["build_record"]["evidence"]["libse_llama_sha256"] == ev["unity_copy"]["evidence"]["sha256"]
              == ev["apk_library"]["evidence"]["sha256"] == hb(w.lib)
              and ev["model_hash"]["evidence"]["sha256"] == hb(w.gguf) and len(ev["model_hash"]["evidence"]["sha256"]) == 64)
        check("device and package identity recorded: serial, model, version, APK path",
              ev["device"]["evidence"]["serial"] == "SERIAL1" and ev["device"]["evidence"]["properties"]["ro.product.model"]
              == "Quest 3" and ev["package"]["evidence"]["versionCode"] == "42" and ev["package"]["evidence"]["apk_paths"] == [BASE])
        check("quantization read from the GGUF: file type 7, MOSTLY_Q8_0",
              ev["model_quantization"]["evidence"]["gguf"]["file_type_name"] == "MOSTLY_Q8_0")
        check(f"all {len(REQUIRED)} header functions usable; the version string found as supporting evidence only",
              ev["exports"]["evidence"]["unusable"] == [] and ev["exports"]["evidence"]["supporting_string_found"] is True)
        check("not extracted at install: not applicable, with the path that was looked at",
              st(rec, "installed_library") == "not_applicable")
        check("read-only: every adb command was devices, version, getprop, pm path, dumpsys, ls or pull",
              all(c[:1] in ALLOWED or c[:2] in ALLOWED for c in adb.calls), str([c for c in adb.calls][:3]))

        scenarios = [
            ("no A1.8c-era record here: continuity unverified, still exit 0", dict(provenance=()),
             lambda r: r["status"]["line"] == "current deployed identity verified; A1.8c binary continuity unverified"
             and r["status"]["exit_code"] == 0),
            ("records here, none matching: continuity contradicted, exit 1", dict(provenance=("r_other",)),
             lambda r: r["continuity"]["status"] == "contradicted" and r["status"]["exit_code"] == 1),
            ("the installed library differs from the build record and the Unity copy: failed, exit 1",
             dict(apks={BASE: apk({"lib/arm64-v8a/libse_llama.so": build_elf(good_symbols(), extra=b"other build")})}),
             lambda r: st(r, "apk_library") == "fail" and r["status"]["current_identity"] == "failed" and r["status"]["exit_code"] == 1),
            ("the Unity copy differs: both the Unity check and the APK check fail", dict(unity=b"another file"),
             lambda r: st(r, "unity_copy") == "fail" and st(r, "apk_library") == "fail"),
            ("no build record: incomplete (never rebuilt), exit 2, other evidence still kept", dict(record=False),
             lambda r: st(r, "build_record") == "incomplete" and st(r, "unity_copy") == "incomplete"
             and st(r, "apk_library") == "incomplete" and r["status"]["exit_code"] == 2),
            ("a build record of another llama.cpp release: failed", dict(llama="b9999 (deadbeef)"),
             lambda r: st(r, "build_record") == "fail"),
            ("a model whose hash differs: failed, and the pulled file is kept as evidence", dict(gguf_kw={"file_type": 1}),
             lambda r: st(r, "model_hash") == "fail" and st(r, "model_quantization") == "fail"),
            ("no model on the headset: both model checks incomplete, exit 2", dict(model_on_device=False),
             lambda r: (st(r, "model_hash"), st(r, "model_quantization")) == ("incomplete", "incomplete") and r["status"]["exit_code"] == 2),
            ("the app's APKs hold no arm64 library: failed", dict(apks={BASE: apk({})}),
             lambda r: st(r, "apk_library") == "fail" and st(r, "exports") == "incomplete"),
            ("split APKs: the library found in exactly one of them passes",
             dict(apks={BASE: apk({}), BASE.replace("base.apk", "split_config.arm64_v8a.apk"):
                        apk({"lib/arm64-v8a/libse_llama.so": build_elf(good_symbols())})}),
             lambda r: st(r, "apk_library") == "pass" and len(r["checks"]["apk_library"]["evidence"]["apks"]) == 2),
            ("an extracted copy equal to the APK's passes", dict(extracted=build_elf(good_symbols())),
             lambda r: st(r, "installed_library") == "pass"),
            ("an extracted copy that differs fails", dict(extracted=b"tampered"),
             lambda r: st(r, "installed_library") == "fail"),
            ("a library missing se_logits: exports fail, naming it",
             dict(lib=build_elf([s for s in good_symbols() if s[0] != "se_logits"])),
             lambda r: st(r, "exports") == "fail" and r["checks"]["exports"]["evidence"]["unusable"] == ["se_logits"]),
            ("an ELF32 library: exports incomplete, never a pass", dict(lib=build_elf(good_symbols(), ei_class=1)),
             lambda r: st(r, "exports") == "incomplete" and r["status"]["exit_code"] == 2),
            ("an x86-64 library: exports fail", dict(lib=build_elf(good_symbols(), machine=62)),
             lambda r: st(r, "exports") == "fail"),
            ("no version string in the library: recorded, supporting only", dict(lib=build_elf(good_symbols(), extra=b"none")),
             lambda r: r["checks"]["exports"]["evidence"]["supporting_string_found"] is False and st(r, "exports") == "pass"),
        ]
        for k, (label, kw, ok) in enumerate(scenarios):
            w2 = World(Path(tmp) / f"s{k}", **kw)
            r, _, o = w2.run(RI)
            check(label, ok(r), json.dumps(r["status"]) + " " + str({c: st(r, c) for c in RI.CURRENT})[:300])
            if label.startswith("a model whose hash differs"):
                check("the differing model file is kept beside the record", (o / "qwen2.5-0.5b-instruct-q8_0.gguf").is_file())
            if label.startswith("the installed library differs"):
                check("the differing library is kept as evidence", hb((o / "libse_llama.so").read_bytes())
                      == r["checks"]["apk_library"]["evidence"]["sha256"])
        for label, kw, want in (("no device", {"devices": ()}, "incomplete"), ("two devices", {"devices": ("A", "B")}, "incomplete")):
            w3 = World(Path(tmp) / label.replace(" ", "_"))
            r, _, _ = w3.run(RI, **kw)
            check(f"{label}: device, package, library and model checks incomplete; local checks still recorded",
                  st(r, "device") == want and st(r, "apk_library") == "incomplete" and st(r, "model_hash") == "incomplete"
                  and st(r, "build_record") == "pass" and r["status"]["exit_code"] == 2)
        w4 = World(Path(tmp) / "boom")
        r, _, o = w4.run(RI, raise_on=("shell", "pm"))
        check("an unexpected error keeps the evidence gathered so far: published, the rest incomplete, exit 2",
              r["error"] == "RuntimeError: simulated failure" and st(r, "build_record") == "pass" and st(r, "package") == "incomplete"
              and "not reached" in r["checks"]["package"]["detail"] and (o / "identity.json").is_file() and r["status"]["exit_code"] == 2)
        EI = importlib.import_module("grounding.evaluation.iref_vla.protocol").EvaluationInputError
        try:
            w.run(RI)
            check("an existing destination is refused", False)
        except EI:
            check("an existing destination is refused", True)


def check_cli(RI):
    print("-- the command line (labelled fake adb)")
    CLI = importlib.import_module("grounding.quest.__main__")
    with tempfile.TemporaryDirectory() as tmp:
        w = World(tmp)
        real = RI.Adb
        RI.Adb = lambda binary="adb": FakeAdb(w.files, devices=())
        try:
            code = CLI.main(["runtime-identity", "--out", str(Path(tmp) / "cli")])
            check("without a device the command publishes its evidence and exits 2",
                  code == 2 and (Path(tmp) / "cli" / "identity.json").is_file())
            check("an existing destination is not run: exit 3", CLI.main(["runtime-identity", "--out", str(Path(tmp) / "cli")]) == 3)
        finally:
            RI.Adb = real


def check_tracked_policy(RI):
    print("-- the tracked policy against its sources")
    pol = RI.load_policy()
    cfg = [(REPO / "runs" / r / "config.yaml").read_text(encoding="utf-8") for r in ("20260930_A1_r027", "20261001_A1_r035")]
    check("the accepted model hash is the full one both A1 run configs record",
          all(f'model_sha256: "{pol["accepted_model_sha256"]}"' in c for c in cfg)
          and pol["accepted_model_sha256"] == "dd753cd62f163c8baa8d2e598e3b61385f31cd46ca04488cd88ba01a9c83eb18")
    pin = (REPO / "native" / "llama.cpp.pin").read_text(encoding="utf-8").split()
    check("the accepted llama.cpp string is the pin's release and short commit", pol["accepted_llama_cpp"] == f"{pin[0]} ({pin[1][:8]})")
    src = (REPO / "tools" / "build_llama.py").read_text(encoding="utf-8")
    check("the build record path is where build_llama.py writes android builds",
          'NATIVE / "out" / ("android-arm64" if target == "android" else "host")' in src
          and pol["build_record"] == "native/out/android-arm64/build.json")
    panel = (REPO / "quest-app" / "Assets" / "SecondEyes" / "Grounding" / "ChatPanel.cs").read_text(encoding="utf-8")
    check("the model file is ChatPanel's ggufFile default", f'string ggufFile = "{pol["model_file"]}"' in panel)
    check("the data folder is llama_headset.py's app folder",
          f'APP_DIR = "{pol["device_data_dir"]}"' in (REPO / "grounding" / "llama_headset.py").read_text(encoding="utf-8"))
    purposes = [re.search(r'^purpose: "([^"]*)"', (REPO / "runs" / r / "config.yaml").read_text(encoding="utf-8"), re.M).group(1)
                for r in pol["provenance_runs"]]
    check("the provenance runs are A1.8c's command-line runs (their purposes)",
          all(p.startswith("A1.8c O18: llama.cpp from adb shell") for p in purposes), str(purposes))
    check("the header declares 23 functions", len(REQUIRED) == 23 and RI.header_functions(HEADER) == REQUIRED)


def main() -> int:
    B = importlib.import_module("grounding.quest.binaries")
    RI = importlib.import_module("grounding.quest.runtime_identity")
    check_elf(B)
    check_gguf(B)
    check_identity(RI)
    check_cli(RI)
    check_tracked_policy(RI)
    print(f"\n{len(PASSES)} passed, {len(FAILS)} failed")
    return 0 if not FAILS else 1


if __name__ == "__main__":
    sys.exit(main())
