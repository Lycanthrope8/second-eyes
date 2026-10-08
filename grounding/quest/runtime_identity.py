"""A2.5 delivery 1, step 2, part 1 (D99(7); ChatGPT's review of 8 October, section 7): the installed artifacts.

Checked on the laptop, with read-only adb commands only (devices, getprop, pm path, dumpsys package, ls, pull); nothing
is written to the headset and no model runs:

- the build record `native/out/android-arm64/build.json`, read as it is, never rebuilt or replaced;
- the Unity project's copy of `libse_llama.so`, and the copy inside the installed APK (and the system's extracted copy,
  if the package has one), against the build record's hash;
- the deployed library's dynamic exports: every function `native/se_llama.h` declares must be a defined, externally
  visible dynamic function in the ARM64 library;
- the model file the app loads, pulled and hashed here against the full accepted SHA-256, with its GGUF quantization.

Two verdicts are kept apart. **Current deployed identity** is verified only when every current check passes. **A1.8c
binary continuity** is linked only when an A1.8c-era build record kept in a run folder names the deployed library's
hash; otherwise it is unverified (no such record here) or contradicted (records here, none matching). A hash recorded
today never becomes an acceptance-era pin. Every check ends as pass, fail, incomplete or not_applicable with its reason;
an unreadable file, an unsupported structure or a missing record is incomplete, never a pass. The evidence folder is
published whatever the verdict: identity.json, report.md, the build record, the deployed library and any provenance
records, plus the pulled model only when its hash differs.
"""
from __future__ import annotations

import datetime
import hashlib
import json
import os
import posixpath
import re
import shutil
import subprocess
import tempfile
import zipfile
from pathlib import Path

from ..evaluation.iref_vla import output
from .publish import publish
from ..evaluation.iref_vla.protocol import (EvaluationInputError, EvaluationOutputError, encode_json, issue, runtime,
                                             sha256, strict_json)
from . import binaries as B

REPO = Path(__file__).resolve().parents[2]
POLICY_PATH = Path(__file__).resolve().parent / "runtime-identity.v1.json"
POLICY_ID = "a25.runtime_identity.v1"
CURRENT = ("header", "build_record", "unity_copy", "device", "package", "apk_library", "installed_library", "exports",
           "model_hash", "model_quantization")
PROPS = ("ro.product.manufacturer", "ro.product.model", "ro.build.fingerprint", "ro.build.version.release",
         "ro.build.version.incremental")


class Adb:
    """The read-only adb commands this check uses. Every call returns (exit code, stdout, stderr)."""

    def __init__(self, binary="adb", timeout=900):
        self.binary, self.timeout = binary, timeout

    def run(self, *args):
        try:
            p = subprocess.run([self.binary, *args], capture_output=True, text=True, timeout=self.timeout,
                               encoding="utf-8", errors="replace")
            return p.returncode, p.stdout, p.stderr
        except FileNotFoundError:
            return 127, "", f"{self.binary} not found"
        except subprocess.TimeoutExpired:
            return 124, "", f"timed out after {self.timeout} s"

    def shell(self, *args):
        return self.run("shell", *args)

    def pull(self, remote, local):
        return self.run("pull", remote, str(local))


def file_sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def header_functions(text: str) -> list:
    """The functions a C header exports with SE_API, in order."""
    return [m.group(1) for m in (re.match(r"^SE_API\b.*?\b(\w+)\s*\(", line) for line in text.splitlines()) if m]


def load_policy(path=None) -> dict:
    p = Path(path) if path is not None else POLICY_PATH
    pol = strict_json("runtime identity policy", p.read_bytes(), "E_IDENTITY_POLICY")
    need = ("package", "device_data_dir", "model_file", "accepted_model_sha256", "accepted_file_type", "accepted_llama_cpp",
            "build_record", "unity_copy", "apk_library_entry", "header", "provenance_runs", "supporting_string")
    if not isinstance(pol, dict) or pol.get("policy_id") != POLICY_ID or any(k not in pol for k in need) \
            or not re.fullmatch(r"[0-9a-f]{64}", str(pol.get("accepted_model_sha256"))):
        raise EvaluationInputError([issue(str(p), "E_IDENTITY_POLICY", f"not a complete {POLICY_ID} policy with a full "
                                                                       "64-hex accepted model hash")])
    return pol


def _check(status, detail, **evidence):
    return {"status": status, "detail": detail, "evidence": evidence}


class _Run:
    """One identity check: gathers evidence into a staging folder; each step records its own status."""

    def __init__(self, pol, adb, repo, staging, progress):
        self.pol, self.adb, self.repo, self.dir, self.say = pol, adb, Path(repo), staging, progress
        self.checks, self.lib, self.build_hash, self.unity_hash, self.device_ok = {}, None, None, None, False

    # ------------------------------------------------------------------ local
    def header(self):
        p = self.repo / self.pol["header"]
        try:
            text = p.read_text(encoding="utf-8")
        except OSError as e:
            self.checks["header"] = _check("incomplete", f"cannot read {self.pol['header']}: {e}")
            return []
        names = header_functions(text)
        self.checks["header"] = _check("pass" if names else "incomplete",
                                       f"{len(names)} functions declared" if names else "no SE_API function found",
                                       path=self.pol["header"], sha256=sha256(text.encode("utf-8")), functions=names)
        return names

    def build_record(self):
        p = self.repo / self.pol["build_record"]
        if not p.is_file():
            self.checks["build_record"] = _check("incomplete", f"{self.pol['build_record']} not found; it is read as it "
                                                               "is and never rebuilt here", path=self.pol["build_record"])
            return
        data = p.read_bytes()
        output._write_file(self.dir / "build.json", data)
        try:
            rec = json.loads(data.decode("utf-8"))
            f = rec["files"]["libse_llama.so"]
            self.build_hash = f["sha256"] if re.fullmatch(r"[0-9a-f]{64}", str(f.get("sha256"))) else None
        except (ValueError, KeyError, TypeError) as e:
            self.checks["build_record"] = _check("incomplete", f"unreadable build record: {type(e).__name__}: {e}",
                                                 path=self.pol["build_record"], sha256=sha256(data), saved_as="build.json")
            return
        ev = dict(path=self.pol["build_record"], sha256=sha256(data), saved_as="build.json", target=rec.get("target"),
                  llama_cpp=rec.get("llama_cpp"), ndk=rec.get("ndk"), arm_arch=rec.get("arm_arch"),
                  libse_llama_sha256=self.build_hash, libse_llama_bytes=f.get("bytes"),
                  file_mtime_utc=datetime.datetime.fromtimestamp(p.stat().st_mtime, datetime.timezone.utc).isoformat())
        if self.build_hash is None:
            self.checks["build_record"] = _check("incomplete", "the record holds no SHA-256 for libse_llama.so", **ev)
        elif rec.get("llama_cpp") != self.pol["accepted_llama_cpp"] or rec.get("target") != "android":
            self.checks["build_record"] = _check("fail", f"an {rec.get('target')!r} build of llama.cpp "
                                                         f"{rec.get('llama_cpp')!r}, not android {self.pol['accepted_llama_cpp']}",
                                                 **ev)
        else:
            self.checks["build_record"] = _check("pass", f"android build of {rec.get('llama_cpp')}", **ev)

    def unity_copy(self):
        p = self.repo / self.pol["unity_copy"]
        if not p.is_file():
            self.checks["unity_copy"] = _check("incomplete", f"{self.pol['unity_copy']} not found", path=self.pol["unity_copy"])
            return
        self.unity_hash = file_sha256(p)
        ev = dict(path=self.pol["unity_copy"], bytes=p.stat().st_size, sha256=self.unity_hash)
        if self.build_hash is None:
            self.checks["unity_copy"] = _check("incomplete", "no build-record hash to compare with", **ev)
        else:
            same = self.unity_hash == self.build_hash
            self.checks["unity_copy"] = _check("pass" if same else "fail", "equals the build record's libse_llama.so"
                                               if same else "differs from the build record's libse_llama.so", **ev)

    # ------------------------------------------------------------------ device
    def device(self):
        code, out, err = self.adb.run("devices")
        if code != 0:
            self.checks["device"] = _check("incomplete", f"adb devices failed ({code}): {err.strip() or out.strip()}")
            return False
        listed = [line.strip() for line in out.splitlines()[1:] if line.strip()]
        serials = [line.split("\t")[0] for line in listed if line.endswith("\tdevice")]
        if len(serials) != 1:
            self.checks["device"] = _check("incomplete", f"{len(serials)} devices ready; exactly one is needed",
                                           adb_devices=listed)
            return False
        self.device_ok = True
        props = {}
        for name in PROPS:
            c, o, _ = self.adb.shell("getprop", name)
            props[name] = o.strip() if c == 0 else None
        code, out, _ = self.adb.run("version")
        ok = all(props.values())
        self.checks["device"] = _check("pass" if ok else "incomplete", "one device" if ok else "some properties unread",
                                       serial=serials[0], properties=props,
                                       adb_version=out.strip().splitlines()[0] if code == 0 and out.strip() else None)
        return True

    def package(self):
        pkg = self.pol["package"]
        code, out, err = self.adb.shell("pm", "path", pkg)
        paths = [line[len("package:"):].strip() for line in out.splitlines() if line.startswith("package:")]
        if code != 0 or not paths:
            self.checks["package"] = _check("incomplete", f"{pkg} is not installed, or pm path failed ({code}): "
                                                          f"{err.strip()}", package=pkg)
            return []
        code, out, _ = self.adb.shell("dumpsys", "package", pkg)
        found = {k: (re.search(rf"\b{k}=(\S+)", out).group(1) if code == 0 and re.search(rf"\b{k}=(\S+)", out) else None)
                 for k in ("versionCode", "versionName")}
        for k in ("firstInstallTime", "lastUpdateTime"):
            m = re.search(rf"\b{k}=([^\r\n]+)", out) if code == 0 else None
            found[k] = m.group(1).strip() if m else None
        self.checks["package"] = _check("pass", f"{len(paths)} APK path(s)", package=pkg, apk_paths=paths, **found)
        return paths

    def apk_library(self, paths, scratch):
        entry, apks, hits = self.pol["apk_library_entry"], [], []
        for k, remote in enumerate(paths):
            local = scratch / f"apk{k}.apk"
            self.say(f"pulling {remote}")
            code, _, err = self.adb.pull(remote, local)
            if code != 0 or not local.is_file():
                apks.append({"path": remote, "pulled": False, "error": err.strip()})
                continue
            rec = {"path": remote, "pulled": True, "bytes": local.stat().st_size, "sha256": file_sha256(local)}
            try:
                with zipfile.ZipFile(local) as z:
                    if entry in z.namelist():
                        hits.append((remote, z.read(entry)))
                        rec["contains_library"] = True
            except zipfile.BadZipFile as e:
                rec["error"] = f"not a readable APK: {e}"
            apks.append(rec)
            local.unlink()
        ev = dict(apks=apks, entry=entry)
        if any(not a["pulled"] or "error" in a for a in apks):
            self.checks["apk_library"] = _check("incomplete", "an APK could not be pulled or read", **ev)
            return
        if not hits:
            self.checks["apk_library"] = _check("fail", f"no APK of the installed app holds {entry}", **ev)
            return
        if len(hits) > 1:
            self.checks["apk_library"] = _check("incomplete", f"{len(hits)} APKs hold {entry}", **ev)
            return
        remote, data = hits[0]
        self.lib = data
        output._write_file(self.dir / "libse_llama.so", data)
        h = sha256(data)
        ev.update(apk=remote, sha256=h, bytes=len(data), saved_as="libse_llama.so")
        refs = {"build record": self.build_hash, "Unity copy": self.unity_hash}
        differs = [n for n, v in refs.items() if v is not None and v != h]
        unknown = [n for n, v in refs.items() if v is None]
        if differs:
            self.checks["apk_library"] = _check("fail", "differs from the " + " and the ".join(differs), **ev)
        elif unknown:
            self.checks["apk_library"] = _check("incomplete", "no hash to compare with from the " + " or the ".join(unknown),
                                                **ev)
        else:
            self.checks["apk_library"] = _check("pass", "equals the build record's and the Unity copy's", **ev)

    def installed_library(self, paths):
        if self.lib is None:
            self.checks["installed_library"] = _check("incomplete", "no APK library to compare with")
            return
        base = next((p for p in paths if p.endswith("/base.apk")), paths[0])
        remote = posixpath.join(posixpath.dirname(base), "lib", "arm64", "libse_llama.so")
        code, out, err = self.adb.shell("ls", "-l", remote)
        if code != 0 and "No such file" in (out + err):
            self.checks["installed_library"] = _check("not_applicable", "not extracted at install: the library is loaded "
                                                                        "from the APK", path=remote)
            return
        with tempfile.TemporaryDirectory() as tmp:
            local = Path(tmp) / "libse_llama.so"
            c, _, e = self.adb.pull(remote, local) if code == 0 else (code, "", err)
            if c != 0 or not local.is_file():
                self.checks["installed_library"] = _check("incomplete", f"cannot read the extracted copy: {(e or err).strip()}",
                                                          path=remote)
                return
            data = local.read_bytes()
        same = sha256(data) == sha256(self.lib)
        if not same:
            output._write_file(self.dir / "installed-libse_llama.so", data)
        self.checks["installed_library"] = _check("pass" if same else "fail", "the extracted copy equals the APK's"
                                                  if same else "the extracted copy differs from the APK's", path=remote,
                                                  sha256=sha256(data), **({} if same else {"saved_as": "installed-libse_llama.so"}))

    def exports(self, required):
        if self.lib is None or not required:
            self.checks["exports"] = _check("incomplete", "no deployed library or no header functions to check")
            return
        try:
            elf = B.elf_dynamic_symbols(self.lib)
        except B.Unsupported as e:
            self.checks["exports"] = _check("incomplete", f"unsupported or unreadable library: {e}")
            return
        fmt = elf["format"]
        results = B.usable_exports(elf["symbols"], required)
        bad = [n for n in required if not results[n]["usable"]]
        others = sorted(n for n, s in elf["symbols"].items() if n not in results and s["defined"] and s["type"] == "FUNC"
                        and s["bind"] in ("GLOBAL", "WEAK") and s["visibility"] in ("DEFAULT", "PROTECTED"))
        ev = dict(format=fmt, required=len(required), results=results, unusable=bad, other_exported_functions=len(others),
                  supporting_string=self.pol["supporting_string"],
                  supporting_string_found=self.pol["supporting_string"].encode("utf-8") in self.lib)
        if not (fmt["machine_is_aarch64"] and fmt["type_is_shared_object"]):
            self.checks["exports"] = _check("fail", f"not an ARM64 shared library (machine {fmt['machine']}, type "
                                                    f"{fmt['type']})", **ev)
        elif bad:
            self.checks["exports"] = _check("fail", f"{len(bad)} of {len(required)} functions not usable: {', '.join(bad)}",
                                            **ev)
        else:
            self.checks["exports"] = _check("pass", f"all {len(required)} functions are usable exports", **ev)

    def model(self):
        remote = f"{self.pol['device_data_dir']}/{self.pol['model_file']}"
        local = self.dir / self.pol["model_file"]
        self.say(f"pulling {remote}")
        code, _, err = self.adb.pull(remote, local)
        if code != 0 or not local.is_file():
            if local.exists():
                local.unlink()
            for k in ("model_hash", "model_quantization"):
                self.checks[k] = _check("incomplete", f"cannot pull {remote}: {err.strip()}", path=remote)
            return
        h, size = file_sha256(local), local.stat().st_size
        same = h == self.pol["accepted_model_sha256"]
        self.checks["model_hash"] = _check("pass" if same else "fail", "equals the accepted A1 model's full SHA-256" if same
                                           else "differs from the accepted A1 model's SHA-256", path=remote, bytes=size,
                                           sha256=h, accepted_sha256=self.pol["accepted_model_sha256"],
                                           **({} if same else {"saved_as": self.pol["model_file"]}))
        try:
            g = B.gguf_header(local)
            want = self.pol["accepted_file_type"]
            if g["file_type"] is None:
                self.checks["model_quantization"] = _check("incomplete", "no general.file_type in the GGUF metadata", gguf=g)
            else:
                ok = g["file_type"] == want
                self.checks["model_quantization"] = _check("pass" if ok else "fail",
                                                           f"general.file_type {g['file_type']} ({g['file_type_name']})"
                                                           + ("" if ok else f", not {want} ({self.pol.get('accepted_file_type_name')})"),
                                                           gguf=g)
        except (B.Unsupported, OSError) as e:
            self.checks["model_quantization"] = _check("incomplete", f"unsupported or unreadable GGUF: {e}")
        if same:
            local.unlink()   # the accepted artifact itself: its hash is the evidence; a differing file is kept

    # ------------------------------------------------------------------ continuity
    def continuity(self):
        deployed = sha256(self.lib) if self.lib is not None else None
        records = []
        for run in self.pol["provenance_runs"]:
            p = self.repo / "runs" / run / "raw" / "build.json"
            if not p.is_file():
                records.append({"run": run, "found": False})
                continue
            data = p.read_bytes()
            (self.dir / "provenance").mkdir(exist_ok=True)
            output._write_file(self.dir / "provenance" / f"{run}-build.json", data)
            try:
                rec = json.loads(data.decode("utf-8"))
                h = rec["files"]["libse_llama.so"]["sha256"]
            except (ValueError, KeyError, TypeError):
                rec, h = {}, None
            records.append({"run": run, "found": True, "sha256": sha256(data), "saved_as": f"provenance/{run}-build.json",
                            "llama_cpp": rec.get("llama_cpp"), "libse_llama_sha256": h,
                            "matches_deployed": deployed is not None and h == deployed})
        linked = [r["run"] for r in records if r.get("matches_deployed")]
        found = [r for r in records if r["found"]]
        if deployed is None:
            status, why = "unverified", "the deployed library could not be read"
        elif linked:
            status, why = "linked", (f"the deployed library's SHA-256 equals the libse_llama.so recorded by {', '.join(linked)}; "
                                     "the in-app acceptance runs recorded no library hash")
        elif found:
            status, why = "contradicted", "A1.8c-era build records exist here and none names the deployed library's hash"
        else:
            status, why = "unverified", "no A1.8c-era build record is in this machine's run folders"
        return {"status": status, "reason": why, "deployed_sha256": deployed, "records": records,
                "rule": self.pol.get("provenance_rule")}


def summarize(checks, continuity) -> dict:
    states = [checks[k]["status"] for k in CURRENT]
    current = "failed" if "fail" in states else "incomplete" if "incomplete" in states else "verified"
    line = f"current deployed identity {current}; A1.8c binary continuity {continuity['status']}"
    code = 1 if current == "failed" or continuity["status"] == "contradicted" else 2 if current == "incomplete" else 0
    return {"current_identity": current, "a18c_continuity": continuity["status"], "line": line, "exit_code": code,
            "failed": [k for k in CURRENT if checks[k]["status"] == "fail"],
            "incomplete": [k for k in CURRENT if checks[k]["status"] == "incomplete"]}


def render(record) -> str:
    s = record["status"]
    L = ["# Runtime identity check (A2.5, D99(7))", "", f"Checked {record['checked_at_utc']} on the laptop; read-only adb.",
         "", f"**{s['line']}.**", "", "| Check | Status | Detail |", "|---|---|---|"]
    for k in CURRENT:
        c = record["checks"][k]
        L.append(f"| {k} | {c['status']} | {c['detail'].replace('|', '/')} |")
    c = record["continuity"]
    L += ["", f"A1.8c binary continuity: **{c['status']}**. {c['reason']}.", ""]
    for r in c["records"]:
        L.append(f"- {r['run']}: " + ("no build record here" if not r["found"] else
                                      f"libse_llama.so {r.get('libse_llama_sha256')}, matches the deployed library: "
                                      f"{'yes' if r.get('matches_deployed') else 'no'}"))
    L += ["", "Loading, runtime behaviour, context allocation, KV-cache types and the version the loaded library reports "
              "are checked later, by the running app; D56 replay acceptance is separate."]
    return "\n".join(L) + "\n"


def code_hashes() -> dict:
    here = Path(__file__).resolve().parent
    return {f"grounding/quest/{p.name}": sha256(p.read_bytes()) for p in sorted(here.glob("*")) if p.suffix in (".py", ".json")}


def check_runtime_identity(*, out, adb=None, repo=None, policy=None, progress=print) -> dict:
    """Run every check, publish the evidence folder whatever the verdict, and return the record."""
    out = output.refuse_existing(out)
    pol_path = Path(policy) if policy is not None else POLICY_PATH
    pol = load_policy(pol_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{out.name}.partial-", dir=str(out.parent)))
    try:
        run = _Run(pol, adb if adb is not None else Adb(), repo if repo is not None else REPO, staging, progress)
        error = None
        try:
            required = run.header()
            run.build_record()
            run.unity_copy()
            paths = run.package() if run.device() else []
            with tempfile.TemporaryDirectory() as scratch:
                if paths:
                    run.apk_library(paths, Path(scratch))
                    run.installed_library(paths)
            run.exports(required)
            if run.device_ok:
                run.model()
        except Exception as e:  # noqa: BLE001  (an unexpected error keeps the evidence gathered so far)
            error = f"{type(e).__name__}: {e}"
        for k in CURRENT:   # a check not reached is incomplete, with the reason
            run.checks.setdefault(k, _check("incomplete", f"not reached: {error}" if error else
                                            ("no single connected device" if not run.device_ok else
                                             "no installed package to read")))
        try:
            continuity = run.continuity()
        except Exception as e:  # noqa: BLE001
            error = error or f"{type(e).__name__}: {e}"
            continuity = {"status": "unverified", "reason": f"not reached: {error}", "deployed_sha256": None,
                          "records": [], "rule": pol.get("provenance_rule")}
        record = {"format_version": 1, "record_type": "a25_runtime_identity", "policy_id": pol["policy_id"],
                  "checked_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
                  "status": summarize(run.checks, continuity), "checks": {k: run.checks[k] for k in CURRENT},
                  "continuity": continuity, "error": error,
                  "notes": ["Read-only adb; nothing was written to the headset and no model ran.",
                            "A hash recorded today is evidence of today's deployment, not an acceptance-era pin.",
                            "Loading, runtime behaviour, context allocation and KV-cache types are later checks."]}
        output._write_file(staging / "identity.json", encode_json(record))
        output._write_file(staging / "report.md", render(record).encode("utf-8"))
        files = {p.relative_to(staging).as_posix(): file_sha256(p) for p in sorted(staging.rglob("*")) if p.is_file()}
        manifest = {"format_version": 1, "record_type": "a25_runtime_identity_manifest", "policy_id": pol["policy_id"],
                    "policy_sha256": sha256(pol_path.read_bytes()), "files": files, "code": code_hashes(),
                    "runtime": runtime()}
        output._write_file(staging / "manifest.json", encode_json(manifest))
        publish(staging, out)
    except OSError as e:
        shutil.rmtree(staging, ignore_errors=True)
        raise EvaluationOutputError([issue(str(out), "E_EVAL_OUTPUT_IO", f"{type(e).__name__}: {e}")]) from e
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return record
