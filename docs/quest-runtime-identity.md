# Runtime identity of the installed app (A2.5; D99(7))

Before the headset replay, and again after any rebuild or before a measurement run, the laptop checks the installed
app's native library and the model file it loads. It uses read-only adb commands only (`devices`, `getprop`,
`pm path`, `dumpsys package`, `ls` and `pull`). Nothing is written to the headset, and no model runs.

The command is `grounding/quest/runtime_identity.py`, with policy `runtime-identity.v1.json`
(`a25.runtime_identity.v1`). The readers it uses are in `grounding/quest/binaries.py`.

This check covers only the installed artifacts. Loading, runtime behaviour, the version the loaded library reports, the
context allocation and the KV-cache types are checked later, by the running app. D56 replay acceptance is separate
again.

## The checks

Each check ends as `pass`, `fail`, `incomplete` or `not_applicable`, with its reason and evidence. An unreadable file,
an unsupported structure or a missing record is `incomplete`, never a pass.

| Check | What it establishes |
|---|---|
| `header` | The functions `native/se_llama.h` exports (23 today) |
| `build_record` | `native/out/android-arm64/build.json`, read as it is, never rebuilt or replaced: an android build of llama.cpp `b11277 (eae11d22)`, with a SHA-256 for `libse_llama.so` |
| `unity_copy` | The Unity project's `libse_llama.so` equals the build record's |
| `device` | Exactly one connected device: its serial, model, OS fingerprint and versions |
| `package` | `com.secondeyes.quest` is installed: its APK path or paths, version and install times |
| `apk_library` | Exactly one installed APK holds `lib/arm64-v8a/libse_llama.so`, and that copy equals both the build record's and the Unity copy's |
| `installed_library` | If the system extracted the library at install, that copy equals the APK's; otherwise `not_applicable` |
| `exports` | Every header function is a defined, externally visible dynamic function export of an ARM64 (AArch64) shared library: in `.dynsym`, `GLOBAL` or `WEAK`, `DEFAULT` or `PROTECTED`, `FUNC`. Missing ones are named individually. Extra exports are counted, never a failure. The embedded `b11277 (eae11d22)` string is recorded as supporting evidence only |
| `model_hash` | The model file the app loads (`ggufFile`, `qwen2.5-0.5b-instruct-q8_0.gguf`, in the app's data folder) is pulled and hashed on the laptop; it must equal the full accepted SHA-256 `dd753cd62f163c8baa8d2e598e3b61385f31cd46ca04488cd88ba01a9c83eb18` (the A1 run configs) |
| `model_quantization` | That file's GGUF metadata: `general.file_type` 7 (`MOSTLY_Q8_0`), with the tensor types counted |

## Two verdicts, kept apart

**Current deployed identity** is `verified` only when every check above passes or is not applicable. A failure makes
it `failed`. Otherwise, anything incomplete makes it `incomplete`.

**A1.8c binary continuity** asks whether today's library is the build A1.8c used. The evidence is the build records
that `grounding/llama_headset.py push` copied into A1.8c's command-line runs, r028–r031 (`runs/<id>/raw/build.json`).
Unchanged sources, matching configuration and matching exports cannot prove that.

- `linked`: one of those records names the deployed library's SHA-256. The in-app acceptance runs (r032–r036) recorded
  no library hash, so a link shows the A1.8c-era build, not that those runs used that exact binary.
- `unverified`: no such record is on this machine.
- `contradicted`: records are here, and none matches.

A hash recorded today is evidence of today's deployment; it never becomes an acceptance-era pin.

## Evidence

The folder is published whatever the verdict, written beside its destination first and never over an existing one.
Repeat checks go to new folders, so each is preserved.

- `identity.json`: every check with its status, reason and evidence, both verdicts, and any unexpected error
- `report.md`: the same, as a table
- `build.json`: the build record, byte for byte
- `libse_llama.so`: the deployed library, from the APK
- `installed-libse_llama.so`: only if the extracted copy differs from the APK's
- `provenance/<run>-build.json`: each A1.8c-era build record found
- the pulled model: only if its hash differs from the accepted one
- `manifest.json`: every file's SHA-256, the policy's and the code's

## Command (laptop, headset connected)

```text
python -m grounding.quest runtime-identity --out NEW_DIR [--adb PATH]
```

Exit codes:

- 0: current deployed identity verified. Continuity is printed separately, as `linked` or `unverified`.
- 1: a check failed, or continuity is contradicted.
- 2: incomplete: nothing failed, but something could not be established.
- 3: not run: an existing destination, an unreadable policy or an output error.

For 0, 1 and 2, the evidence folder is written.

Tests: `python grounding/tests/test_quest_runtime_identity.py`. They use hand-built ARM64 ELF and GGUF files and a
labelled fake adb. Outside the suite, the readers were also compared with binutils' `readelf` on four real ARM64
libraries (one stripped), and with the official `gguf` package on two real GGUF files.
