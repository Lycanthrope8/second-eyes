# C# checks outside Unity (A2.5, A2.7)

Five .NET 8 projects compile the app's own C# files at Unity's language level (C# 9). They need no packages:
`nuget.config` clears every source. They live outside `quest-app/Assets`, so Unity never compiles them.

| Project | What it does | Command (from the repository root) |
|---|---|---|
| `Behavior` | the behaviour checks of the pure cores, against the real goldens (64 checks) | `dotnet run --project tools/csharp-tests/Behavior -- <goldens>\headset` |
| `UnityGlue` | a type check of the runtime, replay, prompting, interactive and parser glue against Unity stubs | `dotnet build tools/csharp-tests/UnityGlue` |
| `PanelPartial` | a type check of `ChatPanel.Session.cs` against a shell of the panel members it uses | `dotnet build tools/csharp-tests/PanelPartial` |
| `Resolution` (A2.7) | the C# port of the parser path against the Python reference, mode by mode (below) | `dotnet run --project tools/csharp-tests/Resolution -- MODE ...` |
| `UnityApi` (A2.7) | the port compiled as a .NET Standard 2.1 library, Unity's API profile: an API Unity lacks fails the build | `dotnet build tools/csharp-tests/UnityApi` |

**The behaviour checks cover:**

- the pipeline: prompts and mappings equal the 32 goldens; the cache (off and exact prefix); refusals; failure and
  recovery;
- the startup guard;
- the intake core: validation, processed-once, scene binding, cancel, stale results;
- the lifecycle: closing admission, release after publication;
- the shutdown orchestration: pending, faulted and cancelled tasks, late handlers;
- the presentation rules.

The goldens folder is `goldens-20261009-053729`. On the laptop it is
`C:\Users\jubay\second-eyes-data\a25-goldens\goldens-20261009-053729\headset`.

**The Resolution project's modes** (A2.7, D116; the golden files come from `python -m grounding.a27 goldens`, or for
`acceptance` from `headset --port-out` on the RTX PC; `docs/a27-headset.md` has the details):

| Mode | Arguments |
|---|---|
| `relations` | `GOLDENS/relations.json grounding/relations/relations.v1.json` |
| `schemas` | `GOLDENS/schemas.json schemas` |
| `validation` | `GOLDENS/validation.json schemas` |
| `resolution` | `GOLDENS/resolution.json schemas grounding/relations/relations.v1.json grounding/relations/directions.v1.json` |
| `reader` | `GOLDENS/reader.json grounding/a24/language.v1.json` |
| `pipeline` | `GOLDENS/pipeline.json schemas` + the two relation configurations + `grounding/a24/language.v1.json grounding/a24/settings.v1.json` |
| `native` | `GOLDENS/native.json TINY.gguf`, with the host build's `se_llama.dll` on PATH |
| `acceptance` | `PORT/port-acceptance.json schemas` + the two relation configurations + `grounding/a24/language.v1.json` |
| `parser-check` | `FOLDER OUT [LIB GGUF]`: the headset's acceptance run on the PC |

Each mode prints how many values it compared and how many differ; every one must differ in none.

**What still needs Unity:** `ChatPanel.cs` itself (its UI types are not stubbed), the real Unity and Android
compilation, and everything on the headset.
