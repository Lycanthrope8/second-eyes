# C# checks outside Unity (A2.5)

Three .NET 8 projects compile the app's own C# files at Unity's language level (C# 9). They need no packages:
`nuget.config` clears every source. They live outside `quest-app/Assets`, so Unity never compiles them.

| Project | What it does | Command (from the repository root) |
|---|---|---|
| `Behavior` | the behaviour checks of the pure cores, against the real goldens (64 checks) | `dotnet run --project tools/csharp-tests/Behavior -- <goldens>\headset` |
| `UnityGlue` | a type check of the runtime, replay, prompting and interactive glue against Unity stubs | `dotnet build tools/csharp-tests/UnityGlue` |
| `PanelPartial` | a type check of `ChatPanel.Session.cs` against a shell of the panel members it uses | `dotnet build tools/csharp-tests/PanelPartial` |

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

**What still needs Unity:** `ChatPanel.cs` itself (its UI types are not stubbed), the real Unity and Android
compilation, and everything on the headset.
