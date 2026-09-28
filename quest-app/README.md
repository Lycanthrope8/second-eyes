# quest-app

The Unity app that runs on the Meta Quest 3. At deployment, all AI inference happens here.

## Setup record

| Item | Value |
|---|---|
| Unity | fill in (A1.3a) |
| Meta XR Core SDK | fill in (A1.3a) |
| XR plug-in provider | fill in (A1.3a) |
| Render pipeline | URP (D14) |
| Package name | `com.secondeyes.quest` (D14) |
| Target frame rate | 72 Hz (D13) |

Update this table whenever a version changes, and log the change in `notes/decisions.md`.

## Where code goes

- Our app-level code goes in `Assets/SecondEyes/` (D14).
- Where each component's C# code lives (O6) is decided when the first component has C# code.
- Commit `Assets/`, `Packages/` and `ProjectSettings/`. Unity's generated folders (`Library/`, `Temp/`, `Logs/`, `Builds/` and similar) are ignored by the root `.gitignore`.

## Build and install

With the headset connected: File → Build Profiles → Android → Build And Run. APKs go in `Builds/`, which Git ignores.
