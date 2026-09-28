# quest-app

The Unity app that runs on the Meta Quest 3. At deployment, all AI inference happens here.

Starts to fill in A1 (headset feasibility). Empty until then.

When A1 starts:

- Decide open item O6 first: whether each component's C# code lives here or in its own component folder.
- The Unity project's root goes in this folder, so `Assets/`, `Packages/` and `ProjectSettings/` sit next to this README. If Unity Hub refuses a folder that isn't empty, create the project elsewhere and move its contents here.
- Unity's generated folders (`Library/`, `Temp/`, `Builds/` and similar) are already ignored by the root `.gitignore`.
