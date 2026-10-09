"""A2.5 (D98, D99): the PC side of the base-model Quest integration.

Delivery 1, step 1: `replay-bundle` (laptop) freezes the replay requests, their float32 references from run
`20261007_A2_r006` and the written integration fixtures into a bundle whose `headset/` part the headset replays;
`replay_inputs` holds the record format, its hash rules and the ordered input checks the headset must reproduce.
Step 2, part 1: `runtime-identity` (laptop, read-only adb) checks the installed app's native library and model file
against the build record and the accepted A1 artifacts. Step 2, part 2: `replay-push` and `replay-pull` move the
headset inputs to the app and the replay's results back (D103). Step 3: `replay-compare` makes the five D101
comparisons per request; `replay-desktop` runs the desktop diagnosis with the host build. `replay-repeat` and
`repeat-compare` run and analyse the repeatability and context-history diagnostic (ChatGPT's proposal). Commands load
lazily.
"""
import importlib

_LAZY = {"build_replay_bundle": "replay_bundle", "verify_replay_bundle": "replay_bundle", "select_replay": "replay_bundle",
         "check_inputs": "replay_inputs", "check_runtime_identity": "runtime_identity", "push": "replay_device",
         "pull": "replay_device",
         "compare_replay": "replay_compare", "run_desktop": "replay_desktop",
         "run_repeat": "replay_repeat", "compare_repeat": "replay_repeat"}
__all__ = list(_LAZY)


def __getattr__(name):
    if name in _LAZY:
        return getattr(importlib.import_module(f".{_LAZY[name]}", __name__), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
