"""The pinned IRef-VLA inputs and the identities the A2.2a import assigns (D74).

Every value here is from the approved A2.2a brief: the five files of Appendix A and the identities of section 5. The
adapter supports exactly this source shape and scene; other scenes or source versions need their own approved step.
"""
from __future__ import annotations

ADAPTER_VERSION = "iref_vla_adapter.v1"
REPOSITORY = "https://github.com/HaochenZ11/IRef-VLA"
COMMIT = "741cf0289a6ee7c8f0763058f40bb3788b76f8e0"
URL_PREFIX = f"https://raw.githubusercontent.com/HaochenZ11/IRef-VLA/{COMMIT}/"
_SAMPLE = "sample_data/Scannet/scene0010_01/"
FILES = {
    "objects": {"name": "scene0010_01_object_result.csv", "path": _SAMPLE + "scene0010_01_object_result.csv",
                "bytes": 18654, "sha256": "3bd5f282abf5cf5c6abe5d106a2fd87ad6090ec810d50b9e66e4d1e17aaf2996"},
    "regions": {"name": "scene0010_01_region_result.csv", "path": _SAMPLE + "scene0010_01_region_result.csv",
                "bytes": 224, "sha256": "1b77150f1e65bb5c377457bdfdb944575864bd04ab88fb94a24cc40510ee0091"},
    "graph": {"name": "scene0010_01_scene_graph.json", "path": _SAMPLE + "scene0010_01_scene_graph.json",
              "bytes": 317449, "sha256": "59eea627704d46b3028989c7eef418804f987955695697fa62c474dd53f27817"},
    "statements": {"name": "scene0010_01_referential_statements.json",
                   "path": _SAMPLE + "scene0010_01_referential_statements.json",
                   "bytes": 4716682, "sha256": "0c077963bcdb0e7f19aa0885ddb547689dfe106da622d8d4b58d4e894d143d08"},
    "vocabulary": {"name": "NYU_Object_Classes.csv", "path": "data/NYU_Object_Classes.csv",
                   "bytes": 14241, "sha256": "3c65ff2e4ad3a60a0a4ad87f01ef864f6848cdff8b8c7792e43e2389d9585f91"},
}
PINNED = {"repository": REPOSITORY, "commit": COMMIT, "url_prefix": URL_PREFIX, "files": FILES}

SCENE_NAME = "scene0010_01"
REGION_ID = "0"
SOURCE_ID = "iref_scannet"
SCENE_ID = "iref.scannet.scene0010_01.full"
FRAME_ID = "iref.scannet.scene0010_01.native"
MAP_ID = "iref.scannet.nyu.sample.v1"
COMMAND_PREFIX = "iref.scannet.scene0010_01.r0.e."
UNKNOWN_SENTINEL = "unknown"
IDENTITY_MATRIX = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
