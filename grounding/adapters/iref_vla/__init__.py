"""The A2.2a IRef-VLA metadata adapter (D74): one pinned public ScanNet sample into the scene contract.

    python -m grounding.adapters.iref_vla --objects CSV --regions CSV --vocabulary CSV
        [--statements JSON] [--graph JSON] --out NEW_FOLDER

The scene stage reads only the object CSV, the region CSV and the vocabulary; expressions become v1 command contexts
with no pose and no action; the source annotations stay in a separate reference-only bundle. See
docs/iref-vla-adapter.md.
"""
from .convert import SceneStage, convert_annotations, convert_commands, convert_scene, crosscheck_graph
from .output import encode, run_import
from .pinned import ADAPTER_VERSION, PINNED
from .sources import AdapterInputError, AdapterOutputError

__all__ = ["ADAPTER_VERSION", "PINNED", "AdapterInputError", "AdapterOutputError", "SceneStage", "convert_annotations",
           "convert_commands", "convert_scene", "crosscheck_graph", "encode", "run_import"]
