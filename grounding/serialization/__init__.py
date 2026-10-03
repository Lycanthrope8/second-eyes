"""The A2.1d offline serializer (D69): validated scene and command-context records into coordinates_v2 or
coordinates_relations_v2 model input. See docs/serialization.md.

    from grounding.serialization import serialize
    result = serialize(scene, command, format="coordinates_relations_v2",
                       relation_config_path="grounding/relations/relations.v1.json",
                       direction_config_path="grounding/relations/directions.v1.json",
                       max_relation_work_units=1190)
    result.status, result.document
"""
from .constants import AUGMENTED, COORDINATES, FORMATS, SERIALIZER_VERSION, work_slots
from .serializer import (SerializationInputError, SerializationResult, TokenCheck, TokenCheckError,
                         canonical_sha256, serialize)

__all__ = ["serialize", "SerializationResult", "TokenCheck", "SerializationInputError", "TokenCheckError",
           "FORMATS", "COORDINATES", "AUGMENTED", "SERIALIZER_VERSION", "work_slots", "canonical_sha256"]
