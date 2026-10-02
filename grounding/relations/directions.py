"""Directional relations, v1 (A2.1c, D68): left, right, in_front_of and behind, in a frame the caller selects:
user_heading, user_to_anchor or object_intrinsic. Offline Python over scene and command records that have passed
the contract validator (grounding/contract/validate.py).

    d = DirectionalRelations(scene_record)            # loads directions.v1.json unless a config is given
    d.evaluate("right", "obj_002", frame="user_to_anchor", anchor_id="obj_001", command=command_record)

Every frame projects horizontal (XY) centre offsets in the scene frame onto a front axis and a right axis. A
requested score above the band is TRUE, below minus the band FALSE, and anything from -band to +band, both edges
included, UNKNOWN. That band is symmetric and separate from A2.1b's le and ge, which are unchanged. Nothing here
reads box sizes or rotations, builds box geometry, or caches a pose-dependent result. There is no language parsing,
automatic frame choice, resolver or serializer. Definitions: docs/directions.md.
"""
from __future__ import annotations

import hashlib
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import jsonschema

sys.path.insert(0, str(Path(__file__).resolve().parent))
import predicates as P  # noqa: E402  (the Truth enum and the strict JSON parser; no box geometry is used here)

REPO = Path(__file__).resolve().parents[2]
CONFIG_FILE = Path(__file__).resolve().parent / "directions.v1.json"
CONFIG_SCHEMA = REPO / "schemas" / "direction-config.v1.json"

RELATIONS = ("right", "left", "in_front_of", "behind")
FRAMES = ("user_heading", "user_to_anchor", "object_intrinsic")
TRUE, FALSE, UNKNOWN = P.TRUE, P.FALSE, P.UNKNOWN


# ---------------------------------------------------------------------------------------------- configuration

@dataclass(frozen=True)
class DirectionConfig:
    """A loaded direction configuration. Immutable, so its identity always matches its values."""
    config_id: str
    status: str
    identity: str        # config_id plus a hash of the parsed content, the same on every platform
    direction_band_m: float
    viewer_anchor_min_horizontal_m: float
    semantic_front_min_horizontal_norm: float


def _finite(name, value) -> float:
    try:
        number = float(value)
    except OverflowError:
        raise ValueError(f"{name} is too large to be a finite number") from None
    if not math.isfinite(number):
        raise ValueError(f"{name} is not finite")
    return number


def load_direction_config(path=CONFIG_FILE) -> DirectionConfig:
    """Load and validate a direction configuration against schemas/direction-config.v1.json."""
    raw = P._strict_json(Path(path).read_bytes())  # A2.1b's strict parser: no BOM, repeated keys, NaN or Infinity
    schema = json.loads(CONFIG_SCHEMA.read_text(encoding="utf-8"))
    jsonschema.validators.validator_for(schema)(schema).validate(raw)
    canonical = json.dumps(raw, sort_keys=True, separators=(",", ":")).encode("utf-8")
    identity = f"{raw['config_id']}@{hashlib.sha256(canonical).hexdigest()[:12]}"
    return DirectionConfig(
        raw["config_id"], raw["status"], identity,
        _finite("direction_band_m", raw["bands"]["direction_band_m"]),
        _finite("viewer_anchor_min_horizontal_m", raw["thresholds"]["viewer_anchor_min_horizontal_m"]),
        _finite("semantic_front_min_horizontal_norm", raw["thresholds"]["semantic_front_min_horizontal_norm"]))


# ---------------------------------------------------------------------------------------------------- results

@dataclass(frozen=True)
class DirectionFieldRef:
    """One source field a directional result consulted. Sources and assumption IDs belong to their own record."""
    record_type: str      # "scene" or "command_context"
    record_id: tuple      # a scene: (scene_id, scene_revision, evidence_profile); a command: (command_id,)
    object_id: object     # the object, for scene-object fields; None for command-pose fields
    field: str            # "geometry.center_m", "semantic_front", "user_pose.position_m" or "user_pose.heading_xy"
    state: str            # "known" or "unknown"
    kind: object          # evidence kind of a known value, else None
    source: str           # source ID within this record
    assumptions: tuple    # assumption IDs within this record
    reason: object        # reason of an unknown value, else None
    heading_source: object  # heading_source of a known user_pose.heading_xy, else None


@dataclass(frozen=True)
class DirectionResult:
    relation: str
    frame: str
    target_id: str
    anchor_id: object        # None for user_heading
    scene: tuple             # (scene_id, scene_revision, evidence_profile)
    command_id: object       # the pose snapshot used by user_heading and user_to_anchor; None for object_intrinsic
    value: object            # predicates.Truth
    reasons: tuple           # empty when TRUE
    requested_score_m: object  # the requested predicate's signed score in metres; None when it can't be computed
    measures: dict           # right_score_m, front_score_m, viewer_anchor_horizontal_m or
    #                          semantic_front_horizontal_norm, whichever were computed
    inputs: tuple            # DirectionFieldRef of every source field consulted
    assumptions: frozenset   # (record_type, record_id, assumption_id) of the known inputs
    evidence: frozenset      # evidence kinds of the known inputs
    config: str


# ------------------------------------------------------------------------------------------------- evaluator

class DirectionalRelations:
    """Directional relations over one scene record, under one configuration. Pose snapshots come with each call
    and nothing that depends on them is kept between calls."""

    def __init__(self, scene_record: dict, config: DirectionConfig = None):
        if not isinstance(scene_record, dict) or scene_record.get("record_type") != "scene":
            raise ValueError("DirectionalRelations needs a scene record")
        self.scene_key = (scene_record["scene_id"], scene_record["scene_revision"], scene_record["evidence_profile"])
        self.frame_id = scene_record["coordinate_frame"]["frame_id"]
        self._objects = {o["object_id"]: o for o in scene_record["objects"]}
        self.cfg = config or load_direction_config()

    # -- input validation (programming errors raise; they never become UNKNOWN)
    def _object(self, object_id):
        if not isinstance(object_id, str) or object_id not in self._objects:
            raise ValueError(f"{object_id!r} is not an object of scene {self.scene_key[0]}")
        return self._objects[object_id]

    def _check_command(self, command):
        if not isinstance(command, dict) or command.get("record_type") != "command_context":
            raise ValueError("command must be a command_context record")
        cid = command.get("command_id")
        if (command["scene_id"], command["scene_revision"]) != self.scene_key[:2]:
            raise ValueError(f"command {cid} is for scene {command['scene_id']} revision {command['scene_revision']}, "
                             f"not {self.scene_key[0]} revision {self.scene_key[1]}")
        if command["user_pose"]["frame_id"] != self.frame_id:
            raise ValueError(f"command {cid}'s pose is in frame {command['user_pose']['frame_id']}, not "
                             f"{self.frame_id}")

    # -- evidence
    def _scene_field(self, obj, field, refs, missing, reason):
        w = obj
        for key in field.split("."):
            w = w[key]
        refs.append(self._ref("scene", self.scene_key, obj["object_id"], field, w))
        if w["state"] != "known":
            missing.append(reason)
            return None
        return w["value"]

    def _pose_field(self, command, name, refs, missing, reason):
        w = command["user_pose"][name]
        refs.append(self._ref("command_context", (command["command_id"],), None, f"user_pose.{name}", w))
        if w["state"] != "known":
            missing.append(reason)
            return None
        return w["value"]

    @staticmethod
    def _ref(record_type, record_id, object_id, field, w):
        if w["state"] == "known":
            ev = w["evidence"]
            return DirectionFieldRef(record_type, record_id, object_id, field, "known", ev["kind"], ev["source"],
                                     tuple(ev["assumptions"]), None, w.get("heading_source"))
        return DirectionFieldRef(record_type, record_id, object_id, field, "unknown", None, w["source"], (),
                                 w["reason"], None)

    def _result(self, relation, frame, target_id, anchor_id, command_id, value, reasons, score, measures, refs):
        known = [r for r in refs if r.state == "known"]
        assumptions = frozenset((r.record_type, r.record_id, a) for r in known for a in r.assumptions)
        return DirectionResult(relation, frame, target_id, anchor_id, self.scene_key, command_id, value,
                               tuple(dict.fromkeys(reasons)), score, dict(measures), tuple(refs), assumptions,
                               frozenset(r.kind for r in known), self.cfg.identity)

    # -- the predicates
    def evaluate(self, relation, target_id, *, frame, anchor_id=None, command=None) -> DirectionResult:
        if relation not in RELATIONS:
            raise ValueError(f"unsupported directional relation {relation!r}")
        if frame not in FRAMES:
            raise ValueError(f"unsupported frame {frame!r}")
        if frame == "user_heading":
            if anchor_id is not None:
                raise ValueError("user_heading takes no anchor")
            if command is None:
                raise ValueError("user_heading needs a command")
        elif anchor_id is None:
            raise ValueError(f"{frame} needs an anchor")
        elif frame == "user_to_anchor" and command is None:
            raise ValueError("user_to_anchor needs a command")
        target = self._object(target_id)
        anchor = self._object(anchor_id) if anchor_id is not None else None
        if command is not None:
            self._check_command(command)
        command_id = command["command_id"] if frame != "object_intrinsic" else None
        if anchor is not None and target_id == anchor_id:
            return self._result(relation, frame, target_id, anchor_id, command_id, FALSE, ["not_distinct"], None,
                                {}, [])

        refs, missing, measures, reasons = [], [], {}, []
        t = self._scene_field(target, "geometry.center_m", refs, missing, f"missing_centre:{target_id}")
        axes = None  # (origin, front_axis, right_axis), all horizontal
        if frame == "user_heading":
            u = self._pose_field(command, "position_m", refs, missing, f"missing_user_position:{command_id}")
            hd = self._pose_field(command, "heading_xy", refs, missing, f"missing_heading:{command_id}")
            if u is not None and hd is not None:
                n = math.hypot(hd[0], hd[1])
                h = (hd[0] / n, hd[1] / n)
                axes = ((u[0], u[1]), h, (h[1], -h[0]))
        elif frame == "user_to_anchor":
            a = self._scene_field(anchor, "geometry.center_m", refs, missing, f"missing_centre:{anchor_id}")
            u = self._pose_field(command, "position_m", refs, missing, f"missing_user_position:{command_id}")
            if u is not None and a is not None:
                w = (a[0] - u[0], a[1] - u[1])
                length = math.hypot(w[0], w[1])
                measures["viewer_anchor_horizontal_m"] = length
                if length <= self.cfg.viewer_anchor_min_horizontal_m:
                    reasons.append("degenerate_viewer_anchor")
                else:
                    v = (w[0] / length, w[1] / length)
                    axes = ((a[0], a[1]), (-v[0], -v[1]), (v[1], -v[0]))  # right from v, not from the front axis
        else:
            a = self._scene_field(anchor, "geometry.center_m", refs, missing, f"missing_centre:{anchor_id}")
            front = self._scene_field(anchor, "semantic_front", refs, missing, f"missing_semantic_front:{anchor_id}")
            if front is not None:
                q = math.hypot(front[0], front[1])  # tested on the stored vector, before normalizing
                measures["semantic_front_horizontal_norm"] = q
                if q <= self.cfg.semantic_front_min_horizontal_norm:
                    reasons.append("vertical_semantic_front")
                elif a is not None:
                    f = (front[0] / q, front[1] / q)
                    axes = ((a[0], a[1]), f, (f[1], -f[0]))
        if missing or reasons or t is None or axes is None:
            return self._result(relation, frame, target_id, anchor_id, command_id, UNKNOWN, missing + reasons,
                                None, measures, refs)

        origin, front_axis, right_axis = axes
        d = (t[0] - origin[0], t[1] - origin[1])
        right_score = d[0] * right_axis[0] + d[1] * right_axis[1]
        front_score = d[0] * front_axis[0] + d[1] * front_axis[1]
        measures["right_score_m"], measures["front_score_m"] = right_score, front_score
        score = {"right": right_score, "left": -right_score, "in_front_of": front_score,
                 "behind": -front_score}[relation]
        band = self.cfg.direction_band_m
        if score > band:
            value, why = TRUE, []
        elif score < -band:
            value, why = FALSE, [f"failed:{relation}"]
        else:
            value, why = UNKNOWN, [f"boundary:{relation}"]
        return self._result(relation, frame, target_id, anchor_id, command_id, value, why, score, measures, refs)
