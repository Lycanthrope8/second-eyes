using System;
using System.Collections.Generic;
using System.Globalization;
using SecondEyes.Grounding.Prompting;

namespace SecondEyes.Grounding.Resolution
{
    // A2.7 delivery 2 (D115): grounding/contract/validate.py's layers after parsing and checks.py (A2.1a, D66; scene v2,
    // D74; scene v3, A4: the perceived representation), free of Unity: the record type and version, the record's JSON Schema, then the semantic rules, then the
    // cross-record rules for records validated together. A layer runs only if the one before it passed. Issues carry
    // the Python's codes and paths; their messages are shorter, since no decision reads them.
    public sealed class Issue
    {
        public string File, Path, Code, Message;
        public bool IsError { get { return Code.StartsWith("E_", StringComparison.Ordinal); } }
        public override string ToString() { return File + ": " + Path + ": " + Code + ": " + Message; }
    }

    public sealed class Contract
    {
        public const double Tolerance = 1e-6;   // D66's validation tolerance for stored geometry

        public static readonly Dictionary<string, string> Files = new Dictionary<string, string>
        {
            { "scene|1", "scene.v1.json" }, { "scene|2", "scene.v2.json" }, { "scene|3", "scene.v3.json" },
            { "command_context|1", "command-context.v1.json" }, { "command_context|2", "command-context.v2.json" }, { "category_map|1", "category-map.v1.json" }
        };

        private static readonly Dictionary<string, string> KeywordCodes = new Dictionary<string, string>
        {
            { "required", "E_SCHEMA_REQUIRED" }, { "additionalProperties", "E_SCHEMA_ADDITIONAL" }, { "type", "E_SCHEMA_TYPE" },
            { "enum", "E_SCHEMA_ENUM" }, { "const", "E_SCHEMA_CONST" }, { "pattern", "E_SCHEMA_PATTERN" },
            { "minItems", "E_SCHEMA_LENGTH" }, { "maxItems", "E_SCHEMA_LENGTH" }, { "minLength", "E_SCHEMA_LENGTH" },
            { "minimum", "E_SCHEMA_RANGE" }, { "maximum", "E_SCHEMA_RANGE" }, { "exclusiveMinimum", "E_SCHEMA_RANGE" },
            { "exclusiveMaximum", "E_SCHEMA_RANGE" }, { "uniqueItems", "E_SCHEMA_UNIQUE" }
        };

        private static readonly Dictionary<string, HashSet<string>> KnownSourceKinds = new Dictionary<string, HashSet<string>>
        {
            { "annotated", new HashSet<string> { "dataset", "fixture" } },
            { "inferred", new HashSet<string> { "dataset", "fixture", "prior_table" } },
            { "assumed", new HashSet<string> { "prior_table" } },
            { "measured", new HashSet<string>() }
        };
        private static readonly Dictionary<string, HashSet<string>> UnknownSourceKinds = new Dictionary<string, HashSet<string>>
        {
            { "not_in_source", new HashSet<string> { "dataset", "fixture" } },
            { "unmapped_label", new HashSet<string> { "dataset", "fixture" } },
            { "source_invalid", new HashSet<string> { "dataset", "fixture" } },
            { "withheld_by_profile", new HashSet<string> { "profile" } },
            { "no_permitted_prior", new HashSet<string> { "prior_table", "profile" } },
            { "not_estimated", null }   // any source kind
        };
        private static readonly HashSet<string> ObjectSourceKinds = new HashSet<string> { "dataset", "fixture" };
        // Scene v3 (A4), as checks.py: a scan's sensor_capture source stands behind inferred and measured values, and
        // objects come only from it.
        private static readonly Dictionary<string, HashSet<string>> KnownSourceKindsV3 = new Dictionary<string, HashSet<string>>
        {
            { "annotated", new HashSet<string> { "dataset", "fixture" } },
            { "inferred", new HashSet<string> { "dataset", "fixture", "prior_table", "sensor_capture" } },
            { "assumed", new HashSet<string> { "prior_table" } },
            { "measured", new HashSet<string> { "sensor_capture" } }
        };
        private static readonly HashSet<string> ObjectSourceKindsV3 = new HashSet<string> { "sensor_capture" };

        /// <summary>Format 2 of the command record exists only for a command typed on the headset (A4): one with a user_input
        /// source. Any other record labelled format 2 stays unsupported, exactly as before format 2 existed.</summary>
        public static bool Typed(JNode record)
        {
            JNode sources = Get(record, "sources");
            if (sources.Kind != JKind.Array) return false;
            foreach (JNode s in sources.Items)
                if (s.Kind == JKind.Object && Get(s, "kind").Kind == JKind.String && Get(s, "kind").Text == "user_input") return true;
            return false;
        }

        // A command typed on the headset (command format 2): its user pose is not in its source, the person's input.
        private static readonly Dictionary<string, HashSet<string>> UnknownSourceKindsTyped = Typed(UnknownSourceKinds);

        private static Dictionary<string, HashSet<string>> Typed(Dictionary<string, HashSet<string>> kinds)
        {
            var d = new Dictionary<string, HashSet<string>>();
            foreach (var kv in kinds) d[kv.Key] = kv.Value == null ? null : new HashSet<string>(kv.Value);   // null: any source kind
            d["not_in_source"].Add("user_input");
            return d;
        }
        // The perceived profile, as checks.py's PERCEIVED_FIELDS: per value, the "evidence kind|source kind" pairs a known
        // value may have, and the reasons an unknown value may give.
        private static readonly Dictionary<string, KeyValuePair<HashSet<string>, HashSet<string>>> PerceivedFields =
            new Dictionary<string, KeyValuePair<HashSet<string>, HashSet<string>>>
        {
            { "category", Allow(new[] { "inferred|sensor_capture" }, "not_estimated") },
            { "geometry.center_m", Allow(new[] { "inferred|sensor_capture" }, "not_estimated") },
            { "geometry.size_m", Allow(new[] { "assumed|prior_table", "inferred|sensor_capture" }, "no_permitted_prior") },   // A4: 3D boxes
            { "geometry.rotation_xyzw", Allow(new[] { "inferred|sensor_capture" }, "not_estimated") },
            { "semantic_front", Allow(new string[0], "not_estimated") },
            { "attributes.colours", Allow(new[] { "inferred|sensor_capture" }, "not_estimated") },
            { "observation.capture_time", Allow(new[] { "measured|sensor_capture" }, "not_estimated") },
            { "observation.detector_confidence", Allow(new[] { "inferred|sensor_capture" }, "not_estimated") },
            { "observation.geometry_uncertainty", Allow(new string[0], "not_estimated") }
        };

        private static KeyValuePair<HashSet<string>, HashSet<string>> Allow(string[] known, string unknown)
        {
            return new KeyValuePair<HashSet<string>, HashSet<string>>(new HashSet<string>(known), new HashSet<string> { unknown });
        }
        private static readonly Dictionary<string, string> PoseSourceKind = new Dictionary<string, string>
            { { "authored", "fixture" }, { "dataset_camera", "dataset" }, { "dataset_viewpoint", "dataset" } };
        private static readonly Dictionary<string, string> PoseHeadingSource = new Dictionary<string, string>
            { { "authored", "authored" }, { "dataset_camera", "camera_yaw" }, { "dataset_viewpoint", "camera_yaw" } };

        /// <summary>The issue code for a failed JSON Schema keyword, as validate.py's KEYWORD_CODES (shared with the
        /// capture record's check, A4).</summary>
        public static string CodeForKeyword(string keyword)
        {
            string code;
            return KeywordCodes.TryGetValue(keyword, out code) ? code : "E_SCHEMA_OTHER";
        }

        private readonly Func<string, Schema> schemaByFile;

        public Contract(Func<string, Schema> schemaByFile) { this.schemaByFile = schemaByFile; }

        /// <summary>Python's dict.get: the field, or JSON null when it is missing or the node isn't an object.</summary>
        public static JNode Get(JNode o, string key) { return o.Kind == JKind.Object && o.Has(key) ? o[key] : JNode.Null; }

        public static string Key(JNode v)
        {
            switch (v.Kind)
            {
                case JKind.String: return "s:" + v.Text;
                case JKind.Number: return "n:" + v.Num.ToString("R", CultureInfo.InvariantCulture);
                case JKind.Bool: return "b:" + v.Bool;
                case JKind.Null: return "null";
                default: return "?";
            }
        }

        /// <summary>Record type, schema and semantic rules for one parsed record.</summary>
        public List<Issue> CheckRecord(string name, JNode record)
        {
            var issues = new List<Issue>();
            if (record.Kind != JKind.Object) { issues.Add(I(name, "$", "E_RECORD_TYPE", "the file's value is not a JSON object")); return issues; }
            JNode rt = Get(record, "record_type"), ver = Get(record, "schema_version");
            string file = null;
            if (rt.Kind == JKind.String && ver.Kind == JKind.Number && ver.IsInteger)
                Files.TryGetValue(rt.Text + "|" + ver.Text, out file);
            if (file == "command-context.v2.json" && !Typed(record)) file = null;   // A4: format 2 is for a command typed on the headset
            if (file == null)
            {
                issues.Add(I(name, "$", "E_RECORD_TYPE", "not a supported record_type and schema_version"));
                return issues;
            }
            var errors = schemaByFile(file).Errors(record);
            if (errors.Count > 0)
            {
                foreach (var e in errors)
                {
                    string code;
                    if (!KeywordCodes.TryGetValue(e.Keyword, out code)) code = "E_SCHEMA_OTHER";
                    issues.Add(I(name, e.Path, code, e.Keyword));
                }
                return issues;
            }
            var r = new Rec(name, record);
            switch (rt.Text)
            {
                case "scene": SceneRules(r); break;
                case "command_context": CommandRules(r); break;
                default: MapRules(r); break;
            }
            return r.Issues;
        }

        /// <summary>All layers after parsing for (name, record) pairs: every issue, errors and warnings.</summary>
        public List<Issue> Validate(List<KeyValuePair<string, JNode>> named)
        {
            var issues = new List<Issue>();
            var passed = new List<KeyValuePair<string, JNode>>();
            foreach (var kv in named)
            {
                var found = CheckRecord(kv.Key, kv.Value);
                issues.AddRange(found);
                if (found.Count == 0) passed.Add(kv);
            }
            issues.AddRange(CheckCross(passed));
            return issues;
        }

        internal static Issue I(string file, string path, string code, string message)
        {
            return new Issue { File = file, Path = path, Code = code, Message = message };
        }

        private static double Norm(JNode v)
        {
            var sq = new List<double>();
            foreach (var x in v.Items) sq.Add(x.Num * x.Num);
            return Math.Sqrt(Py.Sum(sq));
        }

        private sealed class Rec
        {
            public readonly string File;
            public readonly JNode Record;
            public readonly List<Issue> Issues = new List<Issue>();
            public readonly Dictionary<string, string> Sources = new Dictionary<string, string>();
            public readonly HashSet<string> Assumptions = new HashSet<string>();
            private readonly Dictionary<string, HashSet<string>> knownKinds;
            private readonly Dictionary<string, HashSet<string>> unknownKinds;

            public Rec(string file, JNode record)
            {
                File = file;
                Record = record;
                JNode rt = Get(record, "record_type"), ver = Get(record, "schema_version");
                bool v3 = rt.Kind == JKind.String && rt.Text == "scene" && ver.Kind == JKind.Number && ver.Num == 3;
                knownKinds = v3 ? KnownSourceKindsV3 : KnownSourceKinds;
                bool typed = rt.Kind == JKind.String && rt.Text == "command_context" && ver.Kind == JKind.Number && ver.Num == 2;
                unknownKinds = typed ? UnknownSourceKindsTyped : UnknownSourceKinds;
                Numbers(record, "$");
                var sources = Get(record, "sources");
                if (sources.Kind == JKind.Array)
                    for (int i = 0; i < sources.Items.Count; i++)
                    {
                        string id = sources.Items[i]["source_id"].Text;
                        if (Sources.ContainsKey(id)) Add("$.sources[" + i + "].source_id", "E_DUPLICATE_REGISTRY", "source " + id + " repeated");
                        else Sources[id] = sources.Items[i]["kind"].Text;
                    }
                var assumptions = Get(record, "assumptions");
                if (assumptions.Kind == JKind.Array)
                    for (int i = 0; i < assumptions.Items.Count; i++)
                    {
                        string id = assumptions.Items[i]["assumption_id"].Text;
                        if (Assumptions.Contains(id)) Add("$.assumptions[" + i + "].assumption_id", "E_DUPLICATE_REGISTRY", "assumption " + id + " repeated");
                        else Assumptions.Add(id);
                    }
            }

            private void Numbers(JNode v, string path)
            {
                switch (v.Kind)
                {
                    case JKind.Number:
                        if (double.IsInfinity(v.Num) || double.IsNaN(v.Num)) Add(path, "E_NONFINITE", "number is too large for a double");
                        break;
                    case JKind.Object:
                        foreach (var f in v.Fields) Numbers(f.Value, path + "." + f.Key);
                        break;
                    case JKind.Array:
                        for (int i = 0; i < v.Items.Count; i++) Numbers(v.Items[i], path + "[" + i + "]");
                        break;
                }
            }

            public void Add(string path, string code, string message) { Issues.Add(I(File, path, code, message)); }

            public string SourceKind(string path, string sourceId)
            {
                string kind;
                if (!Sources.TryGetValue(sourceId, out kind)) { Add(path, "E_UNRESOLVED_SOURCE", "source " + sourceId + " is not in sources"); return null; }
                return kind;
            }

            public string References(string path, JNode w)
            {
                if (w["state"].Text == "known")
                {
                    var assumptions = w["evidence"]["assumptions"].Items;
                    for (int j = 0; j < assumptions.Count; j++)
                        if (!Assumptions.Contains(assumptions[j].Text))
                            Add(path + ".evidence.assumptions[" + j + "]", "E_UNRESOLVED_ASSUMPTION", "assumption " + assumptions[j].Text + " is not in assumptions");
                    return SourceKind(path + ".evidence.source", w["evidence"]["source"].Text);
                }
                return SourceKind(path + ".source", w["source"].Text);
            }

            public void Provenance(string path, JNode w, string kind)
            {
                if (kind == null) return;
                if (w["state"].Text == "known")
                {
                    string ev = w["evidence"]["kind"].Text;
                    if (!knownKinds[ev].Contains(kind)) Add(path + ".evidence", "E_EVIDENCE_SOURCE", ev + " evidence can't come from a " + kind + " source");
                }
                else
                {
                    var allowed = unknownKinds[w["reason"].Text];
                    if (allowed != null && !allowed.Contains(kind)) Add(path + ".source", "E_EVIDENCE_SOURCE", "reason " + w["reason"].Text + " can't come from a " + kind + " source");
                }
            }

            public void Quaternion(string path, JNode w)
            {
                if (w["state"].Text != "known") return;
                var q = w["value"].Items;
                string support = w["support"].Text;
                double n = Norm(w["value"]);
                if (Math.Abs(n - 1.0) > Tolerance) Add(path + ".value", "E_QUAT_NORM", "norm is not 1 within the tolerance");
                else if (support == "yaw_only" && (Math.Abs(q[0].Num) > Tolerance || Math.Abs(q[1].Num) > Tolerance))
                    Add(path + ".value", "E_QUAT_YAW_TILT", "x and y must be 0 for yaw_only");
                else if (support == "axis_aligned" && (Py.Max(new[] { Math.Abs(q[0].Num), Math.Abs(q[1].Num), Math.Abs(q[2].Num) }) > Tolerance
                                                       || Math.Abs(Math.Abs(q[3].Num) - 1.0) > Tolerance))
                    Add(path + ".value", "E_QUAT_AXIS_ALIGNED", "not the identity");
            }

            public void UnitVector(string path, JNode w)
            {
                if (w["state"].Text == "known" && Math.Abs(Norm(w["value"]) - 1.0) > Tolerance)
                    Add(path + ".value", "E_UNIT_VECTOR", "length is not 1 within the tolerance");
            }
        }

        private static List<KeyValuePair<string, JNode>> ObjectValues(JNode obj, string b)
        {
            JNode g = obj["geometry"], a = obj["attributes"], o = obj["observation"];
            return new List<KeyValuePair<string, JNode>>
            {
                new KeyValuePair<string, JNode>(b + ".category", obj["category"]),
                new KeyValuePair<string, JNode>(b + ".geometry.center_m", g["center_m"]),
                new KeyValuePair<string, JNode>(b + ".geometry.size_m", g["size_m"]),
                new KeyValuePair<string, JNode>(b + ".geometry.rotation_xyzw", g["rotation_xyzw"]),
                new KeyValuePair<string, JNode>(b + ".semantic_front", obj["semantic_front"]),
                new KeyValuePair<string, JNode>(b + ".attributes.colours", a["colours"]),
                new KeyValuePair<string, JNode>(b + ".observation.capture_time", o["capture_time"]),
                new KeyValuePair<string, JNode>(b + ".observation.detector_confidence", o["detector_confidence"]),
                new KeyValuePair<string, JNode>(b + ".observation.geometry_uncertainty", o["geometry_uncertainty"])
            };
        }

        /// <summary>The perceived profile's per-value rules (scene v3), as checks.py's _perceived.</summary>
        private static void Perceived(Rec r, JNode obj, string b, Dictionary<string, string> kinds)
        {
            foreach (var pw in ObjectValues(obj, b))
            {
                string kind;
                if (!kinds.TryGetValue(pw.Key, out kind) || kind == null) continue;   // the unresolved source is already reported
                var allowed = PerceivedFields[pw.Key.Substring(b.Length + 1)];
                JNode w = pw.Value;
                if (w["state"].Text == "known" && !allowed.Key.Contains(w["evidence"]["kind"].Text + "|" + kind))
                    r.Add(pw.Key, "E_PROFILE_PERCEIVED", "a perceived value can't be " + w["evidence"]["kind"].Text + " from a " + kind + " source");
                else if (w["state"].Text == "unknown" && !allowed.Value.Contains(w["reason"].Text))
                    r.Add(pw.Key, "E_PROFILE_PERCEIVED", "a perceived value can't be unknown for reason " + w["reason"].Text);
            }
            JNode views = obj["observation"]["views"], center = obj["geometry"]["center_m"];
            if (views["positions"].Num == 1 && center["state"].Text == "known" && center["evidence"]["assumptions"].Items.Count == 0)
                r.Add(b + ".geometry.center_m.evidence.assumptions", "E_PROFILE_PERCEIVED", "a centre placed from one position names its size prior's assumption");
        }

        /// <summary>Views and association agree, and possible duplicates list each other back (scene v3), as checks.py's _associations.</summary>
        private static void Associations(Rec r, List<JNode> objects)
        {
            var lists = new Dictionary<string, List<string>>();
            foreach (JNode obj in objects)   // a repeated ID keeps its last list, as a Python dict does
            {
                var dup = new List<string>();
                foreach (JNode x in obj["observation"]["association"]["may_duplicate"].Items) dup.Add(x.Text);
                lists[obj["object_id"].Text] = dup;
            }
            for (int i = 0; i < objects.Count; i++)
            {
                JNode obj = objects[i], views = obj["observation"]["views"], assoc = obj["observation"]["association"];
                string b = "$.objects[" + i + "].observation", oid = obj["object_id"].Text, status = assoc["status"].Text;
                var dup = assoc["may_duplicate"].Items;
                if (views["positions"].Num > views["keyframes"].Num) r.Add(b + ".views", "E_ASSOCIATION", "more positions than keyframes");
                if (status == "possible_duplicate" && dup.Count == 0)
                    r.Add(b + ".association.may_duplicate", "E_ASSOCIATION", "a possible duplicate names what it may duplicate");
                else if (status != "possible_duplicate" && dup.Count > 0)
                    r.Add(b + ".association.may_duplicate", "E_ASSOCIATION", "a " + status + " object names no possible duplicates");
                if (status == "multi_position" && views["positions"].Num < 2)
                    r.Add(b + ".association.status", "E_ASSOCIATION", "multi_position needs two or more positions");
                else if (status == "single_position" && views["positions"].Num != 1)
                    r.Add(b + ".association.status", "E_ASSOCIATION", "single_position means exactly one position");
                for (int j = 0; j < dup.Count; j++)
                {
                    string other = dup[j].Text, path = b + ".association.may_duplicate[" + j + "]";
                    if (other == oid) r.Add(path, "E_ASSOCIATION", "an object can't duplicate itself");
                    else if (!lists.ContainsKey(other)) r.Add(path, "E_ASSOCIATION", other + " is not an object of this scene");
                    else if (!lists[other].Contains(oid)) r.Add(path, "E_ASSOCIATION", other + " doesn't list " + oid + " back");
                }
            }
        }

        private static void SceneRules(Rec r)
        {
            JNode scene = r.Record;
            string profile = scene["evidence_profile"].Text;
            double version = scene["schema_version"].Num;
            if (scene["schema_version"].Num == 2)
            {
                JNode frame = scene["coordinate_frame"], conversion = frame["conversion"];
                string sid = conversion["source_id"].Text, kind;
                if (!r.Sources.TryGetValue(sid, out kind) || kind != "dataset")
                    r.Add("$.coordinate_frame.conversion.source_id", "E_FRAME_SOURCE", "conversion source is not a dataset source in sources");
                if (conversion["source_frame_id"].Text != frame["frame_id"].Text)
                    r.Add("$.coordinate_frame.conversion.source_frame_id", "E_FRAME_CONVERSION", "an identity conversion's source frame must be the scene frame");
            }
            if (version == 3)   // the headset's tracking space through the basis change (docs/frames.md)
            {
                string kind;
                if (!r.Sources.TryGetValue(scene["coordinate_frame"]["conversion"]["source_id"].Text, out kind) || kind != "sensor_capture")
                    r.Add("$.coordinate_frame.conversion.source_id", "E_FRAME_SOURCE", "conversion source is not a sensor_capture source in sources");
            }
            HashSet<string> objectKinds = version == 3 ? ObjectSourceKindsV3 : ObjectSourceKinds;
            var seen = new Dictionary<string, int>();
            var objects = scene["objects"].Items;
            for (int i = 0; i < objects.Count; i++)
            {
                JNode obj = objects[i];
                string b = "$.objects[" + i + "]", oid = obj["object_id"].Text;
                if (seen.ContainsKey(oid)) r.Add(b + ".object_id", "E_DUPLICATE_ID", oid + " is also objects[" + seen[oid] + "]");
                else seen[oid] = i;
                foreach (var ps in new[] { new KeyValuePair<string, string>(b + ".source_ref.source_id", obj["source_ref"]["source_id"].Text),
                                           new KeyValuePair<string, string>(b + ".observation.source", obj["observation"]["source"].Text) })
                {
                    string kind = r.SourceKind(ps.Key, ps.Value);
                    if (kind != null && !objectKinds.Contains(kind)) r.Add(ps.Key, "E_EVIDENCE_SOURCE", "objects come from a " + (version == 3 ? "sensor_capture source" : "dataset or fixture") + ", not a " + kind + " source");
                }
                var kinds = new Dictionary<string, string>();
                foreach (var pw in ObjectValues(obj, b))
                {
                    kinds[pw.Key] = r.References(pw.Key, pw.Value);
                    r.Provenance(pw.Key, pw.Value, kinds[pw.Key]);
                }
                r.Quaternion(b + ".geometry.rotation_xyzw", obj["geometry"]["rotation_xyzw"]);
                r.UnitVector(b + ".semantic_front", obj["semantic_front"]);
                if (profile == "restricted")
                {
                    foreach (var pw in new[] { new KeyValuePair<string, JNode>(b + ".geometry.rotation_xyzw", obj["geometry"]["rotation_xyzw"]),
                                               new KeyValuePair<string, JNode>(b + ".semantic_front", obj["semantic_front"]),
                                               new KeyValuePair<string, JNode>(b + ".attributes.colours", obj["attributes"]["colours"]) })
                        if (!(pw.Value["state"].Text == "unknown" && pw.Value["reason"].Text == "withheld_by_profile"))
                            r.Add(pw.Key, "E_PROFILE_WITHHELD", "the restricted profile withholds this field");
                    JNode size = obj["geometry"]["size_m"];
                    if (!((size["state"].Text == "known" && size["evidence"]["kind"].Text == "assumed")
                          || (size["state"].Text == "unknown" && size["reason"].Text == "no_permitted_prior")))
                        r.Add(b + ".geometry.size_m", "E_PROFILE_SIZE", "restricted sizes are a category prior or unknown");
                }
                else if (profile == "perceived")
                    Perceived(r, obj, b, kinds);
                else
                {
                    foreach (var pw in ObjectValues(obj, b))
                    {
                        JNode w = pw.Value;
                        if ((w["state"].Text == "known" && w["evidence"]["kind"].Text == "assumed")
                            || (w["state"].Text == "unknown" && (w["reason"].Text == "withheld_by_profile" || w["reason"].Text == "no_permitted_prior")))
                            r.Add(pw.Key, "E_PROFILE_ANNOTATED", "the annotated profile has no assumed values or profile unknowns");
                    }
                }
            }
            if (profile == "perceived") Associations(r, objects);
        }

        private static void CommandRules(Rec r)
        {
            JNode cmd = r.Record, pose = cmd["user_pose"];
            string kind = pose["pose_kind"].Text;
            if (!SchemaJsonEqual(pose["scene_revision"], cmd["scene_revision"]))
                r.Add("$.user_pose.scene_revision", "E_POSE_REVISION", "differs from the command's revision");
            if (kind == "quest_head") { r.Add("$.user_pose.pose_kind", "E_POSE_RESERVED", "quest_head is reserved for a later version"); return; }
            foreach (var k in new[] { "position_m", "rotation_xyzw", "heading_xy", "snapshot_time" })
            {
                string path = "$.user_pose." + k;
                JNode w = pose[k];
                string sourceKind = r.References(path, w);
                if (w["state"].Text == "unknown") r.Provenance(path, w, sourceKind);
                else if (kind == "none") r.Add(path, "E_POSE_NONE_KNOWN", "pose_kind none has no known components");
                else if (w["evidence"]["kind"].Text != "annotated" || (sourceKind != null && sourceKind != PoseSourceKind[kind]))
                    r.Add(path + ".evidence", "E_POSE_EVIDENCE", "a " + kind + " pose is annotated by a " + PoseSourceKind[kind] + " source in v1");
            }
            JNode heading = pose["heading_xy"];
            if (kind != "none" && heading["state"].Text == "known" && heading["heading_source"].Text != PoseHeadingSource[kind])
                r.Add("$.user_pose.heading_xy.heading_source", "E_HEADING_SOURCE", "the heading's source doesn't fit the pose kind");
            r.Quaternion("$.user_pose.rotation_xyzw", pose["rotation_xyzw"]);
            r.UnitVector("$.user_pose.heading_xy", heading);
        }

        private static bool SchemaJsonEqual(JNode a, JNode b) { return Schema.JsonEqual(a, b); }

        private static void MapRules(Rec r)
        {
            var vocabulary = new HashSet<string>();
            foreach (var x in r.Record["model_vocabulary"].Items) vocabulary.Add(x.Text);
            var seen = new Dictionary<string, int>();
            var entries = r.Record["entries"].Items;
            for (int i = 0; i < entries.Count; i++)
            {
                if (!vocabulary.Contains(entries[i]["model"].Text)) r.Add("$.entries[" + i + "].model", "E_MAP_VOCAB", "not in model_vocabulary");
                string label = entries[i]["source_label"].Text;
                if (seen.ContainsKey(label)) r.Add("$.entries[" + i + "].source_label", "E_MAP_DUPLICATE", "also entries[" + seen[label] + "]");
                else seen[label] = i;
            }
        }

        /// <summary>Rules between records that passed alone.</summary>
        public static List<Issue> CheckCross(List<KeyValuePair<string, JNode>> records)
        {
            var issues = new List<Issue>();
            var scenes = new Dictionary<string, KeyValuePair<string, JNode>>();
            var commands = new Dictionary<string, KeyValuePair<string, JNode>>();
            var maps = new Dictionary<string, KeyValuePair<string, JNode>>();
            var sceneOrder = new List<string>();
            var commandOrder = new List<string>();
            var byScene = new Dictionary<string, List<JNode>>();
            foreach (var kv in records)
            {
                JNode rec = kv.Value;
                string rt = rec["record_type"].Text, key;
                Dictionary<string, KeyValuePair<string, JNode>> table;
                if (rt == "scene") { key = Key(rec["scene_id"]) + "|" + Key(rec["scene_revision"]) + "|" + Key(rec["evidence_profile"]); table = scenes; }
                else if (rt == "command_context") { key = Key(rec["command_id"]); table = commands; }
                else { key = Key(rec["map_id"]); table = maps; }
                if (table.ContainsKey(key)) { issues.Add(I(kv.Key, "$", "E_DUPLICATE_RECORD", rt + " is also in " + table[key].Key)); continue; }
                table[key] = kv;
                if (rt == "scene")
                {
                    sceneOrder.Add(key);
                    string sid = rec["scene_id"].Text;
                    if (!byScene.ContainsKey(sid)) byScene[sid] = new List<JNode>();
                    byScene[sid].Add(rec);
                }
                else if (rt == "command_context") commandOrder.Add(key);
            }
            foreach (var key in commandOrder)
            {
                var kv = commands[key];
                JNode cmd = kv.Value;
                List<JNode> candidates;
                if (!byScene.TryGetValue(cmd["scene_id"].Text, out candidates)) candidates = new List<JNode>();
                var same = new List<JNode>();
                foreach (var s in candidates) if (Schema.JsonEqual(s["scene_revision"], cmd["scene_revision"])) same.Add(s);
                if (candidates.Count == 0) issues.Add(I(kv.Key, "$.scene_id", "W_NOT_CROSS_CHECKED", "scene not given"));
                else if (same.Count == 0) issues.Add(I(kv.Key, "$.scene_revision", "E_SCENE_MISMATCH", "scene given only at other revisions"));
                else
                {
                    bool mismatch = false;
                    foreach (var s in same) mismatch |= s["coordinate_frame"]["frame_id"].Text != cmd["user_pose"]["frame_id"].Text;
                    if (mismatch) issues.Add(I(kv.Key, "$.user_pose.frame_id", "E_FRAME_MISMATCH", "not the scene's frame"));
                }
            }
            foreach (var key in sceneOrder)
            {
                var kv = scenes[key];
                JNode scene = kv.Value;
                KeyValuePair<string, JNode> found;
                if (!maps.TryGetValue(Key(scene["category_map"]), out found))
                {
                    issues.Add(I(kv.Key, "$.category_map", "W_NOT_CROSS_CHECKED", "category map not given"));
                    continue;
                }
                var entries = new HashSet<string>();
                foreach (var e in found.Value["entries"].Items) entries.Add(e["source_label"].Text + "\u0000" + e["standard"].Text + "\u0000" + e["model"].Text);
                var objects = scene["objects"].Items;
                for (int i = 0; i < objects.Count; i++)
                {
                    JNode c = objects[i]["category"];
                    if (c["state"].Text != "known") continue;
                    string triple = objects[i]["source_ref"]["source_label"].Text + "\u0000" + c["value"]["standard"].Text + "\u0000" + c["value"]["model"].Text;
                    if (!entries.Contains(triple)) issues.Add(I(kv.Key, "$.objects[" + i + "].category", "E_CATEGORY_UNMAPPED", "not an entry of the scene's map"));
                }
            }
            return issues;
        }
    }
}
