using System;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using SecondEyes.Grounding.Parsing;
using SecondEyes.Grounding.Prompting;
using SecondEyes.Perception.Frames;
using SecondEyes.Perception.Localization;

namespace SecondEyes.Perception.AutoScan
{
    /// <summary>The automatic scan's settings (schemas/auto-scan-config.v1.json; docs/auto-scan.md).</summary>
    public sealed class AutoScanSettings
    {
        public string ConfigId;
        public double IntervalS, MinScore, SampleSpread, MaxRangeM, FloorYM, FloorMarginM, MinReachM, ReachFactor, ViewpointM, PublishIntervalS;
        public int MinFrames, MaxPointsPerObject, SaveImagesMax;

        public static AutoScanSettings From(JNode n)
        {
            return new AutoScanSettings
            {
                ConfigId = n["config_id"].Text, IntervalS = n["interval_s"].Num, MinScore = n["min_score"].Num,
                SampleSpread = n["sample_spread"].Num, MaxRangeM = n["max_range_m"].Num, FloorYM = n["floor_y_m"].Num,
                FloorMarginM = n["floor_margin_m"].Num, MinReachM = n["min_reach_m"].Num, ReachFactor = n["reach_factor"].Num,
                MinFrames = (int)n["min_frames"].Num, ViewpointM = n["viewpoint_m"].Num,
                MaxPointsPerObject = (int)n["max_points_per_object"].Num, PublishIntervalS = n["publish_interval_s"].Num,
                SaveImagesMax = (int)n["save_images_max"].Num
            };
        }
    }

    /// <summary>One depth sample of a detection: the pixel it was taken at, and what Meta's raycast returned there.</summary>
    public sealed class DepthSample
    {
        public double U, V;
        public string Status;   // the raycast's status in snake case: hit, no_hit, not_ready, hit_point_occluded, ...
        public Vec3? Point;     // in quest_tracking (Unity's axes), when the status is hit
    }

    public sealed class FrameDetection
    {
        public int ClassId;
        public string Label;
        public double Score, X1, Y1, X2, Y2;   // the box in the camera texture's pixels
        public readonly List<DepthSample> Samples = new List<DepthSample>();
    }

    /// <summary>
    /// One detector run while scanning, as one line of frames.jsonl: the camera's pose when its frame was taken, and each
    /// detection with its depth samples. The headset writes the line, and both it and the PC's replay process the line.
    /// </summary>
    public sealed class FrameRecord
    {
        public int Frame;
        public long TimeUs, Epoch;
        public bool Tracked;
        public string Snapshot;
        public Vec3 Position;   // the camera in quest_tracking, as Unity reports it
        public Quat Rotation;
        public readonly List<FrameDetection> Detections = new List<FrameDetection>();

        public JNode ToJson()
        {
            var dets = new List<JNode>();
            foreach (FrameDetection d in Detections)
            {
                var samples = new List<JNode>();
                foreach (DepthSample s in d.Samples)
                    samples.Add(JNode.Obj().Add("u", Localizer.Num(s.U)).Add("v", Localizer.Num(s.V)).Add("status", JNode.Str(s.Status))
                                     .Add("point", s.Point.HasValue ? Localizer.Vec(s.Point.Value) : JNode.Null));
                dets.Add(JNode.Obj().Add("class_id", Localizer.Int(d.ClassId)).Add("label", JNode.Str(d.Label)).Add("score", Localizer.Num(d.Score))
                              .Add("box_px", JNode.Arr(new List<JNode> { Localizer.Num(d.X1), Localizer.Num(d.Y1), Localizer.Num(d.X2), Localizer.Num(d.Y2) }))
                              .Add("samples", JNode.Arr(samples)));
            }
            return JNode.Obj().Add("record_type", JNode.Str("auto_scan_frame")).Add("frame", Localizer.Int(Frame)).Add("time_us", Localizer.Int(TimeUs))
                .Add("epoch", Localizer.Int(Epoch)).Add("tracked", new JNode { Kind = JKind.Bool, Bool = Tracked })
                .Add("snapshot", Snapshot == null ? JNode.Null : JNode.Str(Snapshot)).Add("position", Localizer.Vec(Position))
                .Add("rotation_xyzw", JNode.Arr(new List<JNode> { Localizer.Num(Rotation.X), Localizer.Num(Rotation.Y), Localizer.Num(Rotation.Z), Localizer.Num(Rotation.W) }))
                .Add("detections", JNode.Arr(dets));
        }

        public static FrameRecord FromJson(JNode n)
        {
            var p = n["position"].Items; var q = n["rotation_xyzw"].Items;
            var f = new FrameRecord
            {
                Frame = (int)n["frame"].Num, TimeUs = (long)n["time_us"].Num, Epoch = (long)n["epoch"].Num, Tracked = n["tracked"].Bool,
                Snapshot = n["snapshot"].Kind == JKind.Null ? null : n["snapshot"].Text,
                Position = new Vec3(p[0].Num, p[1].Num, p[2].Num), Rotation = new Quat(q[0].Num, q[1].Num, q[2].Num, q[3].Num)
            };
            foreach (JNode d in n["detections"].Items)
            {
                var b = d["box_px"].Items;
                var fd = new FrameDetection { ClassId = (int)d["class_id"].Num, Label = d["label"].Text, Score = d["score"].Num, X1 = b[0].Num, Y1 = b[1].Num, X2 = b[2].Num, Y2 = b[3].Num };
                foreach (JNode s in d["samples"].Items)
                {
                    Vec3? pt = null;
                    if (s["point"].Kind != JKind.Null) { var v = s["point"].Items; pt = new Vec3(v[0].Num, v[1].Num, v[2].Num); }
                    fd.Samples.Add(new DepthSample { U = s["u"].Num, V = s["v"].Num, Status = s["status"].Text, Point = pt });
                }
                f.Detections.Add(fd);
            }
            return f;
        }
    }

    /// <summary>What one frame did: each detection's fate, in the frame's own order, and whether the objects changed.</summary>
    public sealed class FrameOutcome
    {
        public List<string> Fates = new List<string>();
        public bool Changed;
    }

    /// <summary>
    /// The automatic scan (A4; docs/auto-scan.md), free of Unity: the same code runs on the Quest and in the PC's replay.
    /// Each frame's detections are taken in the order of their boxes; for each:
    /// 1. a label the category map doesn't know, or a score below min_score, is left out;
    /// 2. its depth samples (Meta's raycasts at the box's centre and four points sample_spread of its size away) that hit
    ///    a surface are sorted by distance from the camera, and the median one is its point; with no hit, it is left out;
    /// 3. a point beyond max_range_m, or lower than floor_margin_m below the floor, is left out;
    /// 4. the point joins the nearest object of its category within reach (the larger of min_reach_m and reach_factor
    ///    times the larger of the object's and the sighting's measured size), or starts a new object; an object's centre
    ///    is the per-axis median of its most recent points, its size the median of its sightings' sizes;
    /// 5. objects of one category that come within reach of each other merge;
    /// 6. an object seen in at least min_frames frames is in the scene, its views counting frames and the viewpoints
    ///    (camera positions viewpoint_m apart) it was seen from.
    /// The point is on the object's visible surface, so the scene names the depth_surface assumption.
    /// </summary>
    public sealed class AutoScan
    {
        public const string ScanSourceId = "scan", DepthAssumption = "depth_surface";

        private sealed class Obj
        {
            public int Id, MergedInto = -1;
            public string Standard, Model, Label;
            public readonly List<Vec3> Points = new List<Vec3>();
            public readonly List<double> Sizes = new List<double>();
            public readonly SortedSet<int> Frames = new SortedSet<int>();
            public readonly List<Vec3> Viewpoints = new List<Vec3>();
            public double BestScore, Size;
            public long LastUs;
            public Vec3 Centre;

            public void Add(Vec3 p, double size, int frame, Vec3 camera, double score, long timeUs, AutoScanSettings s)
            {
                Points.Add(p);
                Sizes.Add(size);
                Frames.Add(frame);
                AddViewpoint(camera, s);
                BestScore = Math.Max(BestScore, score);
                LastUs = Math.Max(LastUs, timeUs);
                Trim(s);
                Recompute();
            }

            public void Absorb(Obj b, AutoScanSettings s)
            {
                Points.AddRange(b.Points);
                Sizes.AddRange(b.Sizes);
                Frames.UnionWith(b.Frames);
                foreach (Vec3 v in b.Viewpoints) AddViewpoint(v, s);
                BestScore = Math.Max(BestScore, b.BestScore);
                LastUs = Math.Max(LastUs, b.LastUs);
                Trim(s);
                Recompute();
            }

            private void AddViewpoint(Vec3 camera, AutoScanSettings s)
            {
                if (!Viewpoints.Any(v => (v - camera).Norm < s.ViewpointM)) Viewpoints.Add(camera);
            }

            private void Trim(AutoScanSettings s)
            {
                if (Points.Count > s.MaxPointsPerObject) Points.RemoveRange(0, Points.Count - s.MaxPointsPerObject);
                if (Sizes.Count > s.MaxPointsPerObject) Sizes.RemoveRange(0, Sizes.Count - s.MaxPointsPerObject);
            }

            private void Recompute()
            {
                Centre = new Vec3(Median(Points.Select(x => x.X)), Median(Points.Select(x => x.Y)), Median(Points.Select(x => x.Z)));
                Size = Median(Sizes);
            }
        }

        private readonly string scanId, cameraId, detectorPackage, detectorSha256;
        private readonly double fx;
        private readonly AutoScanSettings s;
        private readonly JNode map, priors;
        private readonly Dictionary<string, JNode> entries = new Dictionary<string, JNode>(StringComparer.Ordinal);
        private readonly Dictionary<string, List<JNode>> sizes = new Dictionary<string, List<JNode>>(StringComparer.Ordinal);
        private readonly List<Obj> objects = new List<Obj>();
        private readonly List<JNode> fates = new List<JNode>();
        private readonly List<int> fateObject = new List<int>();
        private readonly SortedDictionary<string, int> leftOut = new SortedDictionary<string, int>(StringComparer.Ordinal);
        private long epoch = long.MinValue;

        public int Frames { get; private set; }
        public int Sightings { get; private set; }
        public bool Changed { get; private set; }
        public string ScanId => scanId;
        public AutoScanSettings Settings => s;
        public int Confirmed => objects.Count(o => o.MergedInto < 0 && o.Frames.Count >= s.MinFrames);
        public int Tentative => objects.Count(o => o.MergedInto < 0 && o.Frames.Count < s.MinFrames);

        public AutoScan(string scanId, AutoScanSettings settings, JNode categoryMap, JNode sizePriors, string detectorPackage,
                        string detectorSha256, string cameraId, double fx)
        {
            this.scanId = scanId; s = settings; map = categoryMap; priors = sizePriors;
            this.detectorPackage = detectorPackage; this.detectorSha256 = detectorSha256; this.cameraId = cameraId; this.fx = fx;
            foreach (JNode e in map["entries"].Items) entries[e["source_label"].Text] = e;
            foreach (JNode e in priors["entries"].Items)
            {
                if (!sizes.ContainsKey(e["category"].Text)) sizes[e["category"].Text] = new List<JNode>();
                sizes[e["category"].Text].Add(e);
            }
        }

        /// <summary>The first line of frames.jsonl: what the PC's replay needs besides the perception folder.</summary>
        public JNode Header(string folderSha256, int widthPx, int heightPx, double fy, double cx, double cy, int detectorInputPx)
        {
            return JNode.Obj().Add("record_type", JNode.Str("auto_scan_header")).Add("scan_id", JNode.Str(scanId)).Add("config_id", JNode.Str(s.ConfigId))
                .Add("category_map", JNode.Str(map["map_id"].Text)).Add("size_priors", JNode.Str(priors["priors_id"].Text))
                .Add("detector_package", JNode.Str(detectorPackage)).Add("detector_sha256", JNode.Str(detectorSha256)).Add("camera_id", JNode.Str(cameraId))
                .Add("intrinsics", JNode.Arr(new List<JNode> { Localizer.Num(fx), Localizer.Num(fy), Localizer.Num(cx), Localizer.Num(cy) }))
                .Add("texture_px", JNode.Arr(new List<JNode> { Localizer.Int(widthPx), Localizer.Int(heightPx) })).Add("detector_input_px", Localizer.Int(detectorInputPx))
                .Add("perception_folder_sha256", JNode.Str(folderSha256 ?? ""));
        }

        /// <summary>A scan rebuilt from a frames.jsonl header (the PC's replay).</summary>
        public static AutoScan FromHeader(JNode header, AutoScanSettings settings, JNode categoryMap, JNode sizePriors)
        {
            return new AutoScan(header["scan_id"].Text, settings, categoryMap, sizePriors, header["detector_package"].Text,
                                header["detector_sha256"].Text, header["camera_id"].Text, header["intrinsics"].Items[0].Num);
        }

        /// <summary>Where to sample depth in a box: its centre first, then four points spread of its size away.</summary>
        public static List<double[]> SamplePixels(double x1, double y1, double x2, double y2, double spread)
        {
            double cx = (x1 + x2) / 2, cy = (y1 + y2) / 2, dx = spread * (x2 - x1), dy = spread * (y2 - y1);
            var list = new List<double[]> { new[] { cx, cy } };
            if (spread > 0)
            {
                list.Add(new[] { cx - dx, cy }); list.Add(new[] { cx + dx, cy });
                list.Add(new[] { cx, cy - dy }); list.Add(new[] { cx, cy + dy });
            }
            return list;
        }

        /// <summary>The direction through pixel (u, v), in the box's coordinates, in the camera's Unity axes (x right,
        /// y up, z forward): the pinhole with pixel centres at integers, as the localization uses it.</summary>
        public static Vec3 CameraDirection(double u, double v, double fx, double fy, double cx, double cy)
        {
            return new Vec3((u - 0.5 - cx) / fx, -(v - 0.5 - cy) / fy, 1.0).Normalized();
        }

        /// <summary>A point in quest_tracking (Unity: y up) in the scene frame (z up), the basis change of frames.md.</summary>
        public static Vec3 ToScene(Vec3 unity) { return new Vec3(unity.X, unity.Z, unity.Y); }
        public static Vec3 ToUnity(Vec3 scene) { return new Vec3(scene.X, scene.Z, scene.Y); }

        public FrameOutcome Add(FrameRecord input)
        {
            FrameRecord f = FrameRecord.FromJson(input.ToJson());   // exactly what the log holds, so a replay can't differ
            var outcome = new FrameOutcome();
            Frames++;
            if (epoch == long.MinValue) epoch = f.Epoch;
            var order = f.Detections.Select((d, i) => new KeyValuePair<int, FrameDetection>(i, d))
                                    .OrderBy(x => x.Value.X1).ThenBy(x => x.Value.Y1).ThenBy(x => x.Value.ClassId).ThenBy(x => x.Key).ToList();
            Vec3 camera = ToScene(f.Position);
            var byIndex = new string[f.Detections.Count];
            foreach (var kv in order)
            {
                FrameDetection d = kv.Value;
                string fate;
                int obj = -1, hitCount = 0;
                JNode entry;
                if (!f.Tracked) fate = "untracked";
                else if (f.Epoch != epoch) fate = "tracking_reset";
                else if (!entries.TryGetValue(d.Label, out entry)) fate = "unmapped_label";
                else if (d.Score < s.MinScore) fate = "below_min_score";
                else
                {
                    var hits = d.Samples.Where(x => x.Status == "hit" && x.Point.HasValue)
                                        .Select(x => new KeyValuePair<double, Vec3>((x.Point.Value - f.Position).Norm, x.Point.Value))
                                        .OrderBy(x => x.Key).ToList();
                    hitCount = hits.Count;
                    if (hits.Count == 0) fate = "no_depth_hit";
                    else
                    {
                        var pick = hits[(hits.Count - 1) / 2];   // the median distance, the nearer of two middles
                        if (pick.Key > s.MaxRangeM) fate = "too_far";
                        else if (pick.Value.Y < s.FloorYM - s.FloorMarginM) fate = "below_floor";
                        else
                        {
                            double size = Math.Max(d.X2 - d.X1, d.Y2 - d.Y1) * pick.Key / fx;
                            obj = Join(entry, d.Label, ToScene(pick.Value), camera, size, f.Frame, d.Score, f.TimeUs);
                            fate = "kept";
                            Sightings++;
                            outcome.Changed = true;
                        }
                    }
                }
                if (fate != "kept") leftOut[fate] = (leftOut.TryGetValue(fate, out int c) ? c : 0) + 1;
                JNode entryNode = JNode.Obj().Add("keyframe", Localizer.Int(f.Frame)).Add("detection", Localizer.Int(kv.Key)).Add("label", JNode.Str(d.Label))
                                             .Add("score", Localizer.Num(d.Score)).Add("depth_hits", Localizer.Int(hitCount));
                if (fate != "kept") entryNode.Add("fate", JNode.Str(fate));   // a kept one's fate (its object, or tentative) is known only at a snapshot
                fates.Add(entryNode);
                fateObject.Add(obj);
                byIndex[kv.Key] = fate;
            }
            outcome.Fates = byIndex.ToList();
            if (outcome.Changed)
            {
                MergePass();
                Changed = true;
            }
            return outcome;
        }

        private double Reach(double a, double b) { return Math.Max(s.MinReachM, s.ReachFactor * Math.Max(a, b)); }

        private int Join(JNode entry, string label, Vec3 p, Vec3 camera, double size, int frame, double score, long timeUs)
        {
            string standard = entry["standard"].Text;
            Obj best = null;
            double bestD = double.MaxValue;
            foreach (Obj o in objects)
            {
                if (o.MergedInto >= 0 || o.Standard != standard) continue;
                double dist = (o.Centre - p).Norm;
                if (dist <= Reach(o.Size, size) && dist < bestD) { best = o; bestD = dist; }
            }
            if (best == null)
            {
                best = new Obj { Id = objects.Count, Standard = standard, Model = entry["model"].Text, Label = label };
                objects.Add(best);
            }
            best.Add(p, size, frame, camera, score, timeUs, s);
            return best.Id;
        }

        private void MergePass()
        {
            bool again = true;
            while (again)
            {
                again = false;
                var live = objects.Where(o => o.MergedInto < 0).ToList();
                for (int a = 0; a < live.Count && !again; a++)
                    for (int b = a + 1; b < live.Count && !again; b++)
                    {
                        Obj A = live[a], B = live[b];
                        if (A.Standard != B.Standard || (A.Centre - B.Centre).Norm > Reach(A.Size, B.Size)) continue;
                        A.Absorb(B, s);
                        B.MergedInto = A.Id;
                        again = true;
                    }
            }
        }

        private Obj Resolve(int id)
        {
            Obj o = objects[id];
            while (o.MergedInto >= 0) o = objects[o.MergedInto];
            return o;
        }

        /// <summary>The objects in the scene so far, each with its category and its centre in quest_tracking (for labels in view).</summary>
        public List<KeyValuePair<string, Vec3>> ConfirmedInTracking()
        {
            return objects.Where(o => o.MergedInto < 0 && o.Frames.Count >= s.MinFrames)
                          .Select(o => new KeyValuePair<string, Vec3>(o.Standard, ToUnity(o.Centre))).ToList();
        }

        /// <summary>The scene of the confirmed objects (scene v3, perceived) and the trace of how it was made.</summary>
        public LocalizationOutput Snapshot(int revision)
        {
            Changed = false;
            var confirmed = objects.Where(o => o.MergedInto < 0 && o.Frames.Count >= s.MinFrames)
                .OrderBy(o => o.Standard, StringComparer.Ordinal).ThenBy(o => Math.Round(o.Centre.X, 4)).ThenBy(o => Math.Round(o.Centre.Y, 4))
                .ThenBy(o => Math.Round(o.Centre.Z, 4)).ThenBy(o => o.Id).ToList();
            var ids = new Dictionary<int, string>();
            for (int n = 0; n < confirmed.Count; n++) ids[confirmed[n].Id] = "obj_" + (n + 1).ToString("D3", CultureInfo.InvariantCulture);
            string priorsId = priors["priors_id"].Text;
            bool usedSize = false;
            var objs = new List<JNode>();
            foreach (Obj o in confirmed)
            {
                List<JNode> sz;
                bool oneSize = sizes.TryGetValue(o.Standard, out sz) && sz.Count == 1;   // two sizes: neither is claimed
                usedSize |= oneSize;
                JNode size = oneSize
                    ? Localizer.Known(JNode.Arr(sz[0]["size_m"].Items.Select(x => Localizer.Num(x.Num)).ToList()), "assumed", priorsId, new List<JNode> { JNode.Str(Localizer.SizeAssumption) })
                    : Localizer.Unknown("no_permitted_prior", priorsId);
                objs.Add(JNode.Obj()
                    .Add("object_id", JNode.Str(ids[o.Id]))
                    .Add("source_ref", JNode.Obj().Add("source_id", JNode.Str(ScanSourceId)).Add("source_object_id", JNode.Str("track_" + o.Id.ToString(CultureInfo.InvariantCulture)))
                        .Add("source_label", JNode.Str(o.Label)))
                    .Add("category", Localizer.Known(JNode.Obj().Add("standard", JNode.Str(o.Standard)).Add("model", JNode.Str(o.Model)), "inferred", ScanSourceId, new List<JNode>()))
                    .Add("geometry", JNode.Obj().Add("center_m", Localizer.Known(Localizer.Vec(o.Centre), "inferred", ScanSourceId, new List<JNode> { JNode.Str(DepthAssumption) }))
                        .Add("size_m", size).Add("rotation_xyzw", Localizer.Unknown("not_estimated", ScanSourceId)))
                    .Add("semantic_front", Localizer.Unknown("not_estimated", ScanSourceId))
                    .Add("attributes", JNode.Obj().Add("colours", Localizer.Unknown("not_estimated", ScanSourceId)))
                    .Add("observation", JNode.Obj()
                        .Add("source", JNode.Str(ScanSourceId))
                        .Add("capture_time", Localizer.Known(JNode.Obj().Add("clock", JNode.Str("headset_mono")).Add("value_us", Localizer.Int(o.LastUs)), "measured", ScanSourceId, new List<JNode>()))
                        .Add("detector_confidence", Localizer.Known(Localizer.Num(o.BestScore), "inferred", ScanSourceId, new List<JNode>()))
                        .Add("geometry_uncertainty", Localizer.Unknown("not_estimated", ScanSourceId))
                        .Add("views", JNode.Obj().Add("keyframes", Localizer.Int(o.Frames.Count)).Add("positions", Localizer.Int(o.Viewpoints.Count)))
                        .Add("association", JNode.Obj().Add("status", JNode.Str(o.Viewpoints.Count >= 2 ? "multi_position" : "single_position"))
                            .Add("may_duplicate", JNode.Arr(new List<JNode>())))));
            }
            var assumptions = new List<JNode> { JNode.Obj().Add("assumption_id", JNode.Str(DepthAssumption))
                .Add("description", JNode.Str("The object's position is the median of depth points on its visible surface, so it lies toward the camera from its true centre by up to half its depth")) };
            if (usedSize) assumptions.Add(JNode.Obj().Add("assumption_id", JNode.Str(Localizer.SizeAssumption))
                .Add("description", JNode.Str("The object's size is its category's measured size in the size priors (" + priorsId + ")")));
            long ep = epoch == long.MinValue ? 0 : epoch;
            JNode scene = JNode.Obj()
                .Add("schema_version", Localizer.Int(3)).Add("record_type", JNode.Str("scene")).Add("scene_id", JNode.Str(scanId))
                .Add("scene_revision", Localizer.Int(revision)).Add("evidence_profile", JNode.Str("perceived")).Add("category_map", JNode.Str(map["map_id"].Text))
                .Add("coordinate_frame", JNode.Obj().Add("frame_id", JNode.Str("scene")).Add("units", JNode.Str("m")).Add("handedness", JNode.Str("right"))
                    .Add("up_axis", JNode.Str("+z")).Add("origin", JNode.Str("the headset's tracking origin in epoch " + ep.ToString(CultureInfo.InvariantCulture) + ", re-expressed through the basis change (docs/frames.md)"))
                    .Add("conversion", JNode.Obj().Add("kind", JNode.Str("quest_tracking_basis")).Add("source_id", JNode.Str(ScanSourceId))
                        .Add("source_frame_id", JNode.Str("quest_tracking")).Add("tracking_epoch", Localizer.Int(ep))
                        .Add("source_to_scene", JNode.Arr(new List<JNode> { Localizer.Row(1, 0, 0, 0), Localizer.Row(0, 0, 1, 0), Localizer.Row(0, 1, 0, 0), Localizer.Row(0, 0, 0, 1) }))))
                .Add("sources", JNode.Arr(new List<JNode> {
                    JNode.Obj().Add("source_id", JNode.Str(ScanSourceId)).Add("kind", JNode.Str("sensor_capture"))
                        .Add("description", JNode.Str("Auto scan " + scanId + ": camera " + cameraId + ", detector " + detectorPackage + " ("
                                                      + (detectorSha256.Length >= 12 ? detectorSha256.Substring(0, 12) : detectorSha256)
                                                      + "), depth from Meta's environment raycasts, " + s.ConfigId))
                        .Add("release", JNode.Str(scanId)),
                    JNode.Obj().Add("source_id", JNode.Str(priorsId)).Add("kind", JNode.Str("prior_table"))
                        .Add("description", JNode.Str(priors["description"].Text)).Add("release", JNode.Str(priors["release"].Text)) }))
                .Add("assumptions", JNode.Arr(assumptions))
                .Add("objects", JNode.Arr(objs));
            var dets = new List<JNode>();
            for (int i = 0; i < fates.Count; i++)
            {
                JNode d = JParse.Parse(PyJson.Canonical(fates[i]));
                if (fateObject[i] >= 0)
                {
                    Obj o = Resolve(fateObject[i]);
                    string id;
                    if (ids.TryGetValue(o.Id, out id)) d.Add("fate", JNode.Str("object")).Add("object_id", JNode.Str(id));
                    else d.Add("fate", JNode.Str("tentative"));
                }
                dets.Add(d);
            }
            var left = JNode.Obj();
            foreach (var kv in leftOut) left.Add(kv.Key, Localizer.Int(kv.Value));
            JNode trace = JNode.Obj()
                .Add("record_type", JNode.Str("auto_scan_trace")).Add("capture_id", JNode.Str(scanId)).Add("config_id", JNode.Str(s.ConfigId))
                .Add("scene_revision", Localizer.Int(revision)).Add("frames", Localizer.Int(Frames)).Add("sightings", Localizer.Int(Sightings))
                .Add("left_out", left).Add("tentative", Localizer.Int(Tentative))
                .Add("objects", JNode.Arr(confirmed.Select(o => JNode.Obj().Add("object_id", JNode.Str(ids[o.Id])).Add("track", Localizer.Int(o.Id))
                    .Add("frames", Localizer.Int(o.Frames.Count)).Add("viewpoints", Localizer.Int(o.Viewpoints.Count)).Add("size_m", Localizer.Num(o.Size))).ToList()))
                .Add("detections", JNode.Arr(dets));
            return new LocalizationOutput { Scene = scene, Trace = trace };
        }

        private static double Median(IEnumerable<double> xs)
        {
            var a = xs.OrderBy(x => x).ToList();
            if (a.Count == 0) return 0;
            int m = a.Count / 2;
            return a.Count % 2 == 1 ? a[m] : (a[m - 1] + a[m]) / 2;
        }
    }
}
