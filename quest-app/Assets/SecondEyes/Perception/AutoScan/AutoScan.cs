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
    /// <summary>The automatic scan's settings (schemas/auto-scan-config.v2.json, or v1 without Meta's room; docs/auto-scan.md).</summary>
    public sealed class AutoScanSettings
    {
        public string ConfigId;
        public double IntervalS, MinScore, SampleSpread, MaxRangeM, FloorYM, FloorMarginM, MinReachM, ReachFactor, ViewpointM, PublishIntervalS;
        public int MinFrames, MaxPointsPerObject, SaveImagesMax;
        /// <summary>v2: when to ask for Space Setup, how close counts as inside a box or the room, the votes an OTHER box
        /// needs, and the categories whose sightings vote for boxes. A v1 file: never, and no category votes.</summary>
        public string SpaceSetup = "never";
        public double BoxMarginM = 0.1, RoomMarginM = 0.25;
        public int MinBoxVotes = 2;
        public HashSet<string> BoxCategories = new HashSet<string>(StringComparer.Ordinal);
        /// <summary>v3: every object a 3D box. Depth sampled on a sample_grid by sample_grid grid in each detection; a
        /// detection's points within depth_window_m of its median are its surface; points thinned to one per
        /// point_voxel_m, at most box_points_max per object; box sides trimmed by box_trim at either end; boxes of
        /// floor_categories reach the floor; furniture resting on a box within top_allowance_m above it is that box;
        /// box_completion grows a box's hidden side to its size prior. A v1 or v2 file: no boxes.</summary>
        public bool Boxes;
        public int SampleGrid, BoxPointsMax = 800;
        public double DepthWindowM = 0.5, PointVoxelM = 0.03, BoxTrim = 0.03, TopAllowanceM = 0;
        public bool BoxCompletion;
        public HashSet<string> FloorCategories = new HashSet<string>(StringComparer.Ordinal);

        public static AutoScanSettings From(JNode n)
        {
            var s = new AutoScanSettings
            {
                ConfigId = n["config_id"].Text, IntervalS = n["interval_s"].Num, MinScore = n["min_score"].Num,
                SampleSpread = n["sample_spread"].Num, MaxRangeM = n["max_range_m"].Num, FloorYM = n["floor_y_m"].Num,
                FloorMarginM = n["floor_margin_m"].Num, MinReachM = n["min_reach_m"].Num, ReachFactor = n["reach_factor"].Num,
                MinFrames = (int)n["min_frames"].Num, ViewpointM = n["viewpoint_m"].Num,
                MaxPointsPerObject = (int)n["max_points_per_object"].Num, PublishIntervalS = n["publish_interval_s"].Num,
                SaveImagesMax = (int)n["save_images_max"].Num
            };
            if (n.Has("space_setup"))
            {
                s.SpaceSetup = n["space_setup"].Text;
                s.BoxMarginM = n["box_margin_m"].Num;
                s.RoomMarginM = n["room_margin_m"].Num;
                s.MinBoxVotes = (int)n["min_box_votes"].Num;
                s.BoxCategories = new HashSet<string>(n["box_categories"].Items.Select(x => x.Text), StringComparer.Ordinal);
            }
            if (n.Has("sample_grid"))
            {
                s.Boxes = true;
                s.SampleGrid = (int)n["sample_grid"].Num;
                s.DepthWindowM = n["depth_window_m"].Num;
                s.PointVoxelM = n["point_voxel_m"].Num;
                s.BoxPointsMax = (int)n["box_points_max"].Num;
                s.BoxTrim = n["box_trim"].Num;
                s.TopAllowanceM = n["top_allowance_m"].Num;
                s.BoxCompletion = n["box_completion"].Bool;
                s.FloorCategories = new HashSet<string>(n["floor_categories"].Items.Select(x => x.Text), StringComparer.Ordinal);
            }
            return s;
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
        public const string ScanSourceId = "scan", DepthAssumption = "depth_surface", RoomSourceId = "meta_room", VolumeAssumption = "meta_volume",
                            BoxAssumption = "box_centre", CompletionAssumption = "box_completion";

        /// <summary>A box to draw: its label ("?" when unnamed), who named it ("scan", "meta", "detector" or
        /// "unnamed"), and the box in quest_tracking.</summary>
        public sealed class ShownBox { public string Label, Kind; public Box3 Box; }

        /// <summary>What the scan's sightings say about one of Meta's boxes.</summary>
        private sealed class BoxVotes
        {
            public readonly SortedDictionary<string, int> Votes = new SortedDictionary<string, int>(StringComparer.Ordinal);
            public readonly SortedSet<int> Frames = new SortedSet<int>();
            public readonly List<Vec3> Viewpoints = new List<Vec3>();
            public double BestScore;
            public long LastUs;
        }

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
            // v3: the category votes, the surface points (quest_tracking), their voxel, and the box fitted to them.
            public readonly SortedDictionary<string, int> Votes = new SortedDictionary<string, int>(StringComparer.Ordinal);
            public readonly List<Vec3> Surface = new List<Vec3>();
            public double Voxel;
            public Box3 Box;

            public void AddBox(string standard, List<Vec3> patch, Vec3 p, double size, int frame, Vec3 camera, double score, long timeUs, AutoScanSettings s)
            {
                Votes[standard] = (Votes.TryGetValue(standard, out int n) ? n : 0) + 1;
                Surface.AddRange(patch);
                Add(p, size, frame, camera, score, timeUs, s);
            }

            public void AbsorbBox(Obj b, AutoScanSettings s)
            {
                foreach (var kv in b.Votes) Votes[kv.Key] = (Votes.TryGetValue(kv.Key, out int n) ? n : 0) + kv.Value;
                Surface.AddRange(b.Surface);
                Absorb(b, s);
            }

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
        private readonly List<int> fateBox = new List<int>();
        private RoomModel room = RoomModel.None("not loaded");
        private List<Box3> uprights = new List<Box3>();
        private readonly List<BoxVotes> boxVotes = new List<BoxVotes>();
        private readonly SortedDictionary<string, int> leftOut = new SortedDictionary<string, int>(StringComparer.Ordinal);
        private long epoch = long.MinValue;

        public int Frames { get; private set; }
        public int Sightings { get; private set; }
        public bool Changed { get; private set; }
        public string ScanId => scanId;
        public AutoScanSettings Settings => s;
        public int Confirmed => objects.Count(o => o.MergedInto < 0 && o.Frames.Count >= s.MinFrames) + (room.Loaded ? room.Boxes.Count : 0);
        public RoomModel Room => room;

        /// <summary>Meta's room, applied to every frame after it. The scan keeps it exactly as frames.jsonl holds it.</summary>
        public void SetRoom(RoomModel model)
        {
            room = RoomModel.FromJson(JParse.Parse(PyJson.Canonical(model.ToJson())));
            boxVotes.Clear();
            foreach (RoomBox b in room.Boxes) boxVotes.Add(new BoxVotes());
            uprights = room.Boxes.Select(Boxes.Upright).ToList();
            Changed = true;
        }
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

        /// <summary>Where depth is sampled in a detection's box: v3, a grid of sample_grid by sample_grid points at the
        /// centres of equal cells; before, the centre and four points sample_spread of the box away.</summary>
        public static List<double[]> SamplePixels(double x1, double y1, double x2, double y2, AutoScanSettings s)
        {
            if (!s.Boxes) return SamplePixels(x1, y1, x2, y2, s.SampleSpread);
            var list = new List<double[]>();
            int n = s.SampleGrid;
            for (int j = 0; j < n; j++)
                for (int i = 0; i < n; i++)
                    list.Add(new[] { x1 + (i + 0.5) / n * (x2 - x1), y1 + (j + 0.5) / n * (y2 - y1) });
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
                int obj = -1, box = -1, hitCount = 0;
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
                        else if (!room.InRoom(pick.Value, s.RoomMarginM)) fate = "outside_room";
                        else
                        {
                            string std = entry["standard"].Text;
                            RoomBox inside = s.Boxes ? MetaBoxFor(std, pick.Value) : s.BoxCategories.Contains(std) ? room.BoxAt(pick.Value, s.BoxMarginM) : null;
                            if (inside != null)
                            {
                                box = room.Boxes.IndexOf(inside);   // furniture seen inside one of Meta's boxes votes for it
                                Vote(boxVotes[box], entry["standard"].Text, f.Frame, camera, d.Score, f.TimeUs);
                            }
                            else
                            {
                                double size = Math.Max(d.X2 - d.X1, d.Y2 - d.Y1) * pick.Key / fx;
                                obj = s.Boxes ? JoinBox(entry, d.Label, pick.Value, hits.Where(h => Math.Abs(h.Key - pick.Key) <= s.DepthWindowM).Select(h => h.Value).ToList(),
                                                        camera, size, f.Frame, d.Score, f.TimeUs)
                                              : Join(entry, d.Label, ToScene(pick.Value), camera, size, f.Frame, d.Score, f.TimeUs);
                            }
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
                fateBox.Add(box);
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

        private void Vote(BoxVotes v, string standard, int frame, Vec3 camera, double score, long timeUs)
        {
            v.Votes[standard] = (v.Votes.TryGetValue(standard, out int n) ? n : 0) + 1;
            v.Frames.Add(frame);
            if (!v.Viewpoints.Any(p => (p - camera).Norm < s.ViewpointM)) v.Viewpoints.Add(camera);
            v.BestScore = Math.Max(v.BestScore, score);
            v.LastUs = Math.Max(v.LastUs, timeUs);
        }

        /// <summary>A box's name: Meta's when the map has it (Meta's name wins), else the detector's leading category
        /// with at least min_box_votes sightings (ties: the category that sorts first), else none. Returns the map entry,
        /// and who named it: "meta", "detector" or "none".</summary>
        private JNode BoxName(int i, out string by)
        {
            JNode e;
            if (entries.TryGetValue("meta:" + room.Boxes[i].MetaLabel, out e)) { by = "meta"; return e; }
            var lead = boxVotes[i].Votes.OrderByDescending(kv => kv.Value).ThenBy(kv => kv.Key, StringComparer.Ordinal).FirstOrDefault();
            if (lead.Key != null && lead.Value >= s.MinBoxVotes)
            {
                e = EntryFor(lead.Key);
                by = "detector";
                return e;
            }
            by = "none";
            return null;
        }

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

        /// <summary>v3: the sighting joins the object whose box it is inside (furniture joins furniture, whatever the
        /// label, and counts as inside when resting on it), the smallest when several; else the nearest object of its
        /// category within reach; else it starts one. The object's box is fitted again.</summary>
        private int JoinBox(JNode entry, string label, Vec3 pick, List<Vec3> patch, Vec3 camera, double size, int frame, double score, long timeUs)
        {
            string standard = entry["standard"].Text;
            bool furniture = s.BoxCategories.Contains(standard);
            Obj best = null;
            foreach (Obj o in objects)
            {
                if (o.MergedInto >= 0 || o.Box == null || !(o.Standard == standard || (furniture && s.BoxCategories.Contains(o.Standard)))) continue;
                if (o.Box.Outside(pick, furniture ? s.TopAllowanceM : 0) > s.BoxMarginM) continue;
                if (best == null || o.Box.Volume < best.Box.Volume) best = o;
            }
            if (best == null)
            {
                Vec3 p = ToScene(pick);
                double bestD = double.MaxValue;
                foreach (Obj o in objects)
                {
                    if (o.MergedInto >= 0 || o.Standard != standard) continue;
                    double dist = (o.Centre - p).Norm;
                    if (dist <= Reach(o.Size, size) && dist < bestD) { best = o; bestD = dist; }
                }
            }
            if (best == null)
            {
                best = new Obj { Id = objects.Count, Standard = standard, Model = entry["model"].Text, Label = label };
                objects.Add(best);
            }
            best.AddBox(standard, patch.Count > 0 ? patch : new List<Vec3> { pick }, ToScene(pick), size, frame, camera, score, timeUs, s);
            Refit(best);
            return best.Id;
        }

        /// <summary>v3: the object's category is its leading vote (ties to the category that sorts first); its box is
        /// fitted to its thinned surface points and completed to its size prior; its centre is the box's.</summary>
        private void Refit(Obj o)
        {
            string lead = o.Votes.OrderByDescending(kv => kv.Value).ThenBy(kv => kv.Key, StringComparer.Ordinal).First().Key;
            if (lead != o.Standard)
            {
                JNode e = EntryFor(lead);
                o.Standard = lead; o.Model = e["model"].Text; o.Label = e["source_label"].Text;
            }
            o.Voxel = Boxes.Thin(o.Surface, Math.Max(o.Voxel, s.PointVoxelM), s.BoxPointsMax);
            Box3 b = Boxes.Fit(o.Surface, s.BoxTrim, s.FloorCategories.Contains(o.Standard), room.Loaded ? room.FloorY : s.FloorYM);
            List<JNode> pri;
            if (s.BoxCompletion && sizes.TryGetValue(o.Standard, out pri))
                b = Boxes.Complete(b, pri.Select(x => x["size_m"].Items.Select(v => v.Num).ToArray()).ToList(), o.Viewpoints.Select(ToUnity).ToList());
            o.Box = b;
            o.Centre = ToScene(b.Centre);
            o.Size = Math.Max(b.Size.X, b.Size.Z);
        }

        /// <summary>v3: two objects are one when they may be (the same category, or both furniture) and either's centre
        /// is inside the other's box, or, of one category, their centres are within reach.</summary>
        private bool Overlap(Obj a, Obj b)
        {
            bool furniture = s.BoxCategories.Contains(a.Standard) && s.BoxCategories.Contains(b.Standard);
            if (a.Standard != b.Standard && !furniture) return false;
            double top = furniture ? s.TopAllowanceM : 0;
            if (a.Box != null && b.Box != null && (a.Box.Outside(b.Box.Centre, top) <= s.BoxMarginM || b.Box.Outside(a.Box.Centre, top) <= s.BoxMarginM)) return true;
            return a.Standard == b.Standard && (a.Centre - b.Centre).Norm <= Reach(a.Size, b.Size);
        }

        /// <summary>v3: the Meta box a sighting belongs to: for furniture any box, else a box Meta named as the
        /// sighting's category (the monitor inside Meta's screen); inside it within box_margin_m, or resting on its top
        /// within top_allowance_m (a pile on the bed); the smallest when several.</summary>
        private RoomBox MetaBoxFor(string standard, Vec3 p)
        {
            if (!room.Loaded) return null;
            bool furniture = s.BoxCategories.Contains(standard);
            RoomBox best = null;
            for (int i = 0; i < room.Boxes.Count; i++)
            {
                RoomBox b = room.Boxes[i];
                JNode e;
                bool own = entries.TryGetValue("meta:" + b.MetaLabel, out e) && e["standard"].Text == standard;
                if ((!furniture && !own) || uprights[i].Outside(p, s.TopAllowanceM) > s.BoxMarginM) continue;
                if (best == null || b.Volume < best.Volume || (b.Volume == best.Volume && string.CompareOrdinal(b.Id, best.Id) < 0)) best = b;
            }
            return best;
        }

        /// <summary>The map entry for a standard category: the detector's own when it has one, else Meta's.</summary>
        private JNode EntryFor(string standard)
        {
            return map["entries"].Items.Where(x => x["standard"].Text == standard && !x["source_label"].Text.StartsWith("meta:", StringComparison.Ordinal))
                                       .Concat(map["entries"].Items.Where(x => x["standard"].Text == standard)).First();
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
                        if (s.Boxes ? !Overlap(A, B) : A.Standard != B.Standard || (A.Centre - B.Centre).Norm > Reach(A.Size, B.Size)) continue;
                        if (s.Boxes) { A.AbsorbBox(B, s); Refit(A); } else A.Absorb(B, s);
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

        private sealed class Item
        {
            public Obj Depth;
            public int Box = -1;
            public JNode Entry;
            public string By, Sort;
            public Vec3 Centre;
        }

        /// <summary>The scene's objects in their order: Meta's boxes and the confirmed depth objects, by category (unnamed
        /// boxes last), then position. With no room, the depth objects in the order they always had.</summary>
        private List<Item> SceneItems()
        {
            var items = objects.Where(o => o.MergedInto < 0 && o.Frames.Count >= s.MinFrames)
                               .Select(o => new Item { Depth = o, Sort = o.Standard, Centre = o.Centre }).ToList();
            if (room.Loaded)
                for (int i = 0; i < room.Boxes.Count; i++)
                {
                    string by;
                    JNode e = BoxName(i, out by);
                    items.Add(new Item { Box = i, Entry = e, By = by, Sort = e != null ? e["standard"].Text : "\uffff", Centre = ToScene(room.Boxes[i].Centre) });
                }
            return items.OrderBy(x => x.Sort, StringComparer.Ordinal).ThenBy(x => Math.Round(x.Centre.X, 4)).ThenBy(x => Math.Round(x.Centre.Y, 4))
                        .ThenBy(x => Math.Round(x.Centre.Z, 4)).ThenBy(x => x.Box < 0 ? 0 : 1).ThenBy(x => x.Box < 0 ? x.Depth.Id : x.Box).ToList();
        }

        /// <summary>The objects in the scene so far, each with its category ("?" for an unnamed box) and where its label
        /// floats in quest_tracking: a depth object's centre, a box's top.</summary>
        public List<KeyValuePair<string, Vec3>> ConfirmedInTracking()
        {
            return SceneItems().Select(x => x.Box < 0 ? new KeyValuePair<string, Vec3>(x.Depth.Standard, x.Depth.Box != null ? x.Depth.Box.Top : ToUnity(x.Depth.Centre))
                                                      : new KeyValuePair<string, Vec3>(x.Entry != null ? x.Entry["standard"].Text : "?", uprights[x.Box].Top)).ToList();
        }

        /// <summary>v3: every object of the scene so far as the box to draw.</summary>
        public List<ShownBox> ConfirmedBoxes()
        {
            return SceneItems().Where(x => x.Box >= 0 || x.Depth.Box != null).Select(x => x.Box < 0
                ? new ShownBox { Label = x.Depth.Standard, Kind = "scan", Box = x.Depth.Box }
                : new ShownBox { Label = x.Entry != null ? x.Entry["standard"].Text : "?", Kind = x.Entry == null ? "unnamed" : x.By, Box = uprights[x.Box] }).ToList();
        }

        /// <summary>A unit quaternion's component, to 9 decimals: the scene's 4-decimal numbers would leave a turned
        /// box's quaternion off unit length by more than the contract's tolerance.</summary>
        private static JNode Quat(double v) { return RoomModel.Fine(v); }

        /// <summary>A box's size in the scene's axes: along its own x, along its own horizontal z, then its height.</summary>
        private static JNode BoxSize(Box3 b, string source, bool completed)
        {
            return Localizer.Known(Localizer.Vec(new Vec3(b.Size.X, b.Size.Z, b.Size.Y)), "inferred", source,
                                   completed ? new List<JNode> { JNode.Str(CompletionAssumption) } : new List<JNode>());
        }

        /// <summary>A box's turn about the scene's +z (yaw only), from its turn about quest_tracking's +y.</summary>
        private static JNode BoxTurn(Box3 b, string source)
        {
            double[] q = RoomModel.RotationToScene(b.Rotation());
            JNode n = Localizer.Known(JNode.Arr(new List<JNode> { Quat(0), Quat(0), Quat(q[2]), Quat(q[3]) }), "inferred", source, new List<JNode>());
            n.Add("support", JNode.Str("yaw_only"));
            return n;
        }

        /// <summary>The scene of the confirmed objects (scene v3, perceived) and the trace of how it was made.</summary>
        public LocalizationOutput Snapshot(int revision)
        {
            Changed = false;
            List<Item> items = SceneItems();
            var ids = new Dictionary<int, string>();
            var boxIds = new Dictionary<int, string>();
            for (int n = 0; n < items.Count; n++)
            {
                string id = "obj_" + (n + 1).ToString("D3", CultureInfo.InvariantCulture);
                if (items[n].Box < 0) ids[items[n].Depth.Id] = id; else boxIds[items[n].Box] = id;
            }
            var confirmed = items.Where(x => x.Box < 0).Select(x => x.Depth).ToList();
            string priorsId = priors["priors_id"].Text;
            bool usedSize = false, usedVolume = false, usedBox = false, usedCompletion = false;
            var objs = new List<JNode>();
            foreach (Item it in items)
            {
                if (it.Box >= 0)
                {
                    objs.Add(BoxObject(it, boxIds[it.Box], priorsId, ref usedSize));
                    usedVolume = true;
                    continue;
                }
                Obj o = it.Depth;
                List<JNode> sz = null;
                bool oneSize = !s.Boxes && sizes.TryGetValue(o.Standard, out sz) && sz.Count == 1;   // two sizes: neither is claimed
                usedSize |= oneSize;
                usedBox |= s.Boxes;
                usedCompletion |= s.Boxes && o.Box.Completed;
                JNode size = s.Boxes ? BoxSize(o.Box, ScanSourceId, o.Box.Completed)
                    : oneSize ? Localizer.Known(JNode.Arr(sz[0]["size_m"].Items.Select(x => Localizer.Num(x.Num)).ToList()), "assumed", priorsId, new List<JNode> { JNode.Str(Localizer.SizeAssumption) })
                    : Localizer.Unknown("no_permitted_prior", priorsId);
                JNode centre = s.Boxes ? Localizer.Known(Localizer.Vec(ToScene(o.Box.Centre)), "inferred", ScanSourceId, new List<JNode> { JNode.Str(BoxAssumption) })
                                       : Localizer.Known(Localizer.Vec(o.Centre), "inferred", ScanSourceId, new List<JNode> { JNode.Str(DepthAssumption) });
                JNode turn = s.Boxes ? BoxTurn(o.Box, ScanSourceId) : Localizer.Unknown("not_estimated", ScanSourceId);
                objs.Add(JNode.Obj()
                    .Add("object_id", JNode.Str(ids[o.Id]))
                    .Add("source_ref", JNode.Obj().Add("source_id", JNode.Str(ScanSourceId)).Add("source_object_id", JNode.Str("track_" + o.Id.ToString(CultureInfo.InvariantCulture)))
                        .Add("source_label", JNode.Str(o.Label)))
                    .Add("category", Localizer.Known(JNode.Obj().Add("standard", JNode.Str(o.Standard)).Add("model", JNode.Str(o.Model)), "inferred", ScanSourceId, new List<JNode>()))
                    .Add("geometry", JNode.Obj().Add("center_m", centre).Add("size_m", size).Add("rotation_xyzw", turn))
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
            if (usedBox) assumptions.Add(JNode.Obj().Add("assumption_id", JNode.Str(BoxAssumption))
                .Add("description", JNode.Str("The object's centre is its 3D box's centre: the box is fitted to the points its depth samples hit on its surface, each side trimmed of stray points; its size and turn are the box's")));
            if (usedCompletion) assumptions.Add(JNode.Obj().Add("assumption_id", JNode.Str(CompletionAssumption))
                .Add("description", JNode.Str("A side of the box no viewpoint saw is grown to its category's measured size in the size priors (" + priorsId + "), away from where the object was seen")));
            if (usedVolume) assumptions.Add(JNode.Obj().Add("assumption_id", JNode.Str(VolumeAssumption))
                .Add("description", JNode.Str("The object is one of the boxes Meta's Space Setup put around obstacles: its centre is the box's centre; the box's size and rotation are in the trace")));
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
                        .Add("description", JNode.Str(priors["description"].Text)).Add("release", JNode.Str(priors["release"].Text)) }.Concat(RoomSources()).ToList()))
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
                else if (fateBox[i] >= 0) d.Add("fate", JNode.Str("object")).Add("object_id", JNode.Str(boxIds[fateBox[i]]));
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
                .Add("room", RoomTrace(boxIds))
                .Add("detections", JNode.Arr(dets));
            return new LocalizationOutput { Scene = scene, Trace = trace };
        }

        private List<JNode> RoomSources()
        {
            var list = new List<JNode>();
            if (room.Loaded)
                list.Add(JNode.Obj().Add("source_id", JNode.Str(RoomSourceId)).Add("kind", JNode.Str("sensor_capture"))
                    .Add("description", JNode.Str("Meta's room from Space Setup, loaded with MRUK (Meta XR SDK 207): the floor's outline and "
                                                  + room.Boxes.Count.ToString(CultureInfo.InvariantCulture) + " boxes around obstacles"))
                    .Add("release", JNode.Str(scanId)));
            return list;
        }

        /// <summary>One of Meta's boxes as a scene object: Meta's name, else the detector's, else an unknown category.</summary>
        private JNode BoxObject(Item it, string objectId, string priorsId, ref bool usedSize)
        {
            RoomBox b = room.Boxes[it.Box];
            BoxVotes v = boxVotes[it.Box];
            bool seen = v.Frames.Count > 0;
            List<JNode> sz = null;
            bool oneSize = !s.Boxes && it.Entry != null && sizes.TryGetValue(it.Entry["standard"].Text, out sz) && sz.Count == 1;
            usedSize |= oneSize;
            JNode size = oneSize
                ? Localizer.Known(JNode.Arr(sz[0]["size_m"].Items.Select(x => Localizer.Num(x.Num)).ToList()), "assumed", priorsId, new List<JNode> { JNode.Str(Localizer.SizeAssumption) })
                : Localizer.Unknown("no_permitted_prior", priorsId);
            JNode category = it.Entry != null
                ? Localizer.Known(JNode.Obj().Add("standard", JNode.Str(it.Entry["standard"].Text)).Add("model", JNode.Str(it.Entry["model"].Text)),
                                  "inferred", it.By == "meta" ? RoomSourceId : ScanSourceId, new List<JNode>())
                : Localizer.Unknown("not_estimated", RoomSourceId);
            int keyframes = Math.Max(1, v.Frames.Count), positions = Math.Max(1, v.Viewpoints.Count);   // a box no frame saw counts Space Setup's capture once
            return JNode.Obj()
                .Add("object_id", JNode.Str(objectId))
                .Add("source_ref", JNode.Obj().Add("source_id", JNode.Str(RoomSourceId)).Add("source_object_id", JNode.Str(b.Id))
                    .Add("source_label", JNode.Str(it.Entry != null ? it.Entry["source_label"].Text : "meta:" + b.MetaLabel)))   // the map entry that named it
                .Add("category", category)
                .Add("geometry", JNode.Obj().Add("center_m", Localizer.Known(Localizer.Vec(ToScene(b.Centre)), "inferred", RoomSourceId, new List<JNode> { JNode.Str(VolumeAssumption) }))
                    .Add("size_m", s.Boxes ? BoxSize(uprights[it.Box], RoomSourceId, false) : size)
                    .Add("rotation_xyzw", s.Boxes ? BoxTurn(uprights[it.Box], RoomSourceId) : Localizer.Unknown("not_estimated", RoomSourceId)))
                .Add("semantic_front", Localizer.Unknown("not_estimated", RoomSourceId))
                .Add("attributes", JNode.Obj().Add("colours", Localizer.Unknown("not_estimated", RoomSourceId)))
                .Add("observation", JNode.Obj()
                    .Add("source", JNode.Str(RoomSourceId))
                    .Add("capture_time", seen ? Localizer.Known(JNode.Obj().Add("clock", JNode.Str("headset_mono")).Add("value_us", Localizer.Int(v.LastUs)), "measured", ScanSourceId, new List<JNode>())
                                              : Localizer.Unknown("not_estimated", RoomSourceId))
                    .Add("detector_confidence", seen ? Localizer.Known(Localizer.Num(v.BestScore), "inferred", ScanSourceId, new List<JNode>()) : Localizer.Unknown("not_estimated", RoomSourceId))
                    .Add("geometry_uncertainty", Localizer.Unknown("not_estimated", RoomSourceId))
                    .Add("views", JNode.Obj().Add("keyframes", Localizer.Int(keyframes)).Add("positions", Localizer.Int(positions)))
                    .Add("association", JNode.Obj().Add("status", JNode.Str(positions >= 2 ? "multi_position" : "single_position")).Add("may_duplicate", JNode.Arr(new List<JNode>()))));
        }

        /// <summary>Meta's room in the trace: every box, its object, who named it, the votes, and its true size and
        /// rotation in the scene's axes (the scene contract doesn't carry a measured size or rotation yet).</summary>
        private JNode RoomTrace(Dictionary<int, string> boxIds)
        {
            var boxes = new List<JNode>();
            for (int i = 0; i < room.Boxes.Count && room.Loaded; i++)
            {
                RoomBox b = room.Boxes[i];
                string by;
                JNode e = BoxName(i, out by);
                var votes = JNode.Obj();
                foreach (var kv in boxVotes[i].Votes) votes.Add(kv.Key, Localizer.Int(kv.Value));
                boxes.Add(JNode.Obj().Add("object_id", JNode.Str(boxIds[i])).Add("id", JNode.Str(b.Id)).Add("meta_label", JNode.Str(b.MetaLabel))
                    .Add("named_by", JNode.Str(by)).Add("category", JNode.Str(e != null ? e["standard"].Text : "")).Add("votes", votes)
                    .Add("frames", Localizer.Int(boxVotes[i].Frames.Count)).Add("viewpoints", Localizer.Int(boxVotes[i].Viewpoints.Count))
                    .Add("center_m", Localizer.Vec(ToScene(b.Centre))).Add("size_m", Localizer.Vec(new Vec3(uprights[i].Size.X, uprights[i].Size.Z, uprights[i].Size.Y)))
                    .Add("yaw_deg", Localizer.Num(-uprights[i].YawDeg)));
            }
            return JNode.Obj().Add("status", JNode.Str(room.Status)).Add("why", JNode.Str(room.Why)).Add("boxes", JNode.Arr(boxes));
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
