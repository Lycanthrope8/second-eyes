using System;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using SecondEyes.Grounding.Prompting;
using SecondEyes.Perception.Frames;

namespace SecondEyes.Perception.Localization
{
    /// <summary>The localization's thresholds (schemas/localization-config.v1.json; docs/localization.md).</summary>
    public sealed class LocalizationConfig
    {
        public string ConfigId;
        public double MinScore, EdgeMarginPx, SamePositionAngleDeg, RangeRatioTolerance, GapMaxM, MinParallaxDeg, AmbiguityRatio, MaxRangeM;

        public static LocalizationConfig From(JNode n)
        {
            return new LocalizationConfig
            {
                ConfigId = n["config_id"].Text, MinScore = n["min_score"].Num, EdgeMarginPx = n["edge_margin_px"].Num,
                SamePositionAngleDeg = n["same_position_angle_deg"].Num, RangeRatioTolerance = n["range_ratio_tolerance"].Num,
                GapMaxM = n["gap_max_m"].Num, MinParallaxDeg = n["min_parallax_deg"].Num, AmbiguityRatio = n["ambiguity_ratio"].Num,
                MaxRangeM = n["max_range_m"].Num
            };
        }
    }

    /// <summary>A perceived scene (scene v3) and the trace of how it was made (not a contract record).</summary>
    public sealed class LocalizationOutput
    {
        public JNode Scene, Trace;
    }

    /// <summary>
    /// The localization core (A4 Delivery 1, stage 2; docs/localization.md), Unity-free: the same code runs on the Quest and
    /// in the PC's command-line rerun. From a checked capture record, a category map, size priors and a configuration it
    /// builds a perceived scene. In order:
    /// 1. each mapped detection at or above min_score becomes a ray in the scene frame (box centre, pinhole, capture pose);
    ///    a box within edge_margin_px of the image's edge is left out, since its centre and height are cut;
    /// 2. an upright object with a size prior gets a distance interval: at each candidate distance along its ray, the
    ///    prior-sized box is projected through the actual camera pose (its height and tilt included), and the distance is
    ///    solved where the projected height meets the observed one, at six yaws since the object's yaw is unknown; the
    ///    interval is tight for a chair-like box and wide for a long, low one; a category with several sizes gets the union
    ///    of their intervals, and no size of its own in the scene (the size_prior and upright assumptions);
    /// 3. at each standing position, rays of one category within same_position_angle_deg and with agreeing distances merge
    ///    into one sighting: turning in place adds no baseline;
    /// 4. position by position, sightings join tracks by minimum-cost assignment, a link valid only if the rays pass within
    ///    gap_max_m, at least min_parallax_deg apart, in front of both cameras and within max_range_m, and if every distance
    ///    from a size prior falls in its interval, widened by range_ratio_tolerance; this last test removes the ghost
    ///    crossings of identical objects;
    /// 5. a link with a rival within ambiguity_ratio of its cost is not made: both stay separate objects, marked possible
    ///    duplicates of each other, as is any near-tie left unmade;
    /// 6. a track seen from two or more positions gets the least-squares point nearest its rays, and a ray farther than
    ///    gap_max_m from it is split off and marked a possible duplicate (the third-view check); a track seen from one
    ///    position gets its centre at the middle of its distance interval, or none without a size prior.
    /// There is no prior on how many objects of a kind exist. Detections are taken in the order of their boxes, so output
    /// order and object IDs depend only on what was seen, not on the order the detector listed it.
    /// </summary>
    public static class Localizer
    {
        public const string ScanSourceId = "scan", SizeAssumption = "size_prior", UprightAssumption = "upright";

        private sealed class Obs { public int Keyframe, Position; public double X1, Y1; public string Label, Standard, Model; public Vec3 Origin, Dir; public double? Lo, Hi; public double Score; public long TimeUs; }
        private sealed class Sighting { public int Id, Position; public string Label, Standard, Model; public Vec3 Origin, Dir; public double? Lo, Hi; public readonly List<Obs> Members = new List<Obs>(); }
        private sealed class Track { public int Id; public string Label, Standard, Model; public readonly List<Sighting> Sightings = new List<Sighting>(); public Vec3? Centre; public readonly SortedSet<int> MayDuplicate = new SortedSet<int>(); }

        public static LocalizationOutput Run(JNode capture, JNode categoryMap, JNode sizePriors, JNode config)
        {
            var cfg = LocalizationConfig.From(config);
            var map = new Dictionary<string, KeyValuePair<string, string>>(StringComparer.Ordinal);
            foreach (JNode e in categoryMap["entries"].Items) map[e["source_label"].Text] = new KeyValuePair<string, string>(e["standard"].Text, e["model"].Text);
            var priors = new Dictionary<string, List<JNode>>(StringComparer.Ordinal);   // a category may have several sizes
            foreach (JNode e in sizePriors["entries"].Items)
            {
                if (!priors.ContainsKey(e["category"].Text)) priors[e["category"].Text] = new List<JNode>();
                priors[e["category"].Text].Add(e);
            }
            JNode cam = capture["camera"], intr = cam["intrinsics"];
            double width = cam["width_px"].Num, height = cam["height_px"].Num;
            var pin = new PinholeCamera((int)width, (int)height, intr["fx"].Num, intr["fy"].Num, intr["cx"].Num, intr["cy"].Num);
            List<JNode> frames = capture["keyframes"].Items;
            long epoch = (long)frames[0]["tracking"]["epoch"].Num;
            int unmapped = 0, lowScore = 0, edgeCut = 0;

            // 1-2. detections to rays, with distances where a size prior allows
            var obs = new List<Obs>();
            foreach (JNode k in frames)
            {
                JNode pose = k["camera_pose"];
                RigidTransform t = UnityBasis.CameraToScene(cam["camera_id"].Text, epoch, V(pose["translation_m"]), Q(pose["rotation_xyzw"]));
                List<JNode> dets = k["detections"].Items;
                for (int j = 0; j < dets.Count; j++)
                {
                    JNode d = dets[j];
                    if (!map.ContainsKey(d["label"].Text)) { unmapped++; continue; }
                    if (d["score"].Num < cfg.MinScore) { lowScore++; continue; }
                    List<JNode> box = d["box_px"].Items;
                    double x1 = box[0].Num, y1 = box[1].Num, x2 = box[2].Num, y2 = box[3].Num, m = cfg.EdgeMarginPx;
                    if (x1 <= m || y1 <= m || x2 >= width - m || y2 >= height - m) { edgeCut++; continue; }
                    Vec3 rayCam = pin.Ray((x1 + x2) / 2.0 - 0.5, (y1 + y2) / 2.0 - 0.5);   // continuous source pixels to OpenCV's
                    var category = map[d["label"].Text];
                    double? lo = null, hi = null;
                    List<JNode> sizes;
                    Vec3 dir = t.ApplyToDirection(rayCam).Normalized();
                    if (priors.TryGetValue(category.Key, out sizes))
                        foreach (JNode prior in sizes)   // several sizes: the union of their intervals
                        {
                            if (!prior["upright"].Bool) continue;
                            List<JNode> sz = prior["size_m"].Items;
                            double? l, h;
                            Interval(t, pin, dir, new Vec3(sz[0].Num, sz[1].Num, sz[2].Num), y2 - y1, cfg.MaxRangeM, out l, out h);
                            if (!l.HasValue) continue;
                            lo = lo.HasValue ? Math.Min(lo.Value, l.Value) : l;
                            hi = hi.HasValue ? Math.Max(hi.Value, h.Value) : h;
                        }
                    obs.Add(new Obs { Keyframe = (int)k["index"].Num, Position = (int)k["position_index"].Num, X1 = x1, Y1 = y1, Label = d["label"].Text,
                                      Standard = category.Key, Model = category.Value, Origin = t.T, Dir = dir,
                                      Lo = lo, Hi = hi, Score = d["score"].Num, TimeUs = (long)k["capture_time"]["headset_mono_us"].Num });
                }
            }

            // 3. sightings: one per object and standing position
            var sightings = new List<Sighting>();
            double cosSame = Math.Cos(cfg.SamePositionAngleDeg * Math.PI / 180.0);
            foreach (Obs o in obs.OrderBy(x => x.Position).ThenBy(x => x.Keyframe).ThenBy(x => x.X1).ThenBy(x => x.Y1).ThenBy(x => x.Label, StringComparer.Ordinal))
            {
                Sighting hit = sightings.FirstOrDefault(s => s.Position == o.Position && s.Standard == o.Standard && s.Dir.Dot(o.Dir) >= cosSame
                                                             && Agree(s.Lo, s.Hi, o.Lo, o.Hi, cfg.RangeRatioTolerance));
                if (hit == null)
                {
                    hit = new Sighting { Id = sightings.Count, Position = o.Position, Label = o.Label, Standard = o.Standard, Model = o.Model };
                    sightings.Add(hit);
                }
                hit.Members.Add(o);
                hit.Origin = (1.0 / hit.Members.Count) * hit.Members.Aggregate(new Vec3(0, 0, 0), (a, x) => a + x.Origin);
                hit.Dir = hit.Members.Aggregate(new Vec3(0, 0, 0), (a, x) => a + x.Dir).Normalized();
                hit.Lo = Median(hit.Members.Where(x => x.Lo.HasValue).Select(x => x.Lo.Value));
                hit.Hi = Median(hit.Members.Where(x => x.Hi.HasValue).Select(x => x.Hi.Value));
            }

            // 4-6. tracks across positions
            var tracks = new List<Track>();
            var links = new List<JNode>();
            foreach (int pos in sightings.Select(s => s.Position).Distinct().OrderBy(x => x))
            {
                var here = sightings.Where(s => s.Position == pos).ToList();
                var before = tracks.ToList();
                var cost = new double[before.Count, here.Count];
                for (int i = 0; i < before.Count; i++)
                    for (int j = 0; j < here.Count; j++) cost[i, j] = LinkCost(before[i], here[j], cfg);
                int[] assign = Hungarian.Solve(cost);
                var owner = new Dictionary<int, Track>();
                for (int i = 0; i < before.Count; i++)
                {
                    int j = assign[i];
                    if (j < 0) continue;
                    double c = cost[i, j], bar = Math.Max(c, 0.01) * cfg.AmbiguityRatio;
                    bool rival = Enumerable.Range(0, here.Count).Any(jj => jj != j && cost[i, jj] <= bar)
                                 || Enumerable.Range(0, before.Count).Any(ii => ii != i && cost[ii, j] <= bar);
                    links.Add(JNode.Obj().Add("track", Int(before[i].Id)).Add("sighting", Int(here[j].Id)).Add("gap_m", Num(c))
                                         .Add("decision", JNode.Str(rival ? "ambiguous: not merged" : "merged")));
                    if (rival) continue;
                    before[i].Sightings.Add(here[j]);
                    owner[j] = before[i];
                }
                for (int j = 0; j < here.Count; j++)
                    if (!owner.ContainsKey(j))
                    {
                        var t = new Track { Id = tracks.Count, Label = here[j].Label, Standard = here[j].Standard, Model = here[j].Model };
                        t.Sightings.Add(here[j]);
                        tracks.Add(t);
                        owner[j] = t;
                    }
                for (int i = 0; i < before.Count; i++)   // near-ties left unmade: possible duplicates
                    for (int j = 0; j < here.Count; j++)
                    {
                        double c = cost[i, j];
                        if (double.IsInfinity(c) || owner[j] == before[i]) continue;
                        double best = Math.Min(Enumerable.Range(0, here.Count).Min(jj => cost[i, jj]), Enumerable.Range(0, before.Count).Min(ii => cost[ii, j]));
                        if (c <= Math.Max(best, 0.01) * cfg.AmbiguityRatio) Pair(before[i], owner[j]);
                    }
                foreach (Track t in tracks.ToList()) Refit(t, tracks, cfg);
            }

            // the scene
            var ordered = tracks.OrderBy(t => t.Standard, StringComparer.Ordinal).ThenBy(t => Centre(t, out _).HasValue ? 0 : 1)
                                .ThenBy(t => Key(t, 0)).ThenBy(t => Key(t, 1)).ThenBy(t => Key(t, 2))
                                .ThenBy(t => t.Sightings.Min(s => s.Members.Min(o => o.Keyframe))).ThenBy(t => t.Sightings.Min(s => s.Members.Min(o => o.X1))).ToList();
            var ids = new Dictionary<int, string>();
            for (int n = 0; n < ordered.Count; n++) ids[ordered[n].Id] = "obj_" + (n + 1).ToString("D3", CultureInfo.InvariantCulture);
            string priorsId = sizePriors["priors_id"].Text;
            bool usedSize = false, usedUpright = false;
            var objects = new List<JNode>();
            foreach (Track t in ordered)
            {
                bool fromSize;
                Vec3? c = Centre(t, out fromSize);
                List<JNode> sizesFor;
                bool oneSize = priors.TryGetValue(t.Standard, out sizesFor) && sizesFor.Count == 1;   // two sizes: neither is claimed
                var members = t.Sightings.SelectMany(s => s.Members).ToList();
                int positions = t.Sightings.Select(s => s.Position).Distinct().Count();
                var dup = t.MayDuplicate.Select(x => ids[x]).OrderBy(x => x, StringComparer.Ordinal).ToList();
                string status = dup.Count > 0 ? "possible_duplicate" : positions >= 2 ? "multi_position" : "single_position";
                JNode centre;
                if (c.HasValue)
                {
                    var assume = fromSize ? new List<JNode> { JNode.Str(SizeAssumption), JNode.Str(UprightAssumption) } : new List<JNode>();
                    usedSize |= fromSize; usedUpright |= fromSize;
                    centre = Known(Vec(c.Value), "inferred", ScanSourceId, assume);
                }
                else centre = Unknown("not_estimated", ScanSourceId);
                JNode size = oneSize ? Known(JNode.Arr(sizesFor[0]["size_m"].Items.Select(x => Num(x.Num)).ToList()), "assumed", priorsId, new List<JNode> { JNode.Str(SizeAssumption) })
                                     : Unknown("no_permitted_prior", priorsId);
                usedSize |= oneSize;
                objects.Add(JNode.Obj()
                    .Add("object_id", JNode.Str(ids[t.Id]))
                    .Add("source_ref", JNode.Obj().Add("source_id", JNode.Str(ScanSourceId)).Add("source_object_id", JNode.Str("track_" + t.Id.ToString(CultureInfo.InvariantCulture))).Add("source_label", JNode.Str(t.Label)))
                    .Add("category", Known(JNode.Obj().Add("standard", JNode.Str(t.Standard)).Add("model", JNode.Str(t.Model)), "inferred", ScanSourceId, new List<JNode>()))
                    .Add("geometry", JNode.Obj().Add("center_m", centre).Add("size_m", size).Add("rotation_xyzw", Unknown("not_estimated", ScanSourceId)))
                    .Add("semantic_front", Unknown("not_estimated", ScanSourceId))
                    .Add("attributes", JNode.Obj().Add("colours", Unknown("not_estimated", ScanSourceId)))
                    .Add("observation", JNode.Obj()
                        .Add("source", JNode.Str(ScanSourceId))
                        .Add("capture_time", Known(JNode.Obj().Add("clock", JNode.Str("headset_mono")).Add("value_us", Int(members.Max(o => o.TimeUs))), "measured", ScanSourceId, new List<JNode>()))
                        .Add("detector_confidence", Known(Num(members.Max(o => o.Score)), "inferred", ScanSourceId, new List<JNode>()))
                        .Add("geometry_uncertainty", Unknown("not_estimated", ScanSourceId))
                        .Add("views", JNode.Obj().Add("keyframes", Int(members.Count)).Add("positions", Int(positions)))
                        .Add("association", JNode.Obj().Add("status", JNode.Str(status)).Add("may_duplicate", JNode.Arr(dup.Select(JNode.Str).ToList())))));
            }
            var assumptions = new List<JNode>();
            if (usedSize) assumptions.Add(JNode.Obj().Add("assumption_id", JNode.Str(SizeAssumption)).Add("description", JNode.Str("The object's size is its category's measured size in the size priors (" + priorsId + ")")));
            if (usedUpright) assumptions.Add(JNode.Obj().Add("assumption_id", JNode.Str(UprightAssumption)).Add("description", JNode.Str("The object stands upright, so its image height gives its distance")));
            JNode det = capture["detector"];
            JNode scene = JNode.Obj()
                .Add("schema_version", Int(3)).Add("record_type", JNode.Str("scene")).Add("scene_id", JNode.Str(capture["capture_id"].Text))
                .Add("scene_revision", Int(0)).Add("evidence_profile", JNode.Str("perceived")).Add("category_map", JNode.Str(categoryMap["map_id"].Text))
                .Add("coordinate_frame", JNode.Obj().Add("frame_id", JNode.Str("scene")).Add("units", JNode.Str("m")).Add("handedness", JNode.Str("right"))
                    .Add("up_axis", JNode.Str("+z")).Add("origin", JNode.Str("the headset's tracking origin in epoch " + epoch.ToString(CultureInfo.InvariantCulture) + ", re-expressed through the basis change (docs/frames.md)"))
                    .Add("conversion", JNode.Obj().Add("kind", JNode.Str("quest_tracking_basis")).Add("source_id", JNode.Str(ScanSourceId))
                        .Add("source_frame_id", JNode.Str("quest_tracking")).Add("tracking_epoch", Int(epoch))
                        .Add("source_to_scene", JNode.Arr(new List<JNode> { Row(1, 0, 0, 0), Row(0, 0, 1, 0), Row(0, 1, 0, 0), Row(0, 0, 0, 1) }))))
                .Add("sources", JNode.Arr(new List<JNode> {
                    JNode.Obj().Add("source_id", JNode.Str(ScanSourceId)).Add("kind", JNode.Str("sensor_capture"))
                        .Add("description", JNode.Str("Scan " + capture["capture_id"].Text + ": camera " + cam["camera_id"].Text + ", detector " + det["package_id"].Text
                                                      + " (" + det["package_sha256"].Text.Substring(0, 12) + "), localization " + cfg.ConfigId))
                        .Add("release", JNode.Str(capture["capture_id"].Text)),
                    JNode.Obj().Add("source_id", JNode.Str(priorsId)).Add("kind", JNode.Str("prior_table"))
                        .Add("description", JNode.Str(sizePriors["description"].Text)).Add("release", JNode.Str(sizePriors["release"].Text)) }))
                .Add("assumptions", JNode.Arr(assumptions))
                .Add("objects", JNode.Arr(objects));
            JNode trace = JNode.Obj()
                .Add("record_type", JNode.Str("localization_trace")).Add("capture_id", JNode.Str(capture["capture_id"].Text)).Add("config_id", JNode.Str(cfg.ConfigId))
                .Add("left_out", JNode.Obj().Add("unmapped_label", Int(unmapped)).Add("below_min_score", Int(lowScore)).Add("cut_by_the_edge", Int(edgeCut)))
                .Add("sightings", JNode.Arr(sightings.Select(s => JNode.Obj().Add("sighting", Int(s.Id)).Add("position", Int(s.Position)).Add("category", JNode.Str(s.Standard))
                    .Add("keyframes", JNode.Arr(s.Members.Select(o => Int(o.Keyframe)).ToList()))
                    .Add("distance_m", s.Lo.HasValue ? JNode.Arr(new List<JNode> { Num(s.Lo.Value), Num(s.Hi.Value) }) : JNode.Null)).ToList()))
                .Add("links", JNode.Arr(links))
                .Add("objects", JNode.Arr(ordered.Select(t => JNode.Obj().Add("object_id", JNode.Str(ids[t.Id])).Add("track", Int(t.Id))
                    .Add("sightings", JNode.Arr(t.Sightings.Select(s => Int(s.Id)).ToList()))).ToList()));
            return new LocalizationOutput { Scene = scene, Trace = trace };
        }

        /// <summary>Two distance intervals agree when they overlap once widened by the tolerance (or either is missing).</summary>
        private static bool Agree(double? aLo, double? aHi, double? bLo, double? bHi, double tolerance)
        {
            return !aLo.HasValue || !bLo.HasValue || (aLo.Value * (1 - tolerance) <= bHi.Value * (1 + tolerance) && bLo.Value * (1 - tolerance) <= aHi.Value * (1 + tolerance));
        }

        /// <summary>A distance along a ray fits its interval, widened by the tolerance (or there is no interval).</summary>
        private static bool Fits(double? lo, double? hi, double along, double tolerance)
        {
            return !lo.HasValue || (along >= lo.Value * (1 - tolerance) && along <= hi.Value * (1 + tolerance));
        }

        private static double? Median(IEnumerable<double> values)
        {
            var v = values.OrderBy(x => x).ToList();
            return v.Count == 0 ? (double?)null : (v[(v.Count - 1) / 2] + v[v.Count / 2]) / 2.0;
        }

        /// <summary>The distance interval along a ray at which an upright box of the given size, at an unknown yaw, projects to
        /// the observed image height through this camera; none when no yaw can explain the height within max_range_m.</summary>
        private static void Interval(RigidTransform t, PinholeCamera pin, Vec3 dir, Vec3 size, double observed, double maxRange, out double? lo, out double? hi)
        {
            lo = hi = null;
            if (!(observed > 0)) return;
            RigidTransform toCamera = t.Inverse();
            for (int k = 0; k < 6; k++)
            {
                double yaw = k * Math.PI / 6, c = Math.Cos(yaw), s = Math.Sin(yaw);
                Func<double, double> height = r =>   // the box's projected height in pixels at distance r; infinite if it reaches the camera
                {
                    Vec3 centre = t.T + r * dir;
                    double top = double.MaxValue, bottom = double.MinValue;
                    for (int n = 0; n < 8; n++)
                    {
                        double x = ((n & 1) == 0 ? -0.5 : 0.5) * size.X, y = ((n & 2) == 0 ? -0.5 : 0.5) * size.Y, z = ((n & 4) == 0 ? -0.5 : 0.5) * size.Z;
                        Vec3 local = toCamera.ApplyToPoint(centre + new Vec3(c * x - s * y, s * x + c * y, z));
                        double u, v;
                        if (local.Z < 0.05 || !pin.TryProject(local, out u, out v)) return double.PositiveInfinity;
                        top = Math.Min(top, v); bottom = Math.Max(bottom, v);
                    }
                    return bottom - top;
                };
                double near = 0.3, far = maxRange;
                if (height(near) < observed || height(far) > observed) continue;   // no distance in range gives this height
                for (int i = 0; i < 40; i++)
                {
                    double mid = (near + far) / 2;
                    if (height(mid) > observed) near = mid; else far = mid;
                }
                double r0 = (near + far) / 2;
                lo = lo.HasValue ? Math.Min(lo.Value, r0) : r0;
                hi = hi.HasValue ? Math.Max(hi.Value, r0) : r0;
            }
        }

        /// <summary>The cost of joining a sighting to a track: how far their rays pass from each other (or from the track's
        /// centre), or infinity when any test fails.</summary>
        private static double LinkCost(Track track, Sighting s, LocalizationConfig cfg)
        {
            if (track.Standard != s.Standard || track.Sightings.Any(x => x.Position == s.Position)) return double.PositiveInfinity;
            if (track.Centre.HasValue)
            {
                Vec3 w = track.Centre.Value - s.Origin;
                double along = w.Dot(s.Dir);
                if (along <= 0 || along > cfg.MaxRangeM || !Fits(s.Lo, s.Hi, along, cfg.RangeRatioTolerance)) return double.PositiveInfinity;
                double d = (w - along * s.Dir).Norm;
                return d <= cfg.GapMaxM ? d : double.PositiveInfinity;
            }
            Sighting a = track.Sightings[0];
            double b = a.Dir.Dot(s.Dir), denom = 1 - b * b;
            if (denom < 1e-12) return double.PositiveInfinity;
            Vec3 w0 = a.Origin - s.Origin;
            double dd = a.Dir.Dot(w0), e = s.Dir.Dot(w0);
            double t1 = (b * e - dd) / denom, t2 = (e - b * dd) / denom;
            if (t1 <= 0 || t2 <= 0 || t1 > cfg.MaxRangeM || t2 > cfg.MaxRangeM) return double.PositiveInfinity;
            double parallax = Math.Acos(Math.Max(-1, Math.Min(1, b))) * 180.0 / Math.PI;
            if (parallax < cfg.MinParallaxDeg || parallax > 180 - cfg.MinParallaxDeg) return double.PositiveInfinity;
            if (!Fits(a.Lo, a.Hi, t1, cfg.RangeRatioTolerance) || !Fits(s.Lo, s.Hi, t2, cfg.RangeRatioTolerance)) return double.PositiveInfinity;
            double gap = ((a.Origin + t1 * a.Dir) - (s.Origin + t2 * s.Dir)).Norm;
            return gap <= cfg.GapMaxM ? gap : double.PositiveInfinity;
        }

        /// <summary>The least-squares centre of a track seen from two or more positions, after the third-view check.</summary>
        private static void Refit(Track t, List<Track> tracks, LocalizationConfig cfg)
        {
            while (true)
            {
                if (t.Sightings.Select(s => s.Position).Distinct().Count() < 2) { t.Centre = null; return; }
                Vec3? c = LeastSquares(t.Sightings);
                if (!c.HasValue) { t.Centre = null; return; }
                Sighting worst = null;
                double worstD = cfg.GapMaxM;
                foreach (Sighting s in t.Sightings)
                {
                    Vec3 w = c.Value - s.Origin;
                    double d = (w - w.Dot(s.Dir) * s.Dir).Norm;
                    if (d > worstD) { worst = s; worstD = d; }
                }
                if (worst == null) { t.Centre = c; return; }
                t.Sightings.Remove(worst);
                var split = new Track { Id = tracks.Count, Label = worst.Label, Standard = worst.Standard, Model = worst.Model };
                split.Sightings.Add(worst);
                tracks.Add(split);
                Pair(t, split);
            }
        }

        private static Vec3? LeastSquares(List<Sighting> rays)
        {
            double a00 = 0, a01 = 0, a02 = 0, a11 = 0, a12 = 0, a22 = 0, b0 = 0, b1 = 0, b2 = 0;
            foreach (Sighting s in rays)
            {
                Vec3 d = s.Dir, o = s.Origin;
                double m00 = 1 - d.X * d.X, m01 = -d.X * d.Y, m02 = -d.X * d.Z, m11 = 1 - d.Y * d.Y, m12 = -d.Y * d.Z, m22 = 1 - d.Z * d.Z;
                a00 += m00; a01 += m01; a02 += m02; a11 += m11; a12 += m12; a22 += m22;
                b0 += m00 * o.X + m01 * o.Y + m02 * o.Z; b1 += m01 * o.X + m11 * o.Y + m12 * o.Z; b2 += m02 * o.X + m12 * o.Y + m22 * o.Z;
            }
            var a = new Mat3(a00, a01, a02, a01, a11, a12, a02, a12, a22);
            double det = a.Determinant();
            if (Math.Abs(det) < 1e-9) return null;
            Vec3 b = new Vec3(b0, b1, b2);   // Cramer's rule
            return new Vec3(new Mat3(b.X, a01, a02, b.Y, a11, a12, b.Z, a12, a22).Determinant() / det,
                            new Mat3(a00, b.X, a02, a01, b.Y, a12, a02, b.Z, a22).Determinant() / det,
                            new Mat3(a00, a01, b.X, a01, a11, b.Y, a02, a12, b.Z).Determinant() / det);
        }

        private static Vec3? Centre(Track t, out bool fromSize)
        {
            fromSize = false;
            if (t.Centre.HasValue) return t.Centre;
            if (t.Sightings.Select(s => s.Position).Distinct().Count() >= 2) return null;   // seen from two places but no fit: unknown
            Sighting s0 = t.Sightings[0];
            if (!s0.Lo.HasValue) return null;
            fromSize = true;
            return s0.Origin + ((s0.Lo.Value + s0.Hi.Value) / 2) * s0.Dir;
        }

        private static double Key(Track t, int axis)
        {
            Vec3? c = Centre(t, out _);
            if (!c.HasValue) return 0;
            return Math.Round(axis == 0 ? c.Value.X : axis == 1 ? c.Value.Y : c.Value.Z, 4);
        }

        private static void Pair(Track a, Track b) { a.MayDuplicate.Add(b.Id); b.MayDuplicate.Add(a.Id); }

        private static Vec3 V(JNode a) { return new Vec3(a.Items[0].Num, a.Items[1].Num, a.Items[2].Num); }
        private static Quat Q(JNode a) { return new Quat(a.Items[0].Num, a.Items[1].Num, a.Items[2].Num, a.Items[3].Num); }

        private static JNode Known(JNode value, string kind, string source, List<JNode> assumptions)
        {
            return JNode.Obj().Add("state", JNode.Str("known")).Add("value", value)
                .Add("evidence", JNode.Obj().Add("kind", JNode.Str(kind)).Add("source", JNode.Str(source)).Add("assumptions", JNode.Arr(assumptions)));
        }

        private static JNode Unknown(string reason, string source)
        {
            return JNode.Obj().Add("state", JNode.Str("unknown")).Add("reason", JNode.Str(reason)).Add("source", JNode.Str(source));
        }

        private static JNode Vec(Vec3 v) { return JNode.Arr(new List<JNode> { Num(v.X), Num(v.Y), Num(v.Z) }); }
        private static JNode Row(params long[] xs) { return JNode.Arr(xs.Select(Int).ToList()); }

        /// <summary>A number rounded to 0.1 mm (and 1e-4 in general), so the PC and the headset write the same text.</summary>
        public static JNode Num(double v)
        {
            double r = Math.Round(v, 4, MidpointRounding.AwayFromZero);
            if (r == 0) r = 0;   // never "-0"
            return new JNode { Kind = JKind.Number, Num = r, Text = r.ToString("0.0###", CultureInfo.InvariantCulture), IsInteger = false };
        }

        public static JNode Int(long v)
        {
            return new JNode { Kind = JKind.Number, Num = v, Text = v.ToString(CultureInfo.InvariantCulture), IsInteger = true };
        }
    }
}
