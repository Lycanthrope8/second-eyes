using System;
using System.Collections.Generic;
using SecondEyes.Grounding.Prompting;

namespace SecondEyes.Grounding.Resolution
{
    // A2.7 delivery 2 (D115): grounding/relations/predicates.py (A2.1b, D67) without its provenance, free of Unity:
    // near, above, on, below, inside and between, and the closest and farthest ranks 1-3, over a validated scene record.
    // Results are three-valued; conditions combine by AND and OR (FALSE wins an AND, TRUE wins an OR, otherwise
    // UNKNOWN). The decisions and reasons follow the Python exactly. Left out: the fields consulted, with their
    // evidence and assumptions (provenance, which no decision reads), and the configuration's identity hash.
    public enum Truth { True, False, Unknown }

    public static class Tri
    {
        public static Truth And(params Truth[] values)
        {
            foreach (var v in values) if (v == Truth.False) return Truth.False;
            foreach (var v in values) if (v != Truth.True) return Truth.Unknown;
            return Truth.True;
        }

        public static Truth Or(params Truth[] values)
        {
            foreach (var v in values) if (v == Truth.True) return Truth.True;
            foreach (var v in values) if (v != Truth.False) return Truth.Unknown;
            return Truth.False;
        }

        /// <summary>x &lt;= t with band b: TRUE if x &lt;= t - b, FALSE if x &gt; t + b, otherwise UNKNOWN.</summary>
        public static Truth Le(double x, double t, double b)
        {
            if (x <= t - b) return Truth.True;
            if (x > t + b) return Truth.False;
            return Truth.Unknown;
        }

        /// <summary>x &gt;= t with band b: TRUE if x &gt;= t + b, FALSE if x &lt; t - b, otherwise UNKNOWN.</summary>
        public static Truth Ge(double x, double t, double b)
        {
            if (x >= t + b) return Truth.True;
            if (x < t - b) return Truth.False;
            return Truth.Unknown;
        }

        public static string Name(Truth t) { return t == Truth.True ? "TRUE" : (t == Truth.False ? "FALSE" : "UNKNOWN"); }
    }

    /// <summary>A relation configuration (relations.v1.json), already validated against its schema.</summary>
    public sealed class RelationConfig
    {
        public string ConfigId, Status;
        public readonly Dictionary<string, double> Thresholds = new Dictionary<string, double>();
        public readonly Dictionary<string, double> Bands = new Dictionary<string, double>();
        public readonly Dictionary<string, HashSet<string>> Categories = new Dictionary<string, HashSet<string>>();

        public static RelationConfig From(JNode raw)
        {
            var c = new RelationConfig { ConfigId = raw["config_id"].Text, Status = raw["status"].Text };
            foreach (var kv in raw["thresholds"].Fields) c.Thresholds[kv.Key] = kv.Value.Num;
            foreach (var kv in raw["bands"].Fields) c.Bands[kv.Key] = kv.Value.Num;
            foreach (var kv in raw["categories"].Fields)
            {
                var set = new HashSet<string>();
                foreach (var x in kv.Value.Items) set.Add(x.Text);
                c.Categories[kv.Key] = set;
            }
            return c;
        }
    }

    /// <summary>One object's geometry: the model-visible category (or null), the centre (or null), the box (null unless
    /// centre, size and rotation are all known), and why the box is incomplete.</summary>
    public sealed class ObjectGeometry
    {
        public string ObjectId, Category;
        public double[] Centre;
        public Box Box;
        public List<string> Missing = new List<string>();
        /// <summary>Per source field (category, center_m, size_m, rotation_xyzw): known with assumed evidence or with
        /// assumptions. A result that consulted such a field is conditional, as the resolver's flag reports.</summary>
        public readonly Dictionary<string, bool> Conditional = new Dictionary<string, bool>();

        public static bool IsConditional(JNode w)
        {
            if (w["state"].Text != "known") return false;
            var ev = w["evidence"];
            return ev["kind"].Text == "assumed" || ev["assumptions"].Items.Count > 0;
        }

        public static ObjectGeometry Build(JNode obj)
        {
            string oid = obj["object_id"].Text;
            var g = new ObjectGeometry { ObjectId = oid };
            JNode cat = obj["category"], geo = obj["geometry"];
            JNode centre = geo["center_m"], size = geo["size_m"], rot = geo["rotation_xyzw"];
            if (centre["state"].Text != "known") g.Missing.Add("missing_centre:" + oid);
            if (size["state"].Text != "known") g.Missing.Add("missing_size:" + oid);
            if (rot["state"].Text != "known") g.Missing.Add("missing_rotation:" + oid);
            if (centre["state"].Text == "known") g.Centre = Vec(centre["value"]);
            if (g.Missing.Count == 0) g.Box = new Box(Vec(centre["value"]), Vec(size["value"]), Vec(rot["value"]));
            g.Category = cat["state"].Text == "known" ? cat["value"]["model"].Text : null;
            g.Conditional["category"] = IsConditional(cat);
            g.Conditional["center_m"] = IsConditional(centre);
            g.Conditional["size_m"] = IsConditional(size);
            g.Conditional["rotation_xyzw"] = IsConditional(rot);
            return g;
        }

        public static double[] Vec(JNode a)
        {
            var v = new double[a.Items.Count];
            for (int i = 0; i < v.Length; i++) v[i] = a.Items[i].Num;
            return v;
        }
    }

    public sealed class RelationResult
    {
        public string Relation, Basis;
        public string[] ObjectIds;
        public Truth Value;
        public bool Conditional;   // a consulted known field was assumed or carried assumptions
        public List<string> Reasons;
        public Dictionary<string, double?> Measures = new Dictionary<string, double?>();
    }

    public sealed class RankResult
    {
        public string Relation, Anchor, Status;   // status: resolved, ambiguous, no_match or insufficient
        public int Rank, PossibleCount, CombinationsChecked;
        public bool Conditional;
        public List<string> ObjectIds = new List<string>(), Reasons = new List<string>();
        public Dictionary<string, double> Distances = new Dictionary<string, double>();
    }

    /// <summary>The relations of one validated scene record under one configuration.</summary>
    public sealed class Relations
    {
        private readonly Dictionary<string, JNode> objects = new Dictionary<string, JNode>();
        private readonly Dictionary<string, ObjectGeometry> geometry = new Dictionary<string, ObjectGeometry>();
        private readonly RelationConfig cfg;

        public Relations(JNode scene, RelationConfig config)
        {
            foreach (var o in scene["objects"].Items) objects[o["object_id"].Text] = o;
            cfg = config;
        }

        public ObjectGeometry Geo(string oid)
        {
            ObjectGeometry g;
            if (geometry.TryGetValue(oid, out g)) return g;
            JNode o;
            if (!objects.TryGetValue(oid, out o)) throw new ArgumentException(oid + " is not an object of the scene");
            g = ObjectGeometry.Build(o);
            geometry[oid] = g;
            return g;
        }

        private double Th(string k) { return cfg.Thresholds[k]; }
        private double Bd(string k) { return cfg.Bands[k]; }

        private static List<string> Why(params object[] pairs)   // (name, Truth) pairs
        {
            var out_ = new List<string>();
            for (int i = 0; i < pairs.Length; i += 2)
            {
                var v = (Truth)pairs[i + 1];
                if (v == Truth.False) out_.Add("failed:" + (string)pairs[i]);
                else if (v == Truth.Unknown) out_.Add("boundary:" + (string)pairs[i]);
            }
            return out_;
        }

        /// <summary>The fields a result consulted, as (object, field) pairs, in the Python's order.</summary>
        private static List<string[]> Refs(params List<string[]>[] parts)
        {
            var l = new List<string[]>();
            foreach (var p in parts) l.AddRange(p);
            return l;
        }
        private static List<string[]> F(string oid, params string[] fields)
        {
            var l = new List<string[]>();
            foreach (var f in fields) l.Add(new[] { oid, f });
            return l;
        }
        private static List<string[]> BoxRefs(params string[] oids)
        {
            var l = new List<string[]>();
            foreach (var o in oids) l.AddRange(F(o, "center_m", "size_m", "rotation_xyzw"));
            return l;
        }
        private bool AnyConditional(List<string[]> refs)
        {
            foreach (var r in refs) if (Geo(r[0]).Conditional[r[1]]) return true;
            return false;
        }

        private RelationResult Result(string relation, string[] ids, Truth value, IEnumerable<string> reasons,
                                      string basis, Dictionary<string, double?> measures, List<string[]> refs)
        {
            return new RelationResult
            {
                Relation = relation, ObjectIds = ids, Value = value, Conditional = refs != null && AnyConditional(refs),
                Reasons = value == Truth.True ? new List<string>() : Py.Distinct(reasons),
                Basis = value == Truth.True ? basis : null, Measures = measures ?? new Dictionary<string, double?>()
            };
        }

        private static List<string> Cat(params IEnumerable<string>[] parts)
        {
            var l = new List<string>();
            foreach (var p in parts) l.AddRange(p);
            return l;
        }

        private void Overlap(string t, string a, out double? ratio, out Truth c, out List<string> why)
        {
            ratio = Resolution.Geo.FootprintOverlap(Geo(t).Box, Geo(a).Box);
            if (ratio == null) { c = Truth.Unknown; why = new List<string> { "degenerate_footprint" }; return; }
            c = Tri.Ge(ratio.Value, Th("footprint_overlap_min"), Bd("overlap_band"));
            why = Why("overlap", c);
        }

        private RelationResult NotDistinct(string relation, string[] ids)
        {
            return Result(relation, ids, Truth.False, new[] { "not_distinct" }, null, null, null);
        }

        public RelationResult Evaluate(string relation, string target, string[] anchors)
        {
            switch (relation)
            {
                case "near": return Near(target, anchors[0]);
                case "above": return Above(target, anchors[0]);
                case "on": return On(target, anchors[0]);
                case "below": return Below(target, anchors[0]);
                case "inside": return Inside(target, anchors[0]);
                case "between": return Between(target, anchors[0], anchors[1]);
                default: throw new ArgumentException("not a relation of the library: " + relation);
            }
        }

        public RelationResult Near(string t, string a)
        {
            var ids = new[] { t, a };
            if (t == a) return NotDistinct("near", ids);
            ObjectGeometry gt = Geo(t), ga = Geo(a);
            var refs = BoxRefs(t, a);
            if (gt.Box == null || ga.Box == null) return Result("near", ids, Truth.Unknown, Cat(gt.Missing, ga.Missing), null, null, refs);
            double d = Resolution.Geo.BoxDistance(gt.Box, ga.Box);
            var c = Tri.Le(d, Th("near_max_m"), Bd("distance_band_m"));
            return Result("near", ids, c, Why("distance", c), null, new Dictionary<string, double?> { { "distance_m", d } }, refs);
        }

        public RelationResult Above(string t, string a)
        {
            var ids = new[] { t, a };
            if (t == a) return NotDistinct("above", ids);
            ObjectGeometry gt = Geo(t), ga = Geo(a);
            var refs = BoxRefs(t, a);
            if (gt.Box == null || ga.Box == null) return Result("above", ids, Truth.Unknown, Cat(gt.Missing, ga.Missing), null, null, refs);
            double gap = gt.Box.ZMin - ga.Box.ZMax;
            var cGap = Tri.Ge(gap, Th("above_gap_min_m"), Bd("vertical_band_m"));
            Overlap(t, a, out double? ratio, out Truth cOv, out List<string> ovWhy);
            return Result("above", ids, Tri.And(cGap, cOv), Cat(Why("vertical_gap", cGap), ovWhy), null,
                          new Dictionary<string, double?> { { "gap_m", gap }, { "overlap", ratio } }, refs);
        }

        public RelationResult On(string t, string a)
        {
            var ids = new[] { t, a };
            if (t == a) return NotDistinct("on", ids);
            ObjectGeometry gt = Geo(t), ga = Geo(a);
            var catRefs = F(a, "category");
            if (ga.Category == null) return Result("on", ids, Truth.Unknown, new[] { "category_unknown:" + a }, null, null, catRefs);
            if (!cfg.Categories["support"].Contains(ga.Category))
                return Result("on", ids, Truth.Unknown, new[] { "support_not_represented:" + a }, null, null, catRefs);
            var refs = Refs(catRefs, BoxRefs(t, a));
            if (gt.Box == null || ga.Box == null) return Result("on", ids, Truth.Unknown, Cat(gt.Missing, ga.Missing), null, null, refs);
            double tilt = ga.Box.TiltDeg;
            if (tilt > Th("on_support_tilt_max_deg"))
                return Result("on", ids, Truth.Unknown, new[] { "support_tilted:" + a }, null,
                              new Dictionary<string, double?> { { "tilt_deg", tilt } }, refs);
            double gap = gt.Box.ZMin - ga.Box.ZMax;
            double vb = Bd("vertical_band_m");
            var cGap = Tri.Le(gap, Th("on_gap_max_m"), vb);
            var cPen = Tri.Ge(gap, -Th("on_penetration_max_m"), vb);
            Overlap(t, a, out double? ratio, out Truth cOv, out List<string> ovWhy);
            return Result("on", ids, Tri.And(cGap, cPen, cOv), Cat(Why("gap", cGap, "penetration", cPen), ovWhy),
                          "top_face_support", new Dictionary<string, double?> { { "gap_m", gap }, { "overlap", ratio }, { "tilt_deg", tilt } }, refs);
        }

        public RelationResult Below(string t, string a)
        {
            var ids = new[] { t, a };
            if (t == a) return NotDistinct("below", ids);
            ObjectGeometry gt = Geo(t), ga = Geo(a);
            var refs = Refs(BoxRefs(t, a), F(a, "category"));
            if (gt.Box == null || ga.Box == null)
            {
                var extra = ga.Category != null ? new List<string>() : new List<string> { "category_unknown:" + a };
                return Result("below", ids, Truth.Unknown, Cat(gt.Missing, ga.Missing, extra), null, null, refs);
            }
            double vb = Bd("vertical_band_m");
            Overlap(t, a, out double? ratio, out Truth cOv, out List<string> ovWhy);
            double separatedGap = gt.Box.ZMax - ga.Box.ZMin;
            var cSep = Tri.Le(separatedGap, -Th("below_separated_gap_min_m"), vb);
            var separated = Tri.And(cSep, cOv);
            double offset = Math.Abs(gt.Box.ZMin - ga.Box.ZMin);
            double margin = gt.Box.ZMax - ga.Box.ZMax;
            Truth furniture;
            List<string> fWhy;
            if (ga.Category == null) { furniture = Truth.Unknown; fWhy = new List<string> { "category_unknown:" + a }; }
            else if (!cfg.Categories["furniture"].Contains(ga.Category)) { furniture = Truth.False; fWhy = new List<string> { "failed:furniture_category" }; }
            else
            {
                var cOff = Tri.Le(offset, Th("below_furniture_bottom_max_m"), Bd("distance_band_m"));
                var cMar = Tri.Le(margin, -Th("below_furniture_top_margin_m"), vb);
                furniture = Tri.And(cOff, cMar, cOv);
                fWhy = Cat(Why("bottom_offset", cOff, "top_margin", cMar), ovWhy);
            }
            var value = Tri.Or(separated, furniture);
            var names = new List<string>();
            if (separated == Truth.True) names.Add("separated");
            if (furniture == Truth.True) names.Add("beneath_furniture");
            string basis = names.Count > 0 ? string.Join("+", names) : null;
            return Result("below", ids, value, Cat(Why("separated_gap", cSep), ovWhy, fWhy), basis,
                          new Dictionary<string, double?> { { "separated_gap_m", separatedGap }, { "bottom_offset_m", offset },
                                                            { "top_margin_m", margin }, { "overlap", ratio } }, refs);
        }

        public RelationResult Inside(string t, string a)
        {
            var ids = new[] { t, a };
            if (t == a) return NotDistinct("inside", ids);
            ObjectGeometry gt = Geo(t), ga = Geo(a);
            var catRefs = F(a, "category");
            if (ga.Category == null) return Result("inside", ids, Truth.Unknown, new[] { "category_unknown:" + a }, null, null, catRefs);
            if (!cfg.Categories["container"].Contains(ga.Category))
                return Result("inside", ids, Truth.Unknown, new[] { "container_not_represented:" + a }, null, null, catRefs);
            var refs = Refs(catRefs, BoxRefs(t, a));
            if (gt.Box == null || ga.Box == null) return Result("inside", ids, Truth.Unknown, Cat(gt.Missing, ga.Missing), null, null, refs);
            double excess = Resolution.Geo.ContainmentExcess(gt.Box, ga.Box);
            var c = Tri.Le(excess, Th("inside_tolerance_m"), Bd("vertical_band_m"));
            return Result("inside", ids, c, Why("containment", c), "outer_box_containment",
                          new Dictionary<string, double?> { { "excess_m", excess } }, refs);
        }

        public RelationResult Between(string t, string a, string b)
        {
            var ids = new[] { t, a, b };
            if (t == a || t == b || a == b) return NotDistinct("between", ids);
            ObjectGeometry gt = Geo(t), ga = Geo(a), gb = Geo(b);
            var centreRefs = Refs(F(t, "center_m"), F(a, "center_m"), F(b, "center_m"));
            if (gt.Centre == null || ga.Centre == null || gb.Centre == null)
            {
                var missing = new List<string>();
                foreach (var x in new[] { gt, ga, gb }) if (x.Centre == null) missing.Add("missing_centre:" + x.ObjectId);
                return Result("between", ids, Truth.Unknown, missing, null, null, centreRefs);
            }
            double db = Bd("distance_band_m");
            var ab = Resolution.Geo.Sub(gb.Centre, ga.Centre);
            double separation = Resolution.Geo.Norm(ab);
            if (Tri.Ge(separation, Th("between_min_separation_m"), db) != Truth.True)
                return Result("between", ids, Truth.Unknown, new[] { "degenerate_anchors" }, null,
                              new Dictionary<string, double?> { { "separation_m", separation } }, centreRefs);
            double along = Resolution.Geo.Dot(Resolution.Geo.Sub(gt.Centre, ga.Centre), ab) / separation;
            var foot = Resolution.Geo.Add(ga.Centre, Resolution.Geo.Scale(ab, along / separation));
            double lateral = Resolution.Geo.Norm(Resolution.Geo.Sub(gt.Centre, foot));
            double limit = Py.Max(Th("between_lateral_min_m"), Th("between_lateral_ratio") * separation);
            var cA = Tri.Ge(along, 0.0, db);
            var cB = Tri.Le(along, separation, db);
            var cLat = Tri.Le(lateral, limit, db);
            var measures = new Dictionary<string, double?> { { "separation_m", separation }, { "along_m", along },
                                                             { "lateral_m", lateral }, { "lateral_limit_m", limit } };
            var conditions = new List<Truth> { cA, cB, cLat };
            var reasons = Why("endpoint_a", cA, "endpoint_b", cB, "lateral", cLat);
            foreach (var pair in new[] { new object[] { "overlap_a", ga }, new object[] { "overlap_b", gb } })
            {
                string name = (string)pair[0];
                var gx = (ObjectGeometry)pair[1];
                if (gt.Box == null || gx.Box == null)
                {
                    conditions.Add(Truth.Unknown);
                    reasons.AddRange(gt.Missing);
                    reasons.AddRange(gx.Missing);
                    measures[name] = null;
                    continue;
                }
                double? ratio = Resolution.Geo.VerticalOverlap(gt.Box, gx.Box) > 0.0 ? Resolution.Geo.FootprintOverlap(gt.Box, gx.Box) : 0.0;
                if (ratio == null)
                {
                    conditions.Add(Truth.Unknown);
                    reasons.Add("degenerate_footprint");
                }
                else
                {
                    var c = Tri.Le(ratio.Value, Th("between_overlap_max"), Bd("overlap_band"));
                    conditions.Add(c);
                    reasons.AddRange(Why(name, c));
                }
                measures[name] = ratio;
            }
            var refs = Refs(centreRefs, F(t, "size_m", "rotation_xyzw"), F(a, "size_m", "rotation_xyzw"), F(b, "size_m", "rotation_xyzw"));
            return Result("between", ids, Tri.And(conditions.ToArray()), reasons, null, measures, refs);
        }

        /// <summary>Rank k (1-3) of the closest or farthest candidates to an anchor, by centre distance; every
        /// combination of the possible candidates is evaluated, as the Python does (the caller bounds the count).</summary>
        public RankResult Rank(string relation, string anchor, IEnumerable<string> candidates, IEnumerable<string> possible, int k)
        {
            if (relation != "closest" && relation != "farthest") throw new ArgumentException("unknown ranking relation " + relation);
            if (k < 1 || k > 3) throw new ArgumentException("ranks 1, 2 and 3 are supported");
            var definite = new List<string>();
            foreach (var c in Py.Distinct(candidates)) if (c != anchor) definite.Add(c);
            var maybe = new List<string>();
            foreach (var p in Py.Distinct(possible)) if (p != anchor) maybe.Add(p);
            foreach (var m in maybe) if (definite.Contains(m)) throw new ArgumentException("an object can't be both a candidate and a possible candidate");
            var everyone = new List<string> { anchor };
            everyone.AddRange(definite);
            everyone.AddRange(maybe);
            var r = new RankResult { Relation = relation, Anchor = anchor, Rank = k, PossibleCount = maybe.Count };
            foreach (var x in everyone) if (Geo(x).Conditional["center_m"]) r.Conditional = true;
            var anchorCentre = Geo(anchor).Centre;
            if (anchorCentre != null)
            {
                var both = new List<string>(definite);
                both.AddRange(maybe);
                foreach (var x in both)
                {
                    var c = Geo(x).Centre;
                    if (c != null) r.Distances[x] = Resolution.Geo.Norm(Resolution.Geo.Sub(c, anchorCentre));
                }
            }
            var missing = new List<string>();
            foreach (var x in everyone) if (Geo(x).Centre == null) missing.Add("missing_centre:" + x);
            if (missing.Count > 0) { r.Status = "insufficient"; r.Reasons = missing; return r; }
            var outcomes = new Dictionary<string, KeyValuePair<string, List<string>>>();
            int combinations = 0;
            for (int size = 0; size <= maybe.Count; size++)
            {
                foreach (var subset in Combinations(maybe, size))
                {
                    combinations++;
                    var members = new List<string>(definite);
                    members.AddRange(subset);
                    var o = RankOutcome(members, r.Distances, relation, k);
                    outcomes[o.Key + "|" + string.Join(",", o.Value)] = o;
                }
            }
            r.CombinationsChecked = combinations;
            if (outcomes.Count > 1)
            {
                r.Status = "insufficient";
                r.Reasons = new List<string> { "possible_competitors:" + string.Join(",", maybe) };
                return r;
            }
            foreach (var o in outcomes.Values) { r.Status = o.Key; r.ObjectIds = o.Value; }
            r.Reasons = r.Status == "ambiguous" ? new List<string> { "tie" } :
                        (r.Status == "no_match" ? new List<string> { "too_few_candidates" } : new List<string>());
            return r;
        }

        private KeyValuePair<string, List<string>> RankOutcome(List<string> members, Dictionary<string, double> distances, string relation, int k)
        {
            double sign = relation == "closest" ? 1.0 : -1.0;
            var order = new List<string>(members);
            order.Sort((x, y) =>
            {
                double dx = sign * distances[x], dy = sign * distances[y];
                if (dx == dy) return string.CompareOrdinal(x, y);
                return dx < dy ? -1 : 1;
            });
            if (order.Count < k) return new KeyValuePair<string, List<string>>("no_match", new List<string>());
            double tie = Bd("rank_tie_m");
            var linked = new bool[Math.Max(0, order.Count - 1)];
            for (int i = 0; i < order.Count - 1; i++) linked[i] = Math.Abs(distances[order[i]] - distances[order[i + 1]]) <= tie;
            int lo = k - 1, hi = k - 1;
            while (lo > 0 && linked[lo - 1]) lo--;
            while (hi < order.Count - 1 && linked[hi]) hi++;
            if (lo == hi) return new KeyValuePair<string, List<string>>("resolved", new List<string> { order[k - 1] });
            return new KeyValuePair<string, List<string>>("ambiguous", Py.Sorted(order.GetRange(lo, hi - lo + 1)));
        }

        /// <summary>itertools.combinations: subsets of the given size in lexicographic order of positions.</summary>
        public static IEnumerable<List<string>> Combinations(List<string> items, int size)
        {
            var idx = new int[size];
            for (int i = 0; i < size; i++) idx[i] = i;
            if (size > items.Count) yield break;
            while (true)
            {
                var pick = new List<string>(size);
                foreach (int i in idx) pick.Add(items[i]);
                yield return pick;
                int j = size - 1;
                while (j >= 0 && idx[j] == items.Count - size + j) j--;
                if (j < 0) yield break;
                idx[j]++;
                for (int m = j + 1; m < size; m++) idx[m] = idx[m - 1] + 1;
            }
        }
    }
}
