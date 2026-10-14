using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using SecondEyes.Grounding.Prompting;
using SecondEyes.Grounding.Resolution;
using SecondEyes.Grounding.Parsing;

// A2.7 delivery 2: the port against the Python reference's goldens (grounding/a27/goldens.py), value for value.
//   dotnet run --project tools/csharp-tests/Resolution -- relations GOLDENS.json grounding/relations/relations.v1.json
static class Program
{
    static int checks, fails;
    static readonly Dictionary<string, int> failsBy = new Dictionary<string, int>();

    static void Fail(string kind, string detail)
    {
        fails++;
        failsBy[kind] = failsBy.TryGetValue(kind, out int n) ? n + 1 : 1;
        if (failsBy[kind] <= 5) Console.WriteLine("  FAIL  " + kind + ": " + detail);
    }

    static double D(JNode n) { return double.Parse(n.Text, NumberStyles.Float, CultureInfo.InvariantCulture); }
    static bool Same(double a, double b) { return BitConverter.DoubleToInt64Bits(a) == BitConverter.DoubleToInt64Bits(b); }
    static string R(double x) { return x.ToString("R", CultureInfo.InvariantCulture); }

    static void Num(string kind, string what, JNode expected, double? got)
    {
        checks++;
        if (expected.Kind == JKind.Null) { if (got != null) Fail(kind, what + ": expected None, got " + R(got.Value)); return; }
        if (got == null) { Fail(kind, what + ": expected " + expected.Text + ", got None"); return; }
        if (!Same(D(expected), got.Value)) Fail(kind, what + ": expected " + expected.Text + ", got " + R(got.Value));
    }

    static void Eq(string kind, string what, string expected, string got)
    {
        checks++;
        if (expected != got) Fail(kind, what + ": expected " + expected + ", got " + got);
    }

    static string List(JNode a) { var l = new List<string>(); foreach (var x in a.Items) l.Add(x.Text); return string.Join(",", l); }

    static void BoxCheck(string kind, string where, JNode e, Box b)
    {
        for (int i = 0; i < 8; i++) for (int k = 0; k < 3; k++) Num(kind, where + " corner " + i + "." + k, e["corners"].Items[i].Items[k], b.Corners[i][k]);
        Num(kind, where + " z_min", e["z_min"], b.ZMin);
        Num(kind, where + " z_max", e["z_max"], b.ZMax);
        Eq(kind, where + " footprint size", e["footprint"].Items.Count.ToString(), b.Footprint.Count.ToString());
        if (e["footprint"].Items.Count == b.Footprint.Count)
            for (int i = 0; i < b.Footprint.Count; i++) for (int k = 0; k < 2; k++) Num(kind, where + " footprint " + i, e["footprint"].Items[i].Items[k], b.Footprint[i][k]);
        Num(kind, where + " area", e["area"], b.FootprintArea);
        Num(kind, where + " tilt", e["tilt_deg"], b.TiltDeg);
    }

    static void RelCheck(string kind, string where, JNode e, RelationResult r)
    {
        Eq(kind, where + " value", e["value"].Text, Tri.Name(r.Value));
        Eq(kind, where + " reasons", List(e["reasons"]), string.Join(",", r.Reasons));
        Eq(kind, where + " basis", e["basis"].Kind == JKind.Null ? "None" : e["basis"].Text, r.Basis ?? "None");
        Eq(kind, where + " conditional", e["conditional"].Bool.ToString(), r.Conditional.ToString());
        var keys = new List<string>();
        foreach (var kv in e["measures"].Fields) keys.Add(kv.Key);
        var mine = new List<string>(r.Measures.Keys);
        keys.Sort(string.CompareOrdinal); mine.Sort(string.CompareOrdinal);
        Eq(kind, where + " measures", string.Join(",", keys), string.Join(",", mine));
        foreach (var kv in e["measures"].Fields) if (r.Measures.ContainsKey(kv.Key)) Num(kind, where + " " + kv.Key, kv.Value, r.Measures[kv.Key]);
    }

    static void PairGeometry(string kind, string where, JNode e, Box a, Box b)
    {
        Num(kind, where + " distance", e["distance"], Geo.BoxDistance(a, b));
        Eq(kind, where + " intersect", e["intersect"].Bool.ToString(), Geo.BoxesIntersect(a, b).ToString());
        Num(kind, where + " footprint_overlap", e["footprint_overlap"], Geo.FootprintOverlap(a, b));
        Num(kind, where + " vertical_overlap", e["vertical_overlap"], Geo.VerticalOverlap(a, b));
        Num(kind, where + " containment_excess", e["containment_excess"], Geo.ContainmentExcess(a, b));
    }

    static double[] Vec(JNode a) { var v = new double[a.Items.Count]; for (int i = 0; i < v.Length; i++) v[i] = D(a.Items[i]); return v; }

    static int Relations(string goldens, string configPath)
    {
        var g = JParse.Parse(File.ReadAllText(goldens));
        var cfg = RelationConfig.From(JParse.Parse(File.ReadAllText(configPath)));
        var rels = new List<Relations>();
        foreach (var s in g["scenes"].Items) rels.Add(new Relations(s, cfg));
        foreach (var c in g["boxes"].Items)
            BoxCheck("box", c["object_id"].Text, c, rels[(int)c["scene"].Num].Geo(c["object_id"].Text).Box);
        foreach (var c in g["pairs"].Items)
        {
            var rel = rels[(int)c["scene"].Num];
            string t = c["t"].Text, a = c["a"].Text, w = t + "/" + a;
            foreach (var kv in c["relations"].Fields) RelCheck("pair " + kv.Key, w, kv.Value, rel.Evaluate(kv.Key, t, new[] { a }));
            if (c.Has("geometry")) PairGeometry("pair geometry", w, c["geometry"], rel.Geo(t).Box, rel.Geo(a).Box);
        }
        foreach (var c in g["triples"].Items)
        {
            var rel = rels[(int)c["scene"].Num];
            RelCheck("between", c["t"].Text + "/" + c["a"].Text + "/" + c["b"].Text, c["result"], rel.Between(c["t"].Text, c["a"].Text, c["b"].Text));
        }
        foreach (var c in g["ranks"].Items)
        {
            var rel = rels[(int)c["scene"].Num];
            var cands = new List<string>(); foreach (var x in c["candidates"].Items) cands.Add(x.Text);
            var poss = new List<string>(); foreach (var x in c["possible"].Items) poss.Add(x.Text);
            var r = rel.Rank(c["relation"].Text, c["anchor"].Text, cands, poss, (int)c["k"].Num);
            string w = c["relation"].Text + " " + c["k"].Num + " of " + c["anchor"].Text;
            Eq("rank", w + " status", c["status"].Text, r.Status);
            Eq("rank", w + " ids", List(c["object_ids"]), string.Join(",", r.ObjectIds));
            Eq("rank", w + " reasons", List(c["reasons"]), string.Join(",", r.Reasons));
            Eq("rank", w + " combinations", ((int)c["combinations"].Num).ToString(), r.CombinationsChecked.ToString());
            Eq("rank", w + " conditional", c["conditional"].Bool.ToString(), r.Conditional.ToString());
            foreach (var kv in c["distances"].Fields) Num("rank", w + " distance " + kv.Key, kv.Value, r.Distances.ContainsKey(kv.Key) ? r.Distances[kv.Key] : (double?)null);
            Eq("rank", w + " distance keys", c["distances"].Fields.Count.ToString(), r.Distances.Count.ToString());
        }
        int i = 0;
        foreach (var c in g["synthetic"].Items)
        {
            var a = new Box(Vec(c["a"]["centre"]), Vec(c["a"]["size"]), Vec(c["a"]["q"]));
            var b = new Box(Vec(c["b"]["centre"]), Vec(c["b"]["size"]), Vec(c["b"]["q"]));
            BoxCheck("synthetic box", "#" + i + "a", c["box_a"], a);
            BoxCheck("synthetic box", "#" + i + "b", c["box_b"], b);
            PairGeometry("synthetic pair", "#" + i, c, a, b);
            i++;
        }
        var bscenes = new List<Relations>();
        foreach (var sc in g["boundary"]["scenes"].Items) bscenes.Add(new Relations(sc, cfg));
        foreach (var c in g["boundary"]["cases"].Items)
        {
            var rel = bscenes[(int)c["scene"].Num];
            string kind = c["kind"].Text;
            if (kind == "binary")
                RelCheck("boundary " + c["relation"].Text, "scene " + c["scene"].Num, c["result"], rel.Evaluate(c["relation"].Text, c["t"].Text, new[] { c["a"].Text }));
            else if (kind == "between")
                RelCheck("boundary between", "scene " + c["scene"].Num, c["result"], rel.Between(c["t"].Text, c["a"].Text, c["b"].Text));
            else
            {
                var cands = new List<string>(); foreach (var x in c["candidates"].Items) cands.Add(x.Text);
                var r = rel.Rank(c["relation"].Text, c["anchor"].Text, cands, new List<string>(), (int)c["k"].Num);
                string w = "scene " + c["scene"].Num + " " + c["relation"].Text + " " + c["k"].Num;
                Eq("boundary rank", w + " status", c["status"].Text, r.Status);
                Eq("boundary rank", w + " ids", List(c["object_ids"]), string.Join(",", r.ObjectIds));
            }
        }
        Console.WriteLine("boundary: " + g["boundary"]["cases"].Items.Count + " cases in " + g["boundary"]["scenes"].Items.Count + " scenes");
        Console.WriteLine("relations: " + g["scenes"].Items.Count + " scenes, " + g["boxes"].Items.Count + " boxes, " + g["pairs"].Items.Count
                          + " pairs, " + g["triples"].Items.Count + " triples, " + g["ranks"].Items.Count + " ranks, " + g["synthetic"].Items.Count + " synthetic");
        foreach (var kv in failsBy) Console.WriteLine("  failures in " + kv.Key + ": " + kv.Value);
        Console.WriteLine(checks + " values compared, " + fails + " differ");
        return fails == 0 ? 0 : 1;
    }

    static int Schemas(string goldens, string schemasDir)
    {
        var g = JParse.Parse(File.ReadAllText(goldens));
        var schemas = new Dictionary<string, Schema>();
        int cases = 0, valid = 0;
        foreach (var c in g["cases"].Items)
        {
            string name = c["schema"].Text;
            if (!schemas.ContainsKey(name)) schemas[name] = new Schema(JParse.Parse(File.ReadAllText(Path.Combine(schemasDir, name))));
            var want = new List<string>();
            foreach (var e in c["errors"].Items) want.Add(e.Items[0].Text + " " + e.Items[1].Text);
            var got = new List<string>();
            foreach (var e in schemas[name].Errors(c["instance"])) got.Add(e.ToString());
            want.Sort(string.CompareOrdinal); got.Sort(string.CompareOrdinal);
            checks++; cases++;
            if (want.Count == 0) valid++;
            if (string.Join("|", want) != string.Join("|", got))
                Fail("schema " + name, "expected [" + string.Join("; ", want) + "], got [" + string.Join("; ", got) + "]");
        }
        Console.WriteLine("schemas: " + cases + " cases (" + valid + " valid) over " + schemas.Count + " schemas");
        foreach (var kv in failsBy) Console.WriteLine("  failures in " + kv.Key + ": " + kv.Value);
        Console.WriteLine(checks + " cases compared, " + fails + " differ");
        return fails == 0 ? 0 : 1;
    }

    // ---------------------------------------------------------------- validation
    static string Canon(JNode n)
    {
        switch (n.Kind)
        {
            case JKind.Null: return "null";
            case JKind.Bool: return n.Bool ? "true" : "false";
            case JKind.Number: return n.Text;
            case JKind.String: return ResolverValidation.PyJsonString(n.Text);
            case JKind.Array: { var parts = new List<string>(); foreach (var x in n.Items) parts.Add(Canon(x)); return "[" + string.Join(",", parts) + "]"; }
            default:
            {
                var keys = new List<KeyValuePair<string, JNode>>(n.Fields);
                keys.Sort((a, b) => string.CompareOrdinal(a.Key, b.Key));
                var parts = new List<string>();
                foreach (var kv in keys) parts.Add(ResolverValidation.PyJsonString(kv.Key) + ":" + Canon(kv.Value));
                return "{" + string.Join(",", parts) + "}";
            }
        }
    }

    static string S(string x) { return x == null ? "null" : ResolverValidation.PyJsonString(x); }
    static string L(IEnumerable<string> xs) { var p = new List<string>(); foreach (var x in xs) p.Add(S(x)); return "[" + string.Join(",", p) + "]"; }

    static string Canon(List<NormInterpretation> norm)
    {
        var out_ = new List<string>();
        foreach (var it in norm)
        {
            if (it.Kind != "query") { out_.Add(Canon(it.Raw)); continue; }
            var nodeIds = new List<string>(it.Nodes.Keys);
            nodeIds.Sort(string.CompareOrdinal);
            var nodes = new List<string>();
            foreach (var id in nodeIds)
            {
                var n = it.Nodes[id];
                var cons = new List<string>();
                foreach (var c in n.Constraints) cons.Add("{\"anchors\":" + L(c.Anchors) + ",\"frame\":" + S(c.Frame) + ",\"relation\":" + S(c.Relation) + "}");
                string rank = n.RankRelation == null ? "null" : "{\"anchor\":" + S(n.RankAnchor) + ",\"k\":" + n.RankK + ",\"relation\":" + S(n.RankRelation) + "}";
                nodes.Add(S(id) + ":{\"category\":" + S(n.Category) + ",\"colours_all\":" + L(n.ColoursAll) + ",\"constraints\":[" + string.Join(",", cons)
                          + "],\"node_id\":" + S(n.NodeId) + ",\"rank\":" + rank + "}");
            }
            var between = new List<string>();
            foreach (var b in it.Between) between.Add(L(b));
            out_.Add("{\"action\":" + S(it.Action) + ",\"between\":[" + string.Join(",", between) + "],\"interpretation_id\":" + S(it.InterpretationId)
                     + ",\"kind\":\"query\",\"nodes\":{" + string.Join(",", nodes) + "},\"order\":" + L(it.Order) + ",\"root\":" + S(it.Root) + "}");
        }
        return "[" + string.Join(",", out_) + "]";
    }

    static string IssueSet(IEnumerable<Issue> issues)
    {
        var l = new List<string>();
        foreach (var i in issues) l.Add(i.File + " " + i.Path + " " + i.Code);
        l.Sort(string.CompareOrdinal);
        return string.Join(" | ", l);
    }

    static string IssueSet(JNode expected)
    {
        var l = new List<string>();
        foreach (var i in expected.Items) l.Add(i.Items[0].Text + " " + i.Items[1].Text + " " + i.Items[2].Text);
        l.Sort(string.CompareOrdinal);
        return string.Join(" | ", l);
    }

    static int Validation(string goldens, string schemasDir)
    {
        var g = JParse.Parse(File.ReadAllText(goldens));
        var cache = new Dictionary<string, Schema>();
        Func<string, Schema> schema = name =>
        {
            if (!cache.ContainsKey(name)) cache[name] = new Schema(JParse.Parse(File.ReadAllText(Path.Combine(schemasDir, name))));
            return cache[name];
        };
        var contract = new Contract(schema);
        foreach (var c in g["contract"].Items)
        {
            var named = new List<KeyValuePair<string, JNode>>();
            foreach (var r in c["records"].Items) named.Add(new KeyValuePair<string, JNode>(r.Items[0].Text, r.Items[1]));
            Eq("contract", c["name"].Text, IssueSet(c["issues"]), IssueSet(contract.Validate(named)));
        }
        foreach (var q in g["queries"].Items)
        {
            List<NormInterpretation> norm;
            var found = ResolverValidation.CheckQuery(q["query"], schema("grounding-query.v1.json"), out norm);
            Eq("query issues", q["query"].Has("query_id") && q["query"]["query_id"].Kind == JKind.String ? q["query"]["query_id"].Text : "?", IssueSet(q["issues"]), IssueSet(found));
            string want = q["normalized"].Kind == JKind.Null ? "None" : q["normalized"].Text;
            Eq("query normalized", q["query"].Has("query_id") && q["query"]["query_id"].Kind == JKind.String ? q["query"]["query_id"].Text : "?", want, norm == null ? "None" : Canon(norm));
        }
        foreach (var c in g["configs"].Items)
            Eq("config", "resolver config", IssueSet(c["issues"]), IssueSet(ResolverValidation.CheckConfig(c["config"], schema("resolver-config.v1.json"))));
        foreach (var r in g["records"].Items)
        {
            var maps = new List<JNode>(r["maps"].Items);
            var warnings = new List<Issue>();
            var errors = ResolverValidation.CheckRecords(r["scene"], r["command"], maps, contract, warnings);
            Eq("records errors", r["label"].Text, IssueSet(r["errors"]), IssueSet(errors));
            Eq("records warnings", r["label"].Text, IssueSet(r["warnings"]), IssueSet(warnings));
        }
        foreach (var c in g["contexts"].Items)
            Eq("context", "query context", IssueSet(c["issues"]), IssueSet(ResolverValidation.CheckContext(c["query"], c["scene"], c["command"])));
        Console.WriteLine("validation: " + g["contract"].Items.Count + " contract cases, " + g["queries"].Items.Count + " queries, " + g["configs"].Items.Count
                          + " configurations, " + g["records"].Items.Count + " record sets, " + g["contexts"].Items.Count + " contexts");
        foreach (var kv in failsBy) Console.WriteLine("  failures in " + kv.Key + ": " + kv.Value);
        Console.WriteLine(checks + " comparisons, " + fails + " differ");
        return fails == 0 ? 0 : 1;
    }

    // ---------------------------------------------------------------- resolution
    static string Ids(JNode a) { var l = new List<string>(); foreach (var x in a.Items) l.Add(x.Text); return string.Join(",", l); }
    static string Opt(JNode n) { return n.Kind == JKind.Null ? "None" : n.Text; }

    static void RunCheck(string where, JNode e, RunRecord r)
    {
        Eq("run status", where, e["processing_status"].Text, r.ProcessingStatus);
        Eq("run issues", where, IssueSet(e["issues"]), IssueSet(r.Issues));
        Eq("run warnings", where, IssueSet(e["warnings"]), IssueSet(r.Warnings));
        var b = e["budget"];
        Eq("run budget", where, b.Kind == JKind.Null ? "None" : b["name"].Text + " " + b["limit"].Text + " " + b["used"].Text + " " + b["next_required"].Text,
           r.Budget == null ? "None" : r.Budget.Name + " " + r.Budget.Limit + " " + r.Budget.Used + " " + r.Budget.NextRequired);
        if (e["work"].Kind != JKind.Null)
            foreach (var kv in e["work"].Fields) Eq("run work", where + " " + kv.Key, kv.Value.Text, r.Work[kv.Key].ToString());
        var x = e["result"];
        if (x.Kind == JKind.Null) { Eq("run result", where, "None", r.Result == null ? "None" : "a result"); return; }
        if (r.Result == null) { Eq("run result", where, "a result", "None"); return; }
        Eq("result status", where, x["status"].Text, r.Result.Status);
        Eq("result action", where, Opt(x["action"]), r.Result.Action ?? "None");
        Eq("result target", where, Opt(x["target_id"]), r.Result.TargetId ?? "None");
        Eq("result reason code", where, x["reason_code"].Text, r.Result.ReasonCode);
        Eq("result reason codes", where, Ids(x["reason_codes"]), string.Join(",", r.Result.ReasonCodes));
        Eq("result conditional", where, x["conditional"].Bool.ToString(), r.Result.Conditional.ToString());
        var c = x["candidates"];
        Eq("candidates target_ids", where, Ids(c["target_ids"]), string.Join(",", r.Result.TargetIds));
        Eq("candidates uncertain", where, Ids(c["uncertain_match_ids"]), string.Join(",", r.Result.UncertainMatchIds));
        Eq("candidates rank definite", where, Ids(c["rank_definite_eligible_ids"]), string.Join(",", r.Result.RankDefiniteEligibleIds));
        Eq("candidates rank possible", where, Ids(c["rank_possible_eligible_ids"]), string.Join(",", r.Result.RankPossibleEligibleIds));
        Eq("candidates coverage", where, c["coverage"].Text, r.Result.Coverage);
    }

    static int Resolution(string goldens, string schemasDir, string relationsPath, string directionsPath)
    {
        var g = JParse.Parse(File.ReadAllText(goldens));
        var cache = new Dictionary<string, Schema>();
        Func<string, Schema> schema = name =>
        {
            if (!cache.ContainsKey(name)) cache[name] = new Schema(JParse.Parse(File.ReadAllText(Path.Combine(schemasDir, name))));
            return cache[name];
        };
        var setup = ResolverSetup.Load(schema, JParse.Parse(File.ReadAllText(relationsPath)), JParse.Parse(File.ReadAllText(directionsPath)));
        var checker = new Checker(setup, g["protocol"]);
        var recs = g["records"].Items;
        var outcomes = new Dictionary<string, int>();
        foreach (var c in g["cases"].Items)
        {
            string where = c["name"].Text;
            JNode scene = recs[(int)c["scene"].Num], command = recs[(int)c["command"].Num], cmap = recs[(int)c["map"].Num];
            if (c["kind"].Text == "checker")
            {
                var o = checker.Resolve(c["interp"], scene, command, cmap);
                Eq("outcome", where, c["outcome"].Text, o.Outcome);
                if (c["run"].Kind == JKind.Null) Eq("query issues", where, IssueSet(c["query_issues"]), IssueSet(o.QueryIssues));
                else RunCheck(where, c["run"], o.Run);
            }
            else
            {
                var r = Resolver.Resolve(scene, command, c["query"], c["config"], new List<JNode> { cmap }, setup);
                string outcome = r.ProcessingStatus == "completed" ? r.Result.Status : r.ProcessingStatus;
                Eq("outcome", where, c["outcome"].Text, outcome);
                RunCheck(where, c["run"], r);
            }
            string k = c["outcome"].Text;
            outcomes[k] = outcomes.TryGetValue(k, out int n) ? n + 1 : 1;
        }
        var parts = new List<string>();
        foreach (var kv in outcomes) parts.Add(kv.Key + " " + kv.Value);
        Console.WriteLine("resolution: " + g["cases"].Items.Count + " cases (" + string.Join(", ", parts) + ")");
        foreach (var kv in failsBy) Console.WriteLine("  failures in " + kv.Key + ": " + kv.Value);
        Console.WriteLine(checks + " comparisons, " + fails + " differ");
        return fails == 0 ? 0 : 1;
    }

    // ---------------------------------------------------------------- the reader, the pipeline, the native adapter
    static string Repo(string rel) { return Path.Combine(AppContext.BaseDirectory, "../../../../../..", rel); }

    static int Reader(string goldens, string specPath)
    {
        var g = JParse.Parse(File.ReadAllText(goldens));
        var spec = LanguageSpec.From(JParse.Parse(File.ReadAllText(specPath)));
        int valid = 0;
        foreach (var c in g["cases"].Items)
        {
            string text = c["text"].Text, where = "'" + text + "'";
            Parsed p = null;
            try { p = Language.ParseOutput(text, spec); } catch (LanguageError) { }
            Eq("reader verdict", where, c["valid"].Bool ? "valid" : "refused", p != null ? "valid" : "refused");
            if (p == null || !c["valid"].Bool) continue;
            valid++;
            var e = c["parsed"];
            Eq("reader relation", where, e["relation"].Text + " " + (e["k"].Kind == JKind.Null ? "0" : e["k"].Text), p.Relation + " " + p.K);
            Eq("reader target", where, e["target"]["category"].Text + "/" + Opt(e["target"]["colour"]), p.Target.Category + "/" + (p.Target.Colour ?? "None"));
            var anchors = new List<string>();
            foreach (var a in e["anchors"].Items) anchors.Add(a["category"].Text + "/" + Opt(a["colour"]));
            var mine = new List<string>();
            foreach (var a in p.Anchors) mine.Add(a.Category + "/" + (a.Colour ?? "None"));
            Eq("reader anchors", where, string.Join(",", anchors), string.Join(",", mine));
            Eq("reader render", where, c["render"].Text, Language.Render(p));
            var interp = Language.Interpretation(p);
            Eq("reader interpretation", where, c["interpretation"].Text, Canon(interp));
            Eq("reader query", where, c["query"].Text, Canon(Checker.Query(interp, g["scene"], g["command"], ".q")));
        }
        Console.WriteLine("reader: " + g["cases"].Items.Count + " strings (" + valid + " valid)");
        foreach (var kv in failsBy) Console.WriteLine("  failures in " + kv.Key + ": " + kv.Value);
        Console.WriteLine(checks + " comparisons, " + fails + " differ");
        return fails == 0 ? 0 : 1;
    }

    sealed class FakeNative : IParserNative
    {
        public JNode Case;
        public Dictionary<int, byte[]> Bytes;
        public int[] Tokenize(byte[] utf8)
        {
            if (Case["prompt_sha256"].Kind == JKind.Null || SecondEyes.Grounding.Replay.ReplayHash.Sha256Hex(utf8) != Case["prompt_sha256"].Text) return null;
            var ids = new int[Case["prompt_ids"].Items.Count];
            for (int i = 0; i < ids.Length; i++) ids[i] = (int)Case["prompt_ids"].Items[i].Num;
            return ids;
        }
        public int Decode(int[] prompt, int maxNew, int[] output, out int status)
        {
            status = (int)Case["decode_status"].Num;
            if (status < 0) return -1;
            var t = Case["output_tokens"].Items;
            for (int i = 0; i < t.Count; i++) output[i] = (int)t[i].Num;
            return t.Count;
        }
        public byte[] TokenBytes(int token) { byte[] b; return Bytes.TryGetValue(token, out b) ? b : null; }
        public string LastError() { return "fake"; }
    }

    static byte[] Hex(string h) { var b = new byte[h.Length / 2]; for (int i = 0; i < b.Length; i++) b[i] = Convert.ToByte(h.Substring(2 * i, 2), 16); return b; }

    static int Pipeline(string goldens, string schemasDir, string relationsPath, string directionsPath, string specPath, string settingsPath)
    {
        var g = JParse.Parse(File.ReadAllText(goldens));
        var cache = new Dictionary<string, Schema>();
        Func<string, Schema> schema = name =>
        {
            if (!cache.ContainsKey(name)) cache[name] = new Schema(JParse.Parse(File.ReadAllText(Path.Combine(schemasDir, name))));
            return cache[name];
        };
        var setup = ResolverSetup.Load(schema, JParse.Parse(File.ReadAllText(relationsPath)), JParse.Parse(File.ReadAllText(directionsPath)));
        var specNode = JParse.Parse(File.ReadAllText(specPath));
        var assets = new ParserAssets { Spec = LanguageSpec.From(specNode), Prompt = ParserPrompt.From(JParse.Parse(File.ReadAllText(settingsPath))["parser_prompt"]),
                                        Checker = new Checker(setup, g["protocol"]), EndToken = (int)g["end_token"].Num,
                                        MaxNewTokens = (int)specNode["max_new_tokens"].Num, ManifestSha256 = "test" };
        var bytes = new Dictionary<int, byte[]>();
        foreach (var kv in g["token_bytes"].Fields) bytes[int.Parse(kv.Key)] = Hex(kv.Value.Text);
        var kinds = new Dictionary<string, int>();
        foreach (var c in g["cases"].Items)
        {
            var req = new ParserRequest { RequestId = c["name"].Text, Source = "test", SceneText = c["scene_text"].Text, CommandText = c["command_text"].Text, MapText = c["map_text"].Text };
            var o = ParserPipeline.Run(req, assets, new FakeNative { Case = c, Bytes = bytes }, null);
            var e = c["expected"];
            string where = c["name"].Text;
            Eq("pipeline status", where, e["status"].Text, o.Status);
            Eq("pipeline outcome", where, e["outcome"].Text, o.Outcome);
            if (e.Has("target_id"))
            {
                Eq("pipeline target", where, Opt(e["target_id"]), o.TargetObjectId ?? "None");
                Eq("pipeline candidates", where, Ids(e["candidate_ids"]), string.Join(",", o.CandidateIds));
                Eq("pipeline reason code", where, Opt(e["reason_code"]), o.ReasonCode ?? "None");
            }
            if (c["prompt_sha256"].Kind != JKind.Null && o.PromptSha256 != null) Eq("pipeline prompt", where, c["prompt_sha256"].Text, o.PromptSha256);
            kinds[o.Status] = kinds.TryGetValue(o.Status, out int n) ? n + 1 : 1;
        }
        var parts = new List<string>();
        foreach (var kv in kinds) parts.Add(kv.Key + " " + kv.Value);
        Console.WriteLine("pipeline: " + g["cases"].Items.Count + " cases (" + string.Join(", ", parts) + ")");
        foreach (var kv in failsBy) Console.WriteLine("  failures in " + kv.Key + ": " + kv.Value);
        Console.WriteLine(checks + " comparisons, " + fails + " differ");
        return fails == 0 ? 0 : 1;
    }

    static int Native(string goldens, string gguf)
    {
        var g = JParse.Parse(File.ReadAllText(goldens));
        IntPtr model = SecondEyes.Grounding.LlamaNative.se_load_ex(SecondEyes.Grounding.LlamaNative.Utf8(gguf), 1024, 2, 16, 0);
        if (model == IntPtr.Zero) { Console.WriteLine("FAIL  could not load " + gguf); return 1; }
        using (var parser = new SecondEyes.Grounding.LlamaParser(model, g["gbnf"].Text, (int)g["end_token"].Num))
        {
            foreach (var p in g["prompts"].Items)
            {
                var ids = parser.Tokenize(new System.Text.UTF8Encoding(false).GetBytes(p["prompt"].Text));
                Eq("native tokens", "prompt", Ids2(p["native_ids"]), ids == null ? "null" : string.Join(",", ids));
            }
            int maxNew = (int)g["max_new_tokens"].Num;
            foreach (var d in g["decodes"].Items)
            {
                var prompt = new int[d["prompt_ids"].Items.Count];
                for (int i = 0; i < prompt.Length; i++) prompt[i] = (int)d["prompt_ids"].Items[i].Num;
                var output = new int[maxNew];
                int status;
                int n = parser.Decode(prompt, maxNew, output, out status);
                var got = new List<string>();
                for (int i = 0; i < Math.Max(n, 0); i++) got.Add(output[i].ToString());
                Eq("native decode", "tokens", Ids2(d["token_ids"]), string.Join(",", got));
                Eq("native decode", "status", d["status"].Text, status == 0 ? "completed" : (status == 1 ? "length_exhausted" : (status == 2 ? "empty_allowed_set" : "technical_failure")));
            }
            foreach (var m in g["masks"].Items)
            {
                var prefix = new int[m["prefix"].Items.Count];
                for (int i = 0; i < prefix.Length; i++) prefix[i] = (int)m["prefix"].Items[i].Num;
                string sha;
                int count = parser.AllowedAfter(prefix, out sha);
                Eq("native mask", "count", m["count"].Text, count.ToString());
                Eq("native mask", "ids", m["ids_sha256"].Text, sha ?? "null");
            }
        }
        SecondEyes.Grounding.LlamaNative.se_free(model);
        Console.WriteLine("native: " + g["prompts"].Items.Count + " prompts, " + g["decodes"].Items.Count + " decodes, " + g["masks"].Items.Count + " masks through P/Invoke");
        foreach (var kv in failsBy) Console.WriteLine("  failures in " + kv.Key + ": " + kv.Value);
        Console.WriteLine(checks + " comparisons, " + fails + " differ");
        return fails == 0 ? 0 : 1;
    }

    static string Ids2(JNode a) { var l = new List<string>(); foreach (var x in a.Items) l.Add(x.Text); return string.Join(",", l); }

    // ---------------------------------------------------------------- acceptance on real data (criteria 1 and 2)
    static int Acceptance(string file, string schemasDir, string relationsPath, string directionsPath, string specPath)
    {
        var g = JParse.Parse(File.ReadAllText(file));
        var cache = new Dictionary<string, Schema>();
        Func<string, Schema> schema = name =>
        {
            if (!cache.ContainsKey(name)) cache[name] = new Schema(JParse.Parse(File.ReadAllText(Path.Combine(schemasDir, name))));
            return cache[name];
        };
        var setup = ResolverSetup.Load(schema, JParse.Parse(File.ReadAllText(relationsPath)), JParse.Parse(File.ReadAllText(directionsPath)));
        var checker = new Checker(setup, g["protocol"]);
        var spec = LanguageSpec.From(JParse.Parse(File.ReadAllText(specPath)));
        var recs = g["records"].Items;
        var outcomes = new Dictionary<string, int>();
        int headset = 0;
        foreach (var c in g["resolutions"].Items)
        {
            string where = c["parent_command_id"].Text + " " + c["view"].Text;
            var e = c["expected"];
            string outcome, target = "None", cands = "", technical = "False";
            if (c["interp"].Kind == JKind.Null) { outcome = e["outcome"].Text; technical = "True"; cands = e["candidate_ids"].Kind == JKind.Null ? "" : Ids(e["candidate_ids"]); }
            else
            {
                var o = checker.Resolve(c["interp"], recs[(int)c["scene"].Num], recs[(int)c["command"].Num], recs[(int)c["map"].Num]);
                outcome = o.Outcome; target = o.TargetId ?? "None"; cands = string.Join(",", o.CandidateIds);
                if (c.Has("query") && c["query"].Kind != JKind.Null) Eq("acceptance query", where, c["query"].Text, Canon(o.Query));
            }
            Eq("acceptance outcome", where, e["outcome"].Text, outcome);
            Eq("acceptance target", where, Opt(e["target_id"]), target);
            Eq("acceptance candidates", where, e["candidate_ids"].Kind == JKind.Null ? "" : Ids(e["candidate_ids"]), cands);
            Eq("acceptance technical", where, e["technical"].Bool.ToString(), technical);
            outcomes[e["outcome"].Text] = outcomes.TryGetValue(e["outcome"].Text, out int n) ? n + 1 : 1;
            if (c["headset"].Bool) headset++;
        }
        int valid = 0;
        JNode ctxScene = g.Has("reader_context") ? recs[(int)g["reader_context"]["scene"].Num] : null;
        JNode ctxCommand = g.Has("reader_context") ? recs[(int)g["reader_context"]["command"].Num] : null;
        foreach (var r in g["readers"].Items)
        {
            Parsed p = null;
            try { p = Language.ParseOutput(r["text"].Text, spec); } catch (LanguageError) { }
            Eq("acceptance reader verdict", r["text"].Text, r["valid"].Bool ? "valid" : "refused", p != null ? "valid" : "refused");
            if (p == null || !r["valid"].Bool) continue;
            valid++;
            Eq("acceptance reader render", r["text"].Text, r["render"].Text, Language.Render(p));
            Eq("acceptance reader interpretation", r["text"].Text, r["interpretation"].Text, Canon(Language.Interpretation(p)));
            if (ctxScene != null && r.Has("query") && r["query"].Kind != JKind.Null)
                Eq("acceptance reader query", r["text"].Text, r["query"].Text, Canon(Checker.Query(Language.Interpretation(p), ctxScene, ctxCommand, checker.QuerySuffix)));
        }
        var parts = new List<string>();
        foreach (var kv in outcomes) parts.Add(kv.Key + " " + kv.Value);
        Console.WriteLine("acceptance: " + g["resolutions"].Items.Count + " resolutions (" + headset + " in the headset subset; " + string.Join(", ", parts) + "), "
                          + g["readers"].Items.Count + " reader strings (" + valid + " valid)");
        foreach (var kv in failsBy) Console.WriteLine("  failures in " + kv.Key + ": " + kv.Value);
        Console.WriteLine(checks + " comparisons, " + fails + " differ");
        return fails == 0 ? 0 : 1;
    }

    // ---------------------------------------------------------------- the headset acceptance run, on the PC
    sealed class ExpectedNative : IParserNative, IParserMasks
    {
        public Dictionary<string, JNode> Decodes = new Dictionary<string, JNode>();
        public Dictionary<string, JNode> Masks = new Dictionary<string, JNode>();
        public int[] Tokenize(byte[] utf8)
        {
            string key = SecondEyes.Grounding.Replay.ReplayHash.Sha256Hex(utf8);
            JNode d;
            if (!Decodes.TryGetValue(key, out d)) return null;
            var v = new int[d["prompt_ids"].Items.Count];
            for (int i = 0; i < v.Length; i++) v[i] = (int)d["prompt_ids"].Items[i].Num;
            current = d;
            return v;
        }
        JNode current;
        public int Decode(int[] prompt, int maxNew, int[] output, out int status)
        {
            status = current["status"].Text == "completed" ? 0 : (current["status"].Text == "length_exhausted" ? 1 : 2);
            for (int i = 0; i < current["tokens"].Items.Count; i++) output[i] = (int)current["tokens"].Items[i].Num;
            return current["tokens"].Items.Count;
        }
        public byte[] TokenBytes(int token) { return null; }
        public string LastError() { return "expected"; }
        public int AllowedAfter(int[] prefix, out string sha)
        {
            JNode m;
            Masks.TryGetValue(string.Join(",", prefix), out m);
            sha = m == null ? null : m["ids_sha256"].Text;
            return m == null ? -1 : (int)m["count"].Num;
        }
    }

    static int ParserCheck(string folder, string outDir, string lib, string gguf)
    {
        var f = ParserFolder.Load(folder);
        Directory.CreateDirectory(outDir);
        IParserNative native;
        IParserMasks masks;
        IntPtr model = IntPtr.Zero;
        SecondEyes.Grounding.LlamaParser real = null;
        string startup = null, modelSha = f.GgufSha256;   // with expected values, the folder's own candidate
        if (lib == null)
        {
            var e = new ExpectedNative();
            foreach (var line in File.ReadAllLines(Path.Combine(folder, "acceptance", "decodes.jsonl")))
            {
                if (line.Trim().Length == 0) continue;
                var d = JParse.Parse(line);
                e.Decodes[SecondEyes.Grounding.Replay.ReplayHash.Sha256Hex(new System.Text.UTF8Encoding(false).GetBytes(d["prompt"].Text))] = d;
            }
            foreach (var line in File.ReadAllLines(Path.Combine(folder, "acceptance", "masks.jsonl")))
            {
                if (line.Trim().Length == 0) continue;
                var m = JParse.Parse(line);
                e.Masks[Ids2(m["prefix"])] = m;
            }
            native = e; masks = e;
        }
        else
        {
            model = SecondEyes.Grounding.LlamaNative.se_load_ex(SecondEyes.Grounding.LlamaNative.Utf8(gguf), f.Assets.ContextTokens, 2, 16, 0);
            real = new SecondEyes.Grounding.LlamaParser(model, f.Gbnf, f.Assets.EndToken);
            native = real; masks = real;
            startup = f.StartupCheck(real);
            Console.WriteLine("startup check: " + (startup ?? "passed"));
            try { modelSha = f.VerifyModel(gguf); } catch (InvalidDataException e) { modelSha = "mismatch"; Console.WriteLine(e.Message); }
        }
        var sum = ParserAcceptance.Run(f, native, masks, outDir, null, null);
        File.WriteAllText(Path.Combine(outDir, "run.json"), new SecondEyes.Grounding.Replay.JsonWriter().BeginObj().Key("record_type").S("a27_parser_check_run")
            .Key("manifest_sha256").S(f.ManifestSha256).Key("native").S(lib == null ? "expected values (plumbing test)" : "libse_llama (" + lib + ")")
            .Key("gguf_sha256").S(modelSha).Key("startup_check").S(startup ?? "passed")
            .Key("decodes").I(sum.Decodes).Key("masks").I(sum.Masks).Key("resolutions").I(sum.Resolutions).Key("readers").I(sum.Readers)
            .Key("technical").I(sum.Technical).Key("seconds").D(sum.Seconds).End().ToString());
        if (real != null) { real.Dispose(); SecondEyes.Grounding.LlamaNative.se_free(model); }
        Console.WriteLine("parser check: " + sum.Decodes + " decodes, " + sum.Masks + " masks, " + sum.Resolutions + " resolutions, " + sum.Readers + " readers, "
                          + sum.Technical + " technical failures, " + sum.Seconds.ToString("0.0") + " s");
        return 0;
    }

    static int Main(string[] args)
    {
        if ((args.Length == 3 || args.Length == 5) && args[0] == "parser-check")
            return ParserCheck(args[1], args[2], args.Length == 5 ? args[3] : null, args.Length == 5 ? args[4] : null);
        if (args.Length == 6 && args[0] == "acceptance") return Acceptance(args[1], args[2], args[3], args[4], args[5]);
        if (args.Length == 3 && args[0] == "reader") return Reader(args[1], args[2]);
        if (args.Length == 7 && args[0] == "pipeline") return Pipeline(args[1], args[2], args[3], args[4], args[5], args[6]);
        if (args.Length == 3 && args[0] == "native") return Native(args[1], args[2]);
        if (args.Length == 5 && args[0] == "resolution") return Resolution(args[1], args[2], args[3], args[4]);
        if (args.Length == 3 && args[0] == "validation") return Validation(args[1], args[2]);
        if (args.Length == 3 && args[0] == "relations") return Relations(args[1], args[2]);
        if (args.Length == 3 && args[0] == "schemas") return Schemas(args[1], args[2]);
        Console.WriteLine("usage: relations GOLDENS.json RELATION_CONFIG.json | schemas GOLDENS.json SCHEMAS_DIR");
        return 2;
    }
}
