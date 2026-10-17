// The PC rerun of the localization (A4 Delivery 1, stage 2; docs/localization.md): the same Unity-free core the Quest runs,
// from a capture record, a category map, size priors and a configuration. It checks every input, writes scene.json (a
// perceived scene, scene v3) and trace.json, checks the scene against the contract, and prints the objects.
// Run: dotnet run --project tools/localize -- <capture> <category-map> <size-priors> <config> <out-dir> [schemas] [--compare <scene>]
// Or, for the automatic scan (docs/auto-scan.md): --memory <frames.jsonl> <category-map> <size-priors> <autoscan-config> <out-dir>
// [schemas] [--compare <scene>]: the headset's frames.jsonl replayed through the same AutoScan code. With --compare, the
// replay stops at the frame count the headset's trace.json (beside the compared scene) recorded for that scene's revision.
// With --compare, the rerun's scene is compared with another (the headset's, from a pulled scan): the same objects,
// categories, placements and possible duplicates, and centres within 1 mm; any difference is listed and the exit is 1.
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using SecondEyes.Grounding.Parsing;
using SecondEyes.Grounding.Prompting;
using SecondEyes.Grounding.Resolution;
using SecondEyes.Perception.AutoScan;
using SecondEyes.Perception.Capture;
using SecondEyes.Perception.Localization;

static class Program
{
    static int Stop(string what, IEnumerable<string> lines)
    {
        Console.WriteLine("STOP: " + what);
        foreach (string l in lines.Take(10)) Console.WriteLine("  " + l);
        return 1;
    }

    /// <summary>How another scene (the headset's) differs from this rerun's: objects, categories, placements, possible
    /// duplicates, and centres more than 1 mm apart.</summary>
    static List<string> Differences(JNode mine, JNode theirs)
    {
        var d = new List<string>();
        foreach (string k in new[] { "scene_id", "category_map", "evidence_profile" })
            if (mine[k].Text != theirs[k].Text) d.Add(k + ": " + theirs[k].Text + " there, " + mine[k].Text + " here");
        var a = mine["objects"].Items.ToDictionary(x => x["object_id"].Text);
        var b = theirs["objects"].Items.ToDictionary(x => x["object_id"].Text);
        foreach (string id in b.Keys.Except(a.Keys).OrderBy(x => x, StringComparer.Ordinal)) d.Add(id + ": only there");
        foreach (string id in a.Keys.Except(b.Keys).OrderBy(x => x, StringComparer.Ordinal)) d.Add(id + ": only here");
        foreach (string id in a.Keys.Intersect(b.Keys).OrderBy(x => x, StringComparer.Ordinal))
        {
            JNode x = a[id], y = b[id];
            Func<JNode, string> cat = n => n["category"]["value"]["standard"].Text;
            Func<JNode, string> status = n => n["observation"]["association"]["status"].Text;
            Func<JNode, string> dup = n => string.Join(",", n["observation"]["association"]["may_duplicate"].Items.Select(z => z.Text));
            if (cat(x) != cat(y)) d.Add(id + ": " + cat(y) + " there, " + cat(x) + " here");
            if (status(x) != status(y)) d.Add(id + ": " + status(y) + " there, " + status(x) + " here");
            if (dup(x) != dup(y)) d.Add(id + ": possible duplicates [" + dup(y) + "] there, [" + dup(x) + "] here");
            JNode cx = x["geometry"]["center_m"], cy = y["geometry"]["center_m"];
            if (cx["state"].Text != cy["state"].Text) d.Add(id + ": centre " + cy["state"].Text + " there, " + cx["state"].Text + " here");
            else if (cx["state"].Text == "known")
            {
                double s = 0;
                for (int i = 0; i < 3; i++) { double e = cx["value"].Items[i].Num - cy["value"].Items[i].Num; s += e * e; }
                if (Math.Sqrt(s) > 0.001) d.Add(id + ": centres " + Math.Sqrt(s).ToString("0.0000", System.Globalization.CultureInfo.InvariantCulture) + " m apart");
            }
        }
        return d;
    }

    /// <summary>The automatic scan's replay: frames.jsonl through AutoScan, then the same checks and comparison.</summary>
    static int Memory(string[] args, string compare)
    {
        string schemasDir = args.Length > 5 ? args[5] : "schemas";
        var cache = new Dictionary<string, Schema>();
        Func<string, Schema> schema = name =>
        {
            if (!cache.ContainsKey(name)) cache[name] = new Schema(JParse.Parse(File.ReadAllText(Path.Combine(schemasDir, name))));
            return cache[name];
        };
        Func<string, JNode> load = p => JParse.Parse(File.ReadAllText(p));
        JNode map = load(args[1]), priors = load(args[2]), config = load(args[3]);
        var pe = schema("size-priors.v1.json").Errors(priors);
        if (pe.Count > 0) return Stop("the size priors", pe.Select(e => e.Keyword + " " + e.Path));
        var ce = schema("auto-scan-config.v1.json").Errors(config);
        if (ce.Count > 0) return Stop("the automatic scan's settings", ce.Select(e => e.Keyword + " " + e.Path));
        var contract = new Contract(schema);
        var mapIssues = contract.Validate(new List<KeyValuePair<string, JNode>> { new KeyValuePair<string, JNode>(args[1], map) });
        if (mapIssues.Any(i => i.IsError)) return Stop("the category map", mapIssues.Select(i => i.Code + " " + i.Path));
        var lines = File.ReadAllLines(args[0]).Where(l => l.Trim().Length > 0).ToList();
        if (lines.Count == 0) return Stop("frames.jsonl", new[] { "the file is empty" });
        JNode header = JParse.Parse(lines[0]);
        if (header["record_type"].Text != "auto_scan_header") return Stop("frames.jsonl", new[] { "its first line is not an auto_scan_header" });
        foreach (var pair in new[] { ("category map", header["category_map"].Text, map["map_id"].Text), ("size priors", header["size_priors"].Text, priors["priors_id"].Text),
                                     ("settings", header["config_id"].Text, config["config_id"].Text) })
            if (pair.Item2 != pair.Item3) Console.WriteLine("note: the scan used " + pair.Item1 + " " + pair.Item2 + "; this replay uses " + pair.Item3);
        int frames = lines.Count - 1, revision = 1;
        if (compare != null)
        {
            JNode cs = load(compare);
            revision = (int)cs["scene_revision"].Num;
            string tracePath = Path.Combine(Path.GetDirectoryName(Path.GetFullPath(compare)), "trace.json");
            if (File.Exists(tracePath)) frames = Math.Min(frames, (int)load(tracePath)["frames"].Num);
        }
        var scan = AutoScan.FromHeader(header, AutoScanSettings.From(config), map, priors);
        foreach (string line in lines.Skip(1).Take(frames)) scan.Add(FrameRecord.FromJson(JParse.Parse(line)));
        LocalizationOutput o = scan.Snapshot(revision);
        Directory.CreateDirectory(args[4]);
        File.WriteAllText(Path.Combine(args[4], "scene.json"), PyJson.Canonical(o.Scene) + "\n", new UTF8Encoding(false));
        File.WriteAllText(Path.Combine(args[4], "trace.json"), PyJson.Canonical(o.Trace) + "\n", new UTF8Encoding(false));
        var sceneIssues = contract.Validate(new List<KeyValuePair<string, JNode>> {
            new KeyValuePair<string, JNode>("scene.json", o.Scene), new KeyValuePair<string, JNode>(args[1], map) });
        if (sceneIssues.Any(i => i.IsError)) return Stop("the scene is not a valid scene v3", sceneIssues.Where(i => i.IsError).Select(i => i.Code + " " + i.Path + " " + i.Message));
        Console.WriteLine("replayed " + frames + " frames (" + o.Trace["sightings"].Text + " sightings); left out: "
                          + string.Join(", ", o.Trace["left_out"].Fields.Select(kv => kv.Value.Text + " " + kv.Key.Replace('_', ' ')))
                          + "; " + o.Trace["tentative"].Text + " objects seen once");
        foreach (JNode obj in o.Scene["objects"].Items)
        {
            JNode c = obj["geometry"]["center_m"], obs = obj["observation"];
            Console.WriteLine(string.Format("{0}  {1,-10} ({2})  by depth, seen {3} times from {4}", obj["object_id"].Text, obj["category"]["value"]["standard"].Text,
                string.Join(", ", c["value"].Items.Select(x => x.Text)), obs["views"]["keyframes"].Text,
                obs["views"]["positions"].Text + (obs["views"]["positions"].Num == 1 ? " place" : " places")));
        }
        Console.WriteLine("scene.json and trace.json written to " + args[4] + "; the scene passes the contract (scene v3)");
        if (compare == null) return 0;
        var diff = Differences(o.Scene, load(compare));
        if (diff.Count == 0)
        {
            Console.WriteLine("compared with " + compare + ": the same " + o.Scene["objects"].Items.Count
                              + " objects, categories, placements and possible duplicates; centres within 1 mm");
            return 0;
        }
        Console.WriteLine("DIFFERS from " + compare + ":");
        foreach (string line in diff.Take(20)) Console.WriteLine("  " + line);
        return 1;
    }

    static int Main(string[] argv)
    {
        var list = new List<string>(argv);
        string compare = null;
        int at = list.IndexOf("--compare");
        if (at >= 0)
        {
            if (at + 1 >= list.Count) { Console.Error.WriteLine("--compare needs a scene file"); return 2; }
            compare = list[at + 1];
            list.RemoveRange(at, 2);
        }
        bool memory = list.Remove("--memory");
        string[] args = list.ToArray();
        if (args.Length < 5) { Console.Error.WriteLine("usage: localize [--memory] <capture or frames.jsonl> <category-map> <size-priors> <config> <out-dir> [schemas]"); return 2; }
        if (memory) return Memory(args, compare);
        string schemasDir = args.Length > 5 ? args[5] : "schemas";
        var cache = new Dictionary<string, Schema>();
        Func<string, Schema> schema = name =>
        {
            if (!cache.ContainsKey(name)) cache[name] = new Schema(JParse.Parse(File.ReadAllText(Path.Combine(schemasDir, name))));
            return cache[name];
        };
        Func<string, JNode> load = p => JParse.Parse(File.ReadAllText(p));
        JNode capture = load(args[0]), map = load(args[1]), priors = load(args[2]), config = load(args[3]);
        var capIssues = CaptureCheck.Check(args[0], capture, schema(CaptureCheck.SchemaFile));
        if (capIssues.Count > 0) return Stop("the capture record", capIssues.Select(i => i.Code + " " + i.Path + " " + i.Message));
        var pe = schema("size-priors.v1.json").Errors(priors);
        if (pe.Count > 0) return Stop("the size priors", pe.Select(e => e.Keyword + " " + e.Path));
        var ce = schema("localization-config.v1.json").Errors(config);
        if (ce.Count > 0) return Stop("the localization config", ce.Select(e => e.Keyword + " " + e.Path));
        var contract = new Contract(schema);
        var mapIssues = contract.Validate(new List<KeyValuePair<string, JNode>> { new KeyValuePair<string, JNode>(args[1], map) });
        if (mapIssues.Any(i => i.IsError)) return Stop("the category map", mapIssues.Select(i => i.Code + " " + i.Path));
        foreach (var pair in new[] { ("category map", capture["configuration"]["category_map_id"].Text, map["map_id"].Text),
                                     ("size priors", capture["configuration"]["size_priors_id"].Text, priors["priors_id"].Text) })
            if (pair.Item2 != pair.Item3) Console.WriteLine("note: the capture used " + pair.Item1 + " " + pair.Item2 + "; this rerun uses " + pair.Item3);

        LocalizationOutput o = Localizer.Run(capture, map, priors, config);
        Directory.CreateDirectory(args[4]);
        File.WriteAllText(Path.Combine(args[4], "scene.json"), PyJson.Canonical(o.Scene) + "\n", new UTF8Encoding(false));
        File.WriteAllText(Path.Combine(args[4], "trace.json"), PyJson.Canonical(o.Trace) + "\n", new UTF8Encoding(false));
        var sceneIssues = contract.Validate(new List<KeyValuePair<string, JNode>> {
            new KeyValuePair<string, JNode>("scene.json", o.Scene), new KeyValuePair<string, JNode>(args[1], map) });
        if (sceneIssues.Any(i => i.IsError)) return Stop("the scene is not a valid scene v3", sceneIssues.Where(i => i.IsError).Select(i => i.Code + " " + i.Path + " " + i.Message));
        JNode left = o.Trace["left_out"];
        Console.WriteLine("left out: " + left["unmapped_label"].Text + " unmapped, " + left["below_min_score"].Text + " below min_score, "
                          + left["cut_by_the_edge"].Text + " cut by the image edge");
        foreach (JNode obj in o.Scene["objects"].Items)
        {
            JNode c = obj["geometry"]["center_m"], obs = obj["observation"];
            string centre = c["state"].Text == "known" ? "(" + string.Join(", ", c["value"].Items.Select(x => x.Text)) + ")"
                                                       + (c["evidence"]["assumptions"].Items.Count > 0 ? " from its size" : "") : "unknown";
            string dup = obs["association"]["may_duplicate"].Items.Count > 0 ? " with " + string.Join(", ", obs["association"]["may_duplicate"].Items.Select(x => x.Text)) : "";
            Console.WriteLine(string.Format("{0}  {1,-10} {2,-36} {3}{4}; {5} keyframes, {6} positions", obj["object_id"].Text,
                obj["category"]["value"]["standard"].Text, centre, obs["association"]["status"].Text, dup,
                obs["views"]["keyframes"].Text, obs["views"]["positions"].Text));
        }
        Console.WriteLine("scene.json and trace.json written to " + args[4] + "; the scene passes the contract (scene v3)");
        if (compare == null) return 0;
        var diff = Differences(o.Scene, load(compare));
        if (diff.Count == 0)
        {
            Console.WriteLine("compared with " + compare + ": the same " + o.Scene["objects"].Items.Count
                              + " objects, categories, placements and possible duplicates; centres within 1 mm");
            return 0;
        }
        Console.WriteLine("DIFFERS from " + compare + ":");
        foreach (string line in diff.Take(20)) Console.WriteLine("  " + line);
        return 1;
    }
}
