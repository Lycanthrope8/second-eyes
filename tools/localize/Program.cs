// The PC rerun of the localization (A4 Delivery 1, stage 2; docs/localization.md): the same Unity-free core the Quest runs,
// from a capture record, a category map, size priors and a configuration. It checks every input, writes scene.json (a
// perceived scene, scene v3) and trace.json, checks the scene against the contract, and prints the objects.
// Run: dotnet run --project tools/localize -- <capture> <category-map> <size-priors> <config> <out-dir> [schemas]
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using SecondEyes.Grounding.Parsing;
using SecondEyes.Grounding.Prompting;
using SecondEyes.Grounding.Resolution;
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

    static int Main(string[] args)
    {
        if (args.Length < 5) { Console.Error.WriteLine("usage: localize <capture> <category-map> <size-priors> <config> <out-dir> [schemas]"); return 2; }
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
        return 0;
    }
}
