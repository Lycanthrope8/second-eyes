// A4 Delivery 1, stage 1: the C# checks against the shared fixtures, judged by the same rule as the Python suite
// (grounding/tests/test_contract.py): every valid file passes alone, and the valid files pass together with no issue at
// all; every invalid file fails with exactly the code in its name (with its partner file for a cross-record rule), and
// every error is in that file. The expectations are the fixtures' names, written by hand, not output of either checker.
// Run: dotnet run --project tools/csharp-tests/Contract -- grounding/tests/fixtures schemas
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using SecondEyes.Grounding.Parsing;
using SecondEyes.Grounding.Prompting;
using SecondEyes.Grounding.Resolution;
using SecondEyes.Perception.Capture;

static class Program
{
    static int pass, fail, known;
    // Differences that predate A4 and are reported, not hidden: the file is still refused, with another code. Each is
    // checked to stay exactly as recorded; any other result fails.
    static readonly Dictionary<string, string> KnownDifferences = new Dictionary<string, string>
    {
        { "E_NONFINITE__nan.json|E_PARSE_JSON", "the C# strict parser (A2.7, Parsing/ParserCore.cs) refuses the NaN literal as malformed JSON, where Python's parser names it non-finite" }
    };
    // As test_contract.py's PARTNERS: cross-record rules are checked with this valid file alongside.
    static readonly Dictionary<string, string> Partners = new Dictionary<string, string>
    {
        { "E_SCENE_MISMATCH", "scene.a17.annotated.json" }, { "E_FRAME_MISMATCH", "scene.a17.annotated.json" },
        { "E_DUPLICATE_RECORD", "scene.a17.annotated.json" }, { "E_CATEGORY_UNMAPPED", "category-map.fx.json" }
    };

    static void Check(string n, bool ok, string d = "")
    {
        if (ok) { pass++; Console.WriteLine("  ok    " + n); }
        else { fail++; Console.WriteLine("  FAIL  " + n + (d.Length > 0 ? "  [" + d + "]" : "")); }
    }

    /// <summary>validate.py's validate_paths: strict parsing per file, then every layer for the files that parsed.</summary>
    static List<Issue> ValidatePaths(Contract contract, IEnumerable<string> paths)
    {
        var issues = new List<Issue>();
        var parsed = new List<KeyValuePair<string, JNode>>();
        foreach (string path in paths)
        {
            string text = new UTF8Encoding(false, false).GetString(File.ReadAllBytes(path));   // keeps a byte-order mark visible
            string code, message;
            JNode n = StrictJson.Parse(text, out code, out message);
            if (n == null) issues.Add(new Issue { File = path, Path = "$", Code = code, Message = message });
            else parsed.Add(new KeyValuePair<string, JNode>(path, n));
        }
        issues.AddRange(contract.Validate(parsed));
        return issues;
    }

    static string Codes(IEnumerable<Issue> issues) { return string.Join(", ", issues.Select(i => i.Code + " " + i.Path).Take(4)); }

    /// <summary>Where two JSON values first differ, comparing numbers by value and objects by key; null when the same.</summary>
    static string Same(JNode a, JNode b, string path)
    {
        if (a.Kind != b.Kind) return path + ": " + a.Kind + " against " + b.Kind;
        switch (a.Kind)
        {
            case JKind.Number: return a.Num == b.Num ? null : path + ": " + a.Text + " against " + b.Text;
            case JKind.String: return a.Text == b.Text ? null : path + ": \"" + a.Text + "\" against \"" + b.Text + "\"";
            case JKind.Array:
                if (a.Items.Count != b.Items.Count) return path + ": " + a.Items.Count + " items against " + b.Items.Count;
                for (int i = 0; i < a.Items.Count; i++) { string d = Same(a.Items[i], b.Items[i], path + "[" + i + "]"); if (d != null) return d; }
                return null;
            case JKind.Object:
                var ka = a.Fields.Select(f => f.Key).OrderBy(x => x, StringComparer.Ordinal).ToList();
                var kb = b.Fields.Select(f => f.Key).OrderBy(x => x, StringComparer.Ordinal).ToList();
                if (!ka.SequenceEqual(kb)) return path + ": keys " + string.Join(",", ka) + " against " + string.Join(",", kb);
                foreach (string k in ka) { string d = Same(a[k], b[k], path + "." + k); if (d != null) return d; }
                return null;
            default: return a.Kind == JKind.Bool && a.Bool != b.Bool ? path + ": " + a.Bool + " against " + b.Bool : null;
        }
    }

    static int Main(string[] args)
    {
        if (args.Length < 2 || args.Length > 3) { Console.Error.WriteLine("usage: Contract <grounding/tests/fixtures> <schemas> [<built perception folder>]"); return 2; }
        string fixtures = args[0], schemasDir = args[1];
        var cache = new Dictionary<string, Schema>();
        Func<string, Schema> schema = name =>
        {
            if (!cache.ContainsKey(name)) cache[name] = new Schema(JParse.Parse(File.ReadAllText(Path.Combine(schemasDir, name))));
            return cache[name];
        };
        var contract = new Contract(schema);

        Console.WriteLine("The scene contract (formats v1, v2 and v3), command contexts and category maps");
        string valid = Path.Combine(fixtures, "contract", "valid"), invalid = Path.Combine(fixtures, "contract", "invalid");
        var validFiles = Directory.GetFiles(valid, "*.json").OrderBy(f => f, StringComparer.Ordinal).ToList();
        foreach (string f in validFiles)
        {
            var errors = ValidatePaths(contract, new[] { f }).Where(i => i.IsError).ToList();
            Check("valid/" + Path.GetFileName(f) + " passes alone", errors.Count == 0, Codes(errors));
        }
        var together = ValidatePaths(contract, validFiles);
        Check("all " + validFiles.Count + " valid files pass together with no issue", together.Count == 0, Codes(together));
        var covered = new HashSet<string>();
        foreach (string f in Directory.GetFiles(invalid, "*.json").OrderBy(x => x, StringComparer.Ordinal))
        {
            string name = Path.GetFileName(f), code = name.Substring(0, name.IndexOf("__", StringComparison.Ordinal));
            covered.Add(code);
            var paths = new List<string>();
            string partner;
            if (Partners.TryGetValue(code, out partner)) paths.Add(Path.Combine(valid, partner));
            paths.Add(f);
            var errors = ValidatePaths(contract, paths).Where(i => i.IsError).ToList();
            var codes = new SortedSet<string>(errors.Select(i => i.Code));
            bool onlyThis = errors.All(i => i.File == f);
            string why;
            if (codes.Count == 1 && onlyThis && KnownDifferences.TryGetValue(name + "|" + codes.Min, out why))
            {
                known++;
                Console.WriteLine("  known " + "invalid/" + name + ": refused as " + codes.Min + ", not " + code + ": " + why);
                continue;
            }
            Check("invalid/" + name + " fails with exactly " + code, codes.Count == 1 && codes.Contains(code) && onlyThis,
                  "got " + string.Join(", ", codes) + (onlyThis ? "" : " (also in a partner file)"));
        }

        Console.WriteLine("The capture record");
        string cValid = Path.Combine(fixtures, "capture", "valid"), cInvalid = Path.Combine(fixtures, "capture", "invalid");
        Schema captureSchema = schema(CaptureCheck.SchemaFile);
        foreach (string f in Directory.GetFiles(cValid, "*.json").OrderBy(x => x, StringComparer.Ordinal))
        {
            var issues = CaptureCheck.Check(f, JParse.Parse(File.ReadAllText(f)), captureSchema);
            Check("valid/" + Path.GetFileName(f) + " passes", issues.Count == 0, Codes(issues));
        }
        foreach (string f in Directory.GetFiles(cInvalid, "*.json").OrderBy(x => x, StringComparer.Ordinal))
        {
            string name = Path.GetFileName(f), code = name.Substring(0, name.IndexOf("__", StringComparison.Ordinal));
            var codes = new SortedSet<string>(CaptureCheck.Check(f, JParse.Parse(File.ReadAllText(f)), captureSchema).Select(i => i.Code));
            Check("invalid/" + name + " fails with exactly " + code, codes.Count == 1 && codes.Contains(code), "got " + string.Join(", ", codes));
        }
        Check("a missing record is refused as E_RECORD_TYPE",
              CaptureCheck.Check("none", null, captureSchema).Select(i => i.Code).SequenceEqual(new[] { "E_RECORD_TYPE" }));

        Console.WriteLine("Format-2 queries and the association guard: the C# resolver on the perceived bedroom");
        Func<string, JNode> load = p => JParse.Parse(File.ReadAllText(p));
        string grounding = Path.GetFullPath(Path.Combine(fixtures, "..", ".."));
        var setup = ResolverSetup.Load(schema, load(Path.Combine(grounding, "relations", "relations.v1.json")),
                                       load(Path.Combine(grounding, "relations", "directions.v1.json")));
        string av = Path.Combine(fixtures, "association");
        JNode bScene = load(Path.Combine(fixtures, "contract", "valid", "scene.bedroom.perceived_v3.json"));
        JNode bMap = load(Path.Combine(fixtures, "contract", "valid", "category-map.bedroom.json"));
        JNode bCommand = load(Path.Combine(av, "command.bedroom.json"));
        JNode bConfig = load(Path.Combine(fixtures, "resolution", "resolver-config.fixture.json"));
        foreach (JNode c in load(Path.Combine(av, "expected.json"))["cases"].Items)
        {
            string qf = c["query"].Text, nm = qf.Substring(6, qf.Length - 11);
            JNode query = load(Path.Combine(av, qf));
            Check(nm + ": a format-2 query is checked against grounding-query.v2.json", ResolverValidation.QuerySchemaFile(query) == "grounding-query.v2.json");
            RunRecord run = Resolver.Resolve(bScene, bCommand, query, bConfig, new List<JNode> { bMap }, setup);
            JNode want = c["resolver"];
            bool resolverOk = run.ProcessingStatus == "completed" && run.Result != null && run.Result.Status == want["status"].Text
                              && run.Result.TargetId == want["target_id"].Text;
            Check(nm + ": the C# resolver answers " + want["status"].Text + " " + want["target_id"].Text, resolverOk,
                  run.ProcessingStatus + " " + (run.Result == null ? "-" : run.Result.Status + " " + run.Result.TargetId) + " " + Codes(run.Issues));
            if (run.Result == null) continue;
            RunRecord g = AssociationGuard.Guard(bScene, query, run);
            JNode gw = c["guarded"];
            var wantIds = gw["target_ids"].Items.Select(x => x.Text).ToList();
            string wantTarget = gw["target_id"].Kind == JKind.Null ? null : gw["target_id"].Text;
            bool guardOk = g.Result.Status == gw["status"].Text && g.Result.TargetId == wantTarget && g.Result.ReasonCode == gw["reason_code"].Text
                           && g.Result.TargetIds.SequenceEqual(wantIds) && g.Result.ReasonCodes.Contains("uncertain_association") == gw["uncertain_association"].Bool;
            Check(nm + ": after the C# guard, " + gw["status"].Text + " [" + string.Join(", ", wantIds) + "]", guardOk,
                  g.Result.Status + " " + g.Result.TargetId + " [" + string.Join(", ", g.Result.TargetIds) + "] " + string.Join(" ", g.Result.ReasonCodes));
            Check(nm + ": the guard leaves the resolver's own record unchanged", run.Result.Status == want["status"].Text);
        }
        JNode relabelled = JParse.Parse(File.ReadAllText(Path.Combine(av, "query.farthest.json")).Replace("\"schema_version\": 2", "\"schema_version\": 1"));
        List<NormInterpretation> norm;
        JNode annotated2 = JParse.Parse(File.ReadAllText(Path.Combine(av, "query.farthest.json")).Replace("\"perceived\"", "\"annotated\""));
        var a2Issues = ResolverValidation.CheckQuery(annotated2, setup.Schemas(ResolverValidation.QuerySchemaFile(annotated2)), out norm);
        Check("a format-2 query on an annotated scene is checked against format 1 and rejected at $.schema_version, as A2.7 expects",
              ResolverValidation.QuerySchemaFile(annotated2) == "grounding-query.v1.json" && a2Issues.Count == 1
              && a2Issues[0].Code == "E_QUERY_SCHEMA" && a2Issues[0].Path == "$.schema_version",
              string.Join(", ", a2Issues.Select(i => i.Code + " " + i.Path)));
        Check("the same query labelled format 1 fails format 1's schema",
              ResolverValidation.CheckQuery(relabelled, setup.Schemas(ResolverValidation.QuerySchemaFile(relabelled)), out norm).Count > 0);

        Console.WriteLine("A typed command as a command record (item 1)");
        JNode a17Scene = load(Path.Combine(fixtures, "contract", "valid", "scene.a17.annotated.json"));
        JNode a17Map = load(Path.Combine(fixtures, "contract", "valid", "category-map.fx.json"));
        string baseCommand = File.ReadAllText(Path.Combine(fixtures, "contract", "valid", "command.a17.c002.json"));
        string typed = SecondEyes.Grounding.Interactive.TypedCommand.Build(baseCommand, "the box closest to the table", "-t20261016-120000-001");
        JNode built = JParse.Parse(typed), baseNode = JParse.Parse(baseCommand);
        Check("the typed text and its own ID replace the dataset command's", built["text"].Text == "the box closest to the table"
              && built["command_id"].Text == baseNode["command_id"].Text + "-t20261016-120000-001");
        Check("everything else is the dataset command's (scene, revision, pose, sources)",
              built["scene_id"].Text == baseNode["scene_id"].Text && built["scene_revision"].Text == baseNode["scene_revision"].Text
              && SecondEyes.Grounding.Parsing.PyJson.Canonical(built["user_pose"]) == SecondEyes.Grounding.Parsing.PyJson.Canonical(baseNode["user_pose"]));
        var typedIssues = contract.Validate(new List<KeyValuePair<string, JNode>> {
            new KeyValuePair<string, JNode>("scene", a17Scene), new KeyValuePair<string, JNode>("map", a17Map),
            new KeyValuePair<string, JNode>("typed", built) });
        Check("the built record passes the contract with its scene and map", typedIssues.Count == 0, Codes(typedIssues));
        bool rawParses;
        try { JParse.Parse("hello"); rawParses = true; } catch (Exception) { rawParses = false; }
        Check("raw text, as the trial sent it, is not a record the session can read", !rawParses);
        bool emptyRefused;
        try { SecondEyes.Grounding.Interactive.TypedCommand.Build(baseCommand, "  ", "-t"); emptyRefused = false; } catch (ArgumentException) { emptyRefused = true; }
        Check("empty text is refused", emptyRefused);

        Console.WriteLine("A command typed about a scanned room (stage 3)");
        JNode bedScene = load(Path.Combine(fixtures, "contract", "valid", "scene.bedroom.perceived_v3.json"));
        JNode bedMap = load(Path.Combine(fixtures, "contract", "valid", "category-map.bedroom.json"));
        JNode scanCommand = JParse.Parse(SecondEyes.Grounding.Interactive.TypedCommand.ForScan(bedScene, "the chair closest to the table", "fixture.bedroom.t-001"));
        var scanIssues = contract.Validate(new List<KeyValuePair<string, JNode>> {
            new KeyValuePair<string, JNode>("scene", bedScene), new KeyValuePair<string, JNode>("map", bedMap),
            new KeyValuePair<string, JNode>("typed", scanCommand) });
        Check("ForScan builds a format-2 command with a user_input source that passes the contract with its scene",
              scanCommand["schema_version"].Num == 2 && scanCommand["sources"].Items[0]["kind"].Text == "user_input" && scanIssues.Count == 0, Codes(scanIssues));

        Console.WriteLine("The capture builder");
        JNode manifest = load(Path.Combine("quest-app", "Assets", "SecondEyes", "Models", "yolox-nano", "detector-package.json"));
        JNode fixtureCapture = load(Path.Combine(fixtures, "capture", "valid", "capture.bedroom.json"));
        var fc = fixtureCapture["camera"]; var fi = fc["intrinsics"]; var fconf = fixtureCapture["configuration"]; var fclock = fixtureCapture["clock"];
        var builder = new SecondEyes.Perception.Capture.CaptureBuilder(fixtureCapture["capture_id"].Text,
            new SecondEyes.Perception.Capture.CaptureCamera { CameraId = fc["camera_id"].Text, WidthPx = (int)fc["width_px"].Num, HeightPx = (int)fc["height_px"].Num,
                Fx = fi["fx"].Num, Fy = fi["fy"].Num, Cx = fi["cx"].Num, Cy = fi["cy"].Num, IntrinsicsSource = fi["source"].Text, Distortion = fc["distortion"].Text },
            fixtureCapture["app"]["build"].Text, manifest, fixtureCapture["detector"]["package_sha256"].Text, fconf["perception_folder_sha256"].Text,
            fconf["category_map_id"].Text, fconf["size_priors_id"].Text, fclock["sdk_unit"].Text, (long)fclock["offset_us"].Num, fixtureCapture["nominal_step_deg"].Num);
        foreach (JNode k in fixtureCapture["keyframes"].Items)
        {
            JNode img = k["image"];
            builder.Add((int)k["position_index"].Num, (int)k["step_index"].Num, long.Parse(k["capture_time"]["sdk_value"].Text), (long)k["tracking"]["epoch"].Num,
                k["tracking"]["state"].Text == "tracked", k["camera_pose"]["translation_m"].Items.Select(x => x.Num).ToArray(),
                k["camera_pose"]["rotation_xyzw"].Items.Select(x => x.Num).ToArray(),
                k["detections"].Items.Select(d => new SecondEyes.Perception.Capture.CaptureDetection { Label = d["label"].Text, Score = d["score"].Num,
                    X1 = d["box_px"].Items[0].Num, Y1 = d["box_px"].Items[1].Num, X2 = d["box_px"].Items[2].Num, Y2 = d["box_px"].Items[3].Num }).ToList(),
                img["state"].Text == "saved" ? img["file"].Text : null, img["state"].Text == "saved" ? img["sha256"].Text : null);
        }
        JNode rebuilt = builder.Build();
        string where = Same(rebuilt, fixtureCapture, "$");
        Check("rebuilt from its own values, the capture fixture comes out the same, value for value (detector block from the real manifest)", where == null, where ?? "");
        var rebuiltIssues = SecondEyes.Perception.Capture.CaptureCheck.Check("rebuilt", rebuilt, schema(SecondEyes.Perception.Capture.CaptureCheck.SchemaFile));
        Check("and passes the capture check", rebuiltIssues.Count == 0, Codes(rebuiltIssues));
        JNode clipped = JParse.Parse(File.ReadAllText(Path.Combine("quest-app", "Assets", "SecondEyes", "Models", "yolox-nano", "detector-package.json")).Replace("not clipped", "clipped"));
        bool refused = false;
        try { new SecondEyes.Perception.Capture.CaptureBuilder("x", new SecondEyes.Perception.Capture.CaptureCamera(), "b", clipped, "", "", "", "", "ns", 0, 30); }
        catch (ArgumentException) { refused = true; }
        Check("a detector whose boxes are clipped is refused, not mislabelled", refused);

        if (args.Length > 2)
        {
            Console.WriteLine("The perception folder, as the PC built it");
            string pfDir = args[2];
            var pf = SecondEyes.Perception.PerceptionFolder.Load(pfDir);
            Check("it loads: every file matches the manifest, and the configurations, size priors and map pass their checks", pf.ManifestSha256.Length == 64);
            Check("it names the bedroom's map and size priors, and holds scene v3 and the format-2 schemas",
                  pf.CategoryMapId == "map.bedroom.v1" && pf.SizePriorsId == "sizes.bedroom.v1" && pf.Maps.ContainsKey("map.bedroom.v1")
                  && pf.SchemaText("scene.v3.json") != null && pf.SchemaText("grounding-query.v2.json") != null && pf.SchemaText("command-context.v2.json") != null);
            string copy = Path.Combine(Path.GetTempPath(), "pf-tampered-" + Guid.NewGuid().ToString("N"));
            foreach (string file in Directory.GetFiles(pfDir, "*", SearchOption.AllDirectories))
            {
                string to = Path.Combine(copy, Path.GetRelativePath(pfDir, file));
                Directory.CreateDirectory(Path.GetDirectoryName(to));
                File.Copy(file, to);
            }
            File.AppendAllText(Path.Combine(copy, "configs", "scan.default.v1.json"), " ");
            bool caught = false;
            try { SecondEyes.Perception.PerceptionFolder.Load(copy); } catch (InvalidDataException) { caught = true; }
            Check("a copy with one changed file is refused", caught);
            Directory.Delete(copy, true);
        }

        Console.WriteLine(pass + " passed, " + fail + " failed, " + known + " known difference(s) reported");
        return fail == 0 ? 0 : 1;
    }
}
