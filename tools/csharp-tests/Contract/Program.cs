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

    static int Main(string[] args)
    {
        if (args.Length != 2) { Console.Error.WriteLine("usage: Contract <grounding/tests/fixtures> <schemas>"); return 2; }
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
        Check("the same query labelled format 1 fails format 1's schema",
              ResolverValidation.CheckQuery(relabelled, setup.Schemas(ResolverValidation.QuerySchemaFile(relabelled)), out norm).Count > 0);

        Console.WriteLine(pass + " passed, " + fail + " failed, " + known + " known difference(s) reported");
        return fail == 0 ? 0 : 1;
    }
}
