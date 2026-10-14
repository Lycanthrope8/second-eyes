using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Text;
using SecondEyes.Grounding.Prompting;
using SecondEyes.Grounding.Replay;
using SecondEyes.Grounding.Resolution;

namespace SecondEyes.Grounding.Parsing
{
    /// <summary>
    /// A2.7 (D115): the headset's acceptance run over a parser folder's acceptance/ files, free of Unity. Every row is
    /// written as it completes; the PC judges (python -m grounding.a27 headset-judge). Criteria, headset halves:
    /// 7, decodes.jsonl (the native tokenization of each frozen prompt, then its constrained decode); 4, masks.jsonl (the
    /// allowed tokens at every frozen prefix, as count and SHA-256); 1, resolutions.jsonl (A2.4's checker on the
    /// headset); 2, readers.jsonl (the reader's verdict, render and interpretation). A row whose native call or
    /// resolution fails is a technical failure, recorded with its error, never an outcome. Call Run on the model's
    /// worker thread (LlamaRuntime.WithModel).
    /// </summary>
    public static class ParserAcceptance
    {
        private static readonly UTF8Encoding Utf8 = new UTF8Encoding(false);

        private static List<JNode> Lines(string path)
        {
            var l = new List<JNode>();
            foreach (var line in File.ReadAllLines(path, Utf8)) if (line.Trim().Length > 0) l.Add(JParse.Parse(line));
            return l;
        }

        private static int[] Ints(JNode a)
        {
            var v = new int[a.Items.Count];
            for (int i = 0; i < v.Length; i++) v[i] = (int)a.Items[i].Num;
            return v;
        }

        private static JsonWriter IntArr(JsonWriter w, IList<int> xs)
        {
            w.BeginArr();
            foreach (int x in xs) w.I(x);
            return w.End();
        }

        public sealed class Summary { public int Decodes, Masks, Resolutions, Readers, Technical; public double Seconds; }

        /// <summary>The SHA-256 of a query's canonical JSON (Python's json.dumps with sorted keys), as the judge expects.</summary>
        public static string QuerySha(JNode q) { return q == null ? null : ReplayHash.Sha256Hex(Utf8.GetBytes(PyJson.Canonical(q))); }

        public static Summary Run(ParserFolder f, IParserNative native, IParserMasks masks, string outDir, Action<string> progress,
                                  Func<bool> cancelled)
        {
            var sum = new Summary();
            var total = Stopwatch.StartNew();
            string acc = Path.Combine(f.Dir, "acceptance");
            using (var w = new StreamWriter(Path.Combine(outDir, "decodes.jsonl"), false, Utf8))
            {
                var rows = Lines(Path.Combine(acc, "decodes.jsonl"));
                foreach (var r in rows)
                {
                    if (cancelled != null && cancelled()) break;
                    var t = Stopwatch.StartNew();
                    var j = new JsonWriter().BeginObj().Key("parent_command_id").S(r["parent_command_id"].Text);
                    int[] ids = null;
                    string error = null;
                    try { ids = native.Tokenize(Utf8.GetBytes(r["prompt"].Text)); }
                    catch (Exception e) { error = "tokenize: " + e.Message; }
                    double tokMs = t.Elapsed.TotalMilliseconds;
                    j.Key("prompt_ids");
                    IntArr(j, ids ?? new int[0]);
                    var output = new int[f.Assets.MaxNewTokens];
                    int status = -1, n = -1;
                    if (ids != null && ids.Length > 0)
                    {
                        t.Restart();
                        try { n = native.Decode(ids, f.Assets.MaxNewTokens, output, out status); }
                        catch (Exception e) { error = "decode: " + e.Message; n = -1; }
                        if (n < 0 || status < 0) error = error ?? ("decode: " + native.LastError());
                    }
                    else error = error ?? ("tokenize: " + native.LastError());
                    var tokens = new List<int>();
                    for (int i = 0; i < Math.Max(n, 0); i++) tokens.Add(output[i]);
                    string st = error != null ? "technical_failure" : (status == 0 ? "completed" : (status == 1 ? "length_exhausted" : "empty_allowed_set"));
                    if (error != null) sum.Technical++;
                    j.Key("tokens");
                    IntArr(j, tokens);
                    w.WriteLine(j.Key("status").S(st).Key("error").S(error).Key("tokenize_ms").D(tokMs).Key("decode_ms").D(t.Elapsed.TotalMilliseconds)
                                 .End().ToString());
                    w.Flush();
                    sum.Decodes++;
                    if (sum.Decodes % 25 == 0 && progress != null) progress("decodes " + sum.Decodes + "/" + rows.Count);
                }
            }
            using (var w = new StreamWriter(Path.Combine(outDir, "masks.jsonl"), false, Utf8))
            {
                var rows = Lines(Path.Combine(acc, "masks.jsonl"));
                foreach (var r in rows)
                {
                    if (cancelled != null && cancelled()) break;
                    var t = Stopwatch.StartNew();
                    string sha = null, error = null;
                    int count = -1;
                    try { count = masks.AllowedAfter(Ints(r["prefix"]), out sha); }
                    catch (Exception e) { error = e.Message; }
                    if (error != null || count < 0) sum.Technical++;
                    var j = new JsonWriter().BeginObj().Key("prefix");
                    IntArr(j, Ints(r["prefix"]));
                    w.WriteLine(j.Key("count").I(count).Key("ids_sha256").S(sha).Key("error").S(error).Key("ms").D(t.Elapsed.TotalMilliseconds).End().ToString());
                    w.Flush();
                    sum.Masks++;
                    if (sum.Masks % 10 == 0 && progress != null) progress("masks " + sum.Masks + "/" + rows.Count);
                }
            }
            using (var w = new StreamWriter(Path.Combine(outDir, "resolutions.jsonl"), false, Utf8))
            {
                JNode g = JParse.Parse(File.ReadAllText(Path.Combine(acc, "resolutions.json"), Utf8));
                var recs = g["records"].Items;
                foreach (var r in g["resolutions"].Items)
                {
                    if (cancelled != null && cancelled()) break;
                    var t = Stopwatch.StartNew();
                    var j = new JsonWriter().BeginObj().Key("parent_command_id").S(r["parent_command_id"].Text).Key("view").S(r["view"].Text);
                    if (r["interp"].Kind == JKind.Null)
                        j.Key("outcome").S("technical").Key("target_id").Null().Key("candidate_ids").Strs(new string[0]).Key("technical").B(true)
                         .Key("technical_failure").B(false).Key("error").Null();
                    else
                    {
                        try
                        {
                            var o = f.Assets.Checker.Resolve(r["interp"], recs[(int)r["scene"].Num], recs[(int)r["command"].Num], recs[(int)r["map"].Num]);
                            j.Key("outcome").S(o.Outcome).Key("target_id").S(o.TargetId).Key("candidate_ids").Strs(o.CandidateIds.ToArray())
                             .Key("technical").B(false).Key("technical_failure").B(false).Key("error").Null()
                             .Key("query_sha256").S(QuerySha(o.Query));
                        }
                        catch (Exception e)
                        {
                            sum.Technical++;
                            j.Key("outcome").Null().Key("target_id").Null().Key("candidate_ids").Strs(new string[0]).Key("technical").B(false)
                             .Key("technical_failure").B(true).Key("error").S(e.GetType().Name + ": " + e.Message);
                        }
                    }
                    w.WriteLine(j.Key("ms").D(t.Elapsed.TotalMilliseconds).End().ToString());
                    w.Flush();
                    sum.Resolutions++;
                    if (sum.Resolutions % 50 == 0 && progress != null) progress("resolutions " + sum.Resolutions);
                }
            }
            JNode ctx = File.Exists(Path.Combine(acc, "reader-context.json"))
                ? JParse.Parse(File.ReadAllText(Path.Combine(acc, "reader-context.json"), Utf8)) : null;
            using (var w = new StreamWriter(Path.Combine(outDir, "readers.jsonl"), false, Utf8))
            {
                foreach (var r in Lines(Path.Combine(acc, "readers.jsonl")))
                {
                    var j = new JsonWriter().BeginObj().Key("text").S(r["text"].Text);
                    try
                    {
                        var p = Language.ParseOutput(r["text"].Text, f.Assets.Spec);
                        var interp = Language.Interpretation(p);
                        j.Key("valid").B(true).Key("render").S(Language.Render(p)).Key("interpretation").S(PyJson.Canonical(interp))
                         .Key("query_sha256").S(ctx == null ? null
                              : QuerySha(Checker.Query(interp, ctx["scene"], ctx["command"], f.Assets.Checker.QuerySuffix)));
                    }
                    catch (LanguageError) { j.Key("valid").B(false).Key("render").Null().Key("interpretation").Null(); }
                    w.WriteLine(j.End().ToString());
                    sum.Readers++;
                }
            }
            sum.Seconds = total.Elapsed.TotalSeconds;
            return sum;
        }
    }
}
