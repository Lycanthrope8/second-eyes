using System;
using System.Collections.Generic;
using System.Text;
using SecondEyes.Grounding.Replay;

namespace SecondEyes.Grounding.Prompting
{
    // A2.5 delivery 2 (D104(4)): the on-device golden self-check, free of Unity so it is tested on a PC. For each golden,
    // the document, mapping, mapping hash and prompt are rebuilt from the shipped snapshot and command records and compared
    // with the PC's; the runtime's tokenization of the prompt is compared with the PC's token IDs when a tokenizer is given.

    /// <summary>The pinned prompt asset (headset/prompt-asset.json).</summary>
    public sealed class PromptAsset
    {
        public string HeaderLine, SemanticsLine, BeforeSystem, Between, AfterUser, SystemMessage, AskCode, AskTarget;
        public string[] ObjectCodes;
        public Dictionary<string, int> CodeTokenIds = new Dictionary<string, int>();

        public static PromptAsset From(JNode a)
        {
            var p = new PromptAsset
            {
                HeaderLine = a["header_line"].Text, SemanticsLine = a["semantics_line"].Text,
                BeforeSystem = a["wrapper"]["before_system"].Text, Between = a["wrapper"]["between"].Text,
                AfterUser = a["wrapper"]["after_user"].Text, SystemMessage = a["system_message"].Text,
                AskCode = a["ask_code"].Text, AskTarget = a["ask_target"].Text
            };
            var codes = new List<string>();
            foreach (JNode c in a["object_codes"].Items) codes.Add(c.Text);
            p.ObjectCodes = codes.ToArray();
            foreach (var f in a["code_token_ids"].Fields) p.CodeTokenIds[f.Key] = int.Parse(f.Value.Text);
            return p;
        }
    }

    public sealed class GoldenResult
    {
        public string RequestId, Kind, Error, BuiltDocumentSha256, BuiltPromptSha256;
        public bool DocumentOk, MappingOk, MappingHashOk, PromptOk;
        public bool? TokensOk;              // null when no tokenizer was given
        public int RuntimeTokens = -1, FirstTokenDifference = -1;

        public bool AllOk { get { return Error == null && DocumentOk && MappingOk && MappingHashOk && PromptOk && TokensOk != false; } }
    }

    /// <summary>On-device regression checks of the prompt core, run before every golden check (ChatGPT's review of
    /// delivery 2). The expected texts are Python's own output: sorted() and serializer._object_row.</summary>
    public static class PromptSelfChecks
    {
        public sealed class Result { public string Name; public bool Passed; public string Detail; }

        public static List<Result> Run()
        {
            var r = new List<Result>();
            Action<string, bool, string> add = (n, ok, d) => r.Add(new Result { Name = n, Passed = ok, Detail = d });
            string smile = "\U0001F600", fullZ = "\uFF5A", privateUse = "\uE000";
            add("a supplementary-plane character sorts after U+FF5A by code point, though its UTF-16 code units sort first",
                Canon.ComparePython(smile, fullZ) > 0 && string.CompareOrdinal(smile, fullZ) < 0, "");
            add("U+E000 sorts before a supplementary-plane character", Canon.ComparePython(privateUse, smile) < 0, "");
            add("a prefix sorts first, and equal strings compare equal",
                Canon.ComparePython("a", "ab") < 0 && Canon.ComparePython("ab", "a") > 0 && Canon.ComparePython("x", "x") == 0, "");
            var labels = new List<string> { fullZ, smile, privateUse, "red" };
            labels.Sort(Canon.ComparePython);
            add("four labels sort as Python's sorted() does: red, U+E000, U+FF5A, U+1F600",
                string.Join("|", labels) == string.Join("|", new[] { "red", privateUse, fullZ, smile }), string.Join("|", labels));
            JNode o = JParse.Parse("{\"object_id\":\"o1\",\"category\":{\"state\":\"unknown\"},\"attributes\":{\"colours\":{\"state\":\"known\"," +
                "\"value\":[\"\\uff5a\",\"\\ud83d\\ude00\",\"red\"],\"evidence\":{\"kind\":\"annotated\",\"assumptions\":[]}}}," +
                "\"geometry\":{\"center_m\":{\"state\":\"unknown\"},\"size_m\":{\"state\":\"unknown\"},\"rotation_xyzw\":{\"state\":\"unknown\"}}," +
                "\"semantic_front\":{\"state\":\"unknown\"}}");
            string row = Canon.Enc(Coordinates.ObjectRow(o));
            add("an object row's colours sort as Python's _object_row does",
                row == "[\"o1\",null,[\"red\",\"" + fullZ + "\",\"" + smile + "\"],null,null,null,null,null,[]]", row);
            return r;
        }
    }

    public static class GoldenCheck
    {
        private static readonly UTF8Encoding Utf8 = new UTF8Encoding(false);

        private static List<string> Texts(JNode list)
        {
            var r = new List<string>();
            foreach (JNode x in list.Items) r.Add(x.Text);
            return r;
        }

        public static GoldenResult Check(PromptAsset a, JNode golden, string sceneText, string commandText, Func<byte[], int[]> tokenize)
        {
            var r = new GoldenResult { RequestId = golden["request_id"].Text, Kind = golden["kind"].Text };
            try
            {
                JNode scene = JParse.Parse(sceneText), command = JParse.Parse(commandText);
                string document = Coordinates.Document(a.HeaderLine, a.SemanticsLine, scene, command);
                r.BuiltDocumentSha256 = ReplayHash.Sha256Hex(Utf8.GetBytes(document));
                r.DocumentOk = r.BuiltDocumentSha256 == golden["document_sha256"].Text;
                List<KeyValuePair<string, string>> mapping = Choices.SerializedOrder(scene, a.ObjectCodes, a.AskCode, a.AskTarget);
                var codes = new List<string>();
                var targets = new List<string>();
                var ids = new List<int>();
                foreach (var m in mapping) { codes.Add(m.Key); targets.Add(m.Value); ids.Add(a.CodeTokenIds[m.Key]); }
                List<string> wantCodes = Texts(golden["codes"]), wantTargets = Texts(golden["targets"]);
                r.MappingOk = string.Join("\n", codes) == string.Join("\n", wantCodes) && string.Join("\n", targets) == string.Join("\n", wantTargets);
                r.MappingHashOk = ReplayHash.Mapping(codes.ToArray(), targets.ToArray(), ids.ToArray()) == golden["mapping_sha256"].Text;
                byte[] prompt = Utf8.GetBytes(Choices.Prompt(a.BeforeSystem, a.SystemMessage, a.Between, a.AfterUser, mapping, document));
                r.BuiltPromptSha256 = ReplayHash.Sha256Hex(prompt);
                r.PromptOk = r.BuiltPromptSha256 == golden["prompt_sha256"].Text && prompt.Length.ToString() == golden["prompt_bytes"].Text;
                if (tokenize != null)
                {
                    int[] got = tokenize(prompt);
                    List<JNode> want = golden["token_ids"].Items;
                    r.RuntimeTokens = got.Length;
                    int n = Math.Min(got.Length, want.Count);
                    for (int i = 0; i < n; i++)
                        if (got[i].ToString() != want[i].Text) { r.FirstTokenDifference = i; break; }
                    if (r.FirstTokenDifference < 0 && got.Length != want.Count) r.FirstTokenDifference = n;
                    r.TokensOk = r.FirstTokenDifference < 0;
                }
            }
            catch (Exception e)
            {
                r.Error = e.GetType().Name + ": " + e.Message;
            }
            return r;
        }
    }
}
