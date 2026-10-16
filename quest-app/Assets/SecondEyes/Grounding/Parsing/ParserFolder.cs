using System;
using System.Collections.Generic;
using System.IO;
using System.Text;
using SecondEyes.Grounding.Prompting;
using SecondEyes.Grounding.Replay;
using SecondEyes.Grounding.Resolution;

namespace SecondEyes.Grounding.Parsing
{
    /// <summary>
    /// A2.7 (D115): the parser folder pushed to the headset (grounding/a27/headset.py), free of Unity. Loading verifies
    /// every file listed in manifest.json by SHA-256 and size before anything is built from it; the manifest's own
    /// SHA-256 identifies the configuration in every logged outcome. Builds the pipeline's assets: the language
    /// specification, the parser prompt, the resolver (schemas, relation and direction configurations, contract) with
    /// A2.4's checker and the resolver protocol; and holds the grammar, the category maps by ID and the startup check.
    /// </summary>
    public sealed class ParserFolder
    {
        public sealed class StartupPrompt { public string Text, PromptSha256; public int[] Ids; }

        public string Dir, ManifestSha256, GgufFile, GgufSha256, Candidate, Gbnf;
        public ParserAssets Assets;
        public readonly Dictionary<string, string> Maps = new Dictionary<string, string>(StringComparer.Ordinal);
        public readonly List<StartupPrompt> Startup = new List<StartupPrompt>();
        public JNode Manifest;

        public static bool Present(string dir) { return File.Exists(Path.Combine(dir, "manifest.json")); }

        private static readonly UTF8Encoding Utf8 = new UTF8Encoding(false, true);

        public static ParserFolder Load(string dir) { return Load(dir, null, null); }

        /// <summary>A4 (stage 3): with a perception folder, the schemas and category maps the parser folder doesn't hold come
        /// from it, so a scanned room's scene v3, its format-2 queries and commands, and the bedroom's map are found. The
        /// parser folder's own files always come first; without a perception folder nothing changes.</summary>
        public static ParserFolder Load(string dir, Func<string, string> extraSchemaText, IDictionary<string, string> extraMaps)
        {
            byte[] mb = File.ReadAllBytes(Path.Combine(dir, "manifest.json"));
            var f = new ParserFolder { Dir = dir, ManifestSha256 = ReplayHash.Sha256Hex(mb), Manifest = JParse.Parse(Utf8.GetString(mb)) };
            if (f.Manifest["record_type"].Text != "a27_parser_folder") throw new InvalidDataException("not an A2.7 parser folder");
            var texts = new Dictionary<string, string>(StringComparer.Ordinal);
            foreach (var kv in f.Manifest["files"].Fields)
            {
                string path = Path.Combine(dir, kv.Key.Replace('/', Path.DirectorySeparatorChar));
                if (!File.Exists(path)) throw new InvalidDataException(kv.Key + " is missing");
                byte[] b = File.ReadAllBytes(path);
                if (b.Length != (long)kv.Value["bytes"].Num || ReplayHash.Sha256Hex(b) != kv.Value["sha256"].Text)
                    throw new InvalidDataException(kv.Key + " differs from the manifest");
                if (!kv.Key.StartsWith("acceptance/", StringComparison.Ordinal)) texts[kv.Key] = Utf8.GetString(b);
            }
            var cache = new Dictionary<string, Schema>();
            Func<string, Schema> schema = name =>
            {
                Schema s;
                if (!cache.TryGetValue(name, out s))
                {
                    string text;
                    if (!texts.TryGetValue("assets/schemas/" + name, out text) && extraSchemaText != null) text = extraSchemaText(name);
                    if (text == null) throw new KeyNotFoundException("no schema " + name + " in the parser folder" + (extraSchemaText != null ? " or the perception folder" : ""));
                    cache[name] = s = new Schema(JParse.Parse(text));
                }
                return s;
            };
            var setup = ResolverSetup.Load(schema, JParse.Parse(texts["assets/relations.v1.json"]), JParse.Parse(texts["assets/directions.v1.json"]));
            if (setup.ConfigFileIssues.Count > 0) throw new InvalidDataException("a configuration file fails its schema: " + setup.ConfigFileIssues[0]);
            JNode parser = JParse.Parse(texts["assets/parser.json"]);
            JNode specNode = JParse.Parse(texts["assets/language.v1.json"]);
            f.Assets = new ParserAssets
            {
                Spec = LanguageSpec.From(specNode), Prompt = ParserPrompt.From(JParse.Parse(texts["assets/settings.v1.json"])["parser_prompt"]),
                Checker = new Checker(setup, JParse.Parse(texts["assets/resolver-protocol.json"])),
                EndToken = (int)parser["end_token"].Num, MaxNewTokens = (int)parser["max_new_tokens"].Num,
                ContextTokens = (int)parser["context_tokens"].Num, ManifestSha256 = f.ManifestSha256
            };
            f.GgufFile = parser["gguf_file"].Text;
            f.GgufSha256 = parser["gguf_sha256"].Text;
            f.Candidate = parser["candidate"].Text;
            f.Gbnf = texts["assets/grammar.gbnf"];
            foreach (var kv in texts)
                if (kv.Key.StartsWith("maps/", StringComparison.Ordinal))
                {
                    JNode m = JParse.Parse(kv.Value);
                    f.Maps[m["map_id"].Text] = kv.Value;
                }
            if (extraMaps != null)
                foreach (var kv in extraMaps)
                    if (!f.Maps.ContainsKey(kv.Key)) f.Maps[kv.Key] = kv.Value;
            foreach (var p in JParse.Parse(texts["startup-check.json"])["prompts"].Items)
            {
                var ids = new int[p["ids"].Items.Count];
                for (int i = 0; i < ids.Length; i++) ids[i] = (int)p["ids"].Items[i].Num;
                f.Startup.Add(new StartupPrompt { Text = p["text"].Text, PromptSha256 = p["prompt_sha256"].Text, Ids = ids });
            }
            return f;
        }

        /// <summary>The parser's startup check: each fixed prompt, built by this code and tokenized natively, equals the
        /// PC's pinned tokenization. Null when every prompt passes; otherwise the first difference.</summary>
        public string StartupCheck(IParserNative native)
        {
            foreach (var p in Startup)
            {
                string prompt = Assets.Prompt.Build(p.Text);
                byte[] utf8 = new UTF8Encoding(false).GetBytes(prompt);
                if (ReplayHash.Sha256Hex(utf8) != p.PromptSha256) return "the prompt for '" + p.Text + "' differs from the PC's";
                int[] ids = native.Tokenize(utf8);
                if (ids == null || ids.Length != p.Ids.Length) return "the tokenization of '" + p.Text + "' differs in length";
                for (int i = 0; i < ids.Length; i++) if (ids[i] != p.Ids[i]) return "the tokenization of '" + p.Text + "' differs at token " + i;
            }
            return null;
        }

        /// <summary>D117: the model file's SHA-256, which must be the exported candidate's (parser.json). Streams the file
        /// once; InvalidDataException on a mismatch, so no other weights are ever loaded under this folder.</summary>
        public string VerifyModel(string path)
        {
            string sha;
            using (var h = System.Security.Cryptography.SHA256.Create())
            using (var fs = File.OpenRead(path))
            {
                byte[] d = h.ComputeHash(fs);
                var sb = new StringBuilder();
                foreach (byte b in d) sb.Append(b.ToString("x2", System.Globalization.CultureInfo.InvariantCulture));
                sha = sb.ToString();
            }
            if (sha != GgufSha256)
                throw new InvalidDataException("the model file " + Path.GetFileName(path) + " (" + sha + ") is not the exported candidate (" + GgufSha256 + ")");
            return sha;
        }

        /// <summary>The scene's category map, or null when the folder has none for its ID.</summary>
        public string MapFor(JNode scene)
        {
            if (scene.Kind != JKind.Object || !scene.Has("category_map") || scene["category_map"].Kind != JKind.String) return null;
            string m;
            return Maps.TryGetValue(scene["category_map"].Text, out m) ? m : null;
        }
    }
}
