using System;
using System.Collections.Generic;
using System.IO;
using System.Security.Cryptography;
using System.Text;
using SecondEyes.Grounding.Prompting;
using SecondEyes.Grounding.Resolution;

namespace SecondEyes.Perception
{
    /// <summary>
    /// A4 (stage 3): the perception folder pushed to the headset (grounding/quest/perception_device.py), free of Unity.
    /// Loading verifies every file manifest.json lists by size and SHA-256, as the parser folder does, then checks what it
    /// holds: the scan's and the localization's configurations and the size priors against their schemas, the category map
    /// against the contract. The manifest's own SHA-256 identifies the folder in every capture record. The folder also gives
    /// the parser folder the schemas and maps it doesn't hold (scene v3, format-2 queries, results and commands, the
    /// bedroom's map), so a scanned room goes through the accepted parser path.
    /// </summary>
    public sealed class PerceptionFolder
    {
        public string Dir, ManifestSha256, CategoryMapId, SizePriorsId;
        public JNode Manifest, ScanConfig, LocalizationConfig, SizePriors, CategoryMap;
        /// <summary>The automatic scan's settings (auto-scan-config.v1.json or v2, by its schema_version); null in a folder built before it existed.</summary>
        public JNode AutoScanConfig;
        public readonly Dictionary<string, string> Maps = new Dictionary<string, string>(StringComparer.Ordinal);
        private readonly Dictionary<string, string> texts = new Dictionary<string, string>(StringComparer.Ordinal);
        private readonly Dictionary<string, Schema> schemas = new Dictionary<string, Schema>(StringComparer.Ordinal);
        private static readonly UTF8Encoding Utf8 = new UTF8Encoding(false, true);

        public static bool Present(string dir) { return File.Exists(Path.Combine(dir, "manifest.json")); }

        /// <summary>A schema's text by file name, or null when the folder doesn't hold it.</summary>
        public string SchemaText(string name)
        {
            string t;
            return texts.TryGetValue("schemas/" + name, out t) ? t : null;
        }

        public Schema Schema(string name)
        {
            Schema s;
            if (!schemas.TryGetValue(name, out s))
            {
                string t = SchemaText(name);
                if (t == null) throw new KeyNotFoundException("the perception folder holds no schema " + name);
                schemas[name] = s = new Schema(JParse.Parse(t));
            }
            return s;
        }

        public static PerceptionFolder Load(string dir)
        {
            byte[] mb = File.ReadAllBytes(Path.Combine(dir, "manifest.json"));
            var f = new PerceptionFolder { Dir = dir, ManifestSha256 = Sha256Hex(mb), Manifest = JParse.Parse(Utf8.GetString(mb)) };
            if (f.Manifest["record_type"].Text != "a4_perception_folder") throw new InvalidDataException("not a perception folder");
            foreach (var kv in f.Manifest["files"].Fields)
            {
                string path = Path.Combine(dir, kv.Key.Replace('/', Path.DirectorySeparatorChar));
                if (!File.Exists(path)) throw new InvalidDataException(kv.Key + " is missing");
                byte[] b = File.ReadAllBytes(path);
                if (b.Length != (long)kv.Value["bytes"].Num || Sha256Hex(b) != kv.Value["sha256"].Text)
                    throw new InvalidDataException(kv.Key + " differs from the manifest");
                f.texts[kv.Key] = Utf8.GetString(b);
            }
            Func<string, JNode> role = r =>
            {
                string name = f.Manifest[r].Text, t;
                if (name == null || !f.texts.TryGetValue(name, out t)) throw new InvalidDataException("the manifest's " + r + " names no listed file");
                return JParse.Parse(t);
            };
            f.ScanConfig = role("scan_config");
            f.LocalizationConfig = role("localization_config");
            f.SizePriors = role("size_priors");
            f.CategoryMap = role("category_map");
            Expect(f, f.ScanConfig, "scan-config.v1.json", "scan_config");
            Expect(f, f.LocalizationConfig, "localization-config.v1.json", "localization_config");
            Expect(f, f.SizePriors, "size-priors.v1.json", "size_priors");
            if (f.Manifest.Has("autoscan_config"))
            {
                f.AutoScanConfig = role("autoscan_config");
                Expect(f, f.AutoScanConfig, "auto-scan-config.v" + ((int)f.AutoScanConfig["schema_version"].Num).ToString(System.Globalization.CultureInfo.InvariantCulture) + ".json", "autoscan_config");
            }
            var issues = new Contract(f.Schema).Validate(new List<KeyValuePair<string, JNode>> { new KeyValuePair<string, JNode>("category_map", f.CategoryMap) });
            foreach (Issue i in issues)
                if (i.IsError) throw new InvalidDataException("the category map fails the contract: " + i.Code + " " + i.Path);
            f.CategoryMapId = f.CategoryMap["map_id"].Text;
            f.SizePriorsId = f.SizePriors["priors_id"].Text;
            foreach (var kv in f.texts)
                if (kv.Key.StartsWith("maps/", StringComparison.Ordinal)) f.Maps[JParse.Parse(kv.Value)["map_id"].Text] = kv.Value;
            return f;
        }

        private static void Expect(PerceptionFolder f, JNode record, string schema, string role)
        {
            var errors = f.Schema(schema).Errors(record);
            if (errors.Count > 0) throw new InvalidDataException("the " + role + " fails " + schema + ": " + errors[0].Keyword + " " + errors[0].Path);
        }

        private static string Sha256Hex(byte[] b)
        {
            using (var h = SHA256.Create())
            {
                var sb = new StringBuilder(64);
                foreach (byte x in h.ComputeHash(b)) sb.Append(x.ToString("x2"));
                return sb.ToString();
            }
        }
    }
}
