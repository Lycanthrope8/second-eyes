using System;
using System.Collections.Generic;
using System.Text;
using System.Text.RegularExpressions;
using SecondEyes.Grounding.Prompting;

namespace SecondEyes.Grounding.Resolution
{
    // A2.7 delivery 2 (D115): JSON Schema (draft 2020-12) for the keywords the resolution path's eight schemas use, free
    // of Unity: $ref (to #/$defs), type, required, properties, additionalProperties, const, enum, items, minItems,
    // maxItems, minLength, uniqueItems, minimum, maximum, exclusiveMinimum, exclusiveMaximum, pattern, if/then/else and
    // oneOf. Annotations ($schema, $id, title, description) are ignored, and so is format, which Python's jsonschema
    // doesn't assert without a format checker. A schema using any other keyword is refused when loaded, so the checker
    // can't silently skip a rule. Errors follow Python's jsonschema: one per failed keyword at the instance path,
    // descending through $ref, properties, items, additionalProperties, then and else; oneOf fails as one error. The
    // values follow Python's JSON: integer admits a whole-valued float, true never equals 1, a string's length counts
    // code points, and patterns are searched with Python's whitespace for \s and \S.
    public sealed class SchemaError
    {
        public string Path, Keyword;
        public override string ToString() { return Path + " " + Keyword; }
    }

    public sealed class Schema
    {
        private static readonly HashSet<string> Supported = new HashSet<string>
        {
            "$ref", "type", "required", "properties", "additionalProperties", "const", "enum", "items", "minItems",
            "maxItems", "minLength", "uniqueItems", "minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum",
            "pattern", "if", "then", "else", "oneOf", "$defs", "$schema", "$id", "title", "description", "format"
        };
        private readonly JNode root;
        private readonly Dictionary<string, Regex> patterns = new Dictionary<string, Regex>();

        public Schema(JNode schema)
        {
            root = schema;
            Audit(schema, "#");
        }

        private void Audit(JNode s, string where)
        {
            if (s.Kind == JKind.Bool) return;
            if (s.Kind != JKind.Object) throw new ArgumentException("a schema at " + where + " is not an object");
            foreach (var kv in s.Fields)
            {
                if (!Supported.Contains(kv.Key)) throw new ArgumentException("unsupported keyword " + kv.Key + " at " + where);
                switch (kv.Key)
                {
                    case "properties": case "$defs":
                        foreach (var p in kv.Value.Fields) Audit(p.Value, where + "/" + kv.Key + "/" + p.Key);
                        break;
                    case "items": case "additionalProperties": case "if": case "then": case "else":
                        Audit(kv.Value, where + "/" + kv.Key);
                        break;
                    case "oneOf":
                        for (int i = 0; i < kv.Value.Items.Count; i++) Audit(kv.Value.Items[i], where + "/oneOf/" + i);
                        break;
                    case "pattern":
                        if (!patterns.ContainsKey(kv.Value.Text)) patterns[kv.Value.Text] = PyPattern(kv.Value.Text);
                        break;
                    case "$ref":
                        if (!kv.Value.Text.StartsWith("#/$defs/")) throw new ArgumentException("unsupported $ref " + kv.Value.Text);
                        break;
                }
            }
        }

        public List<SchemaError> Errors(JNode instance)
        {
            var out_ = new List<SchemaError>();
            Check(root, instance, "$", out_);
            return out_;
        }

        public bool Valid(JNode instance) { return Errors(instance).Count == 0; }

        private void Check(JNode s, JNode x, string path, List<SchemaError> out_)
        {
            if (s.Kind == JKind.Bool) { if (!s.Bool) Add(out_, path, "false"); return; }
            foreach (var kv in s.Fields)
            {
                JNode v = kv.Value;
                switch (kv.Key)
                {
                    case "$ref":
                        Check(Resolve(v.Text), x, path, out_);
                        break;
                    case "type":
                        bool ok = false;
                        if (v.Kind == JKind.Array) { foreach (var t in v.Items) ok |= IsType(x, t.Text); }
                        else ok = IsType(x, v.Text);
                        if (!ok) Add(out_, path, "type");
                        break;
                    case "required":
                        if (x.Kind == JKind.Object)
                            foreach (var r in v.Items) if (!x.Has(r.Text)) Add(out_, path, "required");
                        break;
                    case "properties":
                        if (x.Kind == JKind.Object)
                            foreach (var f in x.Fields)
                                foreach (var p in v.Fields)
                                    if (p.Key == f.Key) Check(p.Value, f.Value, Child(path, f.Key), out_);
                        break;
                    case "additionalProperties":
                        if (x.Kind == JKind.Object)
                        {
                            var defined = new HashSet<string>();
                            if (s.Has("properties")) foreach (var p in s["properties"].Fields) defined.Add(p.Key);
                            var extra = new List<KeyValuePair<string, JNode>>();
                            foreach (var f in x.Fields) if (!defined.Contains(f.Key)) extra.Add(f);
                            if (v.Kind == JKind.Bool) { if (!v.Bool && extra.Count > 0) Add(out_, path, "additionalProperties"); }
                            else foreach (var f in extra) Check(v, f.Value, Child(path, f.Key), out_);
                        }
                        break;
                    case "const":
                        if (!JsonEqual(x, v)) Add(out_, path, "const");
                        break;
                    case "enum":
                        bool any = false;
                        foreach (var e in v.Items) any |= JsonEqual(x, e);
                        if (!any) Add(out_, path, "enum");
                        break;
                    case "items":
                        if (x.Kind == JKind.Array)
                            for (int i = 0; i < x.Items.Count; i++) Check(v, x.Items[i], path + "[" + i + "]", out_);
                        break;
                    case "minItems":
                        if (x.Kind == JKind.Array && x.Items.Count < v.Num) Add(out_, path, "minItems");
                        break;
                    case "maxItems":
                        if (x.Kind == JKind.Array && x.Items.Count > v.Num) Add(out_, path, "maxItems");
                        break;
                    case "minLength":
                        if (x.Kind == JKind.String && CodePoints(x.Text) < v.Num) Add(out_, path, "minLength");
                        break;
                    case "uniqueItems":
                        if (v.Kind == JKind.Bool && v.Bool && x.Kind == JKind.Array)
                        {
                            bool dup = false;
                            for (int i = 0; i < x.Items.Count && !dup; i++)
                                for (int j = i + 1; j < x.Items.Count && !dup; j++) dup = JsonEqual(x.Items[i], x.Items[j]);
                            if (dup) Add(out_, path, "uniqueItems");
                        }
                        break;
                    case "minimum":
                        if (IsNumber(x) && x.Num < v.Num) Add(out_, path, "minimum");
                        break;
                    case "maximum":
                        if (IsNumber(x) && x.Num > v.Num) Add(out_, path, "maximum");
                        break;
                    case "exclusiveMinimum":
                        if (IsNumber(x) && x.Num <= v.Num) Add(out_, path, "exclusiveMinimum");
                        break;
                    case "exclusiveMaximum":
                        if (IsNumber(x) && x.Num >= v.Num) Add(out_, path, "exclusiveMaximum");
                        break;
                    case "pattern":
                        if (x.Kind == JKind.String && !patterns[v.Text].IsMatch(x.Text)) Add(out_, path, "pattern");
                        break;
                    case "if":
                        var probe = new List<SchemaError>();
                        Check(v, x, path, probe);
                        if (probe.Count == 0) { if (s.Has("then")) Check(s["then"], x, path, out_); }
                        else if (s.Has("else")) Check(s["else"], x, path, out_);
                        break;
                    case "oneOf":
                        int matches = 0;
                        foreach (var sub in v.Items)
                        {
                            var tmp = new List<SchemaError>();
                            Check(sub, x, path, tmp);
                            if (tmp.Count == 0) matches++;
                        }
                        if (matches != 1) Add(out_, path, "oneOf");
                        break;
                }
            }
        }

        private JNode Resolve(string reference)
        {
            JNode n = root;
            foreach (var part in reference.Substring(2).Split('/'))
                n = n[part.Replace("~1", "/").Replace("~0", "~")];
            return n;
        }

        private static void Add(List<SchemaError> out_, string path, string keyword)
        {
            out_.Add(new SchemaError { Path = path, Keyword = keyword });
        }

        private static string Child(string path, string key) { return path + "." + key; }

        private static bool IsNumber(JNode x) { return x.Kind == JKind.Number; }

        private static bool IsType(JNode x, string t)
        {
            switch (t)
            {
                case "object": return x.Kind == JKind.Object;
                case "array": return x.Kind == JKind.Array;
                case "string": return x.Kind == JKind.String;
                case "boolean": return x.Kind == JKind.Bool;
                case "null": return x.Kind == JKind.Null;
                case "number": return x.Kind == JKind.Number;
                case "integer": return x.Kind == JKind.Number && (x.IsInteger || (!double.IsInfinity(x.Num) && Math.Floor(x.Num) == x.Num));
                default: throw new ArgumentException("unknown type " + t);
            }
        }

        /// <summary>Python's JSON equality as jsonschema applies it: numbers by value (1 equals 1.0), a boolean never
        /// equals a number, arrays item by item, objects by their key sets and values.</summary>
        public static bool JsonEqual(JNode a, JNode b)
        {
            if (a.Kind != b.Kind) return false;
            switch (a.Kind)
            {
                case JKind.Null: return true;
                case JKind.Bool: return a.Bool == b.Bool;
                case JKind.Number: return a.Num == b.Num;
                case JKind.String: return a.Text == b.Text;
                case JKind.Array:
                    if (a.Items.Count != b.Items.Count) return false;
                    for (int i = 0; i < a.Items.Count; i++) if (!JsonEqual(a.Items[i], b.Items[i])) return false;
                    return true;
                default:
                    var keys = new HashSet<string>();
                    foreach (var f in a.Fields) keys.Add(f.Key);
                    var other = new HashSet<string>();
                    foreach (var f in b.Fields) other.Add(f.Key);
                    if (!keys.SetEquals(other)) return false;
                    foreach (var f in a.Fields) if (!JsonEqual(f.Value, b[f.Key])) return false;
                    return true;
            }
        }

        /// <summary>Python's len(str): code points, a surrogate pair counting once.</summary>
        public static int CodePoints(string s)
        {
            int n = 0;
            for (int i = 0; i < s.Length; i++) { n++; if (char.IsHighSurrogate(s[i]) && i + 1 < s.Length && char.IsLowSurrogate(s[i + 1])) i++; }
            return n;
        }

        /// <summary>Python's whitespace (str.isspace, as re's \s for text): U+0009 to U+000D, U+001C to U+001F, the
        /// space, U+0085, U+00A0, U+1680, U+2000 to U+200A, U+2028, U+2029, U+202F, U+205F and U+3000.</summary>
        public const string PySpaceClass = "\\t\\n\\x0B\\x0C\\r\\x1C-\\x1F \\x85\\xA0\\u1680\\u2000-\\u200A\\u2028\\u2029\\u202F\\u205F\\u3000";

        /// <summary>A pattern for .NET with Python's search semantics: (?![\s\S]) is the true end (\z), and \S and \s
        /// outside a class use Python's whitespace. .NET's $ already matches before a final newline, as Python's does;
        /// its . already excludes only the newline. Patterns with other constructs keep .NET's meaning.</summary>
        public static Regex PyPattern(string pattern)
        {
            string p = pattern.Replace("(?![\\s\\S])", "\\z");
            var sb = new StringBuilder();
            bool inClass = false;
            for (int i = 0; i < p.Length; i++)
            {
                char c = p[i];
                if (c == '\\' && i + 1 < p.Length)
                {
                    char n = p[i + 1];
                    if (!inClass && n == 'S') { sb.Append("[^").Append(PySpaceClass).Append(']'); i++; continue; }
                    if (!inClass && n == 's') { sb.Append('[').Append(PySpaceClass).Append(']'); i++; continue; }
                    sb.Append(c).Append(n);
                    i++;
                    continue;
                }
                if (c == '[') inClass = true;
                else if (c == ']') inClass = false;
                sb.Append(c);
            }
            return new Regex(sb.ToString(), RegexOptions.CultureInvariant);
        }
    }
}
