using System;
using System.Collections.Generic;
using System.Globalization;
using System.Numerics;
using System.Text;

namespace SecondEyes.Grounding.Prompting
{
    // A2.5 delivery 2 (D98, D99(4), D99(6)): the headset's coordinates_v2 document and prompt, byte for byte the PC's.
    // The references are grounding/serialization/codec.py (canonical spelling), grounding/serialization/serializer.py
    // (the coordinates_v2 lines) and grounding/inference/iref_vla/choices.py (the choices line and the prompt). Free of
    // Unity, so it is tested on a PC against those Python functions.

    public enum JKind { Null, Bool, Number, String, Array, Object }

    /// <summary>A JSON value that keeps object fields in their order and a number's source text.</summary>
    public sealed class JNode
    {
        public JKind Kind;
        public bool Bool;
        public string Text;        // a string's value, or a number's source text
        public double Num;         // a number's value
        public bool IsInteger;     // the number's text has neither fraction nor exponent
        public List<JNode> Items;
        public List<KeyValuePair<string, JNode>> Fields;

        public static readonly JNode Null = new JNode { Kind = JKind.Null };

        public JNode this[string key]
        {
            get
            {
                if (Kind == JKind.Object)
                    foreach (var f in Fields) if (f.Key == key) return f.Value;
                throw new KeyNotFoundException("no field " + key);
            }
        }

        public bool Has(string key)
        {
            if (Kind != JKind.Object) return false;
            foreach (var f in Fields) if (f.Key == key) return true;
            return false;
        }

        public static JNode Str(string s) { return new JNode { Kind = JKind.String, Text = s }; }
        public static JNode Arr(List<JNode> items) { return new JNode { Kind = JKind.Array, Items = items }; }
        public static JNode Obj() { return new JNode { Kind = JKind.Object, Fields = new List<KeyValuePair<string, JNode>>() }; }
        public JNode Add(string key, JNode value) { Fields.Add(new KeyValuePair<string, JNode>(key, value)); return this; }
    }

    /// <summary>A strict JSON reader (RFC 8259) for the snapshot and command records.</summary>
    public static class JParse
    {
        public static JNode Parse(string text)
        {
            int i = 0;
            JNode v = Value(text, ref i);
            Ws(text, ref i);
            if (i != text.Length) throw new FormatException("text after the JSON value at " + i);
            return v;
        }

        private static void Ws(string s, ref int i)
        {
            while (i < s.Length && (s[i] == ' ' || s[i] == '\t' || s[i] == '\n' || s[i] == '\r')) i++;
        }

        private static JNode Value(string s, ref int i)
        {
            Ws(s, ref i);
            if (i >= s.Length) throw new FormatException("unexpected end");
            char c = s[i];
            if (c == '{')
            {
                i++;
                var o = JNode.Obj();
                Ws(s, ref i);
                if (i < s.Length && s[i] == '}') { i++; return o; }
                while (true)
                {
                    Ws(s, ref i);
                    if (i >= s.Length || s[i] != '"') throw new FormatException("expected a field name at " + i);
                    string key = Str(s, ref i);
                    Ws(s, ref i);
                    if (i >= s.Length || s[i] != ':') throw new FormatException("expected ':' at " + i);
                    i++;
                    o.Add(key, Value(s, ref i));
                    Ws(s, ref i);
                    if (i < s.Length && s[i] == ',') { i++; continue; }
                    if (i < s.Length && s[i] == '}') { i++; return o; }
                    throw new FormatException("expected ',' or '}' at " + i);
                }
            }
            if (c == '[')
            {
                i++;
                var items = new List<JNode>();
                Ws(s, ref i);
                if (i < s.Length && s[i] == ']') { i++; return JNode.Arr(items); }
                while (true)
                {
                    items.Add(Value(s, ref i));
                    Ws(s, ref i);
                    if (i < s.Length && s[i] == ',') { i++; continue; }
                    if (i < s.Length && s[i] == ']') { i++; return JNode.Arr(items); }
                    throw new FormatException("expected ',' or ']' at " + i);
                }
            }
            if (c == '"') return JNode.Str(Str(s, ref i));
            if (string.CompareOrdinal(s, i, "true", 0, 4) == 0) { i += 4; return new JNode { Kind = JKind.Bool, Bool = true }; }
            if (string.CompareOrdinal(s, i, "false", 0, 5) == 0) { i += 5; return new JNode { Kind = JKind.Bool, Bool = false }; }
            if (string.CompareOrdinal(s, i, "null", 0, 4) == 0) { i += 4; return JNode.Null; }
            return Number(s, ref i);
        }

        private static JNode Number(string s, ref int i)
        {
            int start = i;
            if (i < s.Length && s[i] == '-') i++;
            if (i < s.Length && s[i] == '0') i++;
            else if (i < s.Length && s[i] >= '1' && s[i] <= '9') while (i < s.Length && char.IsDigit(s[i]) && s[i] < 128) i++;
            else throw new FormatException("bad number at " + start);
            bool integer = true;
            if (i < s.Length && s[i] == '.')
            {
                integer = false;
                i++;
                if (i >= s.Length || s[i] < '0' || s[i] > '9') throw new FormatException("bad fraction at " + start);
                while (i < s.Length && s[i] >= '0' && s[i] <= '9') i++;
            }
            if (i < s.Length && (s[i] == 'e' || s[i] == 'E'))
            {
                integer = false;
                i++;
                if (i < s.Length && (s[i] == '+' || s[i] == '-')) i++;
                if (i >= s.Length || s[i] < '0' || s[i] > '9') throw new FormatException("bad exponent at " + start);
                while (i < s.Length && s[i] >= '0' && s[i] <= '9') i++;
            }
            string text = s.Substring(start, i - start);
            return new JNode { Kind = JKind.Number, Text = text, IsInteger = integer,
                               Num = double.Parse(text, NumberStyles.Float, CultureInfo.InvariantCulture) };
        }

        private static string Str(string s, ref int i)
        {
            i++;   // the opening quote
            var sb = new StringBuilder();
            while (true)
            {
                if (i >= s.Length) throw new FormatException("unterminated string");
                char c = s[i++];
                if (c == '"') return sb.ToString();
                if (c < 0x20) throw new FormatException("a control character in a string");
                if (c != '\\') { sb.Append(c); continue; }
                if (i >= s.Length) throw new FormatException("unterminated escape");
                char e = s[i++];
                switch (e)
                {
                    case '"': sb.Append('"'); break;
                    case '\\': sb.Append('\\'); break;
                    case '/': sb.Append('/'); break;
                    case 'b': sb.Append('\b'); break;
                    case 'f': sb.Append('\f'); break;
                    case 'n': sb.Append('\n'); break;
                    case 'r': sb.Append('\r'); break;
                    case 't': sb.Append('\t'); break;
                    case 'u':
                        if (i + 4 > s.Length) throw new FormatException("short \\u escape");
                        sb.Append((char)int.Parse(s.Substring(i, 4), NumberStyles.HexNumber, CultureInfo.InvariantCulture));
                        i += 4;
                        break;
                    default: throw new FormatException("bad escape \\" + e);
                }
            }
        }
    }

    /// <summary>codec.enc and codec.number: compact JSON in the given field order, text kept as text (Python's
    /// ensure_ascii=False), integers in decimal, integral floats as the equal integer, other floats as Python's repr.</summary>
    public static class Canon
    {
        public static string Enc(JNode v)
        {
            var sb = new StringBuilder();
            Write(sb, v);
            return sb.ToString();
        }

        private static void Write(StringBuilder sb, JNode v)
        {
            switch (v.Kind)
            {
                case JKind.Null: sb.Append("null"); break;
                case JKind.Bool: sb.Append(v.Bool ? "true" : "false"); break;
                case JKind.Number: sb.Append(NumberText(v)); break;
                case JKind.String: Quote(sb, v.Text); break;
                case JKind.Array:
                    sb.Append('[');
                    for (int k = 0; k < v.Items.Count; k++) { if (k > 0) sb.Append(','); Write(sb, v.Items[k]); }
                    sb.Append(']');
                    break;
                default:
                    sb.Append('{');
                    for (int k = 0; k < v.Fields.Count; k++)
                    {
                        if (k > 0) sb.Append(',');
                        Quote(sb, v.Fields[k].Key);
                        sb.Append(':');
                        Write(sb, v.Fields[k].Value);
                    }
                    sb.Append('}');
                    break;
            }
        }

        /// <summary>json.dumps(text, ensure_ascii=False).</summary>
        public static void Quote(StringBuilder sb, string s)
        {
            sb.Append('"');
            foreach (char c in s)
            {
                switch (c)
                {
                    case '"': sb.Append("\\\""); break;
                    case '\\': sb.Append("\\\\"); break;
                    case '\n': sb.Append("\\n"); break;
                    case '\r': sb.Append("\\r"); break;
                    case '\t': sb.Append("\\t"); break;
                    case '\b': sb.Append("\\b"); break;
                    case '\f': sb.Append("\\f"); break;
                    default:
                        if (c < 0x20) sb.Append("\\u").Append(((int)c).ToString("x4", CultureInfo.InvariantCulture));
                        else sb.Append(c);
                        break;
                }
            }
            sb.Append('"');
        }

        private static string NumberText(JNode v)
        {
            if (v.IsInteger)
            {
                BigInteger n = BigInteger.Parse(v.Text, NumberStyles.AllowLeadingSign, CultureInfo.InvariantCulture);
                return n.ToString(CultureInfo.InvariantCulture);   // "-0" becomes "0", as Python's int does
            }
            return Number(v.Num);
        }

        /// <summary>codec.number for a float: both zeros as 0, integral values as the equal integer, else repr.</summary>
        public static string Number(double x)
        {
            if (double.IsNaN(x) || double.IsInfinity(x)) throw new ArgumentException("not a finite number");
            if (x == 0) return "0";
            if (Math.Floor(x) == x) return new BigInteger(x).ToString(CultureInfo.InvariantCulture);
            return Repr(x);
        }

        /// <summary>Python's repr of a finite float: the shortest decimal that reads back as x (the nearest such one,
        /// ties to an even last digit), in fixed notation when -4 &lt; decimal exponent &lt;= 16, else as d.ddde±XX.
        /// Computed with exact integer arithmetic, so it does not depend on the runtime's number formatting.</summary>
        public static string Repr(double x)
        {
            if (x == 0) return BitConverter.DoubleToInt64Bits(x) < 0 ? "-0.0" : "0.0";
            bool neg = x < 0;
            long bits = BitConverter.DoubleToInt64Bits(Math.Abs(x));
            int be = (int)((bits >> 52) & 0x7FF);
            long frac = bits & 0xFFFFFFFFFFFFFL;
            BigInteger m = be == 0 ? new BigInteger(frac) : new BigInteger(frac | (1L << 52));
            int e = be == 0 ? -1074 : be - 1075;          // |x| = m * 2^e
            int s2 = e - 2;                                // work in units of 2^(e-2): |x| = 4m units
            BigInteger v4 = m * 4;
            BigInteger lo4 = v4 - (frac == 0 && be > 1 ? 1 : 2);   // the gap below halves at a power of two
            BigInteger hi4 = v4 + 2;
            bool inclusive = m.IsEven;                     // round-half-even reads the midpoints back as x when m is even
            // the decimal exponent k with 10^k <= |x| < 10^(k+1)
            int k = (int)Math.Floor(Math.Log10(Math.Abs(x)));
            while (Cmp(BigInteger.One, k, v4, s2) > 0) k--;
            while (Cmp(BigInteger.One, k + 1, v4, s2) <= 0) k++;
            for (int p = 1; p <= 17; p++)
            {
                int q = k - p + 1;
                BigInteger num = v4 * (s2 > 0 ? BigInteger.Pow(2, s2) : 1) * (q < 0 ? BigInteger.Pow(10, -q) : 1);
                BigInteger den = (s2 < 0 ? BigInteger.Pow(2, -s2) : 1) * (q > 0 ? BigInteger.Pow(10, q) : 1);
                BigInteger dl = BigInteger.Divide(num, den);
                BigInteger dh = BigInteger.Remainder(num, den).IsZero ? dl : dl + 1;
                bool okL = dl > 0 && Inside(dl, q, lo4, hi4, s2, inclusive);
                bool okH = Inside(dh, q, lo4, hi4, s2, inclusive);
                if (!okL && !okH) continue;
                BigInteger d;
                if (okL && okH && dl != dh)
                {
                    int c = Cmp(dl + dh, q, v4 * 2, s2);   // (dl + dh) 10^q against 2|x|
                    d = c > 0 ? dl : c < 0 ? dh : (dl.IsEven ? dl : dh);
                }
                else d = okL ? dl : dh;
                return (neg ? "-" : "") + Spell(d, q);
            }
            throw new InvalidOperationException("no decimal of at most 17 digits reads back as " + x);
        }

        private static bool Inside(BigInteger d, int q, BigInteger lo4, BigInteger hi4, int s2, bool inclusive)
        {
            int a = Cmp(d, q, lo4, s2), b = Cmp(d, q, hi4, s2);
            return inclusive ? a >= 0 && b <= 0 : a > 0 && b < 0;
        }

        /// <summary>The sign of d * 10^q - n * 2^s2, exactly.</summary>
        private static int Cmp(BigInteger d, int q, BigInteger n, int s2)
        {
            BigInteger l = d, r = n;
            if (q >= 0) l *= BigInteger.Pow(10, q); else r *= BigInteger.Pow(10, -q);
            if (s2 >= 0) r *= BigInteger.Pow(2, s2); else l *= BigInteger.Pow(2, -s2);
            return l.CompareTo(r);
        }

        private static string Spell(BigInteger d, int q)
        {
            string digits = d.ToString(CultureInfo.InvariantCulture);
            int trim = digits.Length;
            while (trim > 1 && digits[trim - 1] == '0') trim--;
            q += digits.Length - trim;
            digits = digits.Substring(0, trim);
            int decpt = digits.Length + q;                 // |x| = 0.digits * 10^decpt
            if (decpt <= -4 || decpt > 16)
            {
                int exp = decpt - 1;
                string mant = digits.Length > 1 ? digits.Substring(0, 1) + "." + digits.Substring(1) : digits;
                return mant + "e" + (exp < 0 ? "-" : "+") + Math.Abs(exp).ToString("00", CultureInfo.InvariantCulture);
            }
            if (decpt <= 0) return "0." + new string('0', -decpt) + digits;
            if (decpt < digits.Length) return digits.Substring(0, decpt) + "." + digits.Substring(decpt);
            return digits + new string('0', decpt - digits.Length) + ".0";
        }
    }

    /// <summary>serializer.py's coordinates_v2 lines that depend on the scene and command. The header and semantics
    /// lines depend only on the format and the relation and direction configs, so they come in as pinned text.</summary>
    public static class Coordinates
    {
        private static readonly string[] EvidenceColumns = { "category", "colours", "center_m", "size_m", "rotation_xyzw", "semantic_front" };
        private static readonly string[] PoseConditional = { "position_m", "heading_xy" };

        private static bool IsKnown(JNode w) { return w["state"].Text == "known"; }
        private static JNode Known(JNode w) { return IsKnown(w) ? w["value"] : JNode.Null; }

        private static bool Conditional(JNode w)
        {
            if (!IsKnown(w)) return false;
            JNode ev = w["evidence"];
            return ev["kind"].Text == "assumed" || (ev["assumptions"].Items != null && ev["assumptions"].Items.Count > 0);
        }

        public static List<JNode> SortedObjects(JNode scene)
        {
            var objects = new List<JNode>(scene["objects"].Items);
            objects.Sort((a, b) => string.CompareOrdinal(a["object_id"].Text, b["object_id"].Text));
            return objects;
        }

        public static JNode ObjectRow(JNode o)
        {
            JNode gm = o["geometry"];
            var wrappers = new[] { o["category"], o["attributes"]["colours"], gm["center_m"], gm["size_m"], gm["rotation_xyzw"], o["semantic_front"] };
            JNode category = Known(o["category"]);
            JNode colours = Known(o["attributes"]["colours"]);
            JNode sortedColours = JNode.Null;
            if (colours.Kind == JKind.Array)
            {
                var list = new List<JNode>(colours.Items);
                list.Sort((a, b) => string.CompareOrdinal(a.Text, b.Text));
                sortedColours = JNode.Arr(list);
            }
            JNode rotation = gm["rotation_xyzw"];
            var conditional = new List<JNode>();
            for (int k = 0; k < EvidenceColumns.Length; k++)
                if (Conditional(wrappers[k])) conditional.Add(JNode.Str(EvidenceColumns[k]));
            return JNode.Arr(new List<JNode> {
                o["object_id"], category.Kind == JKind.Null ? JNode.Null : category["model"], sortedColours,
                Known(gm["center_m"]), Known(gm["size_m"]), Known(rotation),
                IsKnown(rotation) ? rotation["support"] : JNode.Null, Known(o["semantic_front"]), JNode.Arr(conditional) });
        }

        public static string ObjectsLine(JNode scene)
        {
            var rows = new List<JNode>();
            foreach (JNode o in SortedObjects(scene)) rows.Add(ObjectRow(o));
            return Canon.Enc(JNode.Obj().Add("objects", JNode.Arr(rows))) + "\n";
        }

        public static string PoseLine(JNode command)
        {
            JNode pose = command["user_pose"], heading = pose["heading_xy"];
            var conditional = new List<JNode>();
            foreach (string n in PoseConditional) if (Conditional(pose[n])) conditional.Add(JNode.Str(n));
            JNode source = IsKnown(heading) && heading.Has("heading_source") ? heading["heading_source"] : JNode.Null;
            JNode record = JNode.Obj().Add("pose_kind", pose["pose_kind"]).Add("position_m", Known(pose["position_m"]))
                .Add("heading_xy", Known(heading)).Add("heading_source", source).Add("conditional_fields", JNode.Arr(conditional));
            return Canon.Enc(JNode.Obj().Add("pose", record)) + "\n";
        }

        public static string CommandLine(JNode command)
        {
            return Canon.Enc(JNode.Obj().Add("command", command["text"])) + "\n";
        }

        /// <summary>The whole coordinates_v2 document: the two pinned lines, then objects, pose and command.</summary>
        public static string Document(string headerLine, string semanticsLine, JNode scene, JNode command)
        {
            return headerLine + semanticsLine + ObjectsLine(scene) + PoseLine(command) + CommandLine(command);
        }
    }

    /// <summary>choices.py: the choices line and the prompt around the document.</summary>
    public static class Choices
    {
        /// <summary>The proposed stable policy (D99(6), awaiting ChatGPT's confirmation): letters follow the serialized
        /// object order (object IDs in ordinal order), K is ASK last. It depends on the inventory only.</summary>
        public static List<KeyValuePair<string, string>> SerializedOrder(JNode scene, string[] objectCodes, string askCode, string askTarget)
        {
            List<JNode> objects = Coordinates.SortedObjects(scene);
            if (objects.Count > objectCodes.Length)
                throw new ArgumentException(objects.Count + " objects; the interface offers codes for at most " + objectCodes.Length);
            var mapping = new List<KeyValuePair<string, string>>();
            for (int k = 0; k < objects.Count; k++) mapping.Add(new KeyValuePair<string, string>(objectCodes[k], objects[k]["object_id"].Text));
            mapping.Add(new KeyValuePair<string, string>(askCode, askTarget));
            return mapping;
        }

        public static string Line(List<KeyValuePair<string, string>> mapping)
        {
            var pairs = new List<JNode>();
            foreach (var m in mapping) pairs.Add(JNode.Arr(new List<JNode> { JNode.Str(m.Key), JNode.Str(m.Value) }));
            return Canon.Enc(JNode.Obj().Add("choices", JNode.Arr(pairs))) + "\n";
        }

        public static string Prompt(string beforeSystem, string systemMessage, string between, string afterUser,
                                    List<KeyValuePair<string, string>> mapping, string document)
        {
            return beforeSystem + systemMessage + between + Line(mapping) + document + afterUser;
        }
    }
}
