using System.Collections.Generic;
using System.Text;
using SecondEyes.Grounding.Prompting;
using SecondEyes.Grounding.Resolution;

namespace SecondEyes.Grounding.Parsing
{
    /// <summary>A2.7 (D115): Python's json.dumps(sort_keys=True, separators=(",", ":"), ensure_ascii=True) of a parsed
    /// JSON value, so the headset's interpretations compare with the PC's as text. Numbers keep their source text.</summary>
    public static class PyJson
    {
        public static string Canonical(JNode n)
        {
            var sb = new StringBuilder();
            Write(n, sb);
            return sb.ToString();
        }

        private static void Write(JNode n, StringBuilder sb)
        {
            switch (n.Kind)
            {
                case JKind.Null: sb.Append("null"); break;
                case JKind.Bool: sb.Append(n.Bool ? "true" : "false"); break;
                case JKind.Number: sb.Append(n.Text); break;
                case JKind.String: sb.Append(ResolverValidation.PyJsonString(n.Text)); break;
                case JKind.Array:
                    sb.Append('[');
                    for (int i = 0; i < n.Items.Count; i++) { if (i > 0) sb.Append(','); Write(n.Items[i], sb); }
                    sb.Append(']');
                    break;
                default:
                    var fields = new List<KeyValuePair<string, JNode>>(n.Fields);
                    fields.Sort((a, b) => string.CompareOrdinal(a.Key, b.Key));
                    sb.Append('{');
                    for (int i = 0; i < fields.Count; i++)
                    {
                        if (i > 0) sb.Append(',');
                        sb.Append(ResolverValidation.PyJsonString(fields[i].Key)).Append(':');
                        Write(fields[i].Value, sb);
                    }
                    sb.Append('}');
                    break;
            }
        }
    }
}
