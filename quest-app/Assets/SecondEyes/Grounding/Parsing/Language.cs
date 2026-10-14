using System;
using System.Collections.Generic;
using SecondEyes.Grounding.Prompting;
using SecondEyes.Grounding.Resolution;

namespace SecondEyes.Grounding.Parsing
{
    // A2.7 delivery 2 (D115): the parser's v1 language (grounding/a24/language.py), free of Unity: the reader that turns a
    // decoded string into its structure (parse_output, independent of the grammar by construction), the canonical string
    // (render), and the structural conversion to one grounding_query interpretation (interpretation, A2.2b's layout).
    // Form: TARGET | RELATION | ANCHOR, or TARGET | between | ANCHOR | ANCHOR; an object is a category label, optionally
    // followed by " [colour]"; RELATION is near, below, on, between, or closest/farthest with rank 1, 2 or 3.
    public sealed class LanguageError : Exception
    {
        public LanguageError(string message) : base(message) { }
    }

    public sealed class LangObject
    {
        public string Category, Colour;   // colour: null when the command gives none
    }

    public sealed class Parsed
    {
        public LangObject Target;
        public string Relation;            // near, below, on, between, closest or farthest
        public int K;                      // 1-3 for closest and farthest, 0 otherwise
        public List<LangObject> Anchors = new List<LangObject>();
    }

    /// <summary>The language specification the headset loads (language.v1.json): its category labels and colours.</summary>
    public sealed class LanguageSpec
    {
        public readonly HashSet<string> Categories = new HashSet<string>(StringComparer.Ordinal);
        public readonly HashSet<string> Colours = new HashSet<string>(StringComparer.Ordinal);
        public int MaxNewTokens;

        public static LanguageSpec From(JNode spec)
        {
            var s = new LanguageSpec();
            foreach (var x in spec["categories"]["labels"].Items) s.Categories.Add(x.Text);
            foreach (var x in spec["colours"].Items) s.Colours.Add(x.Text);
            s.MaxNewTokens = (int)spec["max_new_tokens"].Num;
            return s;
        }
    }

    public static class Language
    {
        public const string Sep = " | ", Open = " [", Close = "]", Between = "between";
        public static readonly string[] Ranked = { "closest", "farthest" };
        public static readonly string[] Single = { "near", "below", "on" };

        private static LangObject Obj(string text, LanguageSpec spec, string where)
        {
            string cat, col;
            if (text.EndsWith(Close, StringComparison.Ordinal))
            {
                int i = text.LastIndexOf(Open, StringComparison.Ordinal);
                if (i <= 0) throw new LanguageError(where + ": a closing bracket without an opening one");
                cat = text.Substring(0, i);
                col = text.Substring(i + Open.Length, text.Length - i - Open.Length - Close.Length);
                if (!spec.Colours.Contains(col)) throw new LanguageError(where + ": '" + col + "' is not one of the ten colours");
            }
            else { cat = text; col = null; }
            if (!spec.Categories.Contains(cat)) throw new LanguageError(where + ": '" + cat + "' is not a category label");
            return new LangObject { Category = cat, Colour = col };
        }

        /// <summary>The decoded string's structure; LanguageError names the first problem.</summary>
        public static Parsed ParseOutput(string text, LanguageSpec spec)
        {
            if (string.IsNullOrEmpty(text)) throw new LanguageError("empty output");
            var fields = text.Split(new[] { Sep }, StringSplitOptions.None);
            if (fields.Length != 3 && fields.Length != 4)
                throw new LanguageError(fields.Length + " fields; the language has 3 (one anchor) or 4 (between)");
            string rel = fields[1];
            if (fields.Length == 4 && rel != Between) throw new LanguageError("four fields need the relation 'between'");
            if (fields.Length == 3 && rel == Between) throw new LanguageError("'between' needs two anchors");
            string name;
            int k = 0;
            if (Array.IndexOf(Single, rel) >= 0 || rel == Between) name = rel;
            else
            {
                var parts = rel.Split(' ');
                if (parts.Length != 2 || Array.IndexOf(Ranked, parts[0]) < 0 || (parts[1] != "1" && parts[1] != "2" && parts[1] != "3"))
                    throw new LanguageError("'" + rel + "' is not a relation of v1");
                name = parts[0];
                k = parts[1][0] - '0';
            }
            var p = new Parsed { Target = Obj(fields[0], spec, "target"), Relation = name, K = k };
            for (int i = 2; i < fields.Length; i++) p.Anchors.Add(Obj(fields[i], spec, "anchor " + (i - 1)));
            return p;
        }

        /// <summary>The canonical string of a structure, the inverse of ParseOutput.</summary>
        public static string Render(Parsed p)
        {
            Func<LangObject, string> obj = o => o.Category + (o.Colour != null ? Open + o.Colour + Close : "");
            string rel = Array.IndexOf(Ranked, p.Relation) >= 0 ? p.Relation + " " + p.K : p.Relation;
            var parts = new List<string> { obj(p.Target), rel };
            foreach (var a in p.Anchors) parts.Add(obj(a));
            return string.Join(Sep, parts);
        }

        private static JNode Node(string id, LangObject o, List<JNode> constraints, JNode rank)
        {
            var colours = new List<JNode>();
            if (o.Colour != null) colours.Add(JNode.Str(o.Colour));
            return JNode.Obj().Add("node_id", JNode.Str(id)).Add("category", JNode.Str(o.Category)).Add("colours_all", JNode.Arr(colours))
                .Add("constraints", JNode.Arr(constraints ?? new List<JNode>())).Add("rank", rank ?? JNode.Null);
        }

        private static JNode Constraint(string relation, params string[] anchors)
        {
            var a = new List<JNode>();
            foreach (var x in anchors) a.Add(JNode.Str(x));
            return JNode.Obj().Add("relation", JNode.Str(relation)).Add("frame", JNode.Null).Add("anchors", JNode.Arr(a));
        }

        /// <summary>The structure as one grounding_query interpretation: A2.2b's node layout, INSPECT, null frames.</summary>
        public static JNode Interpretation(Parsed p)
        {
            var nodes = new List<JNode>();
            if (Array.IndexOf(Ranked, p.Relation) >= 0)
            {
                var rank = JNode.Obj().Add("relation", JNode.Str(p.Relation)).Add("anchor", JNode.Str("anchor_a")).Add("k", Checker.Int(p.K));
                nodes.Add(Node("target", p.Target, null, rank));
                nodes.Add(Node("anchor_a", p.Anchors[0], null, null));
            }
            else if (p.Relation == Between)
            {
                nodes.Add(Node("target", p.Target, new List<JNode> { Constraint(Between, "anchor_a", "anchor_b") }, null));
                nodes.Add(Node("anchor_a", p.Anchors[0], null, null));
                nodes.Add(Node("anchor_b", p.Anchors[1], null, null));
            }
            else
            {
                nodes.Add(Node("target", p.Target, new List<JNode> { Constraint(p.Relation, "anchor_a") }, null));
                nodes.Add(Node("anchor_a", p.Anchors[0], null, null));
            }
            return JNode.Obj().Add("interpretation_id", JNode.Str("main")).Add("kind", JNode.Str("query")).Add("action", JNode.Str("INSPECT"))
                .Add("root", JNode.Str("target")).Add("nodes", JNode.Arr(nodes));
        }
    }
}
