using System;
using System.Collections.Generic;
using System.Text;
using System.Text.RegularExpressions;
using SecondEyes.Grounding.Prompting;

namespace SecondEyes.Grounding.Resolution
{
    // A2.7 delivery 2 (D115): grounding/resolution/validate.py (A2.1e, D71, D72), free of Unity. Every expected input
    // problem becomes an issue (label, path, code), never an exception. The query and resolver-config schemas are
    // checked by Schema; IDs and labels are re-checked as full-string matches (the schemas' $ lets a final newline
    // through); limits and rank k must be JSON integers, a whole-valued float refused (D72). A valid query is
    // normalized for the evaluation: interpretations by ID; each node's colours sorted, its constraints in canonical
    // order; a dependency order (Kahn's algorithm, the smallest ready node first); the between pairs.
    public sealed class NormConstraint
    {
        public string Relation, Frame;            // frame: null for the v1 relations
        public List<string> Anchors;               // sorted for between
    }

    public sealed class NormNode
    {
        public string NodeId, Category;            // category: null when the node names none
        public List<string> ColoursAll;            // sorted
        public List<NormConstraint> Constraints;   // in canonical order
        public string RankRelation, RankAnchor;    // null when the node has no rank
        public int RankK;
    }

    public sealed class NormInterpretation
    {
        public string InterpretationId, Kind, Action, Reason, Root;   // reason: for unavailable and unsupported
        public JNode Raw;                                            // the interpretation as given
        public List<string> Order;
        public Dictionary<string, NormNode> Nodes;
        public List<string[]> Between;
    }

    public static class ResolverValidation
    {
        private static readonly Regex Id = Schema.PyPattern("\\A(?:[a-z0-9][a-z0-9_.-]{0,127})\\z");
        private static readonly Regex Label = Schema.PyPattern("\\A(?:\\S(?:.*\\S)?)\\z");
        private static readonly Regex LocalId = Schema.PyPattern("\\A(?:[a-z][a-z0-9_]{0,63})\\z");
        private static readonly Dictionary<string, Regex> Rules = new Dictionary<string, Regex>
        {
            { "query_id", Id }, { "scene_id", Id }, { "command_id", Id }, { "config_id", Id }, { "vocabulary_id", Id },
            { "category", Label }, { "colours_all", Label }, { "category_labels", Label }, { "colour_labels", Label },
            { "interpretation_id", LocalId }, { "node_id", LocalId }, { "root", LocalId }, { "anchors", LocalId }, { "anchor", LocalId }
        };

        private static Issue I(string label, string path, string code, string message) { return Contract.I(label, path, code, message); }

        private static void SchemaIssues(string label, JNode record, Schema schema, string code, List<Issue> out_)
        {
            foreach (var e in schema.Errors(record)) out_.Add(I(label, e.Path, code, e.Keyword));
        }

        /// <summary>Every ID and label as a full-string match; the key is the nearest field name on the path.</summary>
        /// <summary>The query's schema, as validate.py's check_query: format 2 only for an integer schema_version 2 with the
        /// perceived profile (A4); every other query, a format-2 query on another profile included, is checked against
        /// format 1, exactly as before.</summary>
        public static string QuerySchemaFile(JNode query)
        {
            JNode v = query != null && query.Kind == JKind.Object ? query["schema_version"] : JNode.Null;
            JNode p = query != null && query.Kind == JKind.Object ? query["evidence_profile"] : JNode.Null;
            return v.Kind == JKind.Number && v.IsInteger && v.Num == 2 && p.Kind == JKind.String && p.Text == "perceived"
                ? "grounding-query.v2.json" : "grounding-query.v1.json";
        }

        public static void FullMatches(string label, JNode value, string path, string key, string code, List<Issue> out_)
        {
            switch (value.Kind)
            {
                case JKind.Object:
                    foreach (var f in value.Fields) FullMatches(label, f.Value, path + "." + f.Key, f.Key, code, out_);
                    break;
                case JKind.Array:
                    for (int i = 0; i < value.Items.Count; i++) FullMatches(label, value.Items[i], path + "[" + i + "]", key, code, out_);
                    break;
                case JKind.String:
                    Regex rule;
                    if (key != null && Rules.TryGetValue(key, out rule) && !rule.IsMatch(value.Text))
                        out_.Add(I(label, path, code, "must match the whole " + key + " pattern"));
                    break;
            }
        }

        public static List<Issue> CheckConfig(JNode config, Schema schema)
        {
            var out_ = new List<Issue>();
            SchemaIssues("resolver_config", config, schema, "E_RESOLVER_CONFIG", out_);
            if (out_.Count == 0)
            {
                FullMatches("resolver_config", config, "$", null, "E_RESOLVER_CONFIG", out_);
                foreach (var f in config["limits"].Fields)
                    if (!(f.Value.Kind == JKind.Number && f.Value.IsInteger))
                        out_.Add(I("resolver_config", "$.limits." + f.Key, "E_RESOLVER_CONFIG", "an integer value is required"));
            }
            return out_;
        }

        private static List<string> Deps(JNode node)
        {
            var deps = new List<string>();
            foreach (var c in node["constraints"].Items) foreach (var a in c["anchors"].Items) deps.Add(a.Text);
            if (node["rank"].Kind != JKind.Null) deps.Add(node["rank"]["anchor"].Text);
            return deps;
        }

        /// <summary>The canonical key of a constraint: compact JSON of [relation, frame, anchors], anchors sorted for
        /// between, as Python's json.dumps writes it (ensure_ascii).</summary>
        public static string ConstraintKey(JNode c, out List<string> anchors)
        {
            anchors = new List<string>();
            foreach (var a in c["anchors"].Items) anchors.Add(a.Text);
            if (c["relation"].Text == "between") anchors.Sort(string.CompareOrdinal);
            var sb = new StringBuilder("[");
            sb.Append(PyJsonString(c["relation"].Text)).Append(',');
            sb.Append(c["frame"].Kind == JKind.Null ? "null" : PyJsonString(c["frame"].Text)).Append(",[");
            for (int i = 0; i < anchors.Count; i++) { if (i > 0) sb.Append(','); sb.Append(PyJsonString(anchors[i])); }
            return sb.Append("]]").ToString();
        }

        /// <summary>json.dumps of a string with ensure_ascii: quotes, backslashes and controls escaped, non-ASCII as \\uXXXX.</summary>
        public static string PyJsonString(string s)
        {
            var sb = new StringBuilder("\"");
            foreach (char ch in s)
            {
                switch (ch)
                {
                    case '"': sb.Append("\\\""); break;
                    case '\\': sb.Append("\\\\"); break;
                    case '\n': sb.Append("\\n"); break;
                    case '\r': sb.Append("\\r"); break;
                    case '\t': sb.Append("\\t"); break;
                    case '\b': sb.Append("\\b"); break;
                    case '\f': sb.Append("\\f"); break;
                    default:
                        if (ch < 0x20 || ch > 0x7e) sb.Append("\\u").Append(((int)ch).ToString("x4"));
                        else sb.Append(ch);
                        break;
                }
            }
            return sb.Append('"').ToString();
        }

        private static List<string> CheckGraph(string label, int i, JNode interp, List<Issue> out_)
        {
            string b = "$.interpretations[" + i + "]";
            var nodes = interp["nodes"].Items;
            var ids = new List<string>();
            foreach (var n in nodes) ids.Add(n["node_id"].Text);
            int found = out_.Count;
            for (int j = 0; j < ids.Count; j++)
            {
                if (ids.IndexOf(ids[j]) != j) out_.Add(I(label, b + ".nodes[" + j + "].node_id", "E_QUERY_GRAPH", "node ID is repeated"));
                if (ids[j].StartsWith("obj_", StringComparison.Ordinal)) out_.Add(I(label, b + ".nodes[" + j + "].node_id", "E_QUERY_GRAPH", "node ID is reserved"));
            }
            string root = interp["root"].Text;
            if (!ids.Contains(root)) out_.Add(I(label, b + ".root", "E_QUERY_GRAPH", "root is not a node"));
            for (int j = 0; j < nodes.Count; j++)
            {
                JNode node = nodes[j];
                var refs = new List<KeyValuePair<string, string>>();
                var cons = node["constraints"].Items;
                for (int k = 0; k < cons.Count; k++)
                    for (int m = 0; m < cons[k]["anchors"].Items.Count; m++)
                        refs.Add(new KeyValuePair<string, string>(b + ".nodes[" + j + "].constraints[" + k + "].anchors[" + m + "]", cons[k]["anchors"].Items[m].Text));
                if (node["rank"].Kind != JKind.Null) refs.Add(new KeyValuePair<string, string>(b + ".nodes[" + j + "].rank.anchor", node["rank"]["anchor"].Text));
                foreach (var r in refs)
                {
                    if (!ids.Contains(r.Value)) out_.Add(I(label, r.Key, "E_QUERY_GRAPH", "not a node of this interpretation"));
                    else if (r.Value == node["node_id"].Text) out_.Add(I(label, r.Key, "E_QUERY_GRAPH", "node depends on itself"));
                }
                var seen = new HashSet<string>();
                for (int k = 0; k < cons.Count; k++)
                {
                    List<string> anchors;
                    string key = ConstraintKey(cons[k], out anchors);
                    if (!seen.Add(key)) out_.Add(I(label, b + ".nodes[" + j + "].constraints[" + k + "]", "E_QUERY_SCHEMA", "constraint repeats an earlier one"));
                }
            }
            if (out_.Count > found) return null;
            var byId = new Dictionary<string, JNode>();
            foreach (var n in nodes) byId[n["node_id"].Text] = n;
            var deps = new Dictionary<string, List<string>>();
            foreach (var nid in ids)
            {
                var d = new List<string>(new HashSet<string>(Deps(byId[nid])));
                d.Sort(string.CompareOrdinal);
                deps[nid] = d;
            }
            var waiting = new Dictionary<string, int>();
            var users = new Dictionary<string, List<string>>();
            foreach (var nid in ids) { waiting[nid] = deps[nid].Count; users[nid] = new List<string>(); }
            foreach (var nid in ids) foreach (var d in deps[nid]) users[d].Add(nid);
            var ready = new SortedSet<string>(StringComparer.Ordinal);
            foreach (var nid in ids) if (waiting[nid] == 0) ready.Add(nid);
            var order = new List<string>();
            while (ready.Count > 0)
            {
                string nid = ready.Min;
                ready.Remove(nid);
                order.Add(nid);
                foreach (var u in users[nid]) { waiting[u]--; if (waiting[u] == 0) ready.Add(u); }
            }
            if (order.Count < ids.Count)
            {
                out_.Add(I(label, b + ".nodes", "E_QUERY_GRAPH", "dependency cycle"));
                return null;
            }
            var reach = new HashSet<string> { root };
            var todo = new List<string> { root };
            while (todo.Count > 0)
            {
                string t = todo[todo.Count - 1];
                todo.RemoveAt(todo.Count - 1);
                foreach (var d in deps[t]) if (reach.Add(d)) todo.Add(d);
            }
            foreach (var nid in ids)
                if (!reach.Contains(nid)) { out_.Add(I(label, b + ".nodes", "E_QUERY_GRAPH", "not reachable from the root")); return null; }
            return order;
        }

        /// <summary>Issues for the query; when there are none, its normalized interpretations.</summary>
        public static List<Issue> CheckQuery(JNode query, Schema schema, out List<NormInterpretation> normalized)
        {
            const string label = "query";
            normalized = null;
            var out_ = new List<Issue>();
            SchemaIssues(label, query, schema, "E_QUERY_SCHEMA", out_);
            if (out_.Count > 0) return out_;
            FullMatches(label, query, "$", null, "E_QUERY_SCHEMA", out_);
            var interps = query["interpretations"].Items;
            for (int i = 0; i < interps.Count; i++)
                if (interps[i]["kind"].Text == "query")
                    for (int j = 0; j < interps[i]["nodes"].Items.Count; j++)
                    {
                        JNode rank = interps[i]["nodes"].Items[j]["rank"];
                        if (rank.Kind != JKind.Null && !(rank["k"].Kind == JKind.Number && rank["k"].IsInteger))
                            out_.Add(I(label, "$.interpretations[" + i + "].nodes[" + j + "].rank.k", "E_QUERY_SCHEMA", "an integer value is required"));
                    }
            if (out_.Count > 0) return out_;
            var iids = new List<string>();
            foreach (var it in interps) iids.Add(it["interpretation_id"].Text);
            for (int i = 0; i < iids.Count; i++)
                if (iids.IndexOf(iids[i]) != i) out_.Add(I(label, "$.interpretations[" + i + "].interpretation_id", "E_QUERY_SCHEMA", "interpretation ID is repeated"));
            var orders = new Dictionary<string, List<string>>();
            for (int i = 0; i < interps.Count; i++)
                if (interps[i]["kind"].Text == "query") orders[interps[i]["interpretation_id"].Text] = CheckGraph(label, i, interps[i], out_);
            if (out_.Count > 0) return out_;
            normalized = Normalize(query, orders);
            return out_;
        }

        public static List<NormInterpretation> Normalize(JNode query, Dictionary<string, List<string>> orders)
        {
            var sorted = new List<JNode>(query["interpretations"].Items);
            sorted.Sort((a, b) => string.CompareOrdinal(a["interpretation_id"].Text, b["interpretation_id"].Text));
            var result = new List<NormInterpretation>();
            foreach (var it in sorted)
            {
                var ni = new NormInterpretation { InterpretationId = it["interpretation_id"].Text, Kind = it["kind"].Text, Raw = it,
                                                  Action = Contract.Get(it, "action").Kind == JKind.String ? it["action"].Text : null,
                                                  Reason = Contract.Get(it, "reason").Kind == JKind.String ? it["reason"].Text : null };
                if (ni.Kind != "query") { result.Add(ni); continue; }
                ni.Root = it["root"].Text;
                ni.Order = orders[ni.InterpretationId];
                ni.Nodes = new Dictionary<string, NormNode>();
                var pairs = new SortedDictionary<string, string[]>(StringComparer.Ordinal);
                foreach (var n in it["nodes"].Items)
                {
                    var nn = new NormNode { NodeId = n["node_id"].Text, Category = n["category"].Kind == JKind.Null ? null : n["category"].Text,
                                            ColoursAll = new List<string>(), Constraints = new List<NormConstraint>() };
                    foreach (var c in n["colours_all"].Items) nn.ColoursAll.Add(c.Text);
                    nn.ColoursAll.Sort(string.CompareOrdinal);
                    var keyed = new List<KeyValuePair<string, NormConstraint>>();
                    foreach (var c in n["constraints"].Items)
                    {
                        List<string> anchors;
                        string key = ConstraintKey(c, out anchors);
                        keyed.Add(new KeyValuePair<string, NormConstraint>(key, new NormConstraint
                        {
                            Relation = c["relation"].Text, Frame = c["frame"].Kind == JKind.Null ? null : c["frame"].Text, Anchors = anchors
                        }));
                    }
                    keyed.Sort((x, y) => string.CompareOrdinal(x.Key, y.Key));
                    foreach (var kv in keyed)
                    {
                        nn.Constraints.Add(kv.Value);
                        if (kv.Value.Relation == "between") pairs[string.Join("\u0000", kv.Value.Anchors)] = kv.Value.Anchors.ToArray();
                    }
                    if (n["rank"].Kind != JKind.Null)
                    {
                        nn.RankRelation = n["rank"]["relation"].Text;
                        nn.RankAnchor = n["rank"]["anchor"].Text;
                        nn.RankK = (int)n["rank"]["k"].Num;
                    }
                    ni.Nodes[nn.NodeId] = nn;
                }
                ni.Between = new List<string[]>(pairs.Values);
                result.Add(ni);
            }
            return result;
        }

        /// <summary>(interpretations, nodes, constraints) as the static limits count them.</summary>
        public static int[] RequestSize(JNode query)
        {
            int interps = query["interpretations"].Items.Count, nodes = 0, cons = 0;
            foreach (var it in query["interpretations"].Items)
                if (it["kind"].Text == "query")
                    foreach (var n in it["nodes"].Items) { nodes++; cons += n["constraints"].Items.Count; }
            return new[] { interps, nodes, cons };
        }

        /// <summary>(errors, warnings) for the scene, command and category maps, through the contract's layers.</summary>
        public static List<Issue> CheckRecords(JNode scene, JNode command, List<JNode> maps, Contract contract, List<Issue> warnings)
        {
            var errors = new List<Issue>();
            foreach (var t in new[] { new object[] { "scene", scene, "scene" }, new object[] { "command", command, "command_context" } })
            {
                var rec = (JNode)t[1];
                var rt = Contract.Get(rec, "record_type");
                if (rec.Kind == JKind.Object && rt.Kind != JKind.Null && !(rt.Kind == JKind.String && rt.Text == (string)t[2]))
                    errors.Add(I((string)t[0], "$.record_type", "E_RECORD_ROLE", "the argument holds another record type"));
            }
            for (int i = 0; i < maps.Count; i++)
            {
                var rt = Contract.Get(maps[i], "record_type");
                if (rt.Kind != JKind.Null && !(rt.Kind == JKind.String && rt.Text == "category_map"))
                    errors.Add(I("category_map[" + i + "]", "$.record_type", "E_RECORD_ROLE", "holds another record type"));
            }
            var named = new List<KeyValuePair<string, JNode>> { new KeyValuePair<string, JNode>("scene", scene), new KeyValuePair<string, JNode>("command", command) };
            for (int i = 0; i < maps.Count; i++) named.Add(new KeyValuePair<string, JNode>("category_map[" + i + "]", maps[i]));
            foreach (var x in contract.Validate(named)) (x.IsError ? errors : warnings).Add(x);
            if (errors.Count > 0) return errors;
            var supplied = new List<string>();
            foreach (var m in maps) supplied.Add(m["map_id"].Text);
            if (maps.Count > 0 && !supplied.Contains(scene["category_map"].Text))
                errors.Add(I("scene", "$.category_map", "E_CATEGORY_MAP_MISMATCH", "the scene's map was not supplied"));
            if (command["scene_id"].Text != scene["scene_id"].Text)
                errors.Add(I("command", "$.scene_id", "E_QUERY_CONTEXT", "the command is for another scene"));
            return errors;
        }

        public static List<Issue> CheckContext(JNode query, JNode scene, JNode command)
        {
            var out_ = new List<Issue>();
            foreach (var kv in new[] { new KeyValuePair<string, JNode>("scene_id", scene["scene_id"]), new KeyValuePair<string, JNode>("scene_revision", scene["scene_revision"]),
                                       new KeyValuePair<string, JNode>("evidence_profile", scene["evidence_profile"]), new KeyValuePair<string, JNode>("command_id", command["command_id"]) })
                if (!Schema.JsonEqual(query[kv.Key], kv.Value)) out_.Add(I("query", "$." + kv.Key, "E_QUERY_CONTEXT", "does not match the supplied record"));
            return out_;
        }
    }
}
