using System;
using System.Collections.Generic;
using SecondEyes.Grounding.Prompting;

namespace SecondEyes.Grounding.Resolution
{
    // A2.7 delivery 2 (D115): grounding/resolution/resolve.py (A2.1e, D71), free of Unity: validation, then the
    // evaluation, then the record and its invariants. Expected input problems give processing_status invalid_input; a
    // work limit gives budget_exceeded with no result; a broken invariant, or a constraint with a frame (directions are
    // not ported: parser v1 has no frames, D115), is a technical failure and throws ResolverFault. Left out: the trace,
    // timings, identity hashes and the assumption records (provenance, kept offline). Kept: the conditional flag, the
    // reason codes, every candidate list and every work counter.
    public sealed class ResolverFault : Exception
    {
        public ResolverFault(string message) : base(message) { }
    }

    public sealed class OverBudget : Exception
    {
        public readonly string Name;
        public readonly long Limit, Used, NextRequired;
        public OverBudget(string name, long limit, long used, long next) : base(name) { Name = name; Limit = limit; Used = used; NextRequired = next; }
    }

    public sealed class ResolverResult
    {
        public string Status, Action, TargetId, ReasonCode, Coverage;
        public List<string> ReasonCodes, TargetIds, UncertainMatchIds, RankDefiniteEligibleIds, RankPossibleEligibleIds;
        public bool Conditional;
    }

    public sealed class RunRecord
    {
        public string ProcessingStatus;   // completed, invalid_input or budget_exceeded
        public ResolverResult Result;     // null unless completed
        public List<Issue> Issues = new List<Issue>(), Warnings = new List<Issue>();
        public OverBudget Budget;
        public Dictionary<string, long> Work = new Dictionary<string, long>();
    }

    /// <summary>What every resolution needs, loaded once and checked at load: the schemas, the contract and the relation
    /// configuration. Problems with the relation or direction configuration are kept as issues, which every resolution
    /// reports as the Python's load_configs would (E_CONFIG).</summary>
    public sealed class ResolverSetup
    {
        public Func<string, Schema> Schemas;
        public Contract Contract;
        public RelationConfig RelationConfig;
        public List<Issue> ConfigFileIssues = new List<Issue>();

        public static ResolverSetup Load(Func<string, Schema> schemas, JNode relationConfig, JNode directionConfig)
        {
            var s = new ResolverSetup { Schemas = schemas, Contract = new Contract(schemas) };
            var rel = schemas("relation-config.v1.json").Errors(relationConfig);
            if (rel.Count > 0) s.ConfigFileIssues.Add(Contract.I("relation_config", "relations.v1.json", "E_CONFIG", "the relation configuration fails its schema at " + rel[0]));
            else s.RelationConfig = RelationConfig.From(relationConfig);
            var dir = schemas("direction-config.v1.json").Errors(directionConfig);
            if (dir.Count > 0) s.ConfigFileIssues.Add(Contract.I("direction_config", "directions.v1.json", "E_CONFIG", "the direction configuration fails its schema at " + dir[0]));
            return s;
        }
    }

    public static class Resolver
    {
        public static readonly string[] Statuses = { "resolved", "ambiguous", "no_match", "insufficient_information", "unsupported" };
        public static readonly Dictionary<string, string> Primary = new Dictionary<string, string>
        {
            { "resolved", "unique_target_consensus" }, { "ambiguous", "nonunique_target_or_interpretation" },
            { "no_match", "no_matching_reference" }, { "insufficient_information", "required_information_unavailable" },
            { "unsupported", "unsupported_interpretation" }
        };
        public static readonly Dictionary<string, string> RankStatus = new Dictionary<string, string>
            { { "resolved", "resolved" }, { "ambiguous", "ambiguous" }, { "no_match", "no_match" }, { "insufficient", "insufficient_information" } };
        public static readonly string[] Work =
        {
            "interpretations", "nodes", "constraints", "candidate_checks", "binding_attempts", "pruned_bindings", "predicate_calls",
            "predicate_cache_hits", "rank_calls", "rank_cache_hits", "rank_subsets", "max_rank_preflight_subsets", "terminal_outcomes"
        };
        public static readonly string[] StaticLimits = { "max_interpretations", "max_nodes", "max_constraints" };
        public static readonly Dictionary<string, string> Dynamic = new Dictionary<string, string>
        {
            { "candidate_checks", "max_candidate_checks" }, { "binding_attempts", "max_binding_attempts" },
            { "predicate_calls", "max_predicate_calls" }, { "rank_subsets", "max_rank_subset_evaluations" }
        };

        /// <summary>Resolve a grounding query against one scene and command.</summary>
        public static RunRecord Resolve(JNode scene, JNode command, JNode query, JNode resolverConfig, List<JNode> maps, ResolverSetup setup)
        {
            var record = new RunRecord();
            foreach (var w in Work) record.Work[w] = 0;
            List<NormInterpretation> normalized;
            var configIssues = ResolverValidation.CheckConfig(resolverConfig, setup.Schemas("resolver-config.v1.json"));
            var queryIssues = ResolverValidation.CheckQuery(query, setup.Schemas("grounding-query.v1.json"), out normalized);
            var recordIssues = ResolverValidation.CheckRecords(scene, command, maps, setup.Contract, record.Warnings);
            var issues = new List<Issue>();
            issues.AddRange(configIssues); issues.AddRange(queryIssues); issues.AddRange(recordIssues);
            if (issues.Count == 0) issues = ResolverValidation.CheckContext(query, scene, command);
            if (issues.Count == 0) issues = new List<Issue>(setup.ConfigFileIssues);
            if (issues.Count > 0)
            {
                record.ProcessingStatus = "invalid_input";
                record.Issues = issues;
                return record;
            }
            var limits = new Dictionary<string, long>();
            foreach (var f in resolverConfig["limits"].Fields) limits[f.Key] = (long)f.Value.Num;
            var run = new Evaluation(scene, command, resolverConfig, limits, setup.RelationConfig, record.Work);
            var sizes = ResolverValidation.RequestSize(query);
            record.Work["interpretations"] = sizes[0]; record.Work["nodes"] = sizes[1]; record.Work["constraints"] = sizes[2];
            try
            {
                for (int i = 0; i < 3; i++)
                    if (sizes[i] > limits[StaticLimits[i]]) throw new OverBudget(StaticLimits[i], limits[StaticLimits[i]], 0, sizes[i]);
                run.Run(normalized);
                record.Result = run.Result(query);
                record.ProcessingStatus = "completed";
            }
            catch (OverBudget e)
            {
                record.ProcessingStatus = "budget_exceeded";
                record.Budget = e;
                record.Result = null;
            }
            var ids = new HashSet<string>();
            foreach (var o in scene["objects"].Items) ids.Add(o["object_id"].Text);
            CheckOutput(record, ids, limits);
            return record;
        }

        /// <summary>The invariants the result must keep; a violation is a resolver fault, never an outcome.</summary>
        public static void CheckOutput(RunRecord record, HashSet<string> sceneIds, Dictionary<string, long> limits)
        {
            var problems = new List<string>();
            var r = record.Result;
            if (r != null)
            {
                foreach (var kv in new[] { new KeyValuePair<string, List<string>>("target_ids", r.TargetIds), new KeyValuePair<string, List<string>>("uncertain_match_ids", r.UncertainMatchIds),
                                           new KeyValuePair<string, List<string>>("rank_definite_eligible_ids", r.RankDefiniteEligibleIds), new KeyValuePair<string, List<string>>("rank_possible_eligible_ids", r.RankPossibleEligibleIds) })
                {
                    var sorted = Py.Sorted(new HashSet<string>(kv.Value));
                    if (string.Join("\u0000", sorted) != string.Join("\u0000", kv.Value)) problems.Add("candidates." + kv.Key);
                    foreach (var x in kv.Value) if (!sceneIds.Contains(x)) { problems.Add("candidates." + kv.Key); break; }
                }
                if (r.ReasonCode != Primary[r.Status] || string.Join("\u0000", Py.Sorted(r.ReasonCodes)) != string.Join("\u0000", r.ReasonCodes)) problems.Add("reason codes");
                if ((r.Status == "resolved") != (r.TargetId != null)) problems.Add("target_id against status");
                if (r.Status == "resolved" && (r.Action == null || !sceneIds.Contains(r.TargetId))) problems.Add("resolved target or action");
            }
            if (record.ProcessingStatus != "invalid_input")
                foreach (var kv in Dynamic) if (record.Work[kv.Key] > limits[kv.Value]) problems.Add("work." + kv.Key + " above " + kv.Value);
            if (problems.Count > 0) throw new ResolverFault("the resolver broke its result invariants: " + string.Join(", ", problems));
        }
    }

    internal sealed class Outcome
    {
        public string Action, Status, Target;
        public HashSet<string> Codes;
        public bool Incomplete;
    }

    internal sealed class Evaluation
    {
        private readonly Dictionary<string, JNode> byId = new Dictionary<string, JNode>();
        private readonly List<string> objects = new List<string>();
        private readonly Dictionary<string, long> limits;
        private readonly HashSet<string> categoryLabels = new HashSet<string>(), colourLabels = new HashSet<string>();
        private readonly Dictionary<string, long> work;
        private readonly JNode scene;
        private readonly RelationConfig relationConfig;
        private Relations relations;
        private readonly Dictionary<string, int> counts = new Dictionary<string, int>();
        private readonly Dictionary<string, RelationResult> memo = new Dictionary<string, RelationResult>();
        private readonly Dictionary<string, RankResult> rankMemo = new Dictionary<string, RankResult>();
        private bool conditional;
        private readonly HashSet<string> targets = new HashSet<string>(), uncertain = new HashSet<string>(),
                                         rankDefinite = new HashSet<string>(), rankPossible = new HashSet<string>();
        private readonly List<Outcome> outcomes = new List<Outcome>();

        public Evaluation(JNode scene, JNode command, JNode config, Dictionary<string, long> limits, RelationConfig rc, Dictionary<string, long> work)
        {
            this.scene = scene;
            foreach (var o in scene["objects"].Items) byId[o["object_id"].Text] = o;
            objects.AddRange(byId.Keys);
            objects.Sort(string.CompareOrdinal);
            this.limits = limits;
            foreach (var x in config["category_labels"].Items) categoryLabels.Add(x.Text);
            foreach (var x in config["colour_labels"].Items) colourLabels.Add(x.Text);
            relationConfig = rc;
            this.work = work;
            foreach (var s in Resolver.Statuses) counts[s] = 0;
        }

        private Relations Rel() { return relations ?? (relations = new Relations(scene, relationConfig)); }

        private void Charge(string counter)
        {
            string name = Resolver.Dynamic[counter];
            long limit = limits[name], used = work[counter];
            if (used + 1 > limit) throw new OverBudget(name, limit, used, 1);
            work[counter] = used + 1;
        }

        private void ConsultWrapper(JNode w) { if (ObjectGeometry.IsConditional(w)) conditional = true; }

        private Truth Attributes(JNode obj, NormNode node, HashSet<string> why)
        {
            Truth value = Truth.True;
            if (node.Category != null)
            {
                JNode w = obj["category"];
                ConsultWrapper(w);
                if (w["state"].Text == "known") value = w["value"]["model"].Text == node.Category ? Truth.True : Truth.False;
                else { value = Truth.Unknown; why.Add("unknown_category"); }
                if (value == Truth.False) return value;
            }
            if (node.ColoursAll.Count > 0)
            {
                JNode w = obj["attributes"]["colours"];
                ConsultWrapper(w);
                Truth colour;
                if (w["state"].Text == "known")
                {
                    var have = new HashSet<string>();
                    foreach (var c in w["value"].Items) have.Add(c.Text);
                    colour = Truth.True;
                    foreach (var c in node.ColoursAll) if (!have.Contains(c)) { colour = Truth.False; break; }
                }
                else { colour = Truth.Unknown; why.Add("unknown_colour"); }
                value = Tri.And(value, colour);
            }
            return value;
        }

        private RelationResult Predicate(Dictionary<string, string> env, NormConstraint c, string target)
        {
            if (c.Frame != null) throw new ResolverFault("directions are not supported on the headset (parser v1 has no frames): " + c.Relation + " " + c.Frame);
            var anchors = new List<string>();
            foreach (var a in c.Anchors) anchors.Add(env[a]);
            string key;
            if (c.Relation == "near") key = c.Relation + "\u0001" + string.Join("\u0000", Py.Sorted(new[] { target, anchors[0] }));
            else if (c.Relation == "between") key = c.Relation + "\u0001" + target + "\u0000" + string.Join("\u0000", Py.Sorted(anchors));
            else key = c.Relation + "\u0001" + target + "\u0000" + string.Join("\u0000", anchors);
            RelationResult result;
            if (memo.TryGetValue(key, out result)) work["predicate_cache_hits"]++;
            else
            {
                Charge("predicate_calls");
                result = Rel().Evaluate(c.Relation, target, anchors.ToArray());
                memo[key] = result;
            }
            if (result.Conditional) conditional = true;
            return result;
        }

        private static HashSet<string> UnknownCodes(RelationResult r)
        {
            var codes = new HashSet<string>();
            foreach (var x in r.Reasons)
            {
                if (x.StartsWith("missing_centre:", StringComparison.Ordinal)) codes.Add("missing_centre");
                else if (x.StartsWith("missing_size:", StringComparison.Ordinal)) codes.Add("missing_size");
                else if (x.StartsWith("missing_rotation:", StringComparison.Ordinal)) codes.Add("missing_rotation");
                else if (x.StartsWith("boundary:", StringComparison.Ordinal)) codes.Add("boundary_relation");
            }
            if (codes.Count == 0) codes.Add("relation_unknown");
            return codes;
        }

        private static HashSet<string> RankCodes(RankResult r)
        {
            var codes = new HashSet<string>();
            foreach (var x in r.Reasons)
            {
                if (x.StartsWith("missing_centre:", StringComparison.Ordinal)) codes.Add("missing_centre");
                else if (x.StartsWith("possible_competitors:", StringComparison.Ordinal)) codes.Add("possible_rank_competitors");
                else if (x == "tie") codes.Add("rank_tie");
                else if (x == "too_few_candidates") codes.Add("too_few_rank_candidates");
            }
            if (r.Status == "insufficient" && codes.Count == 0) codes.Add("relation_unknown");
            return codes;
        }

        private static string Classify(List<string> definite, List<string> possible)
        {
            if (definite.Count >= 2) return "ambiguous";
            if (possible.Count > 0) return "insufficient_information";
            return definite.Count > 0 ? "resolved" : "no_match";
        }

        private void EvaluateNode(NormNode node, Dictionary<string, string> env, List<string> definite, List<string> possible, HashSet<string> causes)
        {
            foreach (var oid in objects)
            {
                Charge("candidate_checks");
                var why = new HashSet<string>();
                Truth value = Attributes(byId[oid], node, why);
                if (value == Truth.False) continue;
                foreach (var c in node.Constraints)
                {
                    var result = Predicate(env, c, oid);
                    if (result.Value == Truth.Unknown) why.UnionWith(UnknownCodes(result));
                    value = Tri.And(value, result.Value);
                    if (value == Truth.False) break;
                }
                if (value == Truth.True) definite.Add(oid);
                else if (value == Truth.Unknown) { possible.Add(oid); causes.UnionWith(why); }
            }
        }

        private static int BitLength(long x)
        {
            x = Math.Abs(x);
            int n = 0;
            while (x > 0) { n++; x >>= 1; }
            return n;
        }

        private RankResult Rank(NormNode node, Dictionary<string, string> env, List<string> definite, List<string> possible, List<string> dx, List<string> px)
        {
            string anchor = env[node.RankAnchor];
            foreach (var x in definite) if (x != anchor) dx.Add(x);
            foreach (var x in possible) if (x != anchor) px.Add(x);
            string key = node.RankRelation + "\u0001" + node.RankK + "\u0001" + anchor + "\u0001" + string.Join("\u0000", dx) + "\u0001" + string.Join("\u0000", px);
            RankResult result;
            if (rankMemo.TryGetValue(key, out result)) work["rank_cache_hits"]++;
            else
            {
                long limit = limits["max_rank_subset_evaluations"], used = work["rank_subsets"];
                long remaining = limit - used;
                if (px.Count >= BitLength(remaining)) throw new OverBudget("max_rank_subset_evaluations", limit, used, remaining + 1);
                long bound = 1L << px.Count;
                result = Rel().Rank(node.RankRelation, anchor, definite, possible, node.RankK);
                if (result.CombinationsChecked < 0 || result.CombinationsChecked > bound) throw new ResolverFault("rank checked more subsets than its bound");
                work["rank_subsets"] += result.CombinationsChecked;
                work["rank_calls"]++;
                work["max_rank_preflight_subsets"] = Math.Max(work["max_rank_preflight_subsets"], bound);
                rankMemo[key] = result;
            }
            if (result.Conditional) conditional = true;
            return result;
        }

        private void Terminal(string action, string status, string target, IEnumerable<string> codes, bool incomplete = false)
        {
            work["terminal_outcomes"]++;
            counts[status]++;
            outcomes.Add(new Outcome { Action = action, Status = status, Target = target, Codes = new HashSet<string>(codes), Incomplete = incomplete });
        }

        public void Run(List<NormInterpretation> interpretations)
        {
            foreach (var it in interpretations)
            {
                if (it.Kind == "unavailable") Terminal(it.Action, "insufficient_information", null, new[] { it.Reason }, true);
                else if (it.Kind == "unsupported") Terminal(it.Action, "unsupported", null, new[] { it.Reason }, true);
                else
                {
                    var outside = new HashSet<string>();
                    foreach (var node in it.Nodes.Values)
                    {
                        if (node.Category != null && !categoryLabels.Contains(node.Category)) outside.Add("unsupported_category_label");
                        foreach (var c in node.ColoursAll) if (!colourLabels.Contains(c)) { outside.Add("unsupported_colour_label"); break; }
                    }
                    if (outside.Count > 0) Terminal(it.Action, "unsupported", null, outside, true);
                    else RunQuery(it);
                }
            }
        }

        private sealed class Frame
        {
            public string Variable, Oid;
            public Dictionary<string, string> Env;
            public int Index;
        }

        private void RunQuery(NormInterpretation it)
        {
            int before = outcomes.Count;
            var stack = new List<Frame> { new Frame { Env = new Dictionary<string, string>(), Index = 0 } };
            while (stack.Count > 0)
            {
                var f = stack[stack.Count - 1];
                stack.RemoveAt(stack.Count - 1);
                var env = f.Env;
                if (f.Variable != null)
                {
                    Charge("binding_attempts");
                    env = new Dictionary<string, string>(env);
                    env[f.Variable] = f.Oid;
                    bool illegal = false;
                    foreach (var pair in it.Between)
                    {
                        string x, y;
                        if (env.TryGetValue(pair[0], out x) && env.TryGetValue(pair[1], out y) && x == y) { illegal = true; break; }
                    }
                    if (illegal) { work["pruned_bindings"]++; continue; }
                }
                var node = it.Nodes[it.Order[f.Index]];
                string nid = node.NodeId;
                var definite = new List<string>();
                var possible = new List<string>();
                var causes = new HashSet<string>();
                EvaluateNode(node, env, definite, possible, causes);
                RankResult ranked = null;
                var dx = new List<string>();
                var px = new List<string>();
                if (node.RankRelation != null) ranked = Rank(node, env, definite, possible, dx, px);
                if (nid == it.Root) { FinishRoot(it.Action, definite, possible, causes, ranked, dx, px); continue; }
                List<string> choices;
                if (ranked == null)
                {
                    if (possible.Count > 0)
                    {
                        var codes = new HashSet<string>(causes) { "uncertain_anchor" };
                        Terminal(it.Action, "insufficient_information", null, codes, true);
                        continue;
                    }
                    if (definite.Count == 0) { Terminal(it.Action, "no_match", null, new[] { "anchor_not_found" }); continue; }
                    choices = definite;
                }
                else
                {
                    if (ranked.Status == "no_match")
                    {
                        var codes = RankCodes(ranked); codes.Add("anchor_not_found");
                        Terminal(it.Action, "no_match", null, codes);
                        continue;
                    }
                    if (ranked.Status == "insufficient")
                    {
                        var codes = RankCodes(ranked); codes.Add("uncertain_anchor"); codes.UnionWith(causes);
                        Terminal(it.Action, "insufficient_information", null, codes, true);
                        continue;
                    }
                    choices = Py.Sorted(ranked.ObjectIds);
                }
                for (int i = choices.Count - 1; i >= 0; i--)
                    stack.Add(new Frame { Variable = nid, Oid = choices[i], Env = env, Index = f.Index + 1 });
            }
            if (outcomes.Count == before) Terminal(it.Action, "no_match", null, new[] { "no_legal_anchor_binding" });
        }

        private void FinishRoot(string action, List<string> definite, List<string> possible, HashSet<string> causes, RankResult ranked,
                                List<string> dx, List<string> px)
        {
            string status, target;
            HashSet<string> codes;
            if (ranked == null)
            {
                status = Classify(definite, possible);
                targets.UnionWith(definite);
                uncertain.UnionWith(possible);
                target = status == "resolved" ? definite[0] : null;
                if (status == "resolved") codes = new HashSet<string>();
                else if (status == "ambiguous") { codes = new HashSet<string>(causes) { "multiple_definite_matches" }; }
                else if (status == "no_match") codes = new HashSet<string> { "target_not_found" };
                else codes = new HashSet<string>(causes);
            }
            else
            {
                status = Resolver.RankStatus[ranked.Status];
                targets.UnionWith(ranked.ObjectIds);
                rankDefinite.UnionWith(dx);
                rankPossible.UnionWith(px);
                target = status == "resolved" ? ranked.ObjectIds[0] : null;
                codes = RankCodes(ranked);
                if (status == "insufficient_information") codes.UnionWith(causes);
            }
            Terminal(action, status, target, codes);
        }

        public ResolverResult Result(JNode query)
        {
            var statuses = new List<string>();
            var resolvedTargets = new HashSet<string>();
            var codes = new HashSet<string>();
            var resolvedActions = new HashSet<string>();
            var actions = new HashSet<string>();
            bool actionsHaveNull = false, incomplete = false;
            foreach (var o in outcomes)
            {
                statuses.Add(o.Status);
                if (o.Status == "resolved") { resolvedTargets.Add(o.Target); resolvedActions.Add(o.Action ?? "\u0000null"); }
                codes.UnionWith(o.Codes);
                if (o.Action == null) actionsHaveNull = true; else actions.Add(o.Action);
                incomplete |= o.Incomplete;
            }
            int distinctActions = actions.Count + (actionsHaveNull ? 1 : 0);
            if (resolvedTargets.Count > 1) codes.Add("divergent_targets");
            if (resolvedActions.Count > 1) codes.Add("divergent_actions");
            if (statuses.Contains("resolved") && statuses.Contains("no_match")) codes.Add("resolved_vs_no_match");
            string status;
            bool allResolved = statuses.TrueForAll(s => s == "resolved"), allNoMatch = statuses.TrueForAll(s => s == "no_match");
            if (statuses.Contains("unsupported")) status = "unsupported";
            else if (statuses.Contains("insufficient_information")) status = "insufficient_information";
            else if (allResolved && resolvedTargets.Count == 1 && distinctActions == 1) status = "resolved";
            else if (allNoMatch) status = "no_match";
            else status = "ambiguous";
            if (conditional) codes.Add("conditional_evidence");
            string onlyAction = null;
            if (distinctActions == 1 && !actionsHaveNull) foreach (var a in actions) onlyAction = a;
            string targetId = null;
            if (status == "resolved") foreach (var t in resolvedTargets) targetId = t;
            return new ResolverResult
            {
                Status = status, Action = onlyAction, TargetId = targetId, ReasonCode = Resolver.Primary[status], ReasonCodes = Py.Sorted(codes),
                TargetIds = Py.Sorted(targets), UncertainMatchIds = Py.Sorted(uncertain), RankDefiniteEligibleIds = Py.Sorted(rankDefinite),
                RankPossibleEligibleIds = Py.Sorted(rankPossible), Coverage = incomplete ? "incomplete_due_to_unresolved_reference" : "evaluated_branch_union",
                Conditional = conditional
            };
        }
    }
}
