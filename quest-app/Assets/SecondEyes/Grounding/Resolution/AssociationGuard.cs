using System;
using System.Collections.Generic;
using System.Linq;
using SecondEyes.Grounding.Prompting;

namespace SecondEyes.Grounding.Resolution
{
    /// <summary>
    /// The association guard (A4 Delivery 1; docs/resolution.md), as grounding/resolution/association.py. In a perceived
    /// scene, a resolved answer that rests on a possible duplicate becomes the existing ambiguity outcome: status
    /// ambiguous, reason nonunique_target_or_interpretation, detail code uncertain_association, and the pair (or group)
    /// as the target candidates. An answer rests on one when its target is one, or when the query ranks with k of 2 or
    /// more and any object eligible for that ranking is one. It runs after the accepted resolver and changes nothing else.
    /// </summary>
    public static class AssociationGuard
    {
        public const string Detail = "uncertain_association";

        public static RunRecord Guard(JNode scene, JNode query, RunRecord run)
        {
            if (scene["evidence_profile"].Text != "perceived" || run.ProcessingStatus != "completed" || run.Result == null) return run;
            ResolverResult r = run.Result;
            if (r.Status != "resolved") return run;
            var dup = new Dictionary<string, List<string>>();
            foreach (JNode o in scene["objects"].Items)
            {
                JNode a = o["observation"]["association"];
                if (a["status"].Text != "possible_duplicate") continue;
                dup[o["object_id"].Text] = a["may_duplicate"].Items.Select(x => x.Text).ToList();
            }
            var involved = new SortedSet<string>(StringComparer.Ordinal);
            List<string> partners;
            if (r.TargetId != null && dup.TryGetValue(r.TargetId, out partners)) { involved.Add(r.TargetId); involved.UnionWith(partners); }
            if (RanksPastFirst(query))
                foreach (string oid in new SortedSet<string>(r.RankDefiniteEligibleIds.Concat(r.RankPossibleEligibleIds), StringComparer.Ordinal))
                    if (dup.TryGetValue(oid, out partners)) { involved.Add(oid); involved.UnionWith(partners); }
            if (involved.Count == 0) return run;
            var codes = new SortedSet<string>(r.ReasonCodes, StringComparer.Ordinal) { Detail };
            var guarded = new ResolverResult
            {
                Status = "ambiguous", Action = r.Action, TargetId = null, ReasonCode = "nonunique_target_or_interpretation",
                Coverage = r.Coverage, ReasonCodes = codes.ToList(), TargetIds = involved.ToList(),
                UncertainMatchIds = new List<string>(r.UncertainMatchIds),
                RankDefiniteEligibleIds = new List<string>(r.RankDefiniteEligibleIds),
                RankPossibleEligibleIds = new List<string>(r.RankPossibleEligibleIds), Conditional = r.Conditional
            };
            return new RunRecord { ProcessingStatus = run.ProcessingStatus, Result = guarded, Issues = run.Issues,
                                   Warnings = run.Warnings, Budget = run.Budget, Work = run.Work };
        }

        private static bool RanksPastFirst(JNode query)
        {
            foreach (JNode it in query["interpretations"].Items)
            {
                if (it["kind"].Text != "query") continue;
                foreach (JNode node in it["nodes"].Items)
                {
                    JNode rank = node["rank"];
                    if (rank.Kind == JKind.Object && rank["k"].Num >= 2) return true;
                }
            }
            return false;
        }
    }
}
