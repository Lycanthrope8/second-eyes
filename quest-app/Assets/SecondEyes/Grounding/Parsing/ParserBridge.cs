using SecondEyes.Grounding.Interactive;
using SecondEyes.Grounding.Prompting;

namespace SecondEyes.Grounding.Parsing
{
    /// <summary>
    /// A2.7 (D115): the parser path inside A2.5's interactive session, free of Unity. A request keeps A2.5's intake
    /// (presets, the ADB inbox, scene binding, staleness, cancellation); the bridge adds the scene's category map from the
    /// parser folder, and maps the parser's outcome onto A2.5's statuses without blurring its kinds: selected becomes
    /// completed with the target; ask stays ask; unsupported, invalid and budget become refused; a technical failure
    /// stays failed; a cancellation is A2.5's cancelled. The exact kind and the resolver's outcome travel in the
    /// outcome's parser block, and the reason names the outcome.
    /// </summary>
    public static class ParserBridge
    {
        public static ParserRequest Request(InteractiveRequest r, ParserFolder f)
        {
            string map = null;
            try { map = f.MapFor(JParse.Parse(r.SceneText)); }
            catch (System.Exception) { map = null; }   // a scene that doesn't parse is refused by the pipeline itself
            return new ParserRequest { RequestId = r.RequestId, Source = r.Source, SnapshotId = r.SnapshotId, SnapshotSha256 = r.SnapshotSha256,
                                       SceneText = r.SceneText, CommandText = r.CommandText, MapText = map };
        }

        public static InteractiveOutcome Outcome(ParserOutcome p)
        {
            var o = new InteractiveOutcome
            {
                RequestId = p.RequestId, Source = p.Source, Path = "parser", CacheMode = "Off", Purpose = "Operational",
                ExecutionPath = p.PromptTokens >= 0 ? "U" : "none", Reason = p.Outcome, Detail = p.Detail ?? "",
                PromptSha256 = p.PromptSha256, Tokens = p.PromptTokens, EvaluatedTokens = p.PromptTokens > 0 ? p.PromptTokens : 0,
                BuildMs = p.ParseMs, TokenizeMs = p.TokenizeMs, EvalMs = p.DecodeMs, ScoreMs = p.ResolveMs, TotalMs = p.TotalMs,
                ParserKind = p.Status, ParserOutcome = p.Outcome, DecodedText = p.DecodedText, ReasonCode = p.ReasonCode,
                AssetsSha256 = p.AssetsSha256, Conditional = p.Conditional, CandidateIds = p.CandidateIds, OutputTokens = p.OutputTokens,
                DecodeMs = p.DecodeMs, ResolveMs = p.ResolveMs
            };
            switch (p.Status)
            {
                case "selected": o.Status = "completed"; o.TargetObjectId = p.TargetObjectId; break;
                case "ask": o.Status = "ask"; break;
                case "unsupported": case "invalid": case "budget": o.Status = "refused"; break;
                default:
                    if (p.Outcome == "cancelled") { o.Status = "cancelled"; o.Reason = "cancelled_before_evaluation"; }
                    else o.Status = "failed";
                    break;
            }
            return o;
        }
    }
}
