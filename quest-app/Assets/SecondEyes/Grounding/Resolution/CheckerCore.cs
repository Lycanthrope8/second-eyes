using System;
using System.Collections.Generic;
using System.Globalization;
using SecondEyes.Grounding.Prompting;

namespace SecondEyes.Grounding.Resolution
{
    // A2.7 delivery 2 (D115): grounding/a24/labels.py's Checker, free of Unity, the accepted resolution path of the parser
    // design: the language's interpretation becomes a grounding_query (language.query); a query that fails validation is
    // invalid_label_query; otherwise the resolver runs with the category map's resolver configuration (predict's
    // resolver_config), and its record becomes the outcome: the result's status when completed, else the processing
    // status (invalid_input, budget_exceeded). A ResolverFault is a technical failure, for the caller to report apart.
    public sealed class CheckerOutcome
    {
        public string Outcome, TargetId, ReasonCode;
        public List<string> CandidateIds = new List<string>();
        public List<Issue> QueryIssues = new List<Issue>();
        public RunRecord Run;
    }

    public sealed class Checker
    {
        private readonly ResolverSetup setup;
        private readonly JNode protocol;   // resolver-protocol.json: config_id, vocabulary_id, limits, colour_labels, query_suffix
        private readonly Dictionary<string, JNode> configs = new Dictionary<string, JNode>();

        public Checker(ResolverSetup setup, JNode resolverProtocol)
        {
            this.setup = setup;
            protocol = resolverProtocol;
        }

        public string QuerySuffix { get { return protocol["query_suffix"].Text; } }

        public static JNode Int(long v)
        {
            return new JNode { Kind = JKind.Number, Num = v, Text = v.ToString(CultureInfo.InvariantCulture), IsInteger = true };
        }

        /// <summary>predict.resolver_config: the protocol's identifiers, limits and colours with the map's sorted
        /// vocabulary, validated once per map. An invalid configuration is a fault, as the Python raises.</summary>
        public JNode Config(JNode cmap)
        {
            string key = cmap["map_id"].Text;
            JNode cfg;
            if (configs.TryGetValue(key, out cfg)) return cfg;
            var labels = new List<string>();
            foreach (var x in cmap["model_vocabulary"].Items) labels.Add(x.Text);
            labels.Sort(string.CompareOrdinal);
            var cats = new List<JNode>();
            foreach (var l in labels) cats.Add(JNode.Str(l));
            cfg = JNode.Obj().Add("schema_version", Int(1)).Add("record_type", JNode.Str("resolver_config"))
                .Add("config_id", protocol["config_id"]).Add("vocabulary_id", protocol["vocabulary_id"])
                .Add("category_labels", JNode.Arr(cats)).Add("colour_labels", protocol["colour_labels"]).Add("limits", protocol["limits"]);
            var bad = ResolverValidation.CheckConfig(cfg, setup.Schemas("resolver-config.v1.json"));
            if (bad.Count > 0) throw new ResolverFault("the resolver configuration does not validate: " + bad[0]);
            configs[key] = cfg;
            return cfg;
        }

        /// <summary>language.query: the full grounding_query, its identity fields from the scene and command.</summary>
        public static JNode Query(JNode interp, JNode scene, JNode command, string suffix)
        {
            return JNode.Obj().Add("schema_version", Int(1)).Add("record_type", JNode.Str("grounding_query"))
                .Add("query_id", JNode.Str(command["command_id"].Text + suffix)).Add("scene_id", scene["scene_id"])
                .Add("scene_revision", scene["scene_revision"]).Add("evidence_profile", scene["evidence_profile"])
                .Add("command_id", command["command_id"]).Add("interpretations", JNode.Arr(new List<JNode> { interp }));
        }

        public CheckerOutcome Resolve(JNode interp, JNode scene, JNode command, JNode cmap)
        {
            var o = new CheckerOutcome();
            var q = Query(interp, scene, command, QuerySuffix);
            List<NormInterpretation> norm;
            o.QueryIssues = ResolverValidation.CheckQuery(q, setup.Schemas("grounding-query.v1.json"), out norm);
            if (o.QueryIssues.Count > 0) { o.Outcome = "invalid_label_query"; return o; }
            o.Run = Resolver.Resolve(scene, command, q, Config(cmap), new List<JNode> { cmap }, setup);
            bool done = o.Run.ProcessingStatus == "completed";
            o.Outcome = done ? o.Run.Result.Status : o.Run.ProcessingStatus;
            if (o.Run.Result != null)
            {
                o.TargetId = o.Outcome == "resolved" ? o.Run.Result.TargetId : null;
                o.CandidateIds = new List<string>(o.Run.Result.TargetIds);
                o.ReasonCode = o.Run.Result.ReasonCode;
            }
            return o;
        }
    }
}
