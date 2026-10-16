using System;
using SecondEyes.Grounding.Parsing;
using SecondEyes.Grounding.Prompting;

namespace SecondEyes.Grounding.Interactive
{
    /// <summary>
    /// A typed command as a command record (A4 Delivery 1, item 1), built as the PC's inbox-send --text builds one
    /// (grounding/quest/inbox.py, build_request): the scene's first dataset command with the person's text and its own
    /// command ID. The session reads a request's CommandText as that record. The trial of 15 October sent the raw text
    /// instead, which the session couldn't parse, so every typed command was refused as invalid_input.
    /// </summary>
    public static class TypedCommand
    {
        public static string Build(string baseCommandJson, string text, string idSuffix)
        {
            if (string.IsNullOrEmpty(text) || text.Trim().Length == 0) throw new ArgumentException("a typed command has text");
            JNode b = JParse.Parse(baseCommandJson);
            if (b == null || b.Kind != JKind.Object || !b.Has("text") || !b.Has("command_id"))
                throw new ArgumentException("the base is not a command record with text and a command ID");
            var o = JNode.Obj();
            foreach (var f in b.Fields)
            {
                if (f.Key == "text") o.Add("text", JNode.Str(text));
                else if (f.Key == "command_id") o.Add("command_id", JNode.Str(f.Value.Text + idSuffix));
                else o.Add(f.Key, f.Value);
            }
            return PyJson.Canonical(o);
        }

        /// <summary>A command typed about a scanned room (A4), which has no dataset command to carry it: a fresh command
        /// record of format 2 whose source is the person's input on the panel, about the scene at its revision, with no user
        /// pose. Every other field of the record follows from the scene.</summary>
        public static string ForScan(JNode scene, string text, string commandId)
        {
            if (string.IsNullOrEmpty(text) || text.Trim().Length == 0) throw new ArgumentException("a typed command has text");
            Func<JNode> unknown = () => JNode.Obj().Add("state", JNode.Str("unknown")).Add("reason", JNode.Str("not_in_source")).Add("source", JNode.Str("panel"));
            JNode pose = JNode.Obj().Add("pose_kind", JNode.Str("none")).Add("frame_id", scene["coordinate_frame"]["frame_id"])
                .Add("scene_revision", scene["scene_revision"]).Add("position_m", unknown()).Add("rotation_xyzw", unknown())
                .Add("heading_xy", unknown()).Add("snapshot_time", unknown());
            JNode o = JNode.Obj().Add("schema_version", JParse.Parse("2")).Add("record_type", JNode.Str("command_context"))
                .Add("command_id", JNode.Str(commandId)).Add("text", JNode.Str(text)).Add("scene_id", scene["scene_id"])
                .Add("scene_revision", scene["scene_revision"])
                .Add("sources", JNode.Arr(new System.Collections.Generic.List<JNode> { JNode.Obj().Add("source_id", JNode.Str("panel"))
                    .Add("kind", JNode.Str("user_input")).Add("description", JNode.Str("Typed on the headset's panel"))
                    .Add("release", JNode.Str(scene["scene_id"].Text)) }))
                .Add("assumptions", JNode.Arr(new System.Collections.Generic.List<JNode>())).Add("user_pose", pose);
            return PyJson.Canonical(o);
        }
    }
}
