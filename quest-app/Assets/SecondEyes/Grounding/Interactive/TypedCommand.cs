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
    }
}
