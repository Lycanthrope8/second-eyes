// A4 Delivery 1, stage 3: Perception/Scan/Glue/RoomScanner.cs compiles here against Stubs.cs, with the real Unity-free code
// it uses; building it is the first check. The stubs of this app's own classes (the event log, the detector runner, the
// frame source) can drift from the real ones, so the second check reads the real source files and confirms that every
// member the scanner calls is declared public in the class the stub puts it in. Unity's own APIs are checked by Unity.
// Run: dotnet run --project tools/csharp-tests/ScanGlue
using System;
using System.IO;
using System.Text.RegularExpressions;

static class Program
{
    static int pass, fail;

    /// <summary>The text of a class's body in a source file: from its declaration to the next top-level type, or the end.</summary>
    static string Body(string file, string type)
    {
        string t = File.ReadAllText(file);
        Match m = Regex.Match(t, @"\n\s*public (static |sealed |partial )*class " + type + @"\b");
        if (!m.Success) return null;
        Match next = Regex.Match(t.Substring(m.Index + m.Length), @"\n    public (static |sealed |partial )*(class|struct|interface|enum) ");
        return next.Success ? t.Substring(m.Index, m.Length + next.Index) : t.Substring(m.Index);
    }

    static void Has(string file, string type, params string[] members)
    {
        string body = Body(Path.Combine("quest-app", "Assets", "SecondEyes", file), type);
        foreach (string member in members)
        {
            bool ok = body != null && Regex.IsMatch(body, @"public [^\n=]*\b" + member + @"\b");
            if (ok) pass++; else fail++;
            Console.WriteLine((ok ? "  ok    " : "  FAIL  ") + type + "." + member + " is public in " + file);
        }
    }

    static int Main()
    {
        Console.WriteLine("RoomScanner compiles against the stubs");
        Console.WriteLine("Every member of this app the stubs give the scanner exists in the real class");
        Has("Logging/EventLog.cs", "EventLog", "SessionId", "Write", "Error", "NowMonoUs");
        Has("Perception/DetectorRunner.cs", "DetectorRunner", "Mode", "IsOn", "IsBusy", "SetOn", "SetMode", "CurrentMode", "ReadyForScan",
            "FrameSource", "ManifestText", "ManifestBytes", "RunScanKeyframe");
        Has("Perception/PassthroughFrameSource.cs", "PassthroughFrameSource", "CurrentTexture", "CameraName", "TryGetIntrinsics", "DescribeApi");
        Console.WriteLine(pass + " passed, " + fail + " failed");
        return fail == 0 ? 0 : 1;
    }
}
