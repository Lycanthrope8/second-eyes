using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using SecondEyes.Grounding.Parsing;
using SecondEyes.Grounding.Prompting;
using SecondEyes.Grounding.Resolution;
using SecondEyes.Perception.AutoScan;
using SecondEyes.Perception.Frames;

/// <summary>
/// Checks of the automatic scan's core (A4; docs/auto-scan.md) on synthetic rooms with known positions: settings,
/// depth sampling, the pinhole, each detection's fate, joining and merging, viewpoints, the replay and the contract.
/// </summary>
static class Program
{
    static int pass, fail;
    static void Check(string n, bool ok, string d = "")
    {
        if (ok) pass++; else fail++;
        Console.WriteLine((ok ? "  ok    " : "  FAIL  ") + n + (ok || d == "" ? "" : "  [" + d + "]"));
    }

    static AutoScanSettings S;
    static JNode Map, Priors;
    const double Fx = 870;

    static AutoScan NewScan() { return new AutoScan("scan.test.1", S, Map, Priors, "yolox_nano_coco.se1", new string('c', 64), "quest_left", Fx); }

    static FrameRecord Frame(int n, Vec3 cam, params FrameDetection[] dets)
    {
        var f = new FrameRecord { Frame = n, TimeUs = 1000000 + n * 500000L, Epoch = 2, Tracked = true, Position = cam, Rotation = new Quat(0, 0, 0, 1) };
        f.Detections.AddRange(dets);
        return f;
    }

    /// <summary>A detection whose five depth samples hit at the given points (null for no hit).</summary>
    static FrameDetection Det(string label, double score, double x1, double y1, double x2, double y2, params Vec3?[] hits)
    {
        var d = new FrameDetection { ClassId = 0, Label = label, Score = score, X1 = x1, Y1 = y1, X2 = x2, Y2 = y2 };
        var px = AutoScan.SamplePixels(x1, y1, x2, y2, S.SampleSpread);
        for (int i = 0; i < px.Count; i++)
        {
            Vec3? h = i < hits.Length ? hits[i] : null;
            d.Samples.Add(new DepthSample { U = px[i][0], V = px[i][1], Status = h.HasValue ? "hit" : "no_hit", Point = h });
        }
        return d;
    }

    static Vec3?[] Around(Vec3 p, double jitter)
    {
        return new Vec3?[] { p, p + new Vec3(jitter, 0, 0), p - new Vec3(jitter, 0, 0), p + new Vec3(0, jitter, 0), p - new Vec3(0, 0, jitter) };
    }

    static List<JNode> Objects(AutoScan a, int rev) { return a.Snapshot(rev).Scene["objects"].Items; }
    static string Cat(JNode o) { return o["category"]["value"]["standard"].Text; }
    static Vec3 Centre(JNode o) { var v = o["geometry"]["center_m"]["value"].Items; return new Vec3(v[0].Num, v[1].Num, v[2].Num); }

    static int Main(string[] args)
    {
        string schemas = args.Length > 0 ? args[0] : "schemas";
        var cache = new Dictionary<string, Schema>();
        Func<string, Schema> schema = n => { if (!cache.ContainsKey(n)) cache[n] = new Schema(JParse.Parse(File.ReadAllText(Path.Combine(schemas, n)))); return cache[n]; };
        JNode cfg = JParse.Parse(File.ReadAllText(Path.Combine("perception", "autoscan", "autoscan.default.v1.json")));
        Map = JParse.Parse(File.ReadAllText(Path.Combine("perception", "bedroom", "category-map.bedroom.v2.json")));
        Priors = JParse.Parse(File.ReadAllText(Path.Combine("perception", "bedroom", "size-priors.bedroom.v1.json")));

        Console.WriteLine("1. The settings");
        var errs = schema("auto-scan-config.v1.json").Errors(cfg);
        Check("autoscan.default.v1 passes its schema", errs.Count == 0, string.Join(", ", errs.Select(e => e.Keyword + " " + e.Path)));
        S = AutoScanSettings.From(cfg);
        Check("its values load: twice a second, two frames to confirm, a 0.3 m least reach", S.IntervalS == 0.5 && S.MinFrames == 2 && S.MinReachM == 0.3 && S.SampleSpread == 0.2);

        Console.WriteLine("2. Where depth is sampled, and the pinhole");
        var px = AutoScan.SamplePixels(100, 200, 300, 400, 0.2);
        Check("five samples, the box's centre first, then 40 px either way across and down", px.Count == 5 && px[0][0] == 200 && px[0][1] == 300
              && px[1][0] == 160 && px[2][0] == 240 && px[3][1] == 260 && px[4][1] == 340);
        Check("with no spread, the centre alone", AutoScan.SamplePixels(0, 0, 10, 10, 0).Count == 1);
        Vec3 c0 = AutoScan.CameraDirection(643.5, 481.5, Fx, Fx, 643, 481);
        Vec3 right = AutoScan.CameraDirection(743.5, 481.5, Fx, Fx, 643, 481), below = AutoScan.CameraDirection(643.5, 581.5, Fx, Fx, 643, 481);
        Check("through the principal point, straight ahead", Math.Abs(c0.X) < 1e-12 && Math.Abs(c0.Y) < 1e-12 && Math.Abs(c0.Z - 1) < 1e-12);
        Check("a pixel to the right points right, one below points down (Unity's y up)", right.X > 0 && Math.Abs(right.Y) < 1e-12 && below.Y < 0 && Math.Abs(below.X) < 1e-12);
        Vec3 u = new Vec3(1, 2, 3);
        Check("the basis change swaps y and z, both ways", AutoScan.ToScene(u).Y == 3 && AutoScan.ToScene(u).Z == 2 && (AutoScan.ToUnity(AutoScan.ToScene(u)) - u).Norm == 0);

        Console.WriteLine("3. A room seen in two frames from two places");
        Vec3 camA = new Vec3(0, 1.5, 0), camB = new Vec3(0.8, 1.5, 0.2);
        Vec3 chair = new Vec3(1.0, 0.45, 3.0), bed = new Vec3(-1.2, 0.25, 2.5), bottle = new Vec3(0.4, 0.85, 2.0);
        var scan = NewScan();
        var f1 = Frame(1, camA,
            Det("chair", 0.8, 800, 300, 1000, 600, Around(chair, 0.02)), Det("bed", 0.7, 100, 500, 600, 700, Around(bed, 0.03)),
            Det("bottle", 0.6, 640, 400, 660, 460, Around(bottle, 0.01)), Det("person", 0.9, 10, 10, 100, 300, Around(new Vec3(-2, 1, 3), 0.02)),
            Det("chair", 0.2, 300, 300, 400, 500, Around(new Vec3(0.2, 0.5, 2), 0.02)), Det("chair", 0.7, 400, 300, 500, 500),
            Det("tv", 0.8, 500, 100, 600, 160, Around(new Vec3(0, 1.2, 7.0), 0.02)), Det("mouse", 0.6, 700, 700, 720, 710, Around(new Vec3(0.3, -0.5, 1.5), 0.01)));
        FrameOutcome o1 = scan.Add(f1);
        Check("after one frame nothing is in the scene: an object needs two", Objects(scan, 1).Count == 0 && scan.Tentative == 3, scan.Tentative.ToString());
        Check("the frame's fates, in the frame's own order", string.Join(",", o1.Fates) == "kept,kept,kept,unmapped_label,below_min_score,no_depth_hit,too_far,below_floor",
              string.Join(",", o1.Fates));
        scan.Add(Frame(2, camB, Det("chair", 0.85, 700, 300, 900, 600, Around(chair + new Vec3(0.03, 0, -0.02), 0.02)),
                       Det("bed", 0.72, 150, 500, 650, 700, Around(bed + new Vec3(0.1, 0, 0.05), 0.03)),
                       Det("bottle", 0.55, 600, 400, 620, 460, Around(bottle + new Vec3(0.01, 0, 0), 0.01))));
        var r = scan.Snapshot(2);
        var objs = r.Scene["objects"].Items;
        Check("after the second: a bed, a bottle and a chair", string.Join(",", objs.Select(Cat)) == "bed,bottle,chair", string.Join(",", objs.Select(Cat)));
        var truth = new Dictionary<string, Vec3> { { "bed", bed }, { "bottle", bottle }, { "chair", chair } };
        foreach (JNode ob in objs)
            Check(Cat(ob) + " within 10 cm of its points (the second frame's points sit up to 11 cm away)", (Centre(ob) - AutoScan.ToScene(truth[Cat(ob)])).Norm < 0.10,
                  (Centre(ob) - AutoScan.ToScene(truth[Cat(ob)])).Norm.ToString("0.000"));
        Check("each is seen from two viewpoints 0.82 m apart: placed from two positions", objs.All(x => x["observation"]["association"]["status"].Text == "multi_position"
              && x["observation"]["views"]["positions"].Num == 2 && x["observation"]["views"]["keyframes"].Num == 2));
        Check("centres name the depth_surface assumption, which the scene declares", objs.All(x => x["geometry"]["center_m"]["evidence"]["assumptions"].Items.Single().Text == "depth_surface")
              && r.Scene["assumptions"].Items.Any(a => a["assumption_id"].Text == "depth_surface"));
        var left = r.Trace["left_out"];
        Check("the trace counts each leave-out reason", left["unmapped_label"].Num == 1 && left["below_min_score"].Num == 1 && left["no_depth_hit"].Num == 1
              && left["too_far"].Num == 1 && left["below_floor"].Num == 1);
        var ids = new HashSet<string>(objs.Select(x => x["object_id"].Text));
        var kept = r.Trace["detections"].Items.Where(d => d["fate"].Text == "object").ToList();
        Check("the trace sends six kept detections to objects in the scene, by keyframe and detection index", kept.Count == 6 && kept.All(d => ids.Contains(d["object_id"].Text))
              && r.Trace["detections"].Items.Count == 11 && r.Trace["detections"].Items.All(d => d.Has("keyframe") && d.Has("detection")));
        var contract = new Contract(schema);
        var issues = contract.Validate(new List<KeyValuePair<string, JNode>> { new KeyValuePair<string, JNode>("scene", r.Scene), new KeyValuePair<string, JNode>("map", Map) });
        Check("the scene passes the contract (scene v3) with the bedroom's map v2", !issues.Any(i => i.IsError), string.Join(", ", issues.Where(i => i.IsError).Select(i => i.Code + " " + i.Path).Take(3)));
        Check("the labels for view are in quest_tracking", scan.ConfirmedInTracking().Any(kv => kv.Key == "chair" && Math.Abs(kv.Value.Y - 0.45) < 0.05));

        Console.WriteLine("4. Depth that misses the object");
        var s4 = NewScan();
        Vec3 near = new Vec3(0, 0.5, 2.0);
        var mixed = new Vec3?[] { near, near + new Vec3(0, 0, 0.05), new Vec3(0, 0.5, 6.0), null, null };   // two on the chair, one through its mesh to the wall
        s4.Add(Frame(1, camA, Det("chair", 0.8, 600, 400, 700, 600, mixed)));
        s4.Add(Frame(2, camA, Det("chair", 0.8, 600, 400, 700, 600, mixed)));
        var c4 = Objects(s4, 1);
        Check("a ray through the chair's mesh to the wall doesn't move it: the median of the hits is on the chair", c4.Count == 1 && Math.Abs(Centre(c4[0]).Y - 2.05) < 1e-6,
              c4.Count == 1 ? Centre(c4[0]).Y.ToString("0.000") : c4.Count.ToString());
        Check("seen twice from one place: a single position", c4.Count == 1 && c4[0]["observation"]["association"]["status"].Text == "single_position");

        Console.WriteLine("5. Joining and merging");
        var s5 = NewScan();
        Vec3 chairL = new Vec3(-0.6, 0.45, 3.0), chairR = new Vec3(0.6, 0.45, 3.0);
        for (int n = 1; n <= 2; n++)
            s5.Add(Frame(n, n == 1 ? camA : camB, Det("chair", 0.8, 400, 300, 560, 600, Around(chairL, 0.02)), Det("chair", 0.8, 720, 300, 880, 600, Around(chairR, 0.02))));
        Check("two chairs 1.2 m apart stay two", Objects(s5, 1).Count(x => Cat(x) == "chair") == 2);
        var s6 = NewScan();
        s6.Add(Frame(1, camA, Det("bed", 0.7, 100, 400, 700, 700, Around(new Vec3(-1.2, 0.25, 2.5), 0.03))));
        s6.Add(Frame(2, camB, Det("bed", 0.7, 300, 400, 900, 700, Around(new Vec3(-0.6, 0.25, 2.9), 0.03))));
        Check("a bed seen at parts 0.72 m apart is one bed: its reach grows with its measured size", Objects(s6, 1).Count == 1, Objects(s6, 1).Count.ToString());
        var s7 = NewScan();
        s7.Add(Frame(1, camA, Det("chair", 0.8, 600, 400, 680, 500, Around(new Vec3(2.0, 0.45, 3.0), 0.0))));
        s7.Add(Frame(2, camA, Det("chair", 0.8, 700, 400, 780, 500, Around(new Vec3(2.35, 0.45, 3.0), 0.0))));
        Check("two small chairs 0.35 m apart start as two", s7.Tentative == 2, s7.Tentative.ToString());
        s7.Add(Frame(3, camB, Det("chair", 0.8, 600, 250, 900, 700, Around(new Vec3(2.18, 0.45, 3.0), 0.0))));
        Check("a closer, larger view joins one and the two merge", s7.Confirmed == 1 && s7.Tentative == 0, s7.Confirmed + "/" + s7.Tentative);

        Console.WriteLine("6. The replay and the order of detections");
        var frames = new List<FrameRecord> { f1 };
        frames.Add(Frame(2, camB, Det("chair", 0.85, 700, 300, 900, 600, Around(chair + new Vec3(0.03, 0, -0.02), 0.02)),
                         Det("bed", 0.72, 150, 500, 650, 700, Around(bed + new Vec3(0.1, 0, 0.05), 0.03))));
        frames.Add(Frame(3, camA, Det("couch", 0.66, 200, 500, 500, 700, Around(new Vec3(-2.5, 0.35, 3.5), 0.03)), Det("bottle", 0.6, 640, 400, 660, 460, Around(bottle, 0.01))));
        frames.Add(Frame(4, camB, Det("couch", 0.7, 250, 500, 550, 700, Around(new Vec3(-2.45, 0.35, 3.5), 0.03))));
        var a = NewScan();
        var lines = new List<string> { PyJson.Canonical(a.Header("f00d", 1280, 960, Fx, 643, 481, 416)) };
        foreach (FrameRecord f in frames) { lines.Add(PyJson.Canonical(f.ToJson())); a.Add(f); }
        var ra = a.Snapshot(4);
        var b = AutoScan.FromHeader(JParse.Parse(lines[0]), S, Map, Priors);
        foreach (string line in lines.Skip(1)) b.Add(FrameRecord.FromJson(JParse.Parse(line)));
        var rb = b.Snapshot(4);
        if (args.Length > 1)   // a sample scan as the headset writes it, for the PC replay (tools/localize --memory)
        {
            Directory.CreateDirectory(args[1]);
            File.WriteAllLines(Path.Combine(args[1], "frames.jsonl"), lines);
            File.WriteAllText(Path.Combine(args[1], "scene.json"), PyJson.Canonical(ra.Scene) + "\n");
            File.WriteAllText(Path.Combine(args[1], "trace.json"), PyJson.Canonical(ra.Trace) + "\n");
            Console.WriteLine("  (a sample scan written to " + args[1] + ")");
        }
        Check("a replay of the logged lines writes the same scene and trace, byte for byte", PyJson.Canonical(ra.Scene) == PyJson.Canonical(rb.Scene)
              && PyJson.Canonical(ra.Trace) == PyJson.Canonical(rb.Trace));
        var rev = NewScan();
        foreach (FrameRecord f in frames)
        {
            var g = FrameRecord.FromJson(f.ToJson());
            var copy = new FrameRecord { Frame = g.Frame, TimeUs = g.TimeUs, Epoch = g.Epoch, Tracked = g.Tracked, Position = g.Position, Rotation = g.Rotation };
            copy.Detections.AddRange(Enumerable.Reverse(g.Detections));
            rev.Add(copy);
        }
        Check("the order the detector lists boxes in changes nothing in the scene", PyJson.Canonical(rev.Snapshot(4).Scene) == PyJson.Canonical(ra.Scene));
        Check("the couch is kept as a couch", ra.Scene["objects"].Items.Any(x => Cat(x) == "couch" && x["category"]["value"]["model"].Text == "couch"));
        var issues6 = contract.Validate(new List<KeyValuePair<string, JNode>> { new KeyValuePair<string, JNode>("scene", ra.Scene), new KeyValuePair<string, JNode>("map", Map) });
        Check("that scene passes the contract too", !issues6.Any(i => i.IsError), string.Join(", ", issues6.Where(i => i.IsError).Select(i => i.Code + " " + i.Path).Take(3)));

        Console.WriteLine("7. Tracking");
        var s8 = NewScan();
        var lost = Frame(1, camA, Det("chair", 0.8, 600, 300, 800, 600, Around(chair, 0.02)));
        lost.Tracked = false;
        Check("a frame taken without tracking is left out", s8.Add(lost).Fates.Single() == "untracked");
        s8.Add(Frame(2, camA, Det("chair", 0.8, 600, 300, 800, 600, Around(chair, 0.02))));
        var moved = Frame(3, camA, Det("chair", 0.8, 600, 300, 800, 600, Around(chair, 0.02)));
        moved.Epoch = 3;
        Check("a frame from a later tracking epoch is left out: its points belong to another origin", s8.Add(moved).Fates.Single() == "tracking_reset");
        var t8 = s8.Snapshot(1).Trace["detections"].Items;
        Check("an object seen once is tentative in the trace", t8.Any(d => d["fate"].Text == "tentative"));

        Console.WriteLine(pass + " passed, " + fail + " failed");
        return fail == 0 ? 0 : 1;
    }
}
