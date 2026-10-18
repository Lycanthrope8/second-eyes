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

    /// <summary>An n by n grid of points on one face of a box (axis 0, 1 or 2 of its own axes; sign -1 or +1).</summary>
    static List<Vec3> Face(Box3 b, int axis, int sign, int n)
    {
        var pts = new List<Vec3>();
        double[] half = { b.Size.X / 2, b.Size.Y / 2, b.Size.Z / 2 };
        for (int i = 0; i < n; i++)
            for (int j = 0; j < n; j++)
            {
                double u = (i + 0.5) / n * 2 - 1, v = (j + 0.5) / n * 2 - 1;
                double[] l = new double[3];
                int a1 = (axis + 1) % 3, a2 = (axis + 2) % 3;
                l[axis] = sign * half[axis]; l[a1] = u * half[a1]; l[a2] = v * half[a2];
                pts.Add(b.World(new Vec3(l[0], l[1], l[2])));
            }
        return pts;
    }

    /// <summary>A detection whose v3 grid of depth samples hits the given points, one per sample, as on the headset.</summary>
    static FrameDetection DetOn(string label, List<Vec3> pts)
    {
        var d = new FrameDetection { ClassId = 0, Label = label, Score = 0.8, X1 = 300, Y1 = 200, X2 = 700, Y2 = 800 };
        var px = AutoScan.SamplePixels(300, 200, 700, 800, S3);
        for (int i = 0; i < px.Count; i++)
        {
            Vec3? h = i < pts.Count ? pts[i] : (Vec3?)null;
            d.Samples.Add(new DepthSample { U = px[i][0], V = px[i][1], Status = h.HasValue ? "hit" : "no_hit", Point = h });
        }
        return d;
    }

    static readonly AutoScanSettings S3 = AutoScanSettings.From(JParse.Parse(File.ReadAllText(Path.Combine("perception", "autoscan", "autoscan.default.v3.json"))));

    /// <summary>9. The 3D boxes (v3).</summary>
    static void BoxChecks(Func<string, Schema> schema, string sampleDir)
    {
        Console.WriteLine("9. The 3D boxes (v3)");
        var s3 = AutoScanSettings.From(JParse.Parse(File.ReadAllText(Path.Combine("perception", "autoscan", "autoscan.default.v3.json"))));
        var map4 = JParse.Parse(File.ReadAllText(Path.Combine("perception", "bedroom", "category-map.bedroom.v4.json")));
        Check("the v3 settings: boxes, a 5 by 5 grid, completion, a side at most 1.5 priors, furniture standing on the floor",
              s3.Boxes && s3.SampleGrid == 5 && s3.BoxCompletion && s3.BoxMaxOverPrior == 1.5 && s3.FloorCategories.SetEquals(new[] { "bed", "chair", "couch", "table" }));
        var grid = AutoScan.SamplePixels(100, 200, 600, 700, s3);
        Check("depth is sampled at the 25 cell centres of the detection's box", grid.Count == 25 && grid[0][0] == 150 && grid[0][1] == 250 && grid[24][0] == 550 && grid[24][1] == 650);

        var truth = new Box3 { Centre = new Vec3(1.0, 0.45, 2.0), Size = new Vec3(0.6, 0.9, 0.45), YawDeg = 25 };
        var pts = new List<Vec3>();
        for (int ax = 0; ax < 3; ax++) foreach (int sg in new[] { -1, 1 }) pts.AddRange(Face(truth, ax, sg, 6));
        for (int k = 0; k < 4; k++) pts.Add(new Vec3(3 + k * 0.1, 1.5, -1));   // stray rays far away
        Box3 fit = Boxes.Fit(pts, 0.03, true, 0);
        Check("a box turned 25 degrees is fitted from its surface points, stray rays trimmed: its turn within 1 degree, sides within 4 cm",
              Math.Abs(fit.YawDeg - 25) <= 1 && Math.Abs(fit.Size.X - 0.6) < 0.04 && Math.Abs(fit.Size.Z - 0.45) < 0.04 && Math.Abs(fit.Size.Y - 0.9) < 0.02
              && (fit.Centre - truth.Centre).Norm < 0.04, fit.YawDeg + " " + fit.Size.X.ToString("0.000") + " " + fit.Size.Y.ToString("0.000") + " " + fit.Size.Z.ToString("0.000"));

        var ell = Face(truth, 2, -1, 6).Concat(Face(truth, 0, 1, 6)).ToList();   // the front and one side: an L of points
        Box3 ellFit = Boxes.Fit(ell, 0.03, true, 0);
        Check("a box seen from its front and one side, an L of points, is fitted turned as it is, not along the L's diagonal, around what was seen",
              Math.Abs(ellFit.YawDeg - 25) <= 1 && Math.Abs(ellFit.Size.X - 0.55) < 0.02 && Math.Abs(ellFit.Size.Z - 0.4125) < 0.02,   // cell centres stop short of the unseen corner
              ellFit.YawDeg + " " + ellFit.Size.X.ToString("0.000") + " " + ellFit.Size.Z.ToString("0.000"));
        var front = Face(truth, 2, -1, 6);   // only the face toward -z seen, from a camera on that side
        Vec3 viewer = truth.World(new Vec3(0, 0.5, -2.0));
        Box3 thin = Boxes.Fit(front, 0.03, true, 0);
        Box3 whole = Boxes.Complete(thin, new List<double[]> { new[] { 0.6, 0.45, 0.9 } }, new List<Vec3> { viewer }, 1.5, new Vec3(0, 0, 0));
        Vec3 depthLocal = truth.Local(whole.Centre);
        Check("a box seen from one side only is completed to the prior, its hidden half grown away from the viewer",
              whole.Completed && Math.Abs(whole.Size.X - 0.6) < 0.04 && Math.Abs(whole.Size.Z - 0.45) < 0.04 && Math.Abs(depthLocal.Z) < 0.03,
              whole.Size.Z.ToString("0.000") + " centre " + depthLocal.Z.ToString("0.000"));
        Box3 stretched = new Box3 { Centre = new Vec3(0, 0.8, 0), Size = new Vec3(0.36, 0.04, 0.07), YawDeg = 0 };
        Box3 mouse = Boxes.Complete(stretched, new List<double[]> { new[] { 0.07, 0.15, 0.04 } }, new List<Vec3> { new Vec3(0, 1.5, -1) }, 1.5, new Vec3(0.1, 0, 0));
        Check("a small thing's side stretched by the desk around it shrinks to its measured size, centred on its points",
              Math.Abs(mouse.Size.X - 0.15) < 1e-9 && Math.Abs(mouse.Centre.X - 0.1) < 1e-9, mouse.Size.X + " at " + mouse.Centre.X);

        double[] up = { Math.Sin(-Math.PI / 4), 0, 0, Math.Cos(-Math.PI / 4) };
        if (RoomModel.Rotate(up, new Vec3(0, 0, 1)).Y < 0) up = new[] { Math.Sin(Math.PI / 4), 0, 0, Math.Cos(Math.PI / 4) };
        var standing = new RoomBox { Id = "anchor-bed", MetaLabel = "BED", Centre = new Vec3(-1, 0.125, 2), Size = new Vec3(2.0, 1.6, 0.25), Rotation = up };
        Box3 bedUp = Boxes.Upright(standing);
        Check("one of Meta's boxes standing on its own z axis, as Space Setup's do: upright, its height that axis, its label on its top",
              Math.Abs(bedUp.Size.Y - 0.25) < 1e-9 && Math.Abs(Math.Max(bedUp.Size.X, bedUp.Size.Z) - 2.0) < 1e-9 && Math.Abs(bedUp.Top.Y - 0.25) < 1e-9,
              bedUp.Size.X + " " + bedUp.Size.Y + " " + bedUp.Size.Z);

        var room = new RoomModel { Status = "loaded", FloorY = 0, CeilingY = 2.6 };
        foreach (var c in new[] { new[] { -3.0, -1.0 }, new[] { 3.0, -1.0 }, new[] { 3.0, 5.0 }, new[] { -3.0, 5.0 } }) room.FloorXz.Add(c);
        room.Boxes.Add(standing);
        var screen = new RoomBox { Id = "anchor-screen", MetaLabel = "SCREEN", Centre = new Vec3(1.5, 1.1, 3.5), Size = new Vec3(0.62, 0.1, 0.37), Rotation = up };
        room.Boxes.Add(screen);
        var a = new AutoScan("scan.box.1", s3, map4, Priors, "yolox_nano_coco.se1", new string('c', 64), "quest_left", Fx);
        var lines = new List<string> { PyJson.Canonical(a.Header("f00d", 1280, 960, Fx, 643, 481, 416)), PyJson.Canonical(room.ToJson()) };
        a.SetRoom(room);
        var chair = new Box3 { Centre = new Vec3(0.5, 0.45, 1.5), Size = new Vec3(0.55, 0.9, 0.45), YawDeg = 10 };
        Vec3 camFront = chair.World(new Vec3(0, 1.0, -2.0)), camSide = chair.World(new Vec3(2.0, 1.0, 0));
        var frames = new List<FrameRecord>();
        for (int n = 1; n <= 3; n++)
            frames.Add(Frame(n, camFront + new Vec3(0.1 * n, 0, 0), DetOn("chair", Face(chair, 2, -1, 5)),
                             DetOn("couch", Face(new Box3 { Centre = new Vec3(-1, 0.41, 2), Size = new Vec3(0.8, 0.02, 0.6) }, 1, 1, 5)),
                             DetOn("tv", Face(new Box3 { Centre = screen.Centre, Size = new Vec3(0.6, 0.35, 0.01) }, 2, -1, 5))));
        frames.Add(Frame(4, camSide, DetOn("chair", Face(chair, 0, 1, 5))));
        foreach (var f in frames) { lines.Add(PyJson.Canonical(f.ToJson())); a.Add(f); }
        var r = a.Snapshot(9);
        var objs = r.Scene["objects"].Items;
        string cats = string.Join(",", objs.Select(o => o["category"]["state"].Text == "known" ? Cat(o) : "?"));
        Check("the scene: the bed and the monitor as Meta's boxes, and one chair, seen from the front and then the side", cats == "bed,chair,television", cats);
        JNode chairObj = objs.Single(o => o["category"]["state"].Text == "known" && Cat(o) == "chair");
        Check("the side view landed inside the chair's box and joined it: one chair seen in 4 frames from 2 places",
              chairObj["observation"]["views"]["keyframes"].Num == 4 && chairObj["observation"]["views"]["positions"].Num == 2);
        var cq = chairObj["geometry"]["rotation_xyzw"]["value"].Items;
        double chairTurn = -2 * Math.Atan2(cq[2].Num, cq[3].Num) * 180 / Math.PI;   // the scene's turn about +z is the opposite of Unity's about +y
        var cs = chairObj["geometry"]["size_m"]["value"].Items;
        Check("its box is the chair's: turned 10 degrees, 0.55 by 0.45 m", Math.Abs(((chairTurn - 10) % 90 + 90) % 90) <= 1.5 || Math.Abs(((chairTurn - 10) % 90 + 90) % 90 - 90) <= 1.5
              && Math.Abs(Math.Max(cs[0].Num, cs[1].Num) - 0.55) < 0.05 && Math.Abs(Math.Min(cs[0].Num, cs[1].Num) - 0.45) < 0.05,
              chairTurn.ToString("0.0") + " " + cs[0].Text + " x " + cs[1].Text);
        JNode rb = r.Trace["room"]["boxes"];
        Check("the monitor's sightings went into Meta's screen box, which Meta named; no second television",
              rb.Items.Single(b => b["id"].Text == "anchor-screen")["votes"]["television"].Num == 3 && objs.Count(o => o["category"]["state"].Text == "known" && Cat(o) == "television") == 1);
        Check("a pile on the mattress, called a couch, 16 cm above the bed's box, counts as the bed; no couch",
              rb.Items.Single(b => b["id"].Text == "anchor-bed")["votes"]["couch"].Num == 3 && !objs.Any(o => o["category"]["state"].Text == "known" && Cat(o) == "couch"));
        JNode g = chairObj["geometry"];
        Check("the chair's box is in the scene: its size, inferred from the scan, completed to the prior, and its turn about +z",
              g["size_m"]["state"].Text == "known" && g["size_m"]["evidence"]["kind"].Text == "inferred" && g["rotation_xyzw"]["support"].Text == "yaw_only"
              && g["center_m"]["evidence"]["assumptions"].Items.Single().Text == AutoScan.BoxAssumption
              && r.Scene["assumptions"].Items.Any(x => x["assumption_id"].Text == AutoScan.BoxAssumption));
        JNode bedObj = objs.Single(o => o["source_ref"]["source_object_id"].Text == "anchor-bed");
        var bs = bedObj["geometry"]["size_m"]["value"].Items;
        Check("Meta's bed carries its true size, height last (0.25 m), and its turn", Math.Abs(bs[2].Num - 0.25) < 1e-9 && bedObj["geometry"]["rotation_xyzw"]["state"].Text == "known");
        var contract = new Contract(schema);
        var issues = contract.Validate(new List<KeyValuePair<string, JNode>> { new KeyValuePair<string, JNode>("scene", r.Scene), new KeyValuePair<string, JNode>("map", map4) });
        Check("the scene with its boxes passes the contract (scene v3, perceived) with the bedroom's map v4", !issues.Any(i => i.IsError),
              string.Join(", ", issues.Where(i => i.IsError).Select(i => i.Code + " " + i.Path)));
        var shown = a.ConfirmedBoxes();
        Check("one box to draw per object: Meta's named by Meta, the chair by the scan", shown.Count == 3 && shown.Count(x => x.Kind == "meta") == 2 && shown.Count(x => x.Kind == "scan") == 1);

        var b2 = AutoScan.FromHeader(JParse.Parse(lines[0]), s3, map4, Priors);
        foreach (string line in lines.Skip(1))
        {
            JNode n = JParse.Parse(line);
            if (n["record_type"].Text == RoomModel.RecordType) b2.SetRoom(RoomModel.FromJson(n)); else b2.Add(FrameRecord.FromJson(n));
        }
        var r2 = b2.Snapshot(9);
        Check("replayed from its lines, the scan fits the same boxes: the same scene and trace byte for byte",
              PyJson.Canonical(r2.Scene) == PyJson.Canonical(r.Scene) && PyJson.Canonical(r2.Trace) == PyJson.Canonical(r.Trace));
        if (sampleDir != null)
        {
            Directory.CreateDirectory(sampleDir);
            File.WriteAllLines(Path.Combine(sampleDir, "frames.jsonl"), lines);
            File.WriteAllText(Path.Combine(sampleDir, "scene.json"), PyJson.Canonical(r.Scene) + "\n");
            File.WriteAllText(Path.Combine(sampleDir, "trace.json"), PyJson.Canonical(r.Trace) + "\n");
            Console.WriteLine("  (a sample scan with 3D boxes written to " + sampleDir + ")");
        }
    }

    /// <summary>10. Resting on a desk, and categories that join (v4).</summary>
    static void SupportChecks(Func<string, Schema> schema, string sampleDir)
    {
        Console.WriteLine("10. Resting on a desk, and categories that join (v4)");
        var s4 = AutoScanSettings.From(JParse.Parse(File.ReadAllText(Path.Combine("perception", "autoscan", "autoscan.default.v4.json"))));
        var map4 = JParse.Parse(File.ReadAllText(Path.Combine("perception", "bedroom", "category-map.bedroom.v4.json")));
        Check("the v4 settings: small things take their support's turn; a bottle and a cup, and a keyboard and a laptop, join",
              s4.AlignToSupport.Contains("keyboard") && s4.RestGapM == 0.08 && s4.JoinGroups.Count == 2
              && s4.JoinGroups.Any(g => g.SetEquals(new[] { "bottle", "cup" })) && s4.JoinGroups.Any(g => g.SetEquals(new[] { "keyboard", "laptop" })));
        var room = new RoomModel { Status = "loaded", FloorY = 0, CeilingY = 2.6 };
        foreach (var c in new[] { new[] { -3.0, -1.0 }, new[] { 3.0, -1.0 }, new[] { 3.0, 5.0 }, new[] { -3.0, 5.0 } }) room.FloorXz.Add(c);
        room.Boxes.Add(Box("anchor-desk", "TABLE", new Vec3(1.0, 0.39, 1.0), new Vec3(1.4, 0.78, 0.7), 20));
        var a = new AutoScan("scan.support.1", s4, map4, Priors, "yolox_nano_coco.se1", new string('c', 64), "quest_left", Fx);
        var lines = new List<string> { PyJson.Canonical(a.Header("f00d", 1280, 960, Fx, 643, 481, 416)), PyJson.Canonical(room.ToJson()) };
        a.SetRoom(room);
        var kb = new Box3 { Centre = new Vec3(1.2, 0.795, 0.9), Size = new Vec3(0.38, 0.03, 0.14), YawDeg = 20 };
        var kbPts = new List<Vec3>();   // its middle samples on the keyboard, the outer ring on the desk around it, along the picture's axes
        for (int j = 0; j < 5; j++)
            for (int i = 0; i < 5; i++)
                kbPts.Add(i >= 1 && i <= 3 && j >= 1 && j <= 3 ? kb.World(new Vec3((i - 2) * 0.12, 0.015, (j - 2) * 0.04))
                                                               : new Vec3(kb.Centre.X + (i - 2) * 0.13, 0.78, kb.Centre.Z + (j - 2) * 0.06));
        var tumbler = new Box3 { Centre = new Vec3(1.5, 0.915, 1.15), Size = new Vec3(0.09, 0.27, 0.09), YawDeg = 0 };
        var laptop = new Box3 { Centre = new Vec3(0.6, 0.905, 1.0), Size = new Vec3(0.34, 0.25, 0.25), YawDeg = 20 };
        var frames = new List<FrameRecord>();
        for (int n = 1; n <= 3; n++)
        {
            var dets = new List<FrameDetection> { DetOn("keyboard", kbPts), DetOn(n == 3 ? "bottle" : "cup", Face(tumbler, 2, -1, 5)), DetOn("laptop", Face(laptop, 2, -1, 5)) };
            if (n <= 2) dets.Add(DetOn("keyboard", Face(new Box3 { Centre = laptop.Centre - new Vec3(0, 0.08, 0), Size = new Vec3(0.3, 0.01, 0.2), YawDeg = 20 }, 1, 1, 5)));
            frames.Add(Frame(n, new Vec3(1.0, 1.5, -0.5) + new Vec3(0.1 * n, 0, 0), dets.ToArray()));
        }
        foreach (var f in frames) { lines.Add(PyJson.Canonical(f.ToJson())); a.Add(f); }
        var r = a.Snapshot(3);
        var objs = r.Scene["objects"].Items;
        string cats = string.Join(",", objs.Select(o => o["category"]["state"].Text == "known" ? Cat(o) : "?"));
        Check("the scene: one cup (the tumbler, called a cup twice and a bottle once), one keyboard, the laptop and Meta's desk", cats == "cup,keyboard,laptop,table", cats);
        Func<string, JNode> one = name => objs.Single(o => o["category"]["state"].Text == "known" && Cat(o) == name);
        Func<JNode, double> turn = o => { var q = o["geometry"]["rotation_xyzw"]["value"].Items; return -2 * Math.Atan2(q[2].Num, q[3].Num) * 180 / Math.PI; };
        double kbTurn = turn(one("keyboard"));
        Check("the keyboard, its outer samples on the desk around it, takes the desk's turn (20 degrees)", Math.Abs(((kbTurn - 20) % 90 + 135) % 90 - 45) < 0.5, kbTurn.ToString("0.0"));
        Check("the laptop's own keyboard joined the laptop: one keyboard, the real one on the desk, and the laptop still a laptop",
              one("laptop")["observation"]["views"]["keyframes"].Num == 3 && (Centre(one("keyboard")) - AutoScan.ToScene(kb.Centre)).Norm < 0.2);
        var contract = new Contract(schema);
        var issues = contract.Validate(new List<KeyValuePair<string, JNode>> { new KeyValuePair<string, JNode>("scene", r.Scene), new KeyValuePair<string, JNode>("map", map4) });
        Check("the scene passes the contract with the bedroom's map v4", !issues.Any(i => i.IsError), string.Join(", ", issues.Where(i => i.IsError).Select(i => i.Code + " " + i.Path)));
        var b2 = AutoScan.FromHeader(JParse.Parse(lines[0]), s4, map4, Priors);
        foreach (string line in lines.Skip(1))
        {
            JNode n = JParse.Parse(line);
            if (n["record_type"].Text == RoomModel.RecordType) b2.SetRoom(RoomModel.FromJson(n)); else b2.Add(FrameRecord.FromJson(n));
        }
        var r2 = b2.Snapshot(3);
        Check("replayed from its lines, the same scene and trace byte for byte", PyJson.Canonical(r2.Scene) == PyJson.Canonical(r.Scene) && PyJson.Canonical(r2.Trace) == PyJson.Canonical(r.Trace));
        if (sampleDir != null)
        {
            Directory.CreateDirectory(sampleDir);
            File.WriteAllLines(Path.Combine(sampleDir, "frames.jsonl"), lines);
            File.WriteAllText(Path.Combine(sampleDir, "scene.json"), PyJson.Canonical(r.Scene) + "\n");
            File.WriteAllText(Path.Combine(sampleDir, "trace.json"), PyJson.Canonical(r.Trace) + "\n");
            Console.WriteLine("  (a sample scan with v4 written to " + sampleDir + ")");
        }
    }

    /// <summary>Points on every face of a box, about step apart: a synthetic stand-in for Meta's room mesh.</summary>
    static List<Vec3> Surface(Box3 b, double step)
    {
        var pts = new List<Vec3>();
        int n = Math.Max(3, (int)Math.Ceiling(Math.Max(b.Size.X, Math.Max(b.Size.Y, b.Size.Z)) / step));
        for (int ax = 0; ax < 3; ax++) foreach (int sg in new[] { -1, 1 }) pts.AddRange(Face(b, ax, sg, n));
        return pts;
    }

    /// <summary>11. Boxes from Meta's room mesh (v5).</summary>
    static void MeshChecks(Func<string, Schema> schema, string sampleDir)
    {
        Console.WriteLine("11. Boxes from Meta's room mesh (v5)");
        var s5 = AutoScanSettings.From(JParse.Parse(File.ReadAllText(Path.Combine("perception", "autoscan", "autoscan.default.v5.json"))));
        var map4 = JParse.Parse(File.ReadAllText(Path.Combine("perception", "bedroom", "category-map.bedroom.v4.json")));
        Check("the v5 settings: boxes from Meta's mesh, thinned to 3 cm, clusters linked within 6 cm", s5.UseMesh && s5.MeshVoxelM == 0.03 && s5.MeshLinkM == 0.06 && s5.ClusterMinPoints == 20);
        var room = new RoomModel { Status = "loaded", FloorY = 0, CeilingY = 2.6, MeshStatus = "loaded" };
        foreach (var c in new[] { new[] { -3.0, -1.0 }, new[] { 3.0, -1.0 }, new[] { 3.0, 5.0 }, new[] { -3.0, 5.0 } }) room.FloorXz.Add(c);
        var desk = Box("anchor-desk", "TABLE", new Vec3(1.0, 0.39, 1.0), new Vec3(1.4, 0.78, 0.7), 20);
        room.Boxes.Add(desk);
        for (double x = -3; x <= 3.001; x += 0.1)   // the floor and the four walls
            for (double z = -1; z <= 5.001; z += 0.1) room.MeshPoints.Add(new Vec3(x, 0, z));
        for (double y = 0.1; y <= 2.5; y += 0.1)
        {
            for (double x = -3; x <= 3.001; x += 0.1) { room.MeshPoints.Add(new Vec3(x, y, -1)); room.MeshPoints.Add(new Vec3(x, y, 5)); }
            for (double z = -1; z <= 5.001; z += 0.1) { room.MeshPoints.Add(new Vec3(-3, y, z)); room.MeshPoints.Add(new Vec3(3, y, z)); }
        }
        room.MeshPoints.AddRange(Surface(Boxes.Upright(desk), 0.05));   // the desk, already Meta's box
        var chair = new Box3 { Centre = new Vec3(-1.0, 0.475, 2.5), Size = new Vec3(0.62, 0.95, 0.58), YawDeg = 35 };   // not the prior's size
        var laptop = new Box3 { Centre = new Vec3(0.8, 0.905, 0.9), Size = new Vec3(0.34, 0.25, 0.25), YawDeg = 20 };
        var bin = new Box3 { Centre = new Vec3(2.2, 0.2, 3.8), Size = new Vec3(0.3, 0.4, 0.3), YawDeg = 0 };
        room.MeshPoints.AddRange(Surface(chair, 0.03));
        room.MeshPoints.AddRange(Surface(laptop, 0.03));
        room.MeshPoints.AddRange(Surface(bin, 0.03));
        room.MeshPoints.AddRange(new[] { new Vec3(-2.0, 1.2, 0.5), new Vec3(-2.02, 1.2, 0.5), new Vec3(-2.0, 1.22, 0.5) });   // a speck of noise
        var a = new AutoScan("scan.mesh.1", s5, map4, Priors, "yolox_nano_coco.se1", new string('c', 64), "quest_left", Fx);
        var lines = new List<string> { PyJson.Canonical(a.Header("f00d", 1280, 960, Fx, 643, 481, 416)), PyJson.Canonical(room.ToJson()) };
        a.SetRoom(room);
        var mouse = new Box3 { Centre = new Vec3(1.3, 0.80, 1.1), Size = new Vec3(0.07, 0.04, 0.15), YawDeg = 20 };
        var frames = new List<FrameRecord>();
        for (int n = 1; n <= 3; n++)
            frames.Add(Frame(n, new Vec3(0.3, 1.5, -0.5) + new Vec3(0.1 * n, 0, 0), DetOn("chair", Face(chair, 2, -1, 5)), DetOn("laptop", Face(laptop, 2, -1, 5)),
                             DetOn("mouse", Face(mouse, 1, 1, 5))));
        foreach (var f in frames) { lines.Add(PyJson.Canonical(f.ToJson())); a.Add(f); }
        var r = a.Snapshot(3);
        var objs = r.Scene["objects"].Items;
        string cats = string.Join(",", objs.Select(o => o["category"]["state"].Text == "known" ? Cat(o) : "?"));
        Check("the scene: the chair and the laptop from the mesh, the mouse from live depth, Meta's desk, and the bin nobody named", cats == "chair,laptop,mouse,table,?", cats);
        JNode mt = r.Trace["mesh"];
        Check("the floor, the walls and Meta's desk are taken off the mesh, and the speck of noise dropped",
              mt["used"].Bool && mt["structure"].Num > 4000 && mt["explained"].Num > 0 && mt["dropped"].Num >= 1 && mt["boxes"].Items.Count == 3,
              "structure " + mt["structure"].Text + ", explained " + mt["explained"].Text + ", dropped " + mt["dropped"].Text + ", boxes " + mt["boxes"].Items.Count);
        Func<string, JNode> one = name => objs.Single(o => o["category"]["state"].Text == "known" && Cat(o) == name);
        JNode ch = one("chair");
        var cs = ch["geometry"]["size_m"]["value"].Items.Select(x => x.Num).ToList();
        var cq = ch["geometry"]["rotation_xyzw"]["value"].Items;
        double chTurn = -2 * Math.Atan2(cq[2].Num, cq[3].Num) * 180 / Math.PI;
        Check("the chair's box is measured from the mesh, every side: its own 0.62 by 0.58 by 0.95 m and its 35-degree turn, not the prior's size",
              Math.Abs(Math.Max(cs[0], cs[1]) - 0.62) < 0.04 && Math.Abs(Math.Min(cs[0], cs[1]) - 0.58) < 0.04 && Math.Abs(cs[2] - 0.95) < 0.03
              && Math.Abs(((chTurn - 35) % 90 + 135) % 90 - 45) <= 2 && ch["geometry"]["center_m"]["evidence"]["assumptions"].Items.Single().Text == AutoScan.MeshAssumption
              && !ch["geometry"]["size_m"]["evidence"]["assumptions"].Items.Any(x => x.Text == AutoScan.CompletionAssumption),
              string.Join(" x ", cs.Select(x => x.ToString("0.00"))) + " at " + chTurn.ToString("0.0"));
        Check("the laptop on the desk is its own mesh box, named by its sightings", one("laptop")["source_ref"]["source_id"].Text == AutoScan.RoomSourceId
              && one("laptop")["geometry"]["center_m"]["evidence"]["assumptions"].Items.Single().Text == AutoScan.MeshAssumption);
        JNode binObj = objs.Single(o => o["category"]["state"].Text == "unknown" && o["source_ref"]["source_id"].Text == AutoScan.RoomSourceId);
        var bs = binObj["geometry"]["size_m"]["value"].Items.Select(x => x.Num).ToList();
        Check("the bin nobody named is still an obstacle with a measured box, its category unknown", Math.Abs(bs[0] - 0.3) < 0.04 && Math.Abs(bs[1] - 0.3) < 0.04 && Math.Abs(bs[2] - 0.4) < 0.04,
              string.Join(" x ", bs.Select(x => x.ToString("0.00"))));
        Check("the mouse, flattened into the desk in the mesh, keeps its box from live depth", one("mouse")["source_ref"]["source_id"].Text == AutoScan.ScanSourceId);
        var shown = a.ConfirmedBoxes();
        Check("boxes to draw: the mesh's named ones, the unnamed one, Meta's desk and the live-depth mouse",
              shown.Count(x => x.Kind == "mesh") == 2 && shown.Count(x => x.Kind == "unnamed") == 1 && shown.Count(x => x.Kind == "meta") == 1 && shown.Count(x => x.Kind == "scan") == 1);
        var contract = new Contract(schema);
        var issues = contract.Validate(new List<KeyValuePair<string, JNode>> { new KeyValuePair<string, JNode>("scene", r.Scene), new KeyValuePair<string, JNode>("map", map4) });
        Check("the scene passes the contract", !issues.Any(i => i.IsError), string.Join(", ", issues.Where(i => i.IsError).Select(i => i.Code + " " + i.Path)));
        var b2 = AutoScan.FromHeader(JParse.Parse(lines[0]), s5, map4, Priors);
        foreach (string line in lines.Skip(1))
        {
            JNode n = JParse.Parse(line);
            if (n["record_type"].Text == RoomModel.RecordType) b2.SetRoom(RoomModel.FromJson(n)); else b2.Add(FrameRecord.FromJson(n));
        }
        var r2 = b2.Snapshot(3);
        Check("replayed from its lines, the mesh included, the same scene and trace byte for byte", PyJson.Canonical(r2.Scene) == PyJson.Canonical(r.Scene) && PyJson.Canonical(r2.Trace) == PyJson.Canonical(r.Trace));
        if (sampleDir != null)
        {
            Directory.CreateDirectory(sampleDir);
            File.WriteAllLines(Path.Combine(sampleDir, "frames.jsonl"), lines);
            File.WriteAllText(Path.Combine(sampleDir, "scene.json"), PyJson.Canonical(r.Scene) + "\n");
            File.WriteAllText(Path.Combine(sampleDir, "trace.json"), PyJson.Canonical(r.Trace) + "\n");
            Console.WriteLine("  (a sample scan with Meta's mesh written to " + sampleDir + ")");
        }
    }

    static string bedObjLabel(List<JNode> objs) { return objs.Single(o => o["source_ref"]["source_object_id"].Text == "anchor-bed")["source_ref"]["source_label"].Text; }

    static RoomBox Box(string id, string label, Vec3 c, Vec3 size, double yawDeg = 0)
    {
        double h = yawDeg * Math.PI / 360;
        return new RoomBox { Id = id, MetaLabel = label, Centre = c, Size = size, Rotation = new[] { 0, Math.Sin(h), 0, Math.Cos(h) } };
    }

    /// <summary>8. Meta's room: boxes named by Meta or the detector, small things on boxes, the room's walls, the replay.</summary>
    static void Room(Func<string, Schema> schema, string sampleDir)
    {
        Console.WriteLine("8. Meta's room");
        var s2 = AutoScanSettings.From(JParse.Parse(File.ReadAllText(Path.Combine("perception", "autoscan", "autoscan.default.v2.json"))));
        var map3 = JParse.Parse(File.ReadAllText(Path.Combine("perception", "bedroom", "category-map.bedroom.v3.json")));
        Check("the v2 settings: Space Setup at every start, furniture votes for boxes, three frames", s2.SpaceSetup == "every_start" && s2.MinFrames == 3
              && s2.BoxCategories.SetEquals(new[] { "bed", "chair", "couch", "table" }) && s2.MinBoxVotes == 2);
        var room = new RoomModel { Status = "loaded", FloorY = 0, CeilingY = 2.6 };
        foreach (var c in new[] { new[] { -3.0, -1.0 }, new[] { 3.0, -1.0 }, new[] { 3.0, 5.0 }, new[] { -3.0, 5.0 } }) room.FloorXz.Add(c);
        Vec3 bedC = new Vec3(-1.2, 0.3, 2.5), chairC = new Vec3(1.0, 0.45, 3.0), deskC = new Vec3(0.4, 0.37, 1.8);
        room.Boxes.Add(Box("anchor-bed", "BED", bedC, new Vec3(2.0, 0.6, 1.6)));
        room.Boxes.Add(Box("anchor-chair", "OTHER", chairC, new Vec3(0.6, 0.9, 0.6), 30));
        room.Boxes.Add(Box("anchor-desk", "OTHER", deskC, new Vec3(1.2, 0.74, 0.6)));
        room.Boxes.Add(Box("anchor-pile", "OTHER", new Vec3(-2.5, 0.2, 0.0), new Vec3(0.5, 0.4, 0.5)));
        room.Boxes.Add(Box("anchor-once", "OTHER", new Vec3(2.5, 0.45, 0.5), new Vec3(0.5, 0.9, 0.5)));

        RoomBox turned = room.Boxes[1];
        Vec3 corner = chairC + RoomModel.Rotate(turned.Rotation, new Vec3(0.29, 0.44, -0.29));
        Vec3 beyond = chairC + RoomModel.Rotate(turned.Rotation, new Vec3(0.35, 0, 0));
        Check("a point just inside a turned box's corner is inside it; one 5 cm beyond its turned side is 5 cm out",
              turned.Outside(corner) == 0 && Math.Abs(turned.Outside(beyond) - 0.05) < 1e-9, turned.Outside(beyond).ToString("0.0000"));
        Check("in the room: inside the outline, or within 25 cm of it, and between floor and ceiling",
              room.InRoom(new Vec3(0, 1, 2), 0.25) && room.InRoom(new Vec3(3.2, 1, 2), 0.25) && !room.InRoom(new Vec3(3.4, 1, 2), 0.25)
              && !room.InRoom(new Vec3(0, -0.3, 2), 0.25) && !room.InRoom(new Vec3(0, 2.9, 2), 0.25));
        double[] q90 = { 0, Math.Sin(Math.PI / 4), 0, Math.Cos(Math.PI / 4) }, qs = RoomModel.RotationToScene(q90);
        Vec3 xTurned = RoomModel.Rotate(qs, new Vec3(1, 0, 0)), unityTurned = AutoScan.ToScene(RoomModel.Rotate(q90, new Vec3(1, 0, 0)));
        Check("a rotation in scene axes turns a direction exactly as the tracking rotation does (90 degrees: x to -y)",
              (xTurned - unityTurned).Norm < 1e-12 && (xTurned - new Vec3(0, -1, 0)).Norm < 1e-12, xTurned.X + "," + xTurned.Y + "," + xTurned.Z);
        Check("the room's line reads back exactly as written", PyJson.Canonical(RoomModel.FromJson(room.ToJson()).ToJson()) == PyJson.Canonical(room.ToJson()));

        var a = new AutoScan("scan.room.1", s2, map3, Priors, "yolox_nano_coco.se1", new string('c', 64), "quest_left", Fx);
        var lines = new List<string> { PyJson.Canonical(a.Header("f00d", 1280, 960, Fx, 643, 481, 416)), PyJson.Canonical(room.ToJson()) };
        a.SetRoom(room);
        Vec3 camA = new Vec3(0, 1.5, 0), camB = new Vec3(0.8, 1.5, 0.2), camC = new Vec3(-0.6, 1.5, 0.3), laptop = new Vec3(0.4, 0.80, 1.8), freeChair = new Vec3(-0.5, 0.45, 4.0);
        var frames = new List<FrameRecord>();
        foreach (var cam in new[] { camA, camB, camC })   // three frames: v2 needs three before a depth object is in the scene
        {
            int n = frames.Count + 1;
            var dets = new List<FrameDetection> {
                Det("couch", 0.7, 100, 500, 600, 700, Around(bedC + new Vec3(0, 0.25, -0.7), 0.03)),
                Det("chair", 0.8, 800, 300, 1000, 600, Around(chairC + new Vec3(0, 0.2, -0.25), 0.02)),
                Det("laptop", 0.75, 640, 400, 700, 450, Around(laptop, 0.01)),
                Det("chair", 0.7, 300, 300, 420, 500, Around(freeChair, 0.02)),
                Det("bottle", 0.6, 1100, 400, 1120, 460, Around(new Vec3(3.6, 0.9, 2.0), 0.01)) };
            if (n == 1) dets.Add(Det("chair", 0.7, 1150, 300, 1250, 500, Around(new Vec3(2.5, 0.6, 0.35), 0.02)));
            frames.Add(Frame(n, cam, dets.ToArray()));
        }
        foreach (var f in frames) { lines.Add(PyJson.Canonical(f.ToJson())); a.Add(f); }
        var r = a.Snapshot(5);
        var objs = r.Scene["objects"].Items;
        string cats = string.Join(",", objs.Select(o => o["category"]["state"].Text == "known" ? Cat(o) : "?"));
        Check("the scene: Meta's bed, the turned box's chair, the free chair, the laptop, and three unnamed boxes", cats == "bed,chair,chair,laptop,?,?,?", cats);
        JNode boxes = r.Trace["room"]["boxes"];
        Func<string, JNode> boxOf = id => boxes.Items.Single(b => b["id"].Text == id);
        Check("Meta's name wins: the bed box is a bed though the detector called it a couch twice",
              boxOf("anchor-bed")["named_by"].Text == "meta" && boxOf("anchor-bed")["category"].Text == "bed" && boxOf("anchor-bed")["votes"]["couch"].Num == 3
              && bedObjLabel(objs) == "meta:BED");
        JNode chairBox = objs.Single(o => o["source_ref"]["source_object_id"].Text == "anchor-chair");
        Check("an OTHER box the detector saw as a chair in three frames from three places is a chair, placed from several positions",
              boxOf("anchor-chair")["named_by"].Text == "detector" && Cat(chairBox) == "chair" && chairBox["observation"]["association"]["status"].Text == "multi_position"
              && chairBox["observation"]["views"]["keyframes"].Num == 3 && chairBox["category"]["evidence"]["source"].Text == AutoScan.ScanSourceId
              && chairBox["source_ref"]["source_label"].Text == "chair");
        Check("a box seen as a chair only once stays unnamed (two votes needed), its one vote in the trace",
              boxOf("anchor-once")["named_by"].Text == "none" && boxOf("anchor-once")["votes"]["chair"].Num == 1);
        Check("the laptop on the desk is its own object, and doesn't name the desk's box", boxOf("anchor-desk")["named_by"].Text == "none"
              && boxOf("anchor-desk")["frames"].Num == 0 && objs.Count(o => o["category"]["state"].Text == "known" && Cat(o) == "laptop") == 1);
        Check("a chair outside every box is placed by depth", objs.Count(o => o["source_ref"]["source_id"].Text == AutoScan.ScanSourceId && Cat(o) == "chair") == 1);
        Check("a bottle beyond the walls (a mirror's reflection) is left out as outside the room", r.Trace["left_out"]["outside_room"].Num == 3);
        JNode bedObj = objs.Single(o => o["source_ref"]["source_object_id"].Text == "anchor-bed");
        Check("a box's object: Meta's room as its source, its centre the box's centre in scene axes, naming the meta_volume assumption",
              bedObj["source_ref"]["source_id"].Text == AutoScan.RoomSourceId && (Centre(bedObj) - AutoScan.ToScene(bedC)).Norm < 1e-9
              && bedObj["geometry"]["center_m"]["evidence"]["assumptions"].Items.Single().Text == AutoScan.VolumeAssumption
              && r.Scene["sources"].Items.Any(x => x["source_id"].Text == AutoScan.RoomSourceId && x["kind"].Text == "sensor_capture"));
        JNode pile = objs.Single(o => o["source_ref"]["source_object_id"].Text == "anchor-pile");
        Check("an unnamed box no frame saw: unknown category, counted once by convention", pile["category"]["state"].Text == "unknown"
              && pile["observation"]["views"]["keyframes"].Num == 1 && pile["observation"]["capture_time"]["state"].Text == "unknown");
        var sz = boxOf("anchor-chair")["size_m"].Items;
        Check("the trace keeps each box's true size in scene axes (height last) and its rotation", sz[0].Num == 0.6 && sz[1].Num == 0.6 && sz[2].Num == 0.9
              && Math.Abs(boxOf("anchor-chair")["yaw_deg"].Num + 30) < 1e-6, boxOf("anchor-chair")["yaw_deg"].Text);
        var dets2 = r.Trace["detections"].Items;
        Check("the trace sends each box's sightings to the box's object", dets2.Count(d => d["fate"].Text == "object" && d["object_id"].Text == chairBox["object_id"].Text) == 3);
        var contract = new Contract(schema);
        var issues = contract.Validate(new List<KeyValuePair<string, JNode>> { new KeyValuePair<string, JNode>("scene", r.Scene), new KeyValuePair<string, JNode>("map", map3) });
        Check("the scene passes the contract (scene v3, perceived) with the bedroom's map v3", !issues.Any(i => i.IsError),
              string.Join(", ", issues.Where(i => i.IsError).Select(i => i.Code + " " + i.Path)));
        var labels = a.ConfirmedInTracking();
        Check("labels float over each box's top, unnamed boxes as ?", labels.Any(kv => kv.Key == "bed" && Math.Abs(kv.Value.Y - 0.6) < 1e-9) && labels.Count(kv => kv.Key == "?") == 3);

        JNode h = JParse.Parse(lines[0]);
        var b2 = AutoScan.FromHeader(h, AutoScanSettings.From(JParse.Parse(File.ReadAllText(Path.Combine("perception", "autoscan", "autoscan.default.v2.json")))), map3, Priors);
        foreach (string line in lines.Skip(1))
        {
            JNode n = JParse.Parse(line);
            if (n["record_type"].Text == RoomModel.RecordType) b2.SetRoom(RoomModel.FromJson(n)); else b2.Add(FrameRecord.FromJson(n));
        }
        var r2 = b2.Snapshot(5);
        if (sampleDir != null)   // a sample scan with Meta's room, as the headset writes it, for the PC replay (tools/localize --memory)
        {
            Directory.CreateDirectory(sampleDir);
            File.WriteAllLines(Path.Combine(sampleDir, "frames.jsonl"), lines);
            File.WriteAllText(Path.Combine(sampleDir, "scene.json"), PyJson.Canonical(r.Scene) + "\n");
            File.WriteAllText(Path.Combine(sampleDir, "trace.json"), PyJson.Canonical(r.Trace) + "\n");
            Console.WriteLine("  (a sample scan with Meta's room written to " + sampleDir + ")");
        }
        Check("replayed from its lines, room included, the scan writes the same scene and trace byte for byte",
              PyJson.Canonical(r2.Scene) == PyJson.Canonical(r.Scene) && PyJson.Canonical(r2.Trace) == PyJson.Canonical(r.Trace));
    }

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

        Room(schema, args.Length > 1 ? Path.Combine(args[1], "room") : null);
        BoxChecks(schema, args.Length > 1 ? Path.Combine(args[1], "boxes") : null);
        SupportChecks(schema, args.Length > 1 ? Path.Combine(args[1], "support") : null);
        MeshChecks(schema, args.Length > 1 ? Path.Combine(args[1], "mesh") : null);

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
