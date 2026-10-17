// The automatic scan rule's checks (A4 Delivery 1, stage 3; docs/scan.md) on simulated head movements at 72 Hz: holds,
// turns and walks, with optional deterministic tremor, a detector that may stay busy after each keyframe, and lost
// tracking. Each keyframe the rule asks for is taken at once. Expectations come from the movements, worked out by hand.
// Run: dotnet run --project tools/csharp-tests/Scan -- schemas
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using SecondEyes.Grounding.Prompting;
using SecondEyes.Grounding.Resolution;
using SecondEyes.Perception.Frames;
using SecondEyes.Perception.Scan;

static class Program
{
    static int pass, fail;
    static void Check(string n, bool ok, string d = "")
    {
        if (ok) { pass++; Console.WriteLine("  ok    " + n); }
        else { fail++; Console.WriteLine("  FAIL  " + n + (d.Length > 0 ? "  [" + d + "]" : "")); }
    }

    sealed class Sim
    {
        const double Dt = 1.0 / 72;
        readonly KeyframeRule rule;
        int frame;
        public double T, Yaw, Pitch, Nod, Tremor, BusyFor, BusyUntil = -1;
        public bool Tracked = true;
        public Vec3 P = new Vec3(0, 1.6, 0);
        public readonly List<(double t, int pos, int step, double yaw)> Taken = new List<(double, int, int, double)>();
        public Sim(ScanSettings s) { rule = new KeyframeRule(s); }
        public KeyframeRule Rule => rule;
        public Sim Hold(double sec) { return Run(sec, 0, 0, 0); }
        public Sim Turn(double deg, double sec) { return Run(sec, deg, 0, 0); }
        public Sim Walk(double dx, double dz, double sec) { return Run(sec, 0, dx, dz); }
        public Sim Tilt(double deg, double sec) { return Run(sec, 0, 0, 0, deg); }
        Sim Run(double sec, double deg, double dx, double dz, double dpitch = 0)   // the exact totals, split evenly over the frames
        {
            int n = Math.Max(1, (int)Math.Round(sec / Dt));
            for (int i = 0; i < n; i++)
            {
                T += Dt; Yaw += deg / n; Pitch += dpitch / n; frame++;
                P = new Vec3(P.X + dx / n, P.Y, P.Z + dz / n);
                double seen = KeyframeRule.Wrap(Yaw + Tremor * Math.Sin(frame * 0.7));
                if (rule.Update(T, P, seen, Pitch + Nod * Math.Sin(frame * 0.9), Tracked, T < BusyUntil))
                {
                    rule.Taken();
                    Taken.Add((T, rule.Position, rule.Step, KeyframeRule.Wrap(Yaw)));
                    BusyUntil = T + BusyFor;
                }
            }
            return this;
        }
    }

    static int Main(string[] args)
    {
        string schemas = args.Length > 0 ? args[0] : "schemas";
        JNode cfg = JParse.Parse(File.ReadAllText(Path.Combine("perception", "scan", "scan.default.v1.json")));
        var errors = new Schema(JParse.Parse(File.ReadAllText(Path.Combine(schemas, "scan-config.v1.json")))).Errors(cfg);
        Check("the default settings pass their schema", errors.Count == 0, string.Join(", ", errors.Select(e => e.Keyword + " " + e.Path)));
        JNode cfg2 = JParse.Parse(File.ReadAllText(Path.Combine("perception", "scan", "scan.default.v2.json")));
        var errors2 = new Schema(JParse.Parse(File.ReadAllText(Path.Combine(schemas, "scan-config.v1.json")))).Errors(cfg2);
        Check("the second settings pass it too, a new spot counting from 1.5 m", errors2.Count == 0 && ScanSettings.From(cfg2).NewPositionM == 1.5);
        ScanSettings S = ScanSettings.From(cfg);
        Func<int, int, ScanSettings> limited = (perPosition, positions) =>
        {
            ScanSettings x = ScanSettings.From(cfg); x.MaxKeyframesPerPosition = perPosition; x.MaxPositions = positions; return x;
        };

        Console.WriteLine("1-2. A full circle in 30-degree turns at one spot, then back to the start");
        var a = new Sim(S).Hold(1);
        for (int k = 1; k < 12; k++) a.Turn(30, 0.8).Hold(1);
        Check("twelve keyframes, all at position 0, steps 0 to 11", a.Taken.Count == 12 && a.Taken.All(x => x.pos == 0)
              && a.Taken.Select(x => x.step).SequenceEqual(Enumerable.Range(0, 12)), string.Join(" ", a.Taken.Select(x => x.pos + "/" + x.step)));
        Check("each facing its own 30 degrees, within 1 degree", a.Taken.Select((x, i) => Math.Abs(KeyframeRule.Wrap(x.yaw - 30 * i)) <= 1).All(b => b));
        a.Turn(30, 0.8).Hold(1);
        Check("turning back to the first direction adds nothing", a.Taken.Count == 12);

        Console.WriteLine("3. Back and forth between two directions");
        var b3 = new Sim(S).Hold(1).Turn(30, 0.8).Hold(1).Turn(-30, 0.8).Hold(1).Turn(30, 0.8).Hold(1);
        Check("two keyframes", b3.Taken.Count == 2, b3.Taken.Count.ToString());

        Console.WriteLine("4. A slow continuous turn (20 degrees a second) is never still");
        Check("no keyframe", new Sim(S).Turn(360, 18).Taken.Count == 0);

        Console.WriteLine("5. Tremor and sway");
        Check("a tremor of 0.5 degrees still counts as holding still", new Sim(S) { Tremor = 0.5 }.Hold(1).Taken.Count == 1);
        Check("a sway of 2.5 degrees does not", new Sim(S) { Tremor = 2.5 }.Hold(2).Taken.Count == 0);

        Console.WriteLine("6. Walking to a new spot");
        var w = new Sim(S).Hold(1).Walk(1.2, 0, 2).Hold(1);
        Check("walking 1.2 m and holding still starts position 1 with a keyframe", w.Taken.Count == 2 && w.Taken[1].pos == 1 && w.Taken[1].step == 0,
              string.Join(" ", w.Taken.Select(x => x.pos + "/" + x.step)));
        Check("walking 0.6 m facing the same way adds nothing", new Sim(S).Hold(1).Walk(0.6, 0, 1).Hold(1).Taken.Count == 1);

        Console.WriteLine("7. A busy detector");
        var busy = new Sim(S) { BusyFor = 1.5 }.Hold(1).Turn(30, 0.3).Hold(2.5);
        Check("the second keyframe waits until the first is read, then comes once", busy.Taken.Count == 2 && busy.Taken[1].t >= busy.Taken[0].t + 1.5,
              busy.Taken.Count + " keyframes; " + string.Join(", ", busy.Taken.Select(x => x.t.ToString("0.00"))));

        Console.WriteLine("8. Lost tracking");
        var lost = new Sim(S) { Tracked = false }.Hold(2);
        Check("no keyframe while tracking is lost", lost.Taken.Count == 0 && lost.Rule.Hint.StartsWith("tracking lost"));
        lost.Tracked = true; lost.Hold(1);
        Check("one once it returns and the head is still", lost.Taken.Count == 1);

        Console.WriteLine("9. Turning across +-180 degrees");
        var wrap = new Sim(S) { Yaw = 170 }.Hold(1).Turn(20, 0.5).Hold(1);
        Check("20 degrees across the seam is too little", wrap.Taken.Count == 1);
        wrap.Turn(10, 0.3).Hold(1);
        Check("30 degrees across it is a new direction, at -160", wrap.Taken.Count == 2 && Math.Abs(wrap.Taken[1].yaw + 160) <= 1,
              string.Join(", ", wrap.Taken.Select(x => x.yaw.ToString("0.0"))));

        Console.WriteLine("10. Taken() with nothing due");
        bool threw = false;
        try { new KeyframeRule(S).Taken(); } catch (InvalidOperationException) { threw = true; }
        Check("is refused", threw);

        Console.WriteLine("11-12. The limits");
        var few = new Sim(limited(3, 4)).Hold(1);
        for (int k = 0; k < 5; k++) few.Turn(30, 0.8).Hold(1);
        Check("three keyframes at most per position", few.Taken.Count == 3);
        Check("then the hint says to move", few.Rule.Hint.StartsWith("move about 1 m"), few.Rule.Hint);
        var two = new Sim(limited(12, 2)).Hold(1).Walk(1.2, 0, 2).Hold(1).Walk(1.2, 0, 2).Hold(1);
        Check("two positions at most", two.Taken.Count == 2 && two.Taken.Max(x => x.pos) == 1);
        Check("then the hint says the scan is complete", two.Rule.Hint.StartsWith("the scan is complete"), two.Rule.Hint);

        Console.WriteLine("13. The camera's model, fitted from its own viewport rays");
        Func<bool, double, List<double[]>> rays = (bottomLeft, noisePx) =>
        {
            var list = new List<double[]>();
            int k = 0;
            foreach (double u in new[] { 0.1, 0.5, 0.9 })
                foreach (double v in new[] { 0.1, 0.5, 0.9 })
                {
                    double px = u * 1280 + noisePx * Math.Sin(++k * 1.7), py = (bottomLeft ? 1 - v : v) * 960 + noisePx * Math.Cos(k * 2.3);
                    double x = (px - 643.0) / 870.0, y = -(py - 478.0) / 870.0, n = Math.Sqrt(x * x + y * y + 1);
                    list.Add(new[] { u, v, x / n, y / n, 1 / n });
                }
            return list;
        };
        CameraFit.Result b = CameraFit.Fit(rays(true, 0), 1280, 960);
        Check("a pinhole with Unity's bottom-left viewport origin is recovered exactly", b != null && b.Origin == "bottom-left" && Math.Abs(b.Fx - 870) < 1e-6
              && Math.Abs(b.Fy - 870) < 1e-6 && Math.Abs(b.Cx - 643) < 1e-6 && Math.Abs(b.Cy - 478) < 1e-6 && b.RmsPx < 1e-6,
              b == null ? "none" : b.Origin + " " + b.Fx + " " + b.Fy + " " + b.Cx + " " + b.Cy);
        CameraFit.Result tl = CameraFit.Fit(rays(false, 0), 1280, 960);
        Check("with a top-left origin, the fit finds that origin", tl != null && tl.Origin == "top-left" && Math.Abs(tl.Fy - 870) < 1e-6 && Math.Abs(tl.Cy - 478) < 1e-6);
        CameraFit.Result noisy = CameraFit.Fit(rays(true, 0.5), 1280, 960);
        Check("with half a pixel of noise it stays within 1 percent, the residual under a pixel", noisy != null && Math.Abs(noisy.Fx / 870 - 1) < 0.01
              && Math.Abs(noisy.Fy / 870 - 1) < 0.01 && noisy.RmsPx < 1, noisy == null ? "none" : noisy.RmsPx.ToString("0.000"));
        Check("too few samples give no fit", CameraFit.Fit(rays(true, 0).GetRange(0, 2), 1280, 960) == null);
        double tfx, tfy, tcx, tcy;
        CameraFit.Crop(869.5435, 869.5435, 643.2365, 638.2825, 1280, 1280, 1280, 960, out tfx, out tfy, out tcx, out tcy);
        Check("the crop model: the square sensor to 1280x960 keeps the focal length and moves cy up 160 rows",
              Math.Abs(tfx - 869.5435) < 1e-9 && Math.Abs(tfy - 869.5435) < 1e-9 && Math.Abs(tcx - 643.2365) < 1e-9 && Math.Abs(tcy - 478.2825) < 1e-9);
        CameraFit.Crop(869.5435, 869.5435, 643.2365, 638.2825, 1280, 1280, 640, 480, out tfx, out tfy, out tcx, out tcy);
        Check("and to 640x480 halves it and crops 80 rows", Math.Abs(tfx - 434.77175) < 1e-9 && Math.Abs(tcy - (638.2825 / 2 - 80)) < 1e-9);

        Console.WriteLine("14. The scan's guidance in view");
        var g = new Sim(S);
        string first = ScanGuide.Text(g.Rule, 0, 0);
        Check("before any keyframe: face part of the room and hold still, no direction done", first.StartsWith("SCAN: face part of the room and hold still")
              && first.Contains("[............]") && first.Contains("spot 1 of up to 4, 0 keyframes"), first.Split('\n')[0]);
        g.Hold(1).Turn(30, 0.8).Hold(1).Turn(30, 0.8).Hold(1).Turn(30, 0.8).Hold(0.2);
        Check("three keyframes 30 degrees apart fill three slots in order", ScanGuide.Directions(g.Rule) == "[###.........]", ScanGuide.Directions(g.Rule));
        Check("between them the hint says to turn about 30 degrees and hold still", ScanGuide.Hint(g.Rule) == "turn about 30 degrees and hold still"
              || ScanGuide.Hint(g.Rule) == "hold still", ScanGuide.Hint(g.Rule));
        g.Hold(1);   // finish holding at 90 degrees, so that direction is taken too
        for (int k = 4; k < 12; k++) g.Turn(30, 0.8).Hold(1);
        Check("after a full circle every slot is done", ScanGuide.Directions(g.Rule) == "[############]", ScanGuide.Directions(g.Rule));
        g.Hold(0.6);
        Check("and the hint says to walk 2 to 3 m to another side of the room", ScanGuide.Hint(g.Rule).StartsWith("walk 2 to 3 m"), ScanGuide.Hint(g.Rule));
        Check("the guidance ends with how to finish", ScanGuide.Text(g.Rule, 12, 0).EndsWith("Done? Right thumbstick brings the panel: press Finish"));

        Console.WriteLine("15. The look-down row (scan.default.v3)");
        JNode cfg3 = JParse.Parse(File.ReadAllText(Path.Combine("perception", "scan", "scan.default.v3.json")));
        var errors3 = new Schema(JParse.Parse(File.ReadAllText(Path.Combine(schemas, "scan-config.v1.json")))).Errors(cfg3);
        ScanSettings S3 = ScanSettings.From(cfg3);
        Check("the third settings pass the schema, with a look-down row from 20 degrees", errors3.Count == 0 && S3.DownPitchDeg == 20 && S3.NewPositionM == 1.5);
        var d3 = new Sim(S3);
        d3.Hold(1).Tilt(-30, 0.5).Hold(1);
        Check("looking down at a desk after a level view in the same direction takes a keyframe in the down row",
              d3.Taken.Count == 2 && d3.Rule.SpotYaws.Count == 1 && d3.Rule.SpotDownYaws.Count == 1, d3.Taken.Count + " taken");
        var d2 = new Sim(ScanSettings.From(cfg2));
        d2.Hold(1).Tilt(-30, 0.5).Hold(1);
        Check("with the v2 settings the same look takes none: one row, and that direction is done", d2.Taken.Count == 1, d2.Taken.Count + " taken");
        d3.Turn(30, 0.6).Hold(1).Turn(30, 0.6).Hold(1);
        Check("turning while looking down fills the down row, not the level row", d3.Rule.SpotDownYaws.Count == 3 && d3.Rule.SpotYaws.Count == 1);
        Check("and the level row still asks for its directions", ScanGuide.Hint(d3.Rule).StartsWith("turn about 30 degrees") || ScanGuide.Hint(d3.Rule) == "hold still",
              ScanGuide.Hint(d3.Rule));
        string rows = ScanGuide.Text(d3.Rule, d3.Taken.Count, 0);
        Check("the guidance shows both rows, counted from the spot's first keyframe", rows.Contains("level [#...........]") && rows.Contains("down  [###.........]"),
              rows.Replace("\n", " | "));
        Check("and says to look down at desks and tables, and to skip walls right beside you", rows.Contains("Look down at desks and tables too; skip walls right beside you"));
        var nod = new Sim(S3) { Nod = 4 };
        nod.Hold(2);
        Check("nodding by 8 degrees while trying to hold still takes no keyframe", nod.Taken.Count == 0);
        var lvl = new Sim(S3);
        lvl.Hold(1);
        for (int k = 1; k < 12; k++) lvl.Turn(30, 0.8).Hold(1);
        lvl.Hold(0.6);
        Check("only the level row decides when a spot is done", lvl.Rule.SpotYaws.Count == 12 && ScanGuide.Hint(lvl.Rule).StartsWith("walk 2 to 3 m"), ScanGuide.Hint(lvl.Rule));
        lvl.Tilt(-30, 0.5).Hold(1);
        Check("after which looking down still takes keyframes there", lvl.Rule.SpotDownYaws.Count == 1);

        Console.WriteLine(pass + " passed, " + fail + " failed");
        return fail == 0 ? 0 : 1;
    }
}
