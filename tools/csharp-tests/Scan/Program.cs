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
        public double T, Yaw, Tremor, BusyFor, BusyUntil = -1;
        public bool Tracked = true;
        public Vec3 P = new Vec3(0, 1.6, 0);
        public readonly List<(double t, int pos, int step, double yaw)> Taken = new List<(double, int, int, double)>();
        public Sim(ScanSettings s) { rule = new KeyframeRule(s); }
        public KeyframeRule Rule => rule;
        public Sim Hold(double sec) { return Run(sec, 0, 0, 0); }
        public Sim Turn(double deg, double sec) { return Run(sec, deg, 0, 0); }
        public Sim Walk(double dx, double dz, double sec) { return Run(sec, 0, dx, dz); }
        Sim Run(double sec, double deg, double dx, double dz)   // the exact totals, split evenly over the frames
        {
            int n = Math.Max(1, (int)Math.Round(sec / Dt));
            for (int i = 0; i < n; i++)
            {
                T += Dt; Yaw += deg / n; frame++;
                P = new Vec3(P.X + dx / n, P.Y, P.Z + dz / n);
                double seen = KeyframeRule.Wrap(Yaw + Tremor * Math.Sin(frame * 0.7));
                if (rule.Update(T, P, seen, Tracked, T < BusyUntil))
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

        Console.WriteLine(pass + " passed, " + fail + " failed");
        return fail == 0 ? 0 : 1;
    }
}
