// The localization core's checks (A4 Delivery 1, stage 2; docs/localization.md) on synthetic scans with exact ground truth:
// boxes placed in a scene, a camera at chosen positions and yaws, each box projected into every view as the detector would
// report it (clipped at the image's edge), and the whole built by CaptureBuilder, as the headset builds it, into a capture
// record that must itself pass the capture check. Every scene the core writes must pass the contract (scene v3) with its category map. Expectations come from the
// geometry (docs/localization.md explains the tolerances), never from the code under test.
// Run: dotnet run --project tools/csharp-tests/Localization -- grounding/tests/fixtures schemas
using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using SecondEyes.Grounding.Parsing;
using SecondEyes.Grounding.Prompting;
using SecondEyes.Grounding.Resolution;
using SecondEyes.Perception.Capture;
using SecondEyes.Perception.Frames;
using SecondEyes.Perception.Localization;

static class Program
{
    static int pass, fail;
    const int W = 1280, H = 960;
    const double F = 870, Cx = 640, Cy = 480, Height = 1.4;
    static readonly PinholeCamera Pin = new PinholeCamera(W, H, F, F, Cx, Cy);
    sealed class Box { public string Label, Category; public Vec3 Centre, Size; }

    static void Check(string n, bool ok, string d = "")
    {
        if (ok) { pass++; Console.WriteLine("  ok    " + n); }
        else { fail++; Console.WriteLine("  FAIL  " + n + (d.Length > 0 ? "  [" + d + "]" : "")); }
    }

    static JNode R(double v) { return new JNode { Kind = JKind.Number, Num = v, Text = v.ToString("R", CultureInfo.InvariantCulture), IsInteger = false }; }
    static JNode I(long v) { return Localizer.Int(v); }
    static JNode S(string s) { return JNode.Str(s); }
    static JNode A(params JNode[] xs) { return JNode.Arr(xs.ToList()); }
    static JNode O(params object[] kv)
    {
        var o = JNode.Obj();
        for (int i = 0; i < kv.Length; i += 2) o.Add((string)kv[i], (JNode)kv[i + 1]);
        return o;
    }

    static Box Chair(double x, double y) { return new Box { Label = "chair", Category = "chair", Centre = new Vec3(x, y, 0.45), Size = new Vec3(0.45, 0.5, 0.9) }; }
    static Box Table(double x, double y) { return new Box { Label = "dining table", Category = "table", Centre = new Vec3(x, y, 0.37), Size = new Vec3(1.2, 0.6, 0.74) }; }
    static Box Bed(double x, double y) { return new Box { Label = "bed", Category = "bed", Centre = new Vec3(x, y, 0.3), Size = new Vec3(1.4, 2.0, 0.6) }; }

    static JNode Manifest;   // the detector package's manifest, as the headset reads it

    /// <summary>A capture of the boxes from each position (scene x, y) at each yaw (degrees, positive to the right), built
    /// through CaptureBuilder exactly as the headset will build one.</summary>
    static JNode Scan(List<Box> boxes, (double x, double y)[] positions, double[] yaws, bool reverse = false, List<CaptureDetection> extra = null)
    {
        var builder = new CaptureBuilder("scan.synthetic",
            new CaptureCamera { CameraId = "quest_left", WidthPx = W, HeightPx = H, Fx = F, Fy = F, Cx = Cx, Cy = Cy, IntrinsicsSource = "synthetic", Distortion = "none" },
            "synthetic", Manifest, new string('0', 63) + "1", new string('b', 64), "map.bedroom", "sizes.bedroom", "ns", 5000, 30);
        long mono = 1200000000;
        int index = 0;
        for (int p = 0; p < positions.Length; p++)
            for (int s = 0; s < yaws.Length; s++)
            {
                double yaw = yaws[s] * Math.PI / 180;
                var pu = new Vec3(positions[p].x, Height, positions[p].y);   // Unity: x right, y up, z forward
                var qu = new Quat(0, Math.Sin(yaw / 2), 0, Math.Cos(yaw / 2));
                RigidTransform toCamera = UnityBasis.CameraToScene("quest_left", 0, pu, qu).Inverse();
                var dets = new List<CaptureDetection>();
                foreach (Box b in boxes)
                {
                    double u0 = double.MaxValue, v0 = double.MaxValue, u1 = double.MinValue, v1 = double.MinValue;
                    bool front = true;
                    for (int c = 0; c < 8 && front; c++)
                    {
                        var corner = b.Centre + new Vec3(((c & 1) == 0 ? -0.5 : 0.5) * b.Size.X, ((c & 2) == 0 ? -0.5 : 0.5) * b.Size.Y,
                                                         ((c & 4) == 0 ? -0.5 : 0.5) * b.Size.Z);
                        Vec3 local = toCamera.ApplyToPoint(corner);
                        double u, v;
                        if (local.Z < 0.1 || !Pin.TryProject(local, out u, out v)) { front = false; break; }
                        u0 = Math.Min(u0, u); v0 = Math.Min(v0, v); u1 = Math.Max(u1, u); v1 = Math.Max(v1, v);
                    }
                    if (!front) continue;
                    double x1 = Math.Max(0, u0 + 0.5), y1 = Math.Max(0, v0 + 0.5), x2 = Math.Min(W, u1 + 0.5), y2 = Math.Min(H, v1 + 0.5);
                    if (x2 - x1 < 2 || y2 - y1 < 2) continue;   // not on the image
                    dets.Add(new CaptureDetection { Label = b.Label, Score = 0.9, X1 = x1, Y1 = y1, X2 = x2, Y2 = y2 });
                }
                if (extra != null && index == 0) dets.AddRange(extra);
                if (reverse) dets.Reverse();
                builder.Add(p, s, (mono - 5000) * 1000, 0, true, new[] { pu.X, pu.Y, pu.Z }, new[] { qu.X, qu.Y, qu.Z, qu.W }, dets);
                mono += 1100000;
                index++;
            }
        return builder.Build();
    }

    static Vec3 Centre(JNode obj) { var v = obj["geometry"]["center_m"]["value"].Items; return new Vec3(v[0].Num, v[1].Num, v[2].Num); }
    static bool Known(JNode obj) { return obj["geometry"]["center_m"]["state"].Text == "known"; }
    static string Status(JNode obj) { return obj["observation"]["association"]["status"].Text; }
    static string Cat(JNode obj) { return obj["category"]["value"]["standard"].Text; }
    static double Nearest(JNode obj, List<Box> boxes) { return boxes.Where(b => b.Category == Cat(obj)).Min(b => (Centre(obj) - b.Centre).Norm); }

    static int Main(string[] args)
    {
        string fixtures = args.Length > 0 ? args[0] : "grounding/tests/fixtures", schemasDir = args.Length > 1 ? args[1] : "schemas";
        var cache = new Dictionary<string, Schema>();
        Func<string, Schema> schema = n => { if (!cache.ContainsKey(n)) cache[n] = new Schema(JParse.Parse(File.ReadAllText(Path.Combine(schemasDir, n)))); return cache[n]; };
        Func<string, JNode> load = p => JParse.Parse(File.ReadAllText(p));
        var contract = new Contract(schema);
        Manifest = load(Path.Combine("quest-app", "Assets", "SecondEyes", "Models", "yolox-nano", "detector-package.json"));
        JNode map = load(Path.Combine(fixtures, "contract", "valid", "category-map.bedroom.json"));
        JNode priors = load(Path.Combine(fixtures, "localization", "size-priors.test.json"));
        JNode config = load(Path.Combine("perception", "localization", "localization.default.v1.json"));
        JNode noChair = JParse.Parse(PyJson.Canonical(priors));
        noChair["entries"].Items.RemoveAll(e => e["category"].Text == "chair");
        Func<JNode, JNode, string, LocalizationOutput> run = (capture, pr, name) =>
        {
            var ci = CaptureCheck.Check(name, capture, schema(CaptureCheck.SchemaFile));
            Check(name + ": the synthetic scan is a valid capture record", ci.Count == 0, string.Join(", ", ci.Select(i => i.Code + " " + i.Path).Take(3)));
            LocalizationOutput o = Localizer.Run(capture, map, pr, config);
            var si = contract.Validate(new List<KeyValuePair<string, JNode>> { new KeyValuePair<string, JNode>("scene", o.Scene), new KeyValuePair<string, JNode>("map", map) });
            Check(name + ": the scene passes the contract (scene v3)", !si.Any(i => i.IsError), string.Join(", ", si.Where(i => i.IsError).Select(i => i.Code + " " + i.Path).Take(3)));
            return o;
        };

        Console.WriteLine("1. Three objects from two positions");
        var room = new List<Box> { Chair(-0.4, 3.0), Table(1.2, 3.6), Bed(-1.6, 4.4) };
        var twoPlaces = new[] { (-0.6, 0.0), (0.9, 0.3) };
        var sweep = new[] { -45.0, -15, 15, 45 };
        var o1 = run(Scan(room, twoPlaces, sweep), priors, "three objects");
        var objs = o1.Scene["objects"].Items;
        Check("three objects, one per box: bed, chair, table", string.Join(",", objs.Select(Cat)) == "bed,chair,table", string.Join(",", objs.Select(Cat)));
        Check("each is placed from both positions, with no possible duplicate", objs.All(x => Status(x) == "multi_position"), string.Join(",", objs.Select(Status)));
        Check("triangulated centres name no assumption", objs.All(x => Known(x) && x["geometry"]["center_m"]["evidence"]["assumptions"].Items.Count == 0));
        foreach (JNode x in objs.Where(Known))
        {
            double tol = Cat(x) == "chair" ? 0.15 : 0.30;   // a box's image centre drifts from its 3D centre more for large boxes
            Check(Cat(x) + " within " + tol.ToString("0.00", CultureInfo.InvariantCulture) + " m of its box",
                  Nearest(x, room) <= tol, Nearest(x, room).ToString("0.000", CultureInfo.InvariantCulture) + " m");
        }

        Console.WriteLine("2. Two identical chairs whose rays cross: the size priors reject the ghosts");
        var pair = new List<Box> { Chair(-0.5, 3.0), Chair(0.5, 3.0) };
        var wide = new[] { (-1.5, 0.0), (1.5, 0.0) };
        var o2 = run(Scan(pair, wide, new[] { -30.0, 0, 30 }), priors, "crossing chairs");
        var chairs = o2.Scene["objects"].Items;
        Check("exactly two chairs, both from two positions, no possible duplicate", chairs.Count == 2 && chairs.All(x => Status(x) == "multi_position"),
              chairs.Count + ": " + string.Join(",", chairs.Select(Status)));
        Check("one at each chair, within 0.15 m", chairs.Count == 2 && chairs.All(x => Known(x) && Nearest(x, pair) <= 0.15)
              && (Centre(chairs[0]) - Centre(chairs[1])).Norm > 0.5);

        Console.WriteLine("3. The same chairs without a chair prior: no ghost passes as a confident object");
        var o3 = run(Scan(pair, wide, new[] { -30.0, 0, 30 }), noChair, "crossing chairs, no prior");
        var c3 = o3.Scene["objects"].Items;
        Check("every confident chair (from two positions, no possible duplicate) is a real one",
              c3.Where(x => Status(x) == "multi_position").All(x => Known(x) && Nearest(x, pair) <= 0.25),
              string.Join("; ", c3.Select(x => Status(x) + (Known(x) ? " " + Nearest(x, pair).ToString("0.00", CultureInfo.InvariantCulture) : " unknown"))));
        Check("an object that isn't confident is said so (possible duplicate or one position)",
              c3.Where(x => !(Known(x) && Nearest(x, pair) <= 0.25)).All(x => Status(x) != "multi_position"));

        Console.WriteLine("4. Turning in place: no baseline, so centres come from size and say so");
        var o4 = run(Scan(room, new[] { (0.0, 0.0) }, new[] { -60.0, -30, 0, 30, 60 }), priors, "one position");
        var c4 = o4.Scene["objects"].Items;
        Check("every object is single_position, seen from one position", c4.Count > 0 && c4.All(x => Status(x) == "single_position"
              && x["observation"]["views"]["positions"].Num == 1), string.Join(",", c4.Select(Status)));
        Check("each centre names the size prior and upright assumptions", c4.All(x => Known(x)
              && string.Join(",", x["geometry"]["center_m"]["evidence"]["assumptions"].Items.Select(a => a.Text)) == "size_prior,upright"));
        foreach (JNode x in c4.Where(Known))
        {
            double dist = (Centre(x) - new Vec3(0, 0, Height)).Norm, err = Nearest(x, room);
            Check(Cat(x) + ": size-based error within a quarter of its distance (" + err.ToString("0.00", CultureInfo.InvariantCulture) + " m at "
                  + dist.ToString("0.0", CultureInfo.InvariantCulture) + " m)", err <= 0.25 * dist);
        }

        Console.WriteLine("5. An object only ever cut by the image's edge is left out, not guessed");
        var o5 = run(Scan(new List<Box> { Table(2.1, 3.0) }, new[] { (0.0, 0.0) }, new[] { 0.0 }), priors, "cut by the edge");
        Check("no object", o5.Scene["objects"].Items.Count == 0);
        Check("the trace counts it as cut by the edge", o5.Trace["left_out"]["cut_by_the_edge"].Num >= 1);

        Console.WriteLine("6. The order the detector lists boxes in changes nothing");
        var o6 = run(Scan(room, twoPlaces, sweep, reverse: true), priors, "reversed detections");
        Check("the scene is identical, text for text", PyJson.Canonical(o6.Scene) == PyJson.Canonical(o1.Scene));

        Console.WriteLine("7. Unmapped labels and low scores are left out and counted");
        var noise = new List<CaptureDetection> { new CaptureDetection { Label = "person", Score = 0.95, X1 = 100, Y1 = 100, X2 = 300, Y2 = 700 },
                                                 new CaptureDetection { Label = "chair", Score = 0.2, X1 = 900, Y1 = 300, X2 = 1000, Y2 = 500 } };
        var o7 = run(Scan(room, twoPlaces, sweep, extra: noise), priors, "with a person and a weak chair");
        Check("the person is counted as unmapped and the weak chair as below min_score",
              o7.Trace["left_out"]["unmapped_label"].Num == 1 && o7.Trace["left_out"]["below_min_score"].Num == 1);
        Check("and the scene is the same as without them", PyJson.Canonical(o7.Scene) == PyJson.Canonical(o1.Scene));

        Console.WriteLine("8. The capture fixture runs end to end");
        JNode fx = load(Path.Combine(fixtures, "capture", "valid", "capture.bedroom.json"));
        var o8 = run(fx, priors, "capture fixture");
        Check("it yields objects", o8.Scene["objects"].Items.Count > 0);

        Console.WriteLine("9. The bedroom's two chairs, 88 and 115 cm tall, with both sizes in the priors");
        JNode bedroomPriors = load(Path.Combine("perception", "bedroom", "size-priors.bedroom.v1.json"));
        JNode bedroomMap = load(Path.Combine("perception", "bedroom", "category-map.bedroom.v1.json"));
        var pe = schema("size-priors.v1.json").Errors(bedroomPriors);
        Check("the bedroom's size priors pass their schema", pe.Count == 0, string.Join(", ", pe.Select(e => e.Keyword + " " + e.Path)));
        var me = contract.Validate(new List<KeyValuePair<string, JNode>> { new KeyValuePair<string, JNode>("map", bedroomMap) });
        Check("the bedroom's category map passes the contract", !me.Any(i => i.IsError), string.Join(", ", me.Select(i => i.Code + " " + i.Path)));
        var chairs2 = new List<Box> { new Box { Label = "chair", Category = "chair", Centre = new Vec3(-0.6, 3.0, 0.44), Size = new Vec3(0.55, 0.45, 0.88) },
                                      new Box { Label = "chair", Category = "chair", Centre = new Vec3(0.7, 3.2, 0.575), Size = new Vec3(0.55, 0.45, 1.15) } };
        JNode realMap = map;
        map = bedroomMap;
        var o9 = run(Scan(chairs2, new[] { (-1.2, 0.0), (1.2, 0.2) }, new[] { -30.0, 0, 30 }), bedroomPriors, "two chairs from two positions");
        var c9 = o9.Scene["objects"].Items;
        Check("both chairs, each from both positions, within 0.15 m", c9.Count == 2 && c9.All(x => Status(x) == "multi_position" && Known(x) && Nearest(x, chairs2) <= 0.15),
              string.Join("; ", c9.Select(x => Status(x) + (Known(x) ? " " + Nearest(x, chairs2).ToString("0.00", CultureInfo.InvariantCulture) : ""))));
        Check("neither claims a size: the priors hold two chair sizes", c9.All(x => x["geometry"]["size_m"]["state"].Text == "unknown"
              && x["geometry"]["size_m"]["reason"].Text == "no_permitted_prior"));
        var o10 = run(Scan(chairs2, new[] { (0.0, 0.0) }, new[] { -30.0, 0, 30 }), bedroomPriors, "two chairs from one position");
        var c10 = o10.Scene["objects"].Items;
        foreach (JNode x in c10.Where(Known))
        {
            double dist = (Centre(x) - new Vec3(0, 0, Height)).Norm, err = Nearest(x, chairs2);
            Check("from one position, each chair's centre comes from both sizes, within a quarter of its distance (" + err.ToString("0.00", CultureInfo.InvariantCulture) + " m at "
                  + dist.ToString("0.0", CultureInfo.InvariantCulture) + " m)", Status(x) == "single_position" && err <= 0.25 * dist);
        }
        Check("two chairs placed from one position", c10.Count == 2 && c10.All(Known));
        map = realMap;

        Console.WriteLine(pass + " passed, " + fail + " failed");
        return fail == 0 ? 0 : 1;
    }
}
