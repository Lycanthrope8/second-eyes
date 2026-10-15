// The frame library's checks (A4 Delivery 1, stage 1; docs/frames.md). Every expected value below is worked out by hand
// from the definitions (the matrices B and F, a quaternion's rotation, the pinhole model), never read from the code
// under test. Run: dotnet run --project tools/csharp-tests/Frames
using System;
using SecondEyes.Perception.Frames;

static class Program
{
    static int pass, fail;
    const double Eps = 1e-9;
    static readonly double H = Math.Sqrt(0.5);   // sin 45° = cos 45°

    static void Check(string n, bool ok, string d = "")
    {
        if (ok) { pass++; Console.WriteLine("  ok    " + n); }
        else { fail++; Console.WriteLine("  FAIL  " + n + (d.Length > 0 ? "  [" + d + "]" : "")); }
    }

    static bool Near(double a, double b, double eps = Eps) { return Math.Abs(a - b) <= eps; }
    static bool Near(Vec3 a, Vec3 b, double eps = Eps) { return Near(a.X, b.X, eps) && Near(a.Y, b.Y, eps) && Near(a.Z, b.Z, eps); }
    static bool Near(Quat a, Quat b, double eps = Eps) { return Near(a.X, b.X, eps) && Near(a.Y, b.Y, eps) && Near(a.Z, b.Z, eps) && Near(a.W, b.W, eps); }
    static string S(Quat q) { return "(" + q.X + ", " + q.Y + ", " + q.Z + ", " + q.W + ")"; }

    static bool Throws(Action a) { try { a(); return false; } catch (ArgumentException) { return true; } catch (InvalidOperationException) { return true; } }

    static Quat AboutAxis(Vec3 axis, double degrees)
    {
        Vec3 u = axis.Normalized();
        double half = degrees * Math.PI / 360.0, s = Math.Sin(half);
        return new Quat(u.X * s, u.Y * s, u.Z * s, Math.Cos(half));
    }

    static int Main()
    {
        Vec3 ex = new Vec3(1, 0, 0), ey = new Vec3(0, 1, 0), ez = new Vec3(0, 0, 1);

        Console.WriteLine("Quaternions");
        Check("the identity quaternion is the identity matrix", Near(Quat.Identity.ToMatrix() * new Vec3(1, 2, 3), new Vec3(1, 2, 3)));
        Quat y90 = new Quat(0, H, 0, H);   // +90° about y by the right-hand rule: z goes to x, x goes to −z
        Check("+90° about y takes z to x", Near(y90.ToMatrix() * ez, ex));
        Check("+90° about y takes x to −z", Near(y90.ToMatrix() * ex, -ez));
        Check("a matrix back to its quaternion", Near(Quat.FromMatrix(y90.ToMatrix()), y90));
        Quat negW = new Quat(0.1, -0.2, 0.3, -0.9).Normalized();
        Quat back = Quat.FromMatrix(negW.ToMatrix());
        Check("a rotation's quaternion comes back with w >= 0 (−q is the same rotation)",
              back.W >= 0 && Near(back, new Quat(-negW.X, -negW.Y, -negW.Z, -negW.W)), S(back));
        Check("a 180° turn about x comes back as (1, 0, 0, 0)", Near(Quat.FromMatrix(new Mat3(1, 0, 0, 0, -1, 0, 0, 0, -1)), new Quat(1, 0, 0, 0)));
        Check("FromMatrix refuses a handedness change (B)", Throws(() => Quat.FromMatrix(UnityBasis.B)));
        Check("FromMatrix refuses a scaled matrix", Throws(() => Quat.FromMatrix(new Mat3(2, 0, 0, 0, 2, 0, 0, 0, 2))));
        Check("a zero quaternion is refused", Throws(() => new Quat(0, 0, 0, 0).ToMatrix()));

        Console.WriteLine("The basis change");
        Check("B takes (1, 2, 3) to (1, 3, 2)", Near(UnityBasis.ToScene(new Vec3(1, 2, 3)), new Vec3(1, 3, 2)));
        Check("B is its own inverse", Near(UnityBasis.ToTracking(UnityBasis.ToScene(new Vec3(4, -5, 6))), new Vec3(4, -5, 6)));
        Check("B and F each have determinant −1", Near(UnityBasis.B.Determinant(), -1) && Near(UnityBasis.F.Determinant(), -1));

        Console.WriteLine("A camera pose from Unity: level, looking along Unity's +z");
        var level = UnityBasis.CameraToScene("quest_left", 0, new Vec3(1, 2, 3), Quat.Identity);
        Check("named T_scene_camera:quest_left in epoch 0", level.Dest == "scene" && level.Src == "camera:quest_left" && level.Epoch == 0);
        Check("its origin is B p = (1, 3, 2)", Near(level.T, new Vec3(1, 3, 2)));
        Check("the optical axis (z) points along scene +y", Near(level.ApplyToDirection(ez), ey));
        Check("the image's down (y) points along scene −z", Near(level.ApplyToDirection(ey), -ez));
        Check("the image's right (x) points along scene +x", Near(level.ApplyToDirection(ex), ex));
        Check("the stored rotation is −90° about x: (−√½, 0, 0, √½)", Near(level.Rotation, new Quat(-H, 0, 0, H)), S(level.Rotation));

        Console.WriteLine("Turned 90° right (Unity: forward goes to +x)");
        var right = UnityBasis.CameraToScene("quest_left", 0, new Vec3(0, 1.5, 0), y90);
        Check("the optical axis points along scene +x", Near(right.ApplyToDirection(ez), ex));
        Check("the image's down still points along scene −z", Near(right.ApplyToDirection(ey), -ez));
        Check("the image's right points along scene −y (facing +x with z up, right is −y)", Near(right.ApplyToDirection(ex), -ey));

        Console.WriteLine("Pitched 30° down (Unity: +30° about x)");
        Quat x30 = new Quat(Math.Sin(Math.PI / 12), 0, 0, Math.Cos(Math.PI / 12));
        var down = UnityBasis.CameraToScene("quest_left", 2, new Vec3(0, 1.5, 0), x30);
        Check("the optical axis points forward and 30° down: (0, cos 30°, −sin 30°)",
              Near(down.ApplyToDirection(ez), new Vec3(0, Math.Cos(Math.PI / 6), -0.5)));

        Console.WriteLine("Scene and Unity agree on where a point is in the image");
        var rng = new Random(20261015);
        bool agree = true, roundTrip = true;
        for (int k = 0; k < 200; k++)
        {
            Vec3 p = new Vec3(rng.NextDouble() * 4 - 2, rng.NextDouble() * 2, rng.NextDouble() * 4 - 2);
            Quat q = AboutAxis(new Vec3(rng.NextDouble() - 0.5, rng.NextDouble() - 0.5, rng.NextDouble() - 0.5), rng.NextDouble() * 360);
            Vec3 world = new Vec3(rng.NextDouble() * 6 - 3, rng.NextDouble() * 3, rng.NextDouble() * 6 - 3);
            // Unity's own way: local = R(q)ᵀ (world − p) in Unity's camera axes; then F (its own inverse) flips y down.
            Vec3 unityLocal = q.ToMatrix().Transposed() * (world - p);
            Vec3 expected = UnityBasis.F * unityLocal;
            var t = UnityBasis.CameraToScene("cam", 1, p, q);
            Vec3 viaScene = t.Inverse().ApplyToPoint(UnityBasis.ToScene(world));
            agree &= Near(viaScene, expected, 1e-9);
            Vec3 p2; Quat q2;
            UnityBasis.SceneToUnityCamera(t, out p2, out q2);
            Quat qc = q.W < 0 ? new Quat(-q.X, -q.Y, -q.Z, -q.W) : q;
            roundTrip &= Near(p2, p, 1e-9) && Near(q2, qc, 1e-9);
        }
        Check("200 random poses: the optical-frame point is the same through scene as through Unity's camera axes", agree);
        Check("200 random poses: SceneToUnityCamera gives back the Unity pose", roundTrip);

        Console.WriteLine("Transforms");
        var a = RigidTransform.FromQuaternion("scene", "camera:c", 3, new Vec3(0.5, -1, 2), AboutAxis(new Vec3(1, 2, 3), 70));
        Vec3 x = new Vec3(0.3, 0.7, -1.1);
        Check("the inverse undoes the transform", Near(a.Inverse().ApplyToPoint(a.ApplyToPoint(x)), x));
        var round = a.Compose(a.Inverse());
        Check("T · T⁻¹ is the identity on scene", round.Dest == "scene" && round.Src == "scene" && Near(round.ApplyToPoint(x), x));
        var mount = RigidTransform.FromQuaternion("camera:c", "camera:d", RigidTransform.NoEpoch, new Vec3(0.1, 0, 0), Quat.Identity);
        var chained = a.Compose(mount);
        Check("composing with a fixed mounting (no epoch) keeps the epoch and the frames",
              chained.Dest == "scene" && chained.Src == "camera:d" && chained.Epoch == 3);
        Check("a composition applies the inner transform first", Near(chained.ApplyToPoint(x), a.ApplyToPoint(mount.ApplyToPoint(x))));
        Check("frames that don't meet can't be composed", Throws(() => a.Compose(a)));
        var otherEpoch = RigidTransform.FromQuaternion("camera:c", "camera:d", 4, new Vec3(0, 0, 0), Quat.Identity);
        Check("transforms from two epochs can't be composed", Throws(() => a.Compose(otherEpoch)));
        Check("a transform refuses a handedness change as its rotation", Throws(() => new RigidTransform("a", "b", 0, UnityBasis.B, new Vec3(0, 0, 0))));
        Check("a transform refuses a non-finite translation", Throws(() => new RigidTransform("a", "b", 0, Mat3.Identity, new Vec3(double.NaN, 0, 0))));
        Check("a transform names both frames", Throws(() => new RigidTransform("", "b", 0, Mat3.Identity, new Vec3(0, 0, 0))));

        Console.WriteLine("The pinhole camera");
        var cam = new PinholeCamera(640, 480, 500, 500, 320, 240);
        double u, v;
        bool inFront = cam.TryProject(new Vec3(0.1, -0.2, 2.0), out u, out v);
        Check("(0.1, −0.2, 2) projects to (345, 190): u = 500·0.05 + 320, v = 500·(−0.1) + 240", inFront && Near(u, 345) && Near(v, 190));
        Check("a point behind the camera has no pixel", !cam.TryProject(new Vec3(0, 0, -1), out u, out v));
        Check("the principal point's ray is the optical axis", Near(cam.Ray(320, 240), ez));
        Check("the ray through (345, 190) points at (0.1, −0.2, 2)", Near(cam.Ray(345, 190), new Vec3(0.1, -0.2, 2.0).Normalized()));
        Check("the image runs half a pixel past the outer pixels' centres",
              cam.OnImage(-0.5, -0.5) && cam.OnImage(639.5, 479.5) && !cam.OnImage(-0.51, 0) && !cam.OnImage(0, 479.51));
        Check("a camera refuses a zero focal length", Throws(() => new PinholeCamera(640, 480, 0, 500, 320, 240)));

        Console.WriteLine("A pixel's ray, carried into the scene, passes through the point it saw");
        bool through = true;
        for (int k = 0; k < 200; k++)
        {
            Vec3 p = new Vec3(rng.NextDouble() * 4 - 2, 1 + rng.NextDouble(), rng.NextDouble() * 4 - 2);
            Quat q = AboutAxis(new Vec3(0, 1, 0), rng.NextDouble() * 360);   // turning in place, as in a scan
            var t = UnityBasis.CameraToScene("cam", 0, p, q);
            Vec3 point = t.ApplyToPoint(new Vec3(rng.NextDouble() - 0.5, rng.NextDouble() - 0.5, 1 + 3 * rng.NextDouble()));
            Vec3 local = t.Inverse().ApplyToPoint(point);
            if (!cam.TryProject(local, out u, out v)) { through = false; break; }
            Vec3 ray = t.ApplyToDirection(cam.Ray(u, v));
            through &= Near(ray, (point - t.T).Normalized(), 1e-9);
        }
        Check("200 random views: the scene ray from the camera through the pixel meets the point", through);

        Console.WriteLine(pass + " passed, " + fail + " failed");
        return fail == 0 ? 0 : 1;
    }
}
