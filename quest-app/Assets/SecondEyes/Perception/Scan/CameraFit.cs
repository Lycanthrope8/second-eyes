using System;
using System.Collections.Generic;

namespace SecondEyes.Perception.Scan
{
    /// <summary>
    /// The camera's model in the texture's pixels (A4 Delivery 1, stage 3; docs/scan.md), free of Unity. Two ways:
    /// - Fit: from rays the camera component itself gives for viewport points, each in the camera's axes (x right, y up,
    ///   z forward), a pinhole fitted by least squares on each axis. The viewport's origin (bottom-left, as Unity's, or
    ///   top-left) is whichever gives positive focal lengths and the smaller residual; the residual, in pixels, says how
    ///   well a pinhole fits at all.
    /// - Crop: from the intrinsics reported for the sensor, the texture taken as the sensor's image scaled uniformly and
    ///   centre-cropped to the texture's size, as Meta's sample code does; the fallback when no fit is available.
    /// </summary>
    public static class CameraFit
    {
        public sealed class Result
        {
            public double Fx, Fy, Cx, Cy, RmsPx;
            public string Origin;
            public int Samples;
        }

        /// <summary>samples: u, v (viewport, 0 to 1), then the ray's x, y, z in the camera's axes. Null without a usable fit.</summary>
        public static Result Fit(IList<double[]> samples, int width, int height)
        {
            if (samples == null || samples.Count < 3) return null;
            Result best = null;
            foreach (bool bottomLeft in new[] { true, false })
            {
                var X = new List<double>(); var Y = new List<double>(); var px = new List<double>(); var py = new List<double>();
                foreach (double[] s in samples)
                {
                    if (s[4] <= 1e-9) continue;   // behind or level with the camera: not a viewport ray
                    X.Add(s[2] / s[4]); Y.Add(-s[3] / s[4]);
                    px.Add(s[0] * width); py.Add((bottomLeft ? 1 - s[1] : s[1]) * height);
                }
                double fx, cx, fy, cy;
                if (!Line(X, px, out fx, out cx) || !Line(Y, py, out fy, out cy) || fx <= 0 || fy <= 0) continue;
                double e = 0;
                for (int i = 0; i < X.Count; i++)
                {
                    double ex = fx * X[i] + cx - px[i], ey = fy * Y[i] + cy - py[i];
                    e += ex * ex + ey * ey;
                }
                double rms = Math.Sqrt(e / (2.0 * X.Count));
                if (best == null || rms < best.RmsPx)
                    best = new Result { Fx = fx, Fy = fy, Cx = cx, Cy = cy, RmsPx = rms, Origin = bottomLeft ? "bottom-left" : "top-left", Samples = X.Count };
            }
            return best;
        }

        /// <summary>The sensor's intrinsics carried to a texture that is the sensor's image scaled uniformly, then cropped to
        /// the texture's size about its centre.</summary>
        public static void Crop(double fx, double fy, double cx, double cy, int sensorW, int sensorH, int texW, int texH,
                                out double tfx, out double tfy, out double tcx, out double tcy)
        {
            double s = Math.Max((double)texW / sensorW, (double)texH / sensorH);
            double offX = (sensorW * s - texW) / 2.0, offY = (sensorH * s - texH) / 2.0;
            tfx = fx * s; tfy = fy * s; tcx = cx * s - offX; tcy = cy * s - offY;
        }

        /// <summary>Least squares y = a x + b; false when x doesn't vary.</summary>
        private static bool Line(List<double> x, List<double> y, out double a, out double b)
        {
            a = b = 0;
            int n = x.Count;
            if (n < 2) return false;
            double mx = 0, my = 0;
            for (int i = 0; i < n; i++) { mx += x[i]; my += y[i]; }
            mx /= n; my /= n;
            double sxx = 0, sxy = 0;
            for (int i = 0; i < n; i++) { sxx += (x[i] - mx) * (x[i] - mx); sxy += (x[i] - mx) * (y[i] - my); }
            if (sxx < 1e-12) return false;
            a = sxy / sxx;
            b = my - a * mx;
            return true;
        }
    }
}
