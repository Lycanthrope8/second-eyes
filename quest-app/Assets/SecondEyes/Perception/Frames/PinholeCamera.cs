using System;

namespace SecondEyes.Perception.Frames
{
    /// <summary>
    /// A pinhole camera in OpenCV's convention (docs/frames.md): u runs right, v down, and the top-left pixel's centre is
    /// (0, 0). Points are in the camera's optical frame (x right, y down, z forward). Lens distortion, where a camera has
    /// any, is removed before this model applies.
    /// </summary>
    public sealed class PinholeCamera
    {
        public int Width { get; }
        public int Height { get; }
        public double Fx { get; }
        public double Fy { get; }
        public double Cx { get; }
        public double Cy { get; }

        public PinholeCamera(int width, int height, double fx, double fy, double cx, double cy)
        {
            if (width <= 0 || height <= 0) throw new ArgumentException("the image has a positive size");
            if (!(fx > 0) || !(fy > 0) || double.IsInfinity(fx) || double.IsInfinity(fy)) throw new ArgumentException("focal lengths are positive and finite");
            if (!Vec3.Finite(cx) || !Vec3.Finite(cy)) throw new ArgumentException("the principal point is finite");
            Width = width; Height = height; Fx = fx; Fy = fy; Cx = cx; Cy = cy;
        }

        /// <summary>The pixel of a point in front of the camera (z &gt; 0); false for any other point.</summary>
        public bool TryProject(Vec3 p, out double u, out double v)
        {
            if (!(p.Z > 0.0) || !p.IsFinite) { u = v = double.NaN; return false; }
            u = Fx * p.X / p.Z + Cx;
            v = Fy * p.Y / p.Z + Cy;
            return true;
        }

        /// <summary>The unit ray through pixel (u, v), in the camera's optical frame.</summary>
        public Vec3 Ray(double u, double v)
        {
            if (!Vec3.Finite(u) || !Vec3.Finite(v)) throw new ArgumentException("a pixel is finite");
            return new Vec3((u - Cx) / Fx, (v - Cy) / Fy, 1.0).Normalized();
        }

        /// <summary>Whether a pixel lies on the image: within half a pixel of the outer pixels' centres.</summary>
        public bool OnImage(double u, double v)
        {
            return u >= -0.5 && v >= -0.5 && u <= Width - 0.5 && v <= Height - 0.5;
        }
    }
}
