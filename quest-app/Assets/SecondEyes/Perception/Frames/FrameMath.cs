using System;
using System.Globalization;

namespace SecondEyes.Perception.Frames
{
    // A4 (docs/frames.md): small, Unity-free geometry for poses and rays, in doubles. Unity's own Vector3 and Quaternion
    // are single precision and left-handed; this code is neither, so the same numbers come out on the PC and the headset.

    /// <summary>A 3-vector: a point in metres or a direction.</summary>
    public readonly struct Vec3
    {
        public readonly double X, Y, Z;

        public Vec3(double x, double y, double z) { X = x; Y = y; Z = z; }

        public static Vec3 operator +(Vec3 a, Vec3 b) { return new Vec3(a.X + b.X, a.Y + b.Y, a.Z + b.Z); }
        public static Vec3 operator -(Vec3 a, Vec3 b) { return new Vec3(a.X - b.X, a.Y - b.Y, a.Z - b.Z); }
        public static Vec3 operator -(Vec3 a) { return new Vec3(-a.X, -a.Y, -a.Z); }
        public static Vec3 operator *(double s, Vec3 a) { return new Vec3(s * a.X, s * a.Y, s * a.Z); }
        public static Vec3 operator *(Vec3 a, double s) { return s * a; }

        public double Dot(Vec3 b) { return X * b.X + Y * b.Y + Z * b.Z; }
        public Vec3 Cross(Vec3 b) { return new Vec3(Y * b.Z - Z * b.Y, Z * b.X - X * b.Z, X * b.Y - Y * b.X); }
        public double Norm { get { return Math.Sqrt(Dot(this)); } }
        public bool IsFinite { get { return Finite(X) && Finite(Y) && Finite(Z); } }

        public Vec3 Normalized()
        {
            double n = Norm;
            if (!(n > 0.0) || double.IsInfinity(n)) throw new ArgumentException("a zero or non-finite vector has no direction");
            return (1.0 / n) * this;
        }

        internal static bool Finite(double v) { return !double.IsNaN(v) && !double.IsInfinity(v); }

        public override string ToString()
        {
            return "(" + X.ToString("R", CultureInfo.InvariantCulture) + ", " + Y.ToString("R", CultureInfo.InvariantCulture)
                   + ", " + Z.ToString("R", CultureInfo.InvariantCulture) + ")";
        }
    }

    /// <summary>A 3x3 matrix, row by row.</summary>
    public readonly struct Mat3
    {
        public readonly double M00, M01, M02, M10, M11, M12, M20, M21, M22;

        public Mat3(double m00, double m01, double m02, double m10, double m11, double m12, double m20, double m21, double m22)
        {
            M00 = m00; M01 = m01; M02 = m02; M10 = m10; M11 = m11; M12 = m12; M20 = m20; M21 = m21; M22 = m22;
        }

        public static Mat3 Identity { get { return new Mat3(1, 0, 0, 0, 1, 0, 0, 0, 1); } }

        public static Mat3 operator *(Mat3 a, Mat3 b)
        {
            return new Mat3(
                a.M00 * b.M00 + a.M01 * b.M10 + a.M02 * b.M20, a.M00 * b.M01 + a.M01 * b.M11 + a.M02 * b.M21, a.M00 * b.M02 + a.M01 * b.M12 + a.M02 * b.M22,
                a.M10 * b.M00 + a.M11 * b.M10 + a.M12 * b.M20, a.M10 * b.M01 + a.M11 * b.M11 + a.M12 * b.M21, a.M10 * b.M02 + a.M11 * b.M12 + a.M12 * b.M22,
                a.M20 * b.M00 + a.M21 * b.M10 + a.M22 * b.M20, a.M20 * b.M01 + a.M21 * b.M11 + a.M22 * b.M21, a.M20 * b.M02 + a.M21 * b.M12 + a.M22 * b.M22);
        }

        public static Vec3 operator *(Mat3 m, Vec3 v)
        {
            return new Vec3(m.M00 * v.X + m.M01 * v.Y + m.M02 * v.Z, m.M10 * v.X + m.M11 * v.Y + m.M12 * v.Z,
                            m.M20 * v.X + m.M21 * v.Y + m.M22 * v.Z);
        }

        public Mat3 Transposed() { return new Mat3(M00, M10, M20, M01, M11, M21, M02, M12, M22); }

        public double Determinant()
        {
            return M00 * (M11 * M22 - M12 * M21) - M01 * (M10 * M22 - M12 * M20) + M02 * (M10 * M21 - M11 * M20);
        }

        /// <summary>Whether this is a proper rotation: orthonormal with determinant +1, within the tolerance.</summary>
        public bool IsRotation(double tolerance)
        {
            Mat3 p = this * Transposed();
            double worst = Math.Max(Math.Max(Math.Max(Math.Abs(p.M00 - 1), Math.Abs(p.M11 - 1)), Math.Abs(p.M22 - 1)),
                                    Math.Max(Math.Max(Math.Abs(p.M01), Math.Abs(p.M02)), Math.Abs(p.M12)));
            return Vec3.Finite(worst) && worst <= tolerance && Math.Abs(Determinant() - 1.0) <= tolerance;
        }
    }

    /// <summary>A rotation as a unit quaternion in the contract's order (x, y, z, w).</summary>
    public readonly struct Quat
    {
        public readonly double X, Y, Z, W;

        public Quat(double x, double y, double z, double w) { X = x; Y = y; Z = z; W = w; }

        public static Quat Identity { get { return new Quat(0, 0, 0, 1); } }

        public double Norm { get { return Math.Sqrt(X * X + Y * Y + Z * Z + W * W); } }

        public Quat Normalized()
        {
            double n = Norm;
            if (!(n > 0.0) || double.IsInfinity(n)) throw new ArgumentException("a zero or non-finite quaternion is no rotation");
            return new Quat(X / n, Y / n, Z / n, W / n);
        }

        /// <summary>The rotation matrix of this quaternion, normalized first.</summary>
        public Mat3 ToMatrix()
        {
            Quat q = Normalized();
            double x = q.X, y = q.Y, z = q.Z, w = q.W;
            return new Mat3(
                1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
                2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
                2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y));
        }

        /// <summary>The unit quaternion of a proper rotation matrix, with w &gt;= 0 so each rotation has one form. Refuses a
        /// matrix that is not a rotation, such as one that still carries a handedness change.</summary>
        public static Quat FromMatrix(Mat3 m)
        {
            if (!m.IsRotation(1e-6)) throw new ArgumentException("not a proper rotation (orthonormal, determinant +1)");
            double trace = m.M00 + m.M11 + m.M22, x, y, z, w;
            if (trace > 0)
            {
                double s = 2.0 * Math.Sqrt(trace + 1.0);
                w = 0.25 * s; x = (m.M21 - m.M12) / s; y = (m.M02 - m.M20) / s; z = (m.M10 - m.M01) / s;
            }
            else if (m.M00 > m.M11 && m.M00 > m.M22)
            {
                double s = 2.0 * Math.Sqrt(1.0 + m.M00 - m.M11 - m.M22);
                w = (m.M21 - m.M12) / s; x = 0.25 * s; y = (m.M01 + m.M10) / s; z = (m.M02 + m.M20) / s;
            }
            else if (m.M11 > m.M22)
            {
                double s = 2.0 * Math.Sqrt(1.0 + m.M11 - m.M00 - m.M22);
                w = (m.M02 - m.M20) / s; x = (m.M01 + m.M10) / s; y = 0.25 * s; z = (m.M12 + m.M21) / s;
            }
            else
            {
                double s = 2.0 * Math.Sqrt(1.0 + m.M22 - m.M00 - m.M11);
                w = (m.M10 - m.M01) / s; x = (m.M02 + m.M20) / s; y = (m.M12 + m.M21) / s; z = 0.25 * s;
            }
            Quat q = new Quat(x, y, z, w).Normalized();
            return q.W < 0 ? new Quat(-q.X, -q.Y, -q.Z, -q.W) : q;
        }
    }
}
