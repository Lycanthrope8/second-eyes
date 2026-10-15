using System;

namespace SecondEyes.Perception.Frames
{
    /// <summary>
    /// T_dest_src (docs/frames.md): maps coordinates in frame Src into frame Dest, x_dest = R x_src + T, with R a proper
    /// rotation. A transform computed in a tracking epoch is valid only in that epoch; NoEpoch marks frames that have none
    /// (a fixed mounting, for instance). Composition checks that the inner frames match and the epochs agree, so a pose
    /// from before a recenter can't be combined with one from after it.
    /// </summary>
    public sealed class RigidTransform
    {
        public const long NoEpoch = -1;
        public const double RotationTolerance = 1e-6;

        public string Dest { get; }
        public string Src { get; }
        public long Epoch { get; }
        public Mat3 R { get; }
        public Vec3 T { get; }

        public RigidTransform(string dest, string src, long epoch, Mat3 r, Vec3 t)
        {
            if (string.IsNullOrEmpty(dest) || string.IsNullOrEmpty(src)) throw new ArgumentException("a transform names both frames");
            if (epoch < NoEpoch) throw new ArgumentException("an epoch is NoEpoch (-1) or a count from 0");
            if (!r.IsRotation(RotationTolerance)) throw new ArgumentException("R must be a proper rotation (orthonormal, determinant +1)");
            if (!t.IsFinite) throw new ArgumentException("T must be finite");
            Dest = dest; Src = src; Epoch = epoch; R = r; T = t;
        }

        public static RigidTransform FromQuaternion(string dest, string src, long epoch, Vec3 translation, Quat rotation)
        {
            return new RigidTransform(dest, src, epoch, rotation.ToMatrix(), translation);
        }

        /// <summary>R as a unit quaternion (x, y, z, w) with w &gt;= 0: the stored form, rotation_xyzw.</summary>
        public Quat Rotation { get { return Quat.FromMatrix(R); } }

        public Vec3 ApplyToPoint(Vec3 p) { return R * p + T; }

        public Vec3 ApplyToDirection(Vec3 d) { return R * d; }

        /// <summary>T_src_dest = (Rᵀ, −Rᵀ T).</summary>
        public RigidTransform Inverse()
        {
            Mat3 rt = R.Transposed();
            return new RigidTransform(Src, Dest, Epoch, rt, -(rt * T));
        }

        /// <summary>This (T_a_b) after inner (T_b_c): T_a_c. The epochs must agree where both have one.</summary>
        public RigidTransform Compose(RigidTransform inner)
        {
            if (inner == null) throw new ArgumentNullException(nameof(inner));
            if (Src != inner.Dest)
                throw new InvalidOperationException("can't compose T_" + Dest + "_" + Src + " with T_" + inner.Dest + "_" + inner.Src);
            if (Epoch != NoEpoch && inner.Epoch != NoEpoch && Epoch != inner.Epoch)
                throw new InvalidOperationException("epochs differ (" + Epoch + " and " + inner.Epoch + "): the tracking moved between them");
            long epoch = Epoch != NoEpoch ? Epoch : inner.Epoch;
            return new RigidTransform(Dest, inner.Src, epoch, R * inner.R, R * inner.T + T);
        }
    }
}
