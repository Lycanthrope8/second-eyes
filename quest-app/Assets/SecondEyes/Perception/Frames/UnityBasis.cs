namespace SecondEyes.Perception.Frames
{
    /// <summary>
    /// The one place where handedness changes (docs/frames.md). Unity reports a camera's pose in quest_tracking as a
    /// position p and a rotation q: x_world = R(q) x_local + p, in Unity's left-handed axes (x right, y up, z forward).
    /// B re-expresses a quest_tracking vector in scene (it swaps y and z); F re-expresses an OpenCV optical-frame vector in
    /// Unity's camera axes (it flips y). A camera pose becomes T_scene_camera = (B R(q) F, B p). B and F each have
    /// determinant −1 and are their own inverses, so the product is a proper rotation and the way back is (B R F, B T).
    /// Quaternion components are never swapped by hand.
    /// </summary>
    public static class UnityBasis
    {
        public const string TrackingFrame = "quest_tracking", SceneFrame = "scene", CameraPrefix = "camera:";

        /// <summary>quest_tracking to scene: (x, y, z) becomes (x, z, y).</summary>
        public static Mat3 B { get { return new Mat3(1, 0, 0, 0, 0, 1, 0, 1, 0); } }

        /// <summary>The OpenCV optical frame to Unity's camera axes: y down becomes y up.</summary>
        public static Mat3 F { get { return new Mat3(1, 0, 0, 0, -1, 0, 0, 0, 1); } }

        /// <summary>A position or direction in quest_tracking, in scene.</summary>
        public static Vec3 ToScene(Vec3 tracking) { return B * tracking; }

        /// <summary>A position or direction in scene, in quest_tracking.</summary>
        public static Vec3 ToTracking(Vec3 scene) { return B * scene; }

        /// <summary>T_scene_camera from the camera pose Unity reports in quest_tracking during the given epoch.</summary>
        public static RigidTransform CameraToScene(string cameraId, long epoch, Vec3 unityPosition, Quat unityRotation)
        {
            if (string.IsNullOrEmpty(cameraId)) throw new System.ArgumentException("a camera has an ID");
            Mat3 r = B * unityRotation.ToMatrix() * F;
            return new RigidTransform(SceneFrame, CameraPrefix + cameraId, epoch, r, B * unityPosition);
        }

        /// <summary>The Unity pose (position, rotation with w &gt;= 0) whose CameraToScene is the given T_scene_camera.</summary>
        public static void SceneToUnityCamera(RigidTransform sceneCamera, out Vec3 unityPosition, out Quat unityRotation)
        {
            if (sceneCamera.Dest != SceneFrame || !sceneCamera.Src.StartsWith(CameraPrefix, System.StringComparison.Ordinal))
                throw new System.ArgumentException("expects T_scene_camera:<id>");
            unityPosition = B * sceneCamera.T;
            unityRotation = Quat.FromMatrix(B * sceneCamera.R * F);
        }
    }
}
