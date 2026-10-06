using UnityEngine;

namespace SecondEyes.Perception
{
    /// <summary>
    /// A source of image frames for on-device perception (A1.10). Detectors read frames only through this
    /// interface, so the same detector serves the passthrough camera now, a recorded video in A4's replay and
    /// the drone's video in A6.
    /// </summary>
    public interface IFrameSource
    {
        /// <summary>A short, stable name for logs, for example "passthrough".</summary>
        string SourceId { get; }

        /// <summary>True while frames are arriving.</summary>
        bool IsDelivering { get; }

        /// <summary>The latest frame as a GPU texture, or null when nothing is arriving.</summary>
        Texture CurrentTexture { get; }

        /// <summary>The frame size in pixels, or (0, 0) when nothing is arriving.</summary>
        Vector2Int Resolution { get; }

        /// <summary>Distinct frames seen since the app started, or -1 when the source cannot tell frames apart.</summary>
        long FramesSeen { get; }
    }
}
