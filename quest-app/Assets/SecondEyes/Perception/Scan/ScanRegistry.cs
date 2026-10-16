namespace SecondEyes.Perception.Scan
{
    /// <summary>
    /// What the panel needs of the headset's scanner (A4 Delivery 1, stage 3; docs/scan.md), free of Unity: the scanner
    /// (Glue/RoomScanner.cs) registers itself here when the app starts, so the panel works through this alone.
    /// </summary>
    public interface IRoomScanner
    {
        /// <summary>"idle", "preparing", "scanning" or "finishing".</summary>
        string Phase { get; }
        /// <summary>The live hint and counts, for the panel's status line.</summary>
        string StatusLine { get; }
        /// <summary>The newest scan's capture ID, its folder's name under files/scans.</summary>
        string LastCaptureId { get; }
        bool TryStart(PerceptionFolder folder, out string why);
        void Finish();
        /// <summary>A finished scan's result, handed over once; null otherwise.</summary>
        ScanResult TakeResult();
    }

    public static class ScanRegistry
    {
        public static IRoomScanner Scanner;
    }
}
