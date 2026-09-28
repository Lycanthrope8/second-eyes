using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Globalization;
using System.IO;
using System.Text;
using System.Threading;
using UnityEngine;
using Debug = UnityEngine.Debug;

namespace SecondEyes.Logging
{
    /// <summary>
    /// Writes the session event log described in docs/logging.md (format v1): one JSON Lines file per
    /// app session, in Application.persistentDataPath/logs/, named "UTC start_session ID.jsonl".
    /// Each event is timestamped when it is logged and queued in memory. A background thread writes
    /// the queue once per second (D17), and right away when the app pauses or quits, so a crash loses
    /// at most the last second.
    /// Put one EventLog on an object in the first scene. Other scripts call EventLog.Mark,
    /// EventLog.Error or EventLog.Write; without an EventLog in the scene, those calls do nothing.
    /// </summary>
    [DefaultExecutionOrder(-1000)]
    public class EventLog : MonoBehaviour
    {
        public const int Format = 1;
        private const int WriteIntervalMs = 1000;
        private static readonly long EpochTicks = new DateTime(1970, 1, 1, 0, 0, 0, DateTimeKind.Utc).Ticks;

        private static EventLog instance;

        private readonly object gate = new object();
        private List<string> pending = new List<string>();
        private List<string> batch = new List<string>();
        private long seq;
        private bool stopping;
        private bool reportedWriteFailure;
        private Thread writer;
        private StreamWriter file;

        /// <summary>ID of the current session, as written in session.start.</summary>
        public static string SessionId { get; private set; }

        /// <summary>Full path of the current session's log file on this device.</summary>
        public static string FilePath { get; private set; }

        /// <summary>Logs one event. dataJson must be a JSON object, for example {"text":"hello"}.</summary>
        public static void Write(string ev, string dataJson)
        {
            EventLog log = instance;
            if (log == null)
            {
                return;
            }
            lock (log.gate)
            {
                if (!log.stopping)
                {
                    log.Enqueue(ev, dataJson);
                }
            }
        }

        /// <summary>Logs a note, for example a button press.</summary>
        public static void Mark(string text)
        {
            var data = new StringBuilder("{\"text\":");
            Json.AppendString(data, text);
            data.Append('}');
            Write("mark", data.ToString());
        }

        /// <summary>Logs something that went wrong.</summary>
        public static void Error(string where, string message)
        {
            var data = new StringBuilder("{\"where\":");
            Json.AppendString(data, where);
            data.Append(",\"message\":");
            Json.AppendString(data, message);
            data.Append('}');
            Write("error", data.ToString());
        }

        private void Awake()
        {
            if (instance != null)
            {
                Debug.LogWarning("[SecondEyes] EventLog: there is already an EventLog; this one is removed.");
                Destroy(this);
                return;
            }

            SessionId = Guid.NewGuid().ToString("N").Substring(0, 8);
            string folder = Path.Combine(Application.persistentDataPath, "logs");
            Directory.CreateDirectory(folder);
            string start = DateTime.UtcNow.ToString("yyyyMMdd'T'HHmmss'Z'", CultureInfo.InvariantCulture);
            FilePath = Path.Combine(folder, start + "_" + SessionId + ".jsonl");
            file = new StreamWriter(FilePath, false, new UTF8Encoding(false)); // UTF-8 without BOM
            instance = this;

            var data = new StringBuilder("{\"format\":");
            data.Append(Format);
            data.Append(",\"session_id\":");
            Json.AppendString(data, SessionId);
            data.Append(",\"app_version\":");
            Json.AppendString(data, Application.version);
            data.Append(",\"os_build\":");
            Json.AppendString(data, OsBuild());
            data.Append('}');
            Write("session.start", data.ToString());

            writer = new Thread(WriteLoop) { IsBackground = true, Name = "SecondEyes EventLog" };
            writer.Start();
            Debug.Log("[SecondEyes] EventLog: writing " + FilePath);
        }

        private void OnApplicationPause(bool paused)
        {
            if (paused)
            {
                lock (gate)
                {
                    Monitor.PulseAll(gate); // write what's queued now, in case the app is killed while paused
                }
            }
        }

        private void OnApplicationQuit()
        {
            Stop();
        }

        private void OnDestroy()
        {
            Stop();
        }

        // Called with the gate held, so seq and mono_us always rise together.
        private void Enqueue(string ev, string dataJson)
        {
            long mono = MonoMicros();
            long utc = (DateTime.UtcNow.Ticks - EpochTicks) / 10;
            var line = new StringBuilder(80 + dataJson.Length);
            line.Append("{\"seq\":").Append(seq++);
            line.Append(",\"mono_us\":").Append(mono);
            line.Append(",\"utc_us\":").Append(utc);
            line.Append(",\"ev\":");
            Json.AppendString(line, ev);
            line.Append(",\"data\":").Append(dataJson).Append('}');
            pending.Add(line.ToString());
        }

        private void Stop()
        {
            if (instance != this)
            {
                return;
            }
            lock (gate)
            {
                if (stopping)
                {
                    return;
                }
                Enqueue("session.end", "{}");
                stopping = true;
                Monitor.PulseAll(gate);
            }
            if (writer != null && !writer.Join(2000))
            {
                Debug.LogWarning("[SecondEyes] EventLog: the writer did not finish in time; the end of the log may be missing.");
            }
            instance = null;
        }

        private void WriteLoop()
        {
            bool last = false;
            while (!last)
            {
                lock (gate)
                {
                    if (!stopping)
                    {
                        Monitor.Wait(gate, WriteIntervalMs);
                    }
                    last = stopping;
                    List<string> swap = pending;
                    pending = batch;
                    batch = swap;
                }
                WriteBatch();
            }
            file.Dispose();
        }

        private void WriteBatch()
        {
            if (batch.Count == 0)
            {
                return;
            }
            try
            {
                foreach (string line in batch)
                {
                    file.Write(line);
                    file.Write('\n');
                }
                file.Flush();
            }
            catch (Exception e)
            {
                if (!reportedWriteFailure)
                {
                    reportedWriteFailure = true;
                    Debug.LogWarning("[SecondEyes] EventLog: writing failed: " + e.Message);
                }
            }
            batch.Clear();
        }

        private static long MonoMicros()
        {
            long ticks = Stopwatch.GetTimestamp();
            long frequency = Stopwatch.Frequency;
            return ticks / frequency * 1000000L + ticks % frequency * 1000000L / frequency;
        }

        private static string OsBuild()
        {
#if UNITY_ANDROID && !UNITY_EDITOR
            try
            {
                using (var build = new AndroidJavaClass("android.os.Build"))
                {
                    return build.GetStatic<string>("FINGERPRINT");
                }
            }
            catch (Exception)
            {
                return null;
            }
#else
            return null;
#endif
        }
    }

    /// <summary>The little JSON the log needs: strings and numbers.</summary>
    public static class Json
    {
        public static void AppendString(StringBuilder sb, string value)
        {
            if (value == null)
            {
                sb.Append("null");
                return;
            }
            sb.Append('"');
            foreach (char c in value)
            {
                switch (c)
                {
                    case '"': sb.Append("\\\""); break;
                    case '\\': sb.Append("\\\\"); break;
                    case '\n': sb.Append("\\n"); break;
                    case '\r': sb.Append("\\r"); break;
                    case '\t': sb.Append("\\t"); break;
                    case '\b': sb.Append("\\b"); break;
                    case '\f': sb.Append("\\f"); break;
                    default:
                        if (c < ' ')
                        {
                            sb.Append("\\u").Append(((int)c).ToString("x4", CultureInfo.InvariantCulture));
                        }
                        else
                        {
                            sb.Append(c);
                        }
                        break;
                }
            }
            sb.Append('"');
        }

        public static void AppendNumber(StringBuilder sb, float value)
        {
            if (float.IsNaN(value) || float.IsInfinity(value))
            {
                sb.Append("null");
                return;
            }
            sb.Append(value.ToString("R", CultureInfo.InvariantCulture));
        }
    }
}
