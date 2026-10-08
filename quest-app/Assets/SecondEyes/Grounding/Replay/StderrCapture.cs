using System;
using System.Runtime.InteropServices;

namespace SecondEyes.Grounding.Replay
{
    /// <summary>
    /// A2.5 (D103): redirects this process's stderr, file descriptor 2, into a file around one native call, so that
    /// llama.cpp's own startup lines (which se_llama writes to stderr when verbose) are kept. stderr is process-wide:
    /// whatever any thread writes to it in that window lands in the file too. Begin flushes C stdio, saves the original
    /// descriptor and points 2 at the file; Dispose flushes again, restores the original and closes the saved copy,
    /// whether the call succeeded, failed or threw. No native rebuild: these are the system C library's calls.
    /// </summary>
    public sealed class StderrCapture : IDisposable
    {
        private const string Libc = "libc";
        private const int StderrFd = 2;
        private const int FileMode = 420;   // 0644

        [DllImport(Libc, SetLastError = true)] private static extern int dup(int fd);
        [DllImport(Libc, SetLastError = true)] private static extern int dup2(int fd, int fd2);
        [DllImport(Libc, SetLastError = true)] private static extern int close(int fd);
        [DllImport(Libc, SetLastError = true)] private static extern int creat([MarshalAs(UnmanagedType.LPStr)] string path, int mode);
        [DllImport(Libc)] private static extern int fflush(IntPtr stream);

        private int saved = -1;
        private bool restored;

        public string Path { get; private set; }
        /// <summary>Set when restoring the original descriptor failed (errno); null otherwise.</summary>
        public string RestoreError { get; private set; }

        private StderrCapture() { }

        public static StderrCapture Begin(string path)
        {
            fflush(IntPtr.Zero);
            int fd = creat(path, FileMode);
            if (fd < 0) throw new InvalidOperationException("creat failed, errno " + Marshal.GetLastWin32Error());
            int copy = dup(StderrFd);
            if (copy < 0)
            {
                int e = Marshal.GetLastWin32Error();
                close(fd);
                throw new InvalidOperationException("dup failed, errno " + e);
            }
            if (dup2(fd, StderrFd) < 0)
            {
                int e = Marshal.GetLastWin32Error();
                close(fd);
                close(copy);
                throw new InvalidOperationException("dup2 failed, errno " + e);
            }
            close(fd);   // descriptor 2 now refers to the file
            return new StderrCapture { saved = copy, Path = path };
        }

        public void Dispose()
        {
            if (restored) return;
            restored = true;
            fflush(IntPtr.Zero);
            if (dup2(saved, StderrFd) < 0) RestoreError = "dup2 restore failed, errno " + Marshal.GetLastWin32Error();
            close(saved);
        }
    }
}
