// Shared by the Second Eyes model menus (A1.7c-1): where a model's files are, and reading them.
#if UNITY_EDITOR
using System;
using System.IO;
using System.Security.Cryptography;
using UnityEngine;

namespace SecondEyes.EditorTools
{
    /// <summary>
    /// Paths and file helpers for Assets/SecondEyes/Models/NAME/ and the model.json that grounding/copy_to_unity.py
    /// writes there.
    /// </summary>
    public static class ModelFiles
    {
        public const string ModelsFolder = "Assets/SecondEyes/Models";
        public const string StreamingAssets = "Assets/StreamingAssets";

        // model.json as grounding/copy_to_unity.py writes it (JsonUtility ignores the rest).
        [Serializable] public class FileRecord { public string file, sha256; public long bytes; }
        [Serializable] public class UnityModel
        {
            public string name, quantization, sentis_file, onnx_folder;
            public FileRecord[] onnx_files, tokenizer_files;
        }

        /// <summary>A problem the user can fix. The menus show it in a dialog, having changed nothing.</summary>
        public sealed class ToolError : Exception
        {
            public ToolError(string message) : base(message) { }
        }

        /// <summary>The Unity project folder, quest-app/.</summary>
        public static string ProjectRoot { get { return Directory.GetParent(Application.dataPath).FullName; } }

        /// <summary>The repository folder, which holds quest-app/ and grounding/.</summary>
        public static string RepoRoot { get { return Directory.GetParent(ProjectRoot).FullName; } }

        /// <summary>The model folder (Assets/SecondEyes/Models/NAME) that an asset path is or is inside; null if none.</summary>
        public static string ModelFolderOf(string assetPath)
        {
            string prefix = ModelsFolder + "/";
            if (string.IsNullOrEmpty(assetPath) || !assetPath.StartsWith(prefix, StringComparison.Ordinal)) return null;
            string name = assetPath.Substring(prefix.Length).Split('/')[0];
            return name.Length == 0 ? null : prefix + name;
        }

        /// <summary>Reads FOLDER/model.json.</summary>
        public static UnityModel ReadModelJson(string folder)
        {
            string name = Path.GetFileName(folder);
            return ReadJson<UnityModel>(Path.Combine(ProjectRoot, folder, "model.json"), folder + "/model.json",
                                        "To write it, run: python grounding/copy_to_unity.py grounding/models/" + name + ".json");
        }

        public static T ReadJson<T>(string path, string shown, string hint) where T : class
        {
            if (!File.Exists(path))
                throw new ToolError(shown + " isn't there." + (hint.Length > 0 ? " " + hint : ""));
            T value;
            try { value = JsonUtility.FromJson<T>(File.ReadAllText(path)); }
            catch (Exception e) { throw new ToolError(shown + " can't be read: " + e.Message); }
            if (value == null) throw new ToolError(shown + " is empty.");
            return value;
        }

        /// <summary>A file's SHA-256 in lowercase hex. Progress (0 to 1) is reported at the start and about every 64 MB.</summary>
        public static string Sha256(string path, Action<float> progress = null)
        {
            using (var sha = SHA256.Create())
            using (var stream = File.OpenRead(path))
            {
                var buffer = new byte[1 << 20];
                long done = 0, length = stream.Length;
                int read, sinceReport = 0;
                if (progress != null) progress(0f);
                while ((read = stream.Read(buffer, 0, buffer.Length)) > 0)
                {
                    sha.TransformBlock(buffer, 0, read, null, 0);
                    done += read;
                    if (progress != null && ++sinceReport == 64)
                    {
                        sinceReport = 0;
                        progress(length > 0 ? (float)done / length : 1f);
                    }
                }
                sha.TransformFinalBlock(buffer, 0, 0);
                return BitConverter.ToString(sha.Hash).Replace("-", "").ToLowerInvariant();
            }
        }
    }
}
#endif
