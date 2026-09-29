// Second Eyes > Convert model (A1.7c-1, D38): converts a model's ONNX export to a .sentis file without importing it.
#if UNITY_EDITOR
using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Reflection;
using System.Runtime.ExceptionServices;
using System.Text;
using UnityEditor;
using UnityEngine;

namespace SecondEyes.EditorTools
{
    /// <summary>
    /// Converts the ONNX export that Assets/SecondEyes/Models/NAME/model.json names into Unity's .sentis format, rounds
    /// its weights as model.json says (Float16 for A1.7c), and saves it in Assets/StreamingAssets/ under model.json's
    /// name (D34). The ONNX file is read from grounding/models/NAME/onnx/ and never enters the project: Unity's importer
    /// stores the full 32-bit model first, which ran out of memory for a 2.5 GB model (D38). The ONNX files' fingerprints
    /// are checked against model.json first. The result is written to Temp/ and only then moved into place, so an
    /// interrupted conversion never leaves a half-written model under the real name, which the headset would copy once
    /// and keep. Unity's ONNX converter isn't public API, so it is found by name (Inference Engine 2.2.1, D36); if that
    /// changes, this stops and says so.
    /// </summary>
    public static class ConvertModel
    {
        [MenuItem("Second Eyes/Convert model", false, 1)]
        static void ConvertSelected()
        {
            var selected = Selection.activeObject;
            string folder = selected == null ? null : ModelFiles.ModelFolderOf(AssetDatabase.GetAssetPath(selected));
            try
            {
                string report = Convert(folder,
                    question => EditorUtility.DisplayDialog("Convert model", question, "Convert", "Cancel"),
                    (step, fraction) => EditorUtility.DisplayProgressBar("Convert model", step, fraction));
                if (report == null) return;   // cancelled
                Debug.Log(report);
                EditorUtility.DisplayDialog("Convert model", "Done. The Console has the details.", "OK");
            }
            catch (ModelFiles.ToolError e)
            {
                Debug.LogError("[Convert model] " + e.Message);
                EditorUtility.DisplayDialog("Convert model", e.Message, "OK");
            }
            catch (OutOfMemoryException e)
            {
                Debug.LogException(e);
                EditorUtility.DisplayDialog("Convert model", "Unity ran out of memory while converting. Nothing was saved " +
                    "under the model's name. Close other programs, restart Unity and try again.", "OK");
            }
            finally
            {
                EditorUtility.ClearProgressBar();
            }
        }

        /// <summary>
        /// Converts the model in FOLDER (Assets/SecondEyes/Models/NAME) and returns the report, or null if confirm said no.
        /// Everything is checked before confirm is asked; a ToolError means nothing was converted.
        /// </summary>
        public static string Convert(string folder, Func<string, bool> confirm, Action<string, float> progress)
        {
            if (folder == null)
                throw new ModelFiles.ToolError("Select the model's folder in " + ModelFiles.ModelsFolder + "/, or a file in it, first.");
            string name = Path.GetFileName(folder);
            var model = ModelFiles.ReadModelJson(folder);
            string descFile = "grounding/models/" + name + ".json";
            string rerun = "run: python grounding/copy_to_unity.py " + descFile;

            var problems = new List<string>();
            if (model.name != name)
                problems.Add(folder + "/model.json is for '" + model.name + "'; " + rerun);
            if (string.IsNullOrEmpty(model.sentis_file) || model.sentis_file.IndexOfAny(new[] { '/', '\\' }) >= 0)
                problems.Add(folder + "/model.json names no .sentis file; " + rerun);
            var files = (model.onnx_files ?? new ModelFiles.FileRecord[0]).Where(f => f != null && !string.IsNullOrEmpty(f.file)).ToArray();
            if (string.IsNullOrEmpty(model.onnx_folder) || !files.Any(f => f.file == "model.onnx"))
                problems.Add(folder + "/model.json doesn't describe the ONNX export; " + rerun);
            string onnxDir = string.IsNullOrEmpty(model.onnx_folder) ? null : Path.Combine(ModelFiles.RepoRoot, model.onnx_folder);
            if (onnxDir != null)
                foreach (var f in files)
                {
                    string path = Path.Combine(onnxDir, f.file), shown = model.onnx_folder + "/" + f.file;
                    if (!File.Exists(path))
                        problems.Add(shown + " isn't there. Export it: python grounding/export_onnx.py " + descFile + "; then " + rerun);
                    else if (new FileInfo(path).Length != f.bytes)
                        problems.Add(shown + " has changed since model.json was written. If you exported again, " + rerun);
                }
            InferenceEngine engine = null;
            try { engine = InferenceEngine.Find(); }
            catch (ModelFiles.ToolError e) { problems.Add(e.Message); }
            if (engine != null && !engine.HasQuantization(model.quantization))
                problems.Add(folder + "/model.json asks for quantization '" + model.quantization + "', which Unity's Inference " +
                             "Engine doesn't have (it has None, " + string.Join(", ", engine.QuantizationNames) + ")");
            if (problems.Count > 0)
                throw new ModelFiles.ToolError("Nothing was converted. Fix these first:\n- " + string.Join("\n- ", problems.ToArray()));

            string target = ModelFiles.StreamingAssets + "/" + model.sentis_file;
            string targetPath = Path.Combine(ModelFiles.ProjectRoot, target);
            string onnxShown = model.onnx_folder + "/model.onnx";
            string question = "Converting takes a few minutes and several GB of memory, and Unity doesn't respond meanwhile. " +
                              "Close big programs such as a web browser first.\n\nConvert " + onnxShown + " to " + target + " now?";
            if (File.Exists(targetPath)) question += "\n\nThat file already exists and will be replaced.";
            if (!confirm(question)) return null;

            var times = new List<string>();
            var watch = System.Diagnostics.Stopwatch.StartNew();
            foreach (var f in files)
            {
                string shown = model.onnx_folder + "/" + f.file;
                if (ModelFiles.Sha256(Path.Combine(onnxDir, f.file), x => progress("Checking " + shown, x)) != f.sha256)
                    throw new ModelFiles.ToolError(shown + " isn't the file model.json describes (its fingerprint differs). " +
                                                   "If you exported again, " + rerun + ". Nothing was converted.");
            }
            Lap(times, "check", watch);

            progress("Converting " + onnxShown + ". This takes a few minutes.", 0.2f);
            object converted = engine.Convert(Path.Combine(onnxDir, "model.onnx"));
            try
            {
                Collect();
                Lap(times, "convert", watch);
                if (model.quantization != "None")
                {
                    progress("Rounding the weights to " + model.quantization, 0.6f);
                    converted = engine.Quantize(converted, model.quantization);
                    Collect();
                    Lap(times, "round", watch);
                }
                progress("Saving " + target, 0.8f);
                string temp = Path.Combine(ModelFiles.ProjectRoot, "Temp", model.sentis_file);
                Directory.CreateDirectory(Path.GetDirectoryName(temp));
                if (File.Exists(temp)) File.Delete(temp);
                engine.Save(temp, converted);
                Directory.CreateDirectory(Path.GetDirectoryName(targetPath));
                if (File.Exists(targetPath)) File.Delete(targetPath);
                File.Move(temp, targetPath);
                Lap(times, "save", watch);
            }
            finally
            {
                engine.DisposeWeights(converted);
                Collect();
            }
            AssetDatabase.ImportAsset(target);
            double gigabytes = new FileInfo(targetPath).Length / 1e9;
            return "[Convert model] Converted " + onnxShown + " (fingerprints match model.json) with Unity Inference Engine " +
                   engine.Version + ", weights " + model.quantization + ", to " + target + " (" +
                   gigabytes.ToString("F2", CultureInfo.InvariantCulture) + " GB). Time: " + string.Join(", ", times.ToArray()) + ".";
        }

        static void Lap(List<string> times, string step, System.Diagnostics.Stopwatch watch)
        {
            times.Add(step + " " + watch.Elapsed.TotalSeconds.ToString("F0", CultureInfo.InvariantCulture) + " s");
            watch.Reset();
            watch.Start();
        }

        static void Collect()
        {
            GC.Collect();
            GC.WaitForPendingFinalizers();
            GC.Collect();
        }

        /// <summary>Unity's Inference Engine, reached by name at run time so that this file compiles whatever Unity changes.</summary>
        sealed class InferenceEngine
        {
            const BindingFlags Any = BindingFlags.Public | BindingFlags.NonPublic;
            ConstructorInfo converterConstructor;
            MethodInfo convert, quantize, save, disposeWeights;
            Type quantizationType;
            public string Version = "(version unknown)";

            public string[] QuantizationNames { get { return Enum.GetNames(quantizationType); } }

            public bool HasQuantization(string name)
            {
                return name == "None" || (name != null && QuantizationNames.Contains(name));
            }

            public static InferenceEngine Find()
            {
                var types = AppDomain.CurrentDomain.GetAssemblies()
                    .Where(a => a.GetName().Name.StartsWith("Unity.InferenceEngine", StringComparison.Ordinal))
                    .SelectMany(TypesOf).ToArray();
                var engine = new InferenceEngine();
                Type modelType = Named(types, "Unity.InferenceEngine", "Model");
                engine.quantizationType = Named(types, "Unity.InferenceEngine", "QuantizationType");
                Type converter = Named(types, null, "ONNXModelConverter");
                Type quantizer = Named(types, "Unity.InferenceEngine", "ModelQuantizer");
                Type writer = Named(types, "Unity.InferenceEngine", "ModelWriter");

                var missing = new List<string>();
                if (modelType == null)
                    missing.Add("Model");
                if (converter != null && modelType != null)
                {
                    engine.converterConstructor = converter.GetConstructors(Any | BindingFlags.Instance).FirstOrDefault(c =>
                    {
                        var ps = c.GetParameters();
                        return ps.Length > 0 && ps[0].ParameterType == typeof(string) && ps.Skip(1).All(p => p.HasDefaultValue);
                    });
                    engine.convert = converter.GetMethods(Any | BindingFlags.Instance).FirstOrDefault(m =>
                        m.Name == "Convert" && modelType.IsAssignableFrom(m.ReturnType) && m.GetParameters().All(p => p.HasDefaultValue));
                }
                if (engine.converterConstructor == null || engine.convert == null)
                    missing.Add("ONNXModelConverter(path).Convert()");
                if (quantizer != null && modelType != null && engine.quantizationType != null)
                    engine.quantize = quantizer.GetMethods(Any | BindingFlags.Static).FirstOrDefault(m =>
                        m.Name == "QuantizeWeights" && IsQuantize(m.GetParameters(), engine.quantizationType, modelType));
                if (engine.quantize == null)
                    missing.Add("ModelQuantizer.QuantizeWeights(QuantizationType, ref Model)");
                if (writer != null && modelType != null)
                    engine.save = writer.GetMethod("Save", BindingFlags.Public | BindingFlags.Static, null, new[] { typeof(string), modelType }, null);
                if (engine.save == null)
                    missing.Add("ModelWriter.Save(string, Model)");
                if (missing.Count > 0)
                    throw new ModelFiles.ToolError("Unity's Inference Engine has no " + string.Join(", no ", missing.ToArray()) +
                        ". Is com.unity.ai.inference 2.2.1 installed (D36)? Another version may have renamed them.");

                engine.disposeWeights = modelType.GetMethod("DisposeWeights", Any | BindingFlags.Instance, null, Type.EmptyTypes, null);
                var package = UnityEditor.PackageManager.PackageInfo.FindForAssembly(modelType.Assembly);
                if (package != null) engine.Version = package.version;
                return engine;
            }

            public object Convert(string onnxPath)
            {
                var ps = converterConstructor.GetParameters();
                var args = new object[ps.Length];
                args[0] = onnxPath;
                for (int i = 1; i < ps.Length; i++) args[i] = ps[i].DefaultValue;
                object converter = Call(() => converterConstructor.Invoke(args));
                object[] convertArgs = convert.GetParameters().Select(p => p.DefaultValue).ToArray();
                return Call(() => convert.Invoke(converter, convertArgs));
            }

            /// <summary>Rounds the weights; returns the model, which QuantizeWeights may have replaced through its ref parameter.</summary>
            public object Quantize(object model, string quantization)
            {
                bool modelFirst = quantize.GetParameters()[0].ParameterType.IsByRef;
                var args = new object[2];
                args[modelFirst ? 0 : 1] = model;
                args[modelFirst ? 1 : 0] = Enum.Parse(quantizationType, quantization);
                Call(() => quantize.Invoke(null, args));
                return args[modelFirst ? 0 : 1];
            }

            public void Save(string path, object model)
            {
                Call(() => save.Invoke(null, new[] { path, model }));
            }

            public void DisposeWeights(object model)
            {
                if (disposeWeights == null || model == null) return;
                try { disposeWeights.Invoke(model, null); }
                catch (Exception e) { Debug.LogWarning("[Convert model] Couldn't free the model's weights: " + e.Message); }
            }

            static bool IsQuantize(ParameterInfo[] ps, Type quantizationType, Type modelType)
            {
                if (ps.Length != 2) return false;
                Func<ParameterInfo, bool> isModel = p => p.ParameterType.IsByRef && p.ParameterType.GetElementType() == modelType;
                return (ps[0].ParameterType == quantizationType && isModel(ps[1])) || (isModel(ps[0]) && ps[1].ParameterType == quantizationType);
            }

            static IEnumerable<Type> TypesOf(Assembly assembly)
            {
                try { return assembly.GetTypes(); }
                catch (ReflectionTypeLoadException e) { return e.Types.Where(t => t != null); }
            }

            static Type Named(Type[] types, string ns, string name)
            {
                return types.FirstOrDefault(t => t.Name == name && t.DeclaringType == null &&
                    (ns == null ? t.Namespace != null && t.Namespace.StartsWith("Unity.InferenceEngine", StringComparison.Ordinal)
                                : t.Namespace == ns));
            }

            /// <summary>Runs a reflected call, rethrowing what the call itself threw (for example OutOfMemoryException).</summary>
            static object Call(Func<object> call)
            {
                try { return call(); }
                catch (TargetInvocationException e)
                {
                    if (e.InnerException != null) ExceptionDispatchInfo.Capture(e.InnerException).Throw();
                    throw;
                }
            }
        }
    }
}
#endif
