# Meta's on-device AI: what we rely on

Facts from Meta's documentation and GitHub that the project depends on, so the source PDFs don't have to be kept. Written in A1.7a (2026-09-28). If Meta changes something, update this page and log it in `notes/decisions.md`.

## Sources

- Meta Horizon OS developer docs, all "Updated: Jul 8, 2026": *AI Building Blocks - Overview*, *Providers and Inference Types*, *Unity Inference Engine*, *Agents and Building Blocks*, *Adding New Providers*, *Troubleshooting and FAQ*, plus the API reference *UnityInferenceEngineProvider Class (Unity SDK v207)*.
- GitHub `oculus-samples/Unity-PassthroughCameraApiSamples`, last updated 2026-08-13 (camera and object detection; for A1.10).
- GitHub `meta-quest/agentic-tools`, last updated 2026-09-23: Meta's toolkit for AI coding assistants (the `metavr` command-line tool and Quest development guides). Nothing on language models or the Inference Engine.

## On-device language models (A1.7, A1.8)

- **Where they run.** Meta's AI Building Blocks pair an *agent* (runtime logic, for example `LlmAgent`) with a *provider* (a ScriptableObject that decides where inference runs). On the headset, the provider is `UnityInferenceEngineProvider`, running on Unity's Inference Engine (formerly Sentis). Namespace: `Meta.XR.BuildingBlocks.AIBlocks`.
- **Requirements.** Unity 6 or newer, Quest 3 or 3S, Meta XR Core SDK v83+ for the blocks, with on-device LLM chat since v85. Our setup: Unity 6000.3.25f1, SDK 207.0.0.
- **Supported models.** SmolLM (135M, 360M), Qwen 0.5B, Phi-2 and Phi-3: decoder-only transformers, text only.
- **Setup.** Convert the model to `.sentis`. Create the provider (Create → Meta → AI → Provider Assets → On-Device → Unity Inference Engine), set its mode to Chat, and fill in `OnDeviceLlmConfig`:
  - vocabulary file (JSON), BPE merges file (TXT) and tokenizer config file (JSON), taken from the model;
  - a chat template string, with `{0}` for the user message and `{1}` for the system message;
  - Max Tokens (default 100) and Max Prompt Length (default 512);
  - architecture fields that must match the model: `maxLayers`, `numKeyValueHeads`, `headDim`, `eosTokenId`.
- **Other provider settings.** Backend (CPU or GPUCompute), Use Streaming Asset (load from `StreamingAssets`), Split Over Frames with Layers Per Frame, Steps Per Frame (50–300).
- **Execution modes.** Blocking takes about 1–5 s and may freeze frames. NonBlocking takes about 5–15 s and keeps the frame rate smooth.
- **API.** `Task<ChatResponse> ChatAsync(ChatRequest req, IProgress<ChatDelta> stream = null, CancellationToken ct = default)` returns the full text, with `stream` receiving tokens as they come. `LlmAgent.SendPromptAsync()` raises `onPromptSent` and `onResponseReceived`. The provider has `WarmUp()`, and agents warm up on-device providers on their own.
- **Conversion and quantization.** Meta → Tools → Unity Inference Engine → ONNX → Sentis Converter imports an ONNX file, optionally quantizes its weights (None = 32-bit, Float16, Uint8) and writes a `.sentis` file. The same is possible in code with `ModelQuantizer.QuantizeWeights(QuantizationType.Float16, ref model)` and `ModelWriter.Save(path, model)`. `.sentis` loads faster and uses less memory than ONNX.
- **Warm-up.** The first run allocates buffers, compiles GPU kernels and uploads weights, which causes a one-time delay of several seconds. Warm up while loading, keep the worker alive, and warm several models up one after another.
- **Changing models.** Agents are provider-agnostic: switching models means switching the provider asset and its files, not the code. Custom providers implement `IChatTask`, inherit from `AIProviderBase`, and appear in the setup wizard through a `CreateAssetMenu` path under `Meta/AI/Provider Assets/`.

## Not in Meta's docs (to find elsewhere)

- Where to download a Qwen 0.5B model in the right form. We export Qwen2.5-0.5B-Instruct to ONNX ourselves.
- Which inputs and outputs the provider expects from the model file.
- Whether the provider can report token probabilities (needed in A1.8).

The last two are answerable from the SDK's C# source in `quest-app/Library/PackageCache/`.

## Camera and object detection (A1.10)

From `oculus-samples/Unity-PassthroughCameraApiSamples`:

- Samples include CameraToWorld (camera pose to world-space rays), MultiObjectDetection (a YOLO model on camera frames with the Inference Engine), CameraViewer, BrightnessEstimation and ShaderSample.
- Requirements: Quest 3 or 3S with Horizon OS v74+, Unity 6000.0.38f1+, Meta MRUK v81+, and Unity Inference Engine v2.2.1 for the detection sample.
