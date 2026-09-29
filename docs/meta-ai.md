# Meta's on-device AI: what we rely on

Facts from Meta's documentation and GitHub that the project depends on, so the source PDFs don't have to be kept. Written in A1.7a (2026-09-28), extended in A1.7c-1. If Meta changes something, update this page and log it in `notes/decisions.md`.

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
- **Conversion and quantization.** Meta → Tools → Unity Inference Engine → ONNX → Sentis Converter imports an ONNX file, optionally quantizes its weights (None = 32-bit, Float16, Uint8) and writes a `.sentis` file. The ONNX file has to be imported into the project (`Assets/`) first; the Inference Engine package imports `.onnx` files as models. Meta's converter is compiled into one of Meta's DLLs; the package cache has no source for it.
- **Unity's importer and large models** (Inference Engine 2.2.1 source, read in A1.7c-1). As soon as an `.onnx` file is in `Assets/`, `ONNXModelImporter` converts it with `ONNXModelConverter` and stores the full 32-bit model with `ModelWriter.SaveModel`. The writer packs the weights in pieces of at most 2^31 − 33 bytes (about 2.1 GB) and holds each piece three times at once: the gathered weights, the FlatBuffer and the final array. For Qwen2.5-0.5B's 2.52 GB of 32-bit weights on a 15.4 GB PC, that ran out of memory. We therefore convert without importing (D38): Second Eyes → Convert model calls `ONNXModelConverter` itself, which isn't public API and is found by name, then `ModelQuantizer.QuantizeWeights` and only then `ModelWriter.Save`, so only the 1.26 GB of 16-bit weights are ever packed. The same is possible in code with `ModelQuantizer.QuantizeWeights(QuantizationType.Float16, ref model)` and `ModelWriter.Save(path, model)`. `.sentis` loads faster and uses less memory than ONNX.
- **Warm-up.** The first run allocates buffers, compiles GPU kernels and uploads weights, which causes a one-time delay of several seconds. Warm up while loading, keep the worker alive, and warm several models up one after another.
- **Changing models.** Agents are provider-agnostic: switching models means switching the provider asset and its files, not the code. Custom providers implement `IChatTask`, inherit from `AIProviderBase`, and appear in the setup wizard through a `CreateAssetMenu` path under `Meta/AI/Provider Assets/`.

## From Meta's source (SDK v207)

Read in A1.7a from `com.meta.xr.sdk.core` in `quest-app/Library/PackageCache/`: `UnityInferenceEngineProvider.cs`, `OnDeviceLLMConfig.cs`, `GPT2Tokenizer.cs` and `TextOnlyLLMRunner.cs`. `grounding/meta_runner.py` mirrors the runner on the PC.

- **Model.** `OnDeviceLLMConfig.cs` names Qwen2.5-0.5B as its example: 24 layers, 2 key-value heads, head size 64, end token 151645, vocabulary 151936. There is no download link, so we export Qwen2.5-0.5B-Instruct to ONNX ourselves (`grounding/export_onnx.py`).
- **Inputs and outputs.** The runner feeds `input_ids`, `attention_mask`, `position_ids`, `past_key_values.N.key` and `past_key_values.N.value`, and reads `logits`, `present.N.key` and `present.N.value`. That is the standard naming of Hugging Face's ONNX export for text generation with a cache.
- **Generation.** The whole prompt runs at once, then one token at a time with the cache. It always takes the most likely token (no sampling), stopping at the end token or after `maxNewTokens`. After every token it copies all layers' caches, so the cost per token grows with the answer's length.
- **Probabilities.** Every token's probability is computed but not handed out. A1.8 needs our own provider for that.
- **Calling it** (`IChatTask.cs`, A1.7c-2). `ChatAsync(new ChatRequest(text), stream)` returns a `ChatResponse` with `text`; `stream` receives one `ChatDelta` (`textFragment`) per answer token, reported straight from the generation loop. The runner builds the prompt with `config.ApplyChatTemplate(user, config.defaultSystemMessage)` and `config.Tokenizer.EncodeAsync`, which our panel repeats to log the same token IDs. Nothing warms the provider up by itself; the panel calls `WarmUp()` and times it.
- **Prompt.** `ChatAsync` passes only the user text; the system message is the config's `defaultSystemMessage`, fixed per provider asset. The default template is Qwen's own format. Prompts over `maxPromptLength` (512) lose their *beginning*, so our instructions would be cut first.
- **Tokenizer.** Despite its name, `GPT2Tokenizer.cs` uses Qwen2's text-splitting pattern, normalizes text to NFC, and reads Qwen's special tokens from `tokenizer_config.json`. It splits `merges.txt` at both line-ending characters, so Windows line endings don't change the tokenization.
- **Loading.** With Use Streaming Asset on, the first use on the headset copies the `.sentis` file from the app into `Application.persistentDataPath`, and later uses load that copy. If a file with the same name is already there, it is used as is, even if it came from an older model or was left incomplete by an interrupted first start; hence the naming rule D34 (`docs/setup/quest-model.md`). `ModelLoader.Load` itself runs on the main thread.
- **Defaults.** The chat defaults to the CPU backend (the comment says it's recommended for language models, for better accuracy), NonBlocking mode and 150 steps per frame. The shape defaults are SmolLM's: 30 layers, 3 key-value heads, head size 64, end token 2. Qwen2.5-0.5B needs 24, 2, 64 and 151645. The default chat template is already Qwen's format.
- **Two backend settings.** The provider's own Backend (default GPUCompute) is used only for object detection and image segmentation. The chat runs on the LLM config's Backend Type.
- **Setting names**, as Second Eyes → Fill chat provider uses them. On the provider: `mode`, `modelFile`, `streamingAssetModel`, `useStreamingAsset`, the hidden `streamingAssetFileName` (Meta's inspector fills it in from the model picked) and `llmConfig`. Inside `llmConfig`: `inferenceExecutionMode`, `stepsPerFrame`, `vocabFile`, `mergesFile`, `tokenizerConfigFile`, `chatTemplateFormat`, `defaultSystemMessage`, `maxLayers`, `numKeyValueHeads`, `headDim`, `eosTokenId`, `maxNewTokens`, `maxPromptLength` and `backendType`. They exist only while the Inference Engine package is installed.
- **Package.** The on-device code only switches on when Unity's Inference Engine package (`com.unity.ai.inference`) version 2.2.1 or newer is installed (define `UNITY_INFERENCE_INSTALLED`).

## Camera and object detection (A1.10)

From `oculus-samples/Unity-PassthroughCameraApiSamples`:

- Samples include CameraToWorld (camera pose to world-space rays), MultiObjectDetection (a YOLO model on camera frames with the Inference Engine), CameraViewer, BrightnessEstimation and ShaderSample.
- Requirements: Quest 3 or 3S with Horizon OS v74+, Unity 6000.0.38f1+, Meta MRUK v81+, and Unity Inference Engine v2.2.1 for the detection sample.
