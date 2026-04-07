<div id="readme-top" align="center">
  <h1 align="center">ComfyUI-VoxCPM2</h1>

  <p align="center">
    <b>English</b> | <a href="./docs/README_zh.md">中文</a>
  </p>

  <p align="center">
    ComfyUI nodes for <strong>VoxCPM2</strong> — tokenizer-free, diffusion autoregressive Text-to-Speech.
    <br>2B parameters, 30 languages, 48kHz audio output, voice design, controllable cloning, and LoRA training.
    <br /><br />
    <a href="https://github.com/Saganaki22/ComfyUI-VoxCPM2/issues/new?labels=bug&template=bug-report---.md">Report Bug</a>
    ·
    <a href="https://github.com/Saganaki22/ComfyUI-VoxCPM2/issues/new?labels=enhancement&template=feature-request---.md">Request Feature</a>
  </p>
</div>

<div align="center">

[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![Forks][forks-shield]][forks-url]

</div>

<img width="1646" height="1241" alt="Screenshot 2026-04-07 142144" src="https://github.com/user-attachments/assets/ac7acaf5-018f-4906-bb7e-3fd06a5be048" />


<br>

## About

[VoxCPM2](https://github.com/OpenBMB/VoxCPM) is a tokenizer-free Text-to-Speech model trained on over 2 million hours of multilingual speech data. Built on a MiniCPM-4 backbone with AudioVAE V2, it outputs **48kHz studio-quality audio** and supports **30 languages** with no language tag needed.

This custom node provides two inference nodes and a full LoRA training pipeline, all integrated directly into ComfyUI — based on the original [ComfyUI-VoxCPM](https://github.com/wildminder/ComfyUI-VoxCPM) by [@wildminder](https://github.com/wildminder).

**Key Features:**
* **30-Language Multilingual** — Input text in any supported language, no language tag needed
* **Voice Design** — Generate a novel voice from a natural-language description alone (gender, age, tone, emotion, pace)
* **Controllable Voice Cloning** — Clone any voice from a short reference clip, with optional style guidance
* **Ultimate Cloning** — Provide reference audio + transcript for maximum fidelity reproduction
* **48kHz Studio-Quality Output** — Accepts 16kHz reference audio, outputs 48kHz via AudioVAE V2's built-in super-resolution
* **LoRA Support** — Load fine-tuned LoRA checkpoints for specific voice styles
* **Native LoRA Training** — Train LoRA adapters directly within ComfyUI
* **Automatic Model Management** — Models are downloaded and managed by ComfyUI to save VRAM
* **Torch Compile** — Optional `torch.compile` optimization for faster inference

## Installation

### Via ComfyUI Manager (Recommended)

Search for `ComfyUI-VoxCPM2` and click "Install".

### Manual

1. Clone into your `ComfyUI/custom_nodes/` directory:
   ```sh
   git clone https://github.com/Saganaki22/ComfyUI-VoxCPM2.git
   ```

2. Install dependencies:
   ```sh
   cd ComfyUI-VoxCPM2
   pip install -r requirements.txt
   ```

3. Restart ComfyUI. Nodes appear under the `audio/tts` category.

## Models

The model is downloaded automatically on first use to `ComfyUI/models/tts/VoxCPM/`.

| Model | Parameters | Sample Rate | Description | Hugging Face |
|:---|:---:|:---:|:---|:---|
| **VoxCPM2** | 2B | 48kHz | Latest release. 30 languages, voice design, controllable cloning. | [openbmb/VoxCPM2](https://huggingface.co/openbmb/VoxCPM2) |

## Nodes

### VoxCPM2 TTS

Text-to-speech with optional voice design. No reference audio needed.

| Input | Type | Default | Description |
|:---|:---|:---:|:---|
| `model_name` | Combo | — | Select the VoxCPM2 model |
| `lora_name` | Combo | None | LoRA checkpoint from `models/loras` |
| `voice_description` | String | — | Voice design prompt (e.g. "A young woman, gentle and sweet voice"). Auto-wrapped in parentheses and prepended to text |
| `text` | String | — | Target text to synthesize |
| `cfg_value` | Float | 2.0 | Classifier-Free Guidance scale (1.0–10.0) |
| `inference_timesteps` | Int | 10 | Diffusion steps. More = better quality, slower |
| `max_tokens` | Int | 4096 | Max generation length (64–8192) |
| `normalize_text` | Toggle | Normalize | Auto-process numbers, abbreviations, punctuation |
| `seed` | Int | -1 | Reproducibility seed (-1 = random) |
| `force_offload` | Toggle | Auto | Force VRAM offload after generation |
| `dtype` | Combo | auto | Model dtype: `auto` (native bf16, fp16 on older GPUs), `bf16`, `fp16` |
| `device` | Combo | cuda | Inference device (cuda, mps, cpu) |
| `retry_max_attempts` | Int | 3 | Auto-retries on bad generation (0–10) |
| `retry_threshold` | Float | 6.0 | Threshold for detecting bad generations |
| `torch_compile` | Toggle | Standard | Enable `torch.compile` optimization |

### VoxCPM2 Voice Clone

Voice cloning with controllable and ultimate modes.

| Input | Type | Default | Description |
|:---|:---|:---:|:---|
| `model_name` | Combo | — | Select the VoxCPM2 model |
| `lora_name` | Combo | None | LoRA checkpoint from `models/loras` |
| `voice_description` | String | — | Style control (e.g. "slightly faster, cheerful tone"). Auto-wrapped in parentheses and prepended to text |
| `text` | String | — | Target text to synthesize |
| `reference_audio` | Audio | **Required** | Reference audio for voice cloning |
| `prompt_text` | String | — | Transcript of reference audio. Provide for **Ultimate Cloning** (highest fidelity). Leave empty for **Controllable Cloning** |
| `cfg_value` | Float | 2.0 | Classifier-Free Guidance scale (1.0–10.0) |
| `inference_timesteps` | Int | 10 | Diffusion steps. More = better quality, slower |
| `max_tokens` | Int | 4096 | Max generation length (64–8192) |
| `normalize_text` | Toggle | Normalize | Auto-process numbers, abbreviations, punctuation |
| `seed` | Int | -1 | Reproducibility seed (-1 = random) |
| `force_offload` | Toggle | Auto | Force VRAM offload after generation |
| `dtype` | Combo | auto | Model dtype: `auto` (native bf16, fp16 on older GPUs), `bf16`, `fp16` |
| `device` | Combo | cuda | Inference device (cuda, mps, cpu) |
| `retry_max_attempts` | Int | 3 | Auto-retries on bad generation (0–10) |
| `retry_threshold` | Float | 6.0 | Threshold for detecting bad generations |
| `torch_compile` | Toggle | Standard | Enable `torch.compile` optimization |

## Usage

### Text-to-Speech (Zero-Shot)
1. Add the **VoxCPM2 TTS** node to your workflow.
2. Type your text in the `text` field.
3. Optionally describe a voice in `voice_description` (e.g. "A deep male voice, calm and authoritative").
4. Queue the prompt.

### Voice Design
The `voice_description` field lets you create any voice without reference audio:
- "A young woman, gentle and sweet voice"
- "An old man with a gravelly, slow voice"
- "A child, excited and energetic"

The description is automatically wrapped in parentheses and prepended to your text, matching the VoxCPM2 API format `(description)text`.

### Controllable Voice Cloning
1. Add the **VoxCPM2 Voice Clone** node.
2. Connect a `Load Audio` node to `reference_audio`.
3. Enter your target text in `text`.
4. Optionally add style guidance in `voice_description` (e.g. "slightly faster, cheerful tone").
5. Leave `prompt_text` empty.

### Ultimate Cloning (Highest Fidelity)
1. Same as above, but also provide the **exact transcript** of the reference audio in `prompt_text`.
2. The model uses audio-continuation cloning to reproduce every vocal nuance.

## LoRA Support

### Inference
1. Place `.safetensors` LoRA files in `ComfyUI/models/loras/`.
2. Select your LoRA in the `lora_name` dropdown.

### Training
Train custom LoRA adapters directly in ComfyUI using the training nodes (`VoxCPM2 Train Config`, `VoxCPM2 Dataset Maker`, `VoxCPM2 LoRA Trainer`).

**[Click here for the full LoRA Training Guide](docs/readme-lora-training.md)**

## Tips for Best Results

### Voice Cloning
- Use **clean, high-quality reference audio** (5–15 seconds of continuous speech)
- For **Ultimate Cloning**, provide an accurate verbatim transcript in `prompt_text`
- Punctuation in the transcript helps the model capture intonation

### Generation Quality
- **`cfg_value` (default 2.0):** Raise for more adherence to the prompt, lower for more natural variation
- **`inference_timesteps` (default 10):** 5–10 for fast drafts, 15–25 for higher quality
- **`normalize_text`:** Keep ON for natural language input. Turn OFF only for phoneme input like `{HH AH0 L OW1}`

## Supported Languages (30)

Arabic, Burmese, Chinese, Danish, Dutch, English, Finnish, French, German, Greek, Hebrew, Hindi, Indonesian, Italian, Japanese, Khmer, Korean, Lao, Malay, Norwegian, Polish, Portuguese, Russian, Spanish, Swahili, Swedish, Tagalog, Thai, Turkish, Vietnamese

Chinese Dialects: Sichuan, Cantonese, Wu, Northeastern, Henan, Shaanxi, Shandong, Tianjin, Southern Min

## Limitations

- Voice Design and Style Control results may vary between runs; generating 1–3 times is recommended
- Performance varies across languages depending on training data availability
- Occasional instability with very long or highly expressive inputs
- **Strictly forbidden** to use for impersonation, fraud, or disinformation. AI-generated content should be clearly labeled.

## License

The VoxCPM model and its components are subject to the [Apache-2.0 License](https://github.com/OpenBMB/VoxCPM/blob/main/LICENSE) provided by OpenBMB.

## Acknowledgments

- **[@wildminder](https://github.com/wildminder)** for the original [ComfyUI-VoxCPM](https://github.com/wildminder/ComfyUI-VoxCPM) this project is based on
- **OpenBMB & ModelBest** for creating and open-sourcing [VoxCPM](https://github.com/OpenBMB/VoxCPM)
- **The ComfyUI team** for their powerful and extensible platform

<!-- MARKDOWN LINKS & IMAGES -->
[stars-shield]: https://img.shields.io/github/stars/Saganaki22/ComfyUI-VoxCPM2.svg?style=for-the-badge
[stars-url]: https://github.com/Saganaki22/ComfyUI-VoxCPM2/stargazers
[issues-shield]: https://img.shields.io/github/issues/Saganaki22/ComfyUI-VoxCPM2.svg?style=for-the-badge
[issues-url]: https://github.com/Saganaki22/ComfyUI-VoxCPM2/issues
[forks-shield]: https://img.shields.io/github/forks/Saganaki22/ComfyUI-VoxCPM2.svg?style=for-the-badge
[forks-url]: https://github.com/Saganaki22/ComfyUI-VoxCPM2/network/members
