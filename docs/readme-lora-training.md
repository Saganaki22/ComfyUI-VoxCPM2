# VoxCPM2 LoRA Training Guide

This guide details how to fine-tune VoxCPM2 models using LoRA (Low-Rank Adaptation) directly within ComfyUI. This process allows you to clone voices or adapt the model's style using a small dataset of audio samples.

---

## Prerequisites

1.  **Hardware**: NVIDIA GPU with at least 12GB VRAM (24GB recommended for higher batch sizes).
2.  **Dataset**: A collection of high-quality `.wav` audio files and corresponding text transcripts.
3.  **Base Model**: Ensure `VoxCPM2` is downloaded in `models/tts/VoxCPM`.
4.  **Training Dependencies**: The training nodes require two extra packages not installed by default (`argbind` and `datasets`). Install them using the method that matches your setup:

    **Standalone Python / venv:**
    ```sh
    pip install argbind datasets
    ```

    **ComfyUI Portable (embedded Python):**
    ```sh
    # From the ComfyUI_windows_portable folder:
    python_embeded\python.exe -m pip install argbind datasets
    ```

    **Using uv (any setup):**
    ```sh
    uv pip install argbind datasets
    ```

    If these are missing, the training nodes will show an error with the install command when you try to use them.

---

## Data Preparation

Your training data must consist of pairs of audio files and text transcripts.

### 1. Folder Structure
Organize your data into a single directory. The audio and text files must share the same filename (excluding extension).

```
my_dataset/
├── voice_001.wav
├── voice_001.txt
├── voice_002.wav
├── voice_002.txt
└── ...
```

### 2. Audio Requirements (`.wav`)
*   **Format**: WAV (PCM)
*   **Sample Rate**: 16kHz is the encoder input rate for both VoxCPM2 AudioVAE versions. Training audio is automatically resampled to match the encoder — any source sample rate works, but providing 16kHz avoids unnecessary resampling.
*   **Length**: Short clips between 3 to 10 seconds work best. Avoid clips longer than 15 seconds to prevent VRAM issues.
*   **Quality**: Clean, background-noise-free speech is critical.
*   **Language**: Any of the 30 supported languages.

### 3. Transcript Requirements (`.txt`)
*   **Content**: The exact spoken text corresponding to the audio file.
*   **Language**: Supports all 30 VoxCPM2 languages.
*   **Normalization**: Raw text is accepted. The training pipeline handles basic tokenization.

---

## Training Workflow

The training process involves three specific nodes connected in sequence.

### Step 1: Create Dataset Manifest (`VoxCPM2 Dataset Maker`)
This node scans your folder and generates a `train.jsonl` file required by the training engine.

*   **Inputs**:
    *   `audio_directory`: Absolute path to your dataset folder (e.g., `C:\AI\data\my_voice`).
    *   `output_filename`: Defaults to `train.jsonl`.
*   **Output**: Path string to the generated JSONL file.

### Step 2: Configure Training Parameters (`VoxCPM2 Train Config`)
This node aggregates all hyperparameters.

#### Key Parameters:
*   **`learning_rate`** (Default: `1e-4`):
    *   Controls how fast the model learns.
    *   *Recommendation*: Start with `1e-4`. If the loss explodes (NaN), reduce to `5e-5`.
*   **`lora_rank`** (Default: `32`):
    *   The dimension of the low-rank matrices. Higher values capture more detail but require more VRAM and data.
    *   *Recommendation*: `32` or `64`.
*   **`lora_alpha`** (Default: `16`):
    *   Scaling factor. A common rule of thumb is `alpha = rank / 2`.
*   **`grad_accum_steps`** (Default: `1`):
    *   Simulates a larger batch size. Since the physical batch size is locked to 1 for stability, increase this to 4 or 8 to stabilize gradients.
*   **`warmup_steps`**: Steps to ramp up the learning rate. Usually 5-10% of total steps.
*   **`max_batch_tokens`**: Limits the amount of audio processed at once. Lower this if you encounter Out-Of-Memory (OOM) errors.
*   **`sample_rate`** (Default: `16000`): Must match the AudioVAE encoder input rate. The trainer auto-detects the correct value from the loaded model and will warn + override if the config disagrees.
*   **`enable_lm_lora`** / **`enable_dit_lora`** / **`enable_proj_lora`**: Choose which model components to apply LoRA to. LM and DiT are enabled by default.

### Step 3: Run Training (`VoxCPM2 LoRA Trainer`)
This is the execution node. **Warning**: Running this node will block the ComfyUI interface until training completes.

*   **Inputs**:
    *   `base_model_name`: Select `VoxCPM2`.
    *   `train_config`: Connect from the Config node.
    *   `dataset_path`: Connect from the Dataset Maker node.
    *   `output_name`: The name of the subfolder in `models/loras` where checkpoints will be saved.
    *   `max_steps`: Total training duration.
        *   *Rule of Thumb*: For a dataset of ~5 minutes, try 1000-2000 steps.
    *   `save_every_steps`: Checkpoint interval.

---

## Monitoring & Results

### Console Output
Open the ComfyUI console window to see real-time logs:
```
Step 10/1000, Loss: 2.145, LR: 0.00001000
Step 20/1000, Loss: 1.892, LR: 0.00002000
```
*   **Loss**: Should generally decrease. If it stays at 0.0000, something is wrong with the setup.
*   **Loss Spike**: Sudden increases are normal but should recover.

### Output Files
After training, check `ComfyUI/models/loras/[output_name]/`:
1.  **`*.safetensors`**: The LoRA weight files.
2.  **`lora_config.json`**: Configuration metadata required for loading.

---

## Using Your LoRA

1.  Refresh your ComfyUI browser page.
2.  In the **VoxCPM2 TTS** or **VoxCPM2 Voice Clone** node:
    *   Set **`model_name`** to `VoxCPM2`.
    *   In the **`lora_name`** dropdown, select your newly trained LoRA (e.g., `my_voice_step_2000.safetensors`).
3.  Generate audio!

> **Tip**: If the effect is too strong or distorted, training might have overfitted. Try an earlier checkpoint or reduce the `learning_rate` and retrain.
