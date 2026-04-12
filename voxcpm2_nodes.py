import os
import re
import tempfile
import torch
import torchaudio
import logging
from typing import List

import folder_paths
import comfy.model_management as model_management
from comfy_api.latest import ComfyExtension, io, ui

from .modules.model_info import AVAILABLE_VOXCPM_MODELS
from .modules.loader import VoxCPMModelHandler
from .modules.patcher import VoxCPMPatcher

from .voxcpm2_train_nodes import VoxCPM_TrainConfig, VoxCPM_DatasetMaker, VoxCPM_LoraTrainer

logger = logging.getLogger(__name__)

VOXCPM_PATCHER_CACHE = {}
MAX_REFERENCE_AUDIO_SECONDS = 50.0

# ---------------------------------------------------------------------------
# ASR (Auto Speech Recognition) — lazy-loaded singleton
# ---------------------------------------------------------------------------
_ASR_MODEL = None


def get_asr_model():
    """Lazily load and cache the FunASR SenseVoiceSmall model for auto-transcription."""
    global _ASR_MODEL
    if _ASR_MODEL is not None:
        return _ASR_MODEL
    try:
        from funasr import AutoModel
        from huggingface_hub import snapshot_download

        asr_model_ref = "FunAudioLLM/SenseVoiceSmall"

        # Cache in ComfyUI's audio_encoders directory
        encoders_dir = os.path.join(folder_paths.models_dir, "audio_encoders")
        os.makedirs(encoders_dir, exist_ok=True)
        asr_local_dir = os.path.join(encoders_dir, "SenseVoiceSmall")

        # Download only if not already cached locally
        if not os.path.isdir(asr_local_dir) or not os.listdir(asr_local_dir):
            logger.info(f"Downloading ASR model '{asr_model_ref}' to {asr_local_dir}...")
            snapshot_download(
                repo_id=asr_model_ref,
                local_dir=asr_local_dir,
                local_dir_use_symlinks=False,
            )
        else:
            logger.info(f"Using cached ASR model from: {asr_local_dir}")

        logger.info("Loading ASR model on CPU ...")
        _ASR_MODEL = AutoModel(
            model=asr_local_dir,
            disable_update=True,
            log_level="INFO",
            device="cpu",
        )
        logger.info("ASR model loaded.")
        return _ASR_MODEL
    except ImportError:
        raise ImportError(
            "ASR requires funasr. Install with: pip install funasr"
        )


def transcribe_audio(wav_path: str) -> str:
    """Run ASR on a wav file and return the transcribed text."""
    asr_model = get_asr_model()
    res = asr_model.generate(input=wav_path, language="auto", use_itn=True)
    if not res:
        return ""
    first_item = res[0]
    if isinstance(first_item, dict):
        return re.sub(r"<\|[^|>]*\|>", "", str(first_item.get("text", ""))).strip()
    return ""


def offload_asr():
    """Release the ASR model from memory."""
    global _ASR_MODEL
    if _ASR_MODEL is not None:
        del _ASR_MODEL
        _ASR_MODEL = None
        import gc
        gc.collect()
        logger.info("ASR model offloaded.")


# ---------------------------------------------------------------------------
# Audio utilities
# ---------------------------------------------------------------------------


def get_available_devices():
    devices = []
    if torch.cuda.is_available():
        devices.append("cuda")
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        devices.append("mps")
    devices.append("cpu")
    return devices


def set_seed(seed: int):
    import random
    if seed < 0:
        seed = random.randint(0, 2**31 - 1)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    try:
        import numpy as np
        np.random.seed(seed & 0xFFFFFFFF)
    except ImportError:
        pass


def _save_audio_to_temp(waveform: torch.Tensor, sample_rate: int) -> str:
    """Save a ComfyUI audio tensor to a temporary WAV file and return the path."""
    # Ensure shape is [1, T]
    if waveform.dim() == 3:
        waveform = waveform[0]
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp.close()
    try:
        torchaudio.save(tmp.name, waveform, sample_rate)
    except Exception:
        try:
            os.unlink(tmp.name)
        except OSError:
            pass
        raise
    return tmp.name


def _get_audio_duration_seconds(wav_path: str) -> float:
    """Get audio duration in seconds using soundfile."""
    import soundfile as sf
    info = sf.info(wav_path)
    return float(info.frames) / float(info.samplerate)


def _validate_reference_audio_duration(wav_path: str):
    """Raise ValueError if reference audio exceeds the maximum allowed duration."""
    duration = _get_audio_duration_seconds(wav_path)
    if duration > MAX_REFERENCE_AUDIO_SECONDS:
        raise ValueError(
            f"Reference audio is {duration:.1f}s — max allowed is {MAX_REFERENCE_AUDIO_SECONDS:.0f}s. "
            "Trim the audio and try again."
        )


def _normalize_loudness(wav_path: str):
    """Normalize audio loudness to -20 LUFS in-place."""
    audio, sr = torchaudio.load(wav_path)
    loudness = torchaudio.functional.loudness(audio, sr)
    # loudness returns per-channel tensor — average to scalar
    loudness_val = loudness.mean().item()
    normalized_audio = torchaudio.functional.gain(audio, -20 - loudness_val)
    torchaudio.save(wav_path, normalized_audio, sr)


def _load_patcher(model_name, device, torch_compile, dtype="auto", optimize=False):
    """Get or create a patcher for the given model/device/config.
    Fully unloads any existing patchers with a different config to free VRAM/RAM."""
    if device == "cuda":
        load_device = model_management.get_torch_device()
        offload_device = model_management.intermediate_device()
    else:
        load_device = torch.device("cpu")
        offload_device = torch.device("cpu")

    cache_key = f"{model_name}_{device}_opt{optimize}_compile{torch_compile}_dtype{dtype}"

    # Fast path: already cached with the right key, skip eviction
    if cache_key in VOXCPM_PATCHER_CACHE:
        return VOXCPM_PATCHER_CACHE[cache_key]

    # Evict all cached patchers that don't match the requested config
    stale_keys = [k for k in VOXCPM_PATCHER_CACHE if k != cache_key]
    for key in stale_keys:
        old_patcher = VOXCPM_PATCHER_CACHE.pop(key)
        old_patcher.force_unload()

    if cache_key not in VOXCPM_PATCHER_CACHE:
        handler = VoxCPMModelHandler(model_name, optimize=optimize, torch_compile=torch_compile, dtype=dtype)
        patcher = VoxCPMPatcher(handler, load_device=load_device, offload_device=offload_device, size=handler.size)
        VOXCPM_PATCHER_CACHE[cache_key] = patcher

    return VOXCPM_PATCHER_CACHE[cache_key]


class VoxCPM2TTSNode(io.ComfyNode):
    """VoxCPM2 Text-to-Speech node. Supports voice design via parenthetical descriptions."""
    CATEGORY = "audio/tts"

    @classmethod
    def define_schema(cls) -> io.Schema:
        model_names = list(AVAILABLE_VOXCPM_MODELS.keys())

        available_devices = get_available_devices()
        default_device = available_devices[0]

        lora_list = ["None"] + folder_paths.get_filename_list("loras")

        return io.Schema(
            node_id="VoxCPM2_TTS",
            display_name="VoxCPM TTS",
            category=cls.CATEGORY,
            description="Generate speech with any VoxCPM model. Supports voice design via parenthetical descriptions like '(gentle female voice)Hello world'.",
            inputs=[
                io.Combo.Input("model_name", options=model_names, default=model_names[0], tooltip="Select the VoxCPM model to use."),
                io.Combo.Input("lora_name", options=lora_list, default="None", tooltip="LoRA checkpoint from models/loras. Set to None to disable."),
                io.String.Input("voice_description", multiline=True, default="", tooltip="Voice design description (optional). E.g. 'A young woman, gentle and sweet voice'. Wrapped in parentheses and prepended to text."),
                io.String.Input("text", multiline=True, default="Hello, welcome to VoxCPM!", tooltip="Target text to synthesize into speech."),
                io.Float.Input("cfg_value", default=2.0, min=1.0, max=10.0, step=0.1, tooltip="Classifier-Free Guidance scale. Higher = more adherence to prompt, lower = more natural variation."),
                io.Int.Input("inference_timesteps", default=10, min=1, max=100, step=1, tooltip="Number of diffusion steps. More steps = better quality but slower."),
                io.Int.Input("max_tokens", default=4096, min=64, max=8192, tooltip="Maximum generation length in tokens. Controls max audio duration."),
                io.Boolean.Input("normalize_text", default=True, label_on="Normalize", label_off="Raw", tooltip="Auto-process numbers, abbreviations, and punctuation. Turn OFF for phoneme input."),
                io.Int.Input("seed", default=42, min=-1, max=0xFFFFFFFFFFFFFFFF, tooltip="Random seed for reproducibility. -1 = random each run."),
                io.Boolean.Input("force_offload", default=False, label_on="Force Offload", label_off="Auto", tooltip="Fully unload model from VRAM and RAM after generation."),
                io.Combo.Input("dtype", options=["auto", "bf16", "fp16"], default="auto", tooltip="Model dtype. Auto uses native bf16 (fp16 on older GPUs)."),
                io.Combo.Input("device", options=available_devices, default=default_device, tooltip="Inference device."),
                io.Boolean.Input("torch_compile", default=False, label_on="Torch Compile", label_off="Standard", tooltip="Enable torch.compile optimization (first run compiles kernels, subsequent runs are faster)."),
            ],
            outputs=[
                io.Audio.Output(display_name="Generated Audio"),
            ],
        )

    @classmethod
    def execute(cls, model_name, lora_name, device, text, cfg_value, inference_timesteps,
                max_tokens, normalize_text, seed, force_offload,
                torch_compile,
                voice_description="", dtype="auto", **kwargs):

        # Prepend voice description in parentheses if provided
        if voice_description and voice_description.strip():
            text = f"({voice_description.strip()}){text}"

        patcher = _load_patcher(model_name, device, torch_compile, dtype)
        model_management.load_model_gpu(patcher)
        voxcpm_model = patcher.model.model

        if not voxcpm_model:
            raise RuntimeError(f"Failed to load model '{model_name}'.")

        # LoRA handling
        if lora_name != "None":
            lora_path = folder_paths.get_full_path("loras", lora_name)
            if not lora_path:
                raise FileNotFoundError(f"LoRA file not found: {lora_name}")
            try:
                voxcpm_model.load_lora(lora_path)
                voxcpm_model.set_lora_enabled(True)
            except Exception as e:
                raise RuntimeError(f"Failed to load LoRA '{lora_name}': {e}")
        else:
            voxcpm_model.set_lora_enabled(False)

        set_seed(seed)

        try:
            wav_array = voxcpm_model.generate(
                text=text,
                cfg_value=cfg_value,
                inference_timesteps=inference_timesteps,
                max_len=max_tokens,
                normalize=normalize_text,
            )

            output_tensor = torch.from_numpy(wav_array).float().unsqueeze(0).unsqueeze(0)
            output_sr = voxcpm_model.tts_model.sample_rate
            output_audio = {"waveform": output_tensor, "sample_rate": output_sr}

            logger.info(f"VoxCPM TTS generation complete ({model_name}).")

            if force_offload:
                cache_key = f"{model_name}_{device}_opt{patcher.model.optimize}_compile{torch_compile}_dtype{dtype}"
                patcher.force_unload()
                VOXCPM_PATCHER_CACHE.pop(cache_key, None)
                offload_asr()

            return io.NodeOutput(output_audio, ui=ui.PreviewAudio(output_audio, cls=cls))

        except Exception as e:
            logger.error(f"VoxCPM2 TTS error: {e}")
            raise e


class VoxCPM2CloneNode(io.ComfyNode):
    """VoxCPM2 Voice Cloning node. Supports controllable and ultimate cloning."""
    CATEGORY = "audio/tts"

    @classmethod
    def define_schema(cls) -> io.Schema:
        model_names = list(AVAILABLE_VOXCPM_MODELS.keys())

        available_devices = get_available_devices()
        default_device = available_devices[0]

        lora_list = ["None"] + folder_paths.get_filename_list("loras")

        return io.Schema(
            node_id="VoxCPM2_Clone",
            display_name="VoxCPM Voice Clone",
            category=cls.CATEGORY,
            description="Clone a voice using any VoxCPM model. VoxCPM2: controllable (reference audio only) or ultimate (reference + transcript). V1 models: transcript required.",
            inputs=[
                io.Combo.Input("model_name", options=model_names, default=model_names[0], tooltip="Select the VoxCPM model to use."),
                io.Combo.Input("lora_name", options=lora_list, default="None", tooltip="LoRA checkpoint from models/loras. Set to None to disable."),
                io.String.Input("voice_description", multiline=True, default="", tooltip="Style control description (optional). E.g. 'slightly faster, cheerful tone'. Wrapped in parentheses and prepended to text."),
                io.String.Input("text", multiline=True, default="This is a cloned voice generated by VoxCPM.", tooltip="Target text to synthesize into speech."),
                io.Audio.Input("reference_audio", tooltip="Reference audio for voice cloning. Required."),
                io.String.Input("prompt_text", multiline=True, tooltip="Transcript of the reference audio. Provide for Ultimate Cloning (highest fidelity). Leave empty for Controllable Cloning (VoxCPM2 only)."),
                io.Float.Input("cfg_value", default=2.0, min=1.0, max=10.0, step=0.1, tooltip="Classifier-Free Guidance scale. Higher = more adherence to prompt, lower = more natural variation."),
                io.Int.Input("inference_timesteps", default=10, min=1, max=100, step=1, tooltip="Number of diffusion steps. More steps = better quality but slower."),
                io.Int.Input("max_tokens", default=4096, min=64, max=8192, tooltip="Maximum generation length in tokens. Controls max audio duration."),
                io.Boolean.Input("normalize_text", default=True, label_on="Normalize", label_off="Raw", tooltip="Auto-process numbers, abbreviations, and punctuation. Turn OFF for phoneme input."),
                io.Boolean.Input("enable_denoiser", default=False, label_on="Denoise", label_off="Off", tooltip="Denoise reference audio before cloning. Requires modelscope package."),
                io.Int.Input("seed", default=42, min=-1, max=0xFFFFFFFFFFFFFFFF, tooltip="Random seed for reproducibility. -1 = random each run."),
                io.Boolean.Input("force_offload", default=False, label_on="Force Offload", label_off="Auto", tooltip="Fully unload model from VRAM and RAM after generation."),
                io.Combo.Input("dtype", options=["auto", "bf16", "fp16"], default="auto", tooltip="Model dtype. Auto uses native bf16 (fp16 on older GPUs)."),
                io.Combo.Input("device", options=available_devices, default=default_device, tooltip="Inference device."),
                io.Boolean.Input("enable_asr", default=False, label_on="ASR", label_off="Off", tooltip="Auto-transcribe reference audio to text using SenseVoiceSmall ASR. Requires funasr package. Ignored when prompt_text is provided. First run downloads the model (~400MB)."),
                io.Int.Input("retry_max_attempts", default=3, min=0, max=10, step=1, tooltip="Auto-retry on bad generation (babbling/silence). 0 = no retries."),
                io.Float.Input("retry_threshold", default=6.0, min=2.0, max=20.0, step=0.1, tooltip="Threshold for detecting bad generations based on audio/text length ratio."),
                io.Boolean.Input("torch_compile", default=False, label_on="Torch Compile", label_off="Standard", tooltip="Enable torch.compile optimization (first run compiles kernels, subsequent runs are faster)."),
            ],
            outputs=[
                io.Audio.Output(display_name="Cloned Audio"),
            ],
        )

    @classmethod
    def execute(cls, model_name, lora_name, device, text, cfg_value, inference_timesteps,
                max_tokens, normalize_text, enable_denoiser, seed, force_offload,
                enable_asr, retry_max_attempts, retry_threshold, torch_compile,
                reference_audio=None, prompt_text=None, voice_description="", dtype="auto", **kwargs):

        # Voice description handling
        has_voice_desc = bool(voice_description and voice_description.strip())
        has_prompt = bool(prompt_text and prompt_text.strip())

        if has_voice_desc and (has_prompt or enable_asr):
            logger.warning("voice_description is ignored in Ultimate Cloning mode (prompt_text/ASR provided).")
        elif has_voice_desc:
            logger.info(f"Controllable Clone — voice_description: \"{voice_description.strip()}\"")
            text = f"({voice_description.strip()}){text}"

        if reference_audio is None:
            raise ValueError("Reference audio is required for voice cloning.")

        # Save reference audio to temp file for duration check and later use
        ref_waveform = reference_audio['waveform']
        ref_sample_rate = reference_audio['sample_rate']
        ref_wav_path = _save_audio_to_temp(ref_waveform, ref_sample_rate)

        # Validate reference audio duration
        _validate_reference_audio_duration(ref_wav_path)

        # ASR auto-transcription: only when no prompt_text is provided
        if enable_asr and (not prompt_text or not prompt_text.strip()):
            logger.info("Running ASR on reference audio...")
            try:
                prompt_text = transcribe_audio(ref_wav_path)
                if prompt_text:
                    logger.info(f"ASR result: {prompt_text[:80]}{'...' if len(prompt_text) > 80 else ''}")
                else:
                    logger.warning("ASR returned empty transcript. Proceeding without transcript.")
            except Exception as e:
                logger.warning(f"ASR failed: {e}. Proceeding without transcript.")
        elif enable_asr and prompt_text and prompt_text.strip():
            logger.info("ASR enabled but prompt_text already provided — using manual transcript instead.")

        patcher = _load_patcher(model_name, device, torch_compile, dtype)
        model_management.load_model_gpu(patcher)
        voxcpm_model = patcher.model.model

        if not voxcpm_model:
            raise RuntimeError(f"Failed to load model '{model_name}'.")

        is_v2 = voxcpm_model._is_v2

        # V1 models require prompt_text for voice cloning (no controllable mode)
        has_transcript = prompt_text and prompt_text.strip()
        if not is_v2 and not has_transcript:
            raise ValueError(
                f"Model '{model_name}' does not support controllable cloning (no transcript). "
                "Provide prompt_text or enable ASR for ultimate cloning, or use a VoxCPM2 model."
            )

        # LoRA handling
        if lora_name != "None":
            lora_path = folder_paths.get_full_path("loras", lora_name)
            if not lora_path:
                raise FileNotFoundError(f"LoRA file not found: {lora_name}")
            try:
                voxcpm_model.load_lora(lora_path)
                voxcpm_model.set_lora_enabled(True)
            except Exception as e:
                raise RuntimeError(f"Failed to load LoRA '{lora_name}': {e}")
        else:
            voxcpm_model.set_lora_enabled(False)

        set_seed(seed)

        try:
            if has_transcript and is_v2:
                # V2 ultimate cloning: reference + prompt + text
                wav_array = voxcpm_model.generate(
                    text=text,
                    prompt_text=prompt_text,
                    prompt_wav_path=ref_wav_path,
                    reference_wav_path=ref_wav_path,
                    cfg_value=cfg_value,
                    inference_timesteps=inference_timesteps,
                    max_len=max_tokens,
                    normalize=normalize_text,
                    denoise=enable_denoiser,
                    retry_badcase=retry_max_attempts > 0,
                    retry_badcase_max_times=retry_max_attempts,
                    retry_badcase_ratio_threshold=retry_threshold,
                )
            elif has_transcript and not is_v2:
                # V1 cloning: prompt_wav + prompt_text only (no reference_wav_path)
                wav_array = voxcpm_model.generate(
                    text=text,
                    prompt_text=prompt_text,
                    prompt_wav_path=ref_wav_path,
                    cfg_value=cfg_value,
                    inference_timesteps=inference_timesteps,
                    max_len=max_tokens,
                    normalize=normalize_text,
                    denoise=enable_denoiser,
                    retry_badcase=retry_max_attempts > 0,
                    retry_badcase_max_times=retry_max_attempts,
                    retry_badcase_ratio_threshold=retry_threshold,
                )
            else:
                # V2 controllable cloning: reference only
                wav_array = voxcpm_model.generate(
                    text=text,
                    reference_wav_path=ref_wav_path,
                    cfg_value=cfg_value,
                    inference_timesteps=inference_timesteps,
                    max_len=max_tokens,
                    normalize=normalize_text,
                    denoise=enable_denoiser,
                    retry_badcase=retry_max_attempts > 0,
                    retry_badcase_max_times=retry_max_attempts,
                    retry_badcase_ratio_threshold=retry_threshold,
                )

            output_tensor = torch.from_numpy(wav_array).float().unsqueeze(0).unsqueeze(0)
            output_sr = voxcpm_model.tts_model.sample_rate

            # Loudness normalization when denoiser is active
            if enable_denoiser:
                norm_tmp = None
                try:
                    norm_tmp = _save_audio_to_temp(output_tensor, output_sr)
                    _normalize_loudness(norm_tmp)
                    norm_waveform, _ = torchaudio.load(norm_tmp)
                    output_tensor = norm_waveform.unsqueeze(0)
                except Exception as e:
                    logger.warning(f"Loudness normalization failed: {e}. Using raw output.")
                finally:
                    if norm_tmp:
                        try:
                            os.unlink(norm_tmp)
                        except OSError:
                            pass

            output_audio = {"waveform": output_tensor, "sample_rate": output_sr}

            logger.info(f"VoxCPM voice cloning complete ({model_name}).")

            if force_offload:
                cache_key = f"{model_name}_{device}_opt{patcher.model.optimize}_compile{torch_compile}_dtype{dtype}"
                patcher.force_unload()
                VOXCPM_PATCHER_CACHE.pop(cache_key, None)
                offload_asr()

            return io.NodeOutput(output_audio, ui=ui.PreviewAudio(output_audio, cls=cls))

        except Exception as e:
            logger.error(f"VoxCPM2 clone error: {e}")
            raise e

        finally:
            # Always clean up the temp file
            try:
                os.unlink(ref_wav_path)
            except OSError:
                pass


class VoxCPMExtension(ComfyExtension):
    async def get_node_list(self) -> List[type[io.ComfyNode]]:
        return [
            VoxCPM2TTSNode,
            VoxCPM2CloneNode,
            VoxCPM_TrainConfig,
            VoxCPM_DatasetMaker,
            VoxCPM_LoraTrainer,
        ]

async def comfy_entrypoint() -> VoxCPMExtension:
    return VoxCPMExtension()
