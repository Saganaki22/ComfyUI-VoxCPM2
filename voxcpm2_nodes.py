import os
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


def get_available_devices():
    devices = []
    if torch.cuda.is_available():
        devices.append("cuda")
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        devices.append("mps")
    devices.append("cpu")
    return devices


def set_seed(seed: int):
    if seed < 0:
        import random as _random
        seed = _random.randint(0, 2**31 - 1)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _save_audio_to_temp(waveform: torch.Tensor, sample_rate: int) -> str:
    """Save a ComfyUI audio tensor to a temporary WAV file and return the path."""
    # Ensure shape is [1, T]
    if waveform.dim() == 3:
        waveform = waveform[0]
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp.close()
    torchaudio.save(tmp.name, waveform, sample_rate)
    return tmp.name


def _load_patcher(model_name, device, torch_compile, dtype="auto"):
    """Get or create a patcher for the given model/device/config.
    Fully unloads any existing patchers with a different config to free VRAM/RAM."""
    if device == "cuda":
        load_device = model_management.get_torch_device()
        offload_device = model_management.intermediate_device()
    else:
        load_device = torch.device("cpu")
        offload_device = torch.device("cpu")

    cache_key = f"{model_name}_{device}_opt{torch_compile}_dtype{dtype}"

    # Evict all cached patchers that don't match the requested config
    stale_keys = [k for k in VOXCPM_PATCHER_CACHE if k != cache_key]
    for key in stale_keys:
        old_patcher = VOXCPM_PATCHER_CACHE.pop(key)
        old_patcher.force_unload()

    if cache_key not in VOXCPM_PATCHER_CACHE:
        handler = VoxCPMModelHandler(model_name, torch_compile=torch_compile, dtype=dtype)
        patcher = VoxCPMPatcher(handler, load_device=load_device, offload_device=offload_device, size=handler.size)
        VOXCPM_PATCHER_CACHE[cache_key] = patcher

    return VOXCPM_PATCHER_CACHE[cache_key]


class VoxCPM2TTSNode(io.ComfyNode):
    """VoxCPM2 Text-to-Speech node. Supports voice design via parenthetical descriptions."""
    CATEGORY = "audio/tts"

    @classmethod
    def define_schema(cls) -> io.Schema:
        model_names = [m for m in AVAILABLE_VOXCPM_MODELS.keys() if "VoxCPM2" in m]
        if not model_names:
            model_names = list(AVAILABLE_VOXCPM_MODELS.keys())

        available_devices = get_available_devices()
        default_device = available_devices[0]

        lora_list = ["None"] + folder_paths.get_filename_list("loras")

        return io.Schema(
            node_id="VoxCPM2_TTS",
            display_name="VoxCPM2 TTS",
            category=cls.CATEGORY,
            description="Generate speech with VoxCPM2. Supports voice design via parenthetical descriptions like '(gentle female voice)Hello world'.",
            inputs=[
                io.Combo.Input("model_name", options=model_names, default=model_names[0], tooltip="Select the VoxCPM2 model to use."),
                io.Combo.Input("lora_name", options=lora_list, default="None", tooltip="LoRA checkpoint from models/loras. Set to None to disable."),
                io.String.Input("voice_description", multiline=True, default="", tooltip="Voice design description (optional). E.g. 'A young woman, gentle and sweet voice'. Wrapped in parentheses and prepended to text."),
                io.String.Input("text", multiline=True, default="Hello, welcome to VoxCPM2!", tooltip="Target text to synthesize into speech."),
                io.Float.Input("cfg_value", default=2.0, min=1.0, max=10.0, step=0.1, tooltip="Classifier-Free Guidance scale. Higher = more adherence to prompt, lower = more natural variation."),
                io.Int.Input("inference_timesteps", default=10, min=1, max=100, step=1, tooltip="Number of diffusion steps. More steps = better quality but slower."),
                io.Int.Input("max_tokens", default=4096, min=64, max=8192, tooltip="Maximum generation length in tokens. Controls max audio duration."),
                io.Boolean.Input("normalize_text", default=True, label_on="Normalize", label_off="Raw", tooltip="Auto-process numbers, abbreviations, and punctuation. Turn OFF for phoneme input."),
                io.Int.Input("seed", default=42, min=-1, max=0xFFFFFFFFFFFFFFFF, tooltip="Random seed for reproducibility. -1 = random each run."),
                io.Boolean.Input("force_offload", default=False, label_on="Force Offload", label_off="Auto", tooltip="Fully unload model from VRAM and RAM after generation."),
                io.Combo.Input("dtype", options=["auto", "bf16", "fp16"], default="auto", tooltip="Model dtype. Auto uses native bf16 (fp16 on older GPUs)."),
                io.Combo.Input("device", options=available_devices, default=default_device, tooltip="Inference device."),
                io.Int.Input("retry_max_attempts", default=3, min=0, max=10, step=1, tooltip="Auto-retry on bad generation (babbling/silence). 0 = no retries."),
                io.Float.Input("retry_threshold", default=6.0, min=2.0, max=20.0, step=0.1, tooltip="Threshold for detecting bad generations based on audio/text length ratio."),
                io.Boolean.Input("torch_compile", default=False, label_on="Torch Compile", label_off="Standard", tooltip="Enable torch.compile optimization for faster inference."),
            ],
            outputs=[
                io.Audio.Output(display_name="Generated Audio"),
            ],
        )

    @classmethod
    def execute(cls, model_name, lora_name, device, text, cfg_value, inference_timesteps,
                max_tokens, normalize_text, seed, force_offload,
                retry_max_attempts, retry_threshold, torch_compile,
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
                retry_badcase=retry_max_attempts > 0,
                retry_badcase_max_times=retry_max_attempts,
                retry_badcase_ratio_threshold=retry_threshold,
            )

            output_tensor = torch.from_numpy(wav_array).float().unsqueeze(0).unsqueeze(0)
            output_sr = voxcpm_model.tts_model.sample_rate
            output_audio = {"waveform": output_tensor, "sample_rate": output_sr}

            logger.info("VoxCPM2 TTS generation complete.")

            if force_offload:
                cache_key = f"{model_name}_{device}_opt{torch_compile}_dtype{dtype}"
                patcher.force_unload()
                VOXCPM_PATCHER_CACHE.pop(cache_key, None)

            return io.NodeOutput(output_audio, ui=ui.PreviewAudio(output_audio, cls=cls))

        except Exception as e:
            logger.error(f"VoxCPM2 TTS error: {e}")
            raise e


class VoxCPM2CloneNode(io.ComfyNode):
    """VoxCPM2 Voice Cloning node. Supports controllable and ultimate cloning."""
    CATEGORY = "audio/tts"

    @classmethod
    def define_schema(cls) -> io.Schema:
        model_names = [m for m in AVAILABLE_VOXCPM_MODELS.keys() if "VoxCPM2" in m]
        if not model_names:
            model_names = list(AVAILABLE_VOXCPM_MODELS.keys())

        available_devices = get_available_devices()
        default_device = available_devices[0]

        lora_list = ["None"] + folder_paths.get_filename_list("loras")

        return io.Schema(
            node_id="VoxCPM2_Clone",
            display_name="VoxCPM2 Voice Clone",
            category=cls.CATEGORY,
            description="Clone a voice using VoxCPM2. Controllable: reference audio only. Ultimate: reference audio + transcript.",
            inputs=[
                io.Combo.Input("model_name", options=model_names, default=model_names[0], tooltip="Select the VoxCPM2 model to use."),
                io.Combo.Input("lora_name", options=lora_list, default="None", tooltip="LoRA checkpoint from models/loras. Set to None to disable."),
                io.String.Input("voice_description", multiline=True, default="", tooltip="Style control description (optional). E.g. 'slightly faster, cheerful tone'. Wrapped in parentheses and prepended to text."),
                io.String.Input("text", multiline=True, default="This is a cloned voice generated by VoxCPM2.", tooltip="Target text to synthesize into speech."),
                io.Audio.Input("reference_audio", tooltip="Reference audio for voice cloning. Required."),
                io.String.Input("prompt_text", multiline=True, tooltip="Transcript of the reference audio. Provide for Ultimate Cloning (highest fidelity). Leave empty for Controllable Cloning."),
                io.Float.Input("cfg_value", default=2.0, min=1.0, max=10.0, step=0.1, tooltip="Classifier-Free Guidance scale. Higher = more adherence to prompt, lower = more natural variation."),
                io.Int.Input("inference_timesteps", default=10, min=1, max=100, step=1, tooltip="Number of diffusion steps. More steps = better quality but slower."),
                io.Int.Input("max_tokens", default=4096, min=64, max=8192, tooltip="Maximum generation length in tokens. Controls max audio duration."),
                io.Boolean.Input("normalize_text", default=True, label_on="Normalize", label_off="Raw", tooltip="Auto-process numbers, abbreviations, and punctuation. Turn OFF for phoneme input."),
                io.Int.Input("seed", default=42, min=-1, max=0xFFFFFFFFFFFFFFFF, tooltip="Random seed for reproducibility. -1 = random each run."),
                io.Boolean.Input("force_offload", default=False, label_on="Force Offload", label_off="Auto", tooltip="Fully unload model from VRAM and RAM after generation."),
                io.Combo.Input("dtype", options=["auto", "bf16", "fp16"], default="auto", tooltip="Model dtype. Auto uses native bf16 (fp16 on older GPUs)."),
                io.Combo.Input("device", options=available_devices, default=default_device, tooltip="Inference device."),
                io.Int.Input("retry_max_attempts", default=3, min=0, max=10, step=1, tooltip="Auto-retry on bad generation (babbling/silence). 0 = no retries."),
                io.Float.Input("retry_threshold", default=6.0, min=2.0, max=20.0, step=0.1, tooltip="Threshold for detecting bad generations based on audio/text length ratio."),
                io.Boolean.Input("torch_compile", default=False, label_on="Torch Compile", label_off="Standard", tooltip="Enable torch.compile optimization for faster inference."),
            ],
            outputs=[
                io.Audio.Output(display_name="Cloned Audio"),
            ],
        )

    @classmethod
    def execute(cls, model_name, lora_name, device, text, cfg_value, inference_timesteps,
                max_tokens, normalize_text, seed, force_offload,
                retry_max_attempts, retry_threshold, torch_compile,
                reference_audio=None, prompt_text=None, voice_description="", dtype="auto", **kwargs):

        # Prepend voice description in parentheses if provided
        if voice_description and voice_description.strip():
            text = f"({voice_description.strip()}){text}"

        if reference_audio is None:
            raise ValueError("Reference audio is required for voice cloning.")

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

        # Save reference audio to temp file for the pip package API
        ref_waveform = reference_audio['waveform']
        ref_sample_rate = reference_audio['sample_rate']
        ref_wav_path = _save_audio_to_temp(ref_waveform, ref_sample_rate)

        try:
            # Determine cloning mode based on whether prompt_text is provided
            # Ultimate cloning: reference_wav_path + prompt_wav_path + prompt_text
            # Controllable cloning: reference_wav_path only
            has_transcript = prompt_text and prompt_text.strip()

            if has_transcript:
                # Ultimate cloning: use reference as both structural isolation and continuation
                wav_array = voxcpm_model.generate(
                    text=text,
                    prompt_text=prompt_text,
                    prompt_wav_path=ref_wav_path,
                    reference_wav_path=ref_wav_path,
                    cfg_value=cfg_value,
                    inference_timesteps=inference_timesteps,
                    max_len=max_tokens,
                    normalize=normalize_text,
                    retry_badcase=retry_max_attempts > 0,
                    retry_badcase_max_times=retry_max_attempts,
                    retry_badcase_ratio_threshold=retry_threshold,
                )
            else:
                # Controllable cloning: reference only
                wav_array = voxcpm_model.generate(
                    text=text,
                    reference_wav_path=ref_wav_path,
                    cfg_value=cfg_value,
                    inference_timesteps=inference_timesteps,
                    max_len=max_tokens,
                    normalize=normalize_text,
                    retry_badcase=retry_max_attempts > 0,
                    retry_badcase_max_times=retry_max_attempts,
                    retry_badcase_ratio_threshold=retry_threshold,
                )

            output_tensor = torch.from_numpy(wav_array).float().unsqueeze(0).unsqueeze(0)
            output_sr = voxcpm_model.tts_model.sample_rate
            output_audio = {"waveform": output_tensor, "sample_rate": output_sr}

            logger.info("VoxCPM2 voice cloning complete.")

            if force_offload:
                cache_key = f"{model_name}_{device}_opt{torch_compile}_dtype{dtype}"
                patcher.force_unload()
                VOXCPM_PATCHER_CACHE.pop(cache_key, None)

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
