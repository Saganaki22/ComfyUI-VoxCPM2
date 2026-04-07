import os
import json
import logging
import torch

import folder_paths

from ..modules.model_info import AVAILABLE_VOXCPM_MODELS

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.propagate = False
if not logger.hasHandlers():
    handler = logging.StreamHandler()
    formatter = logging.Formatter(f"[ComfyUI-VoxCPM] %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)

LOADED_MODELS_CACHE = {}


def _resolve_dtype(dtype: str) -> str:
    """Resolve effective dtype, auto-detecting if GPU doesn't support bf16."""
    if dtype in ("bf16", "fp16"):
        return dtype

    # "auto" — use bf16 if GPU supports it, otherwise fp16
    if torch.cuda.is_available():
        major, _ = torch.cuda.get_device_capability()
        if major >= 8:
            return "bf16"
        else:
            logger.info(f"GPU compute capability sm_{major}x — bf16 not supported, defaulting to fp16")
            return "fp16"

    return "auto"  # CPU — no dtype conversion; caller skips the conversion branch when "auto"


def _detect_architecture(model_path: str) -> str:
    """Detect model architecture from config.json."""
    config_path = os.path.join(model_path, "config.json")
    if not os.path.exists(config_path):
        logger.warning(f"config.json not found at {config_path}, assuming voxcpm")
        return "voxcpm"
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)
    return config.get("architecture", "voxcpm").lower()


class VoxCPMModelHandler(torch.nn.Module):
    """
    A lightweight handler for a VoxCPM model. It acts as a container
    that ComfyUI's ModelPatcher can manage, while the actual heavy model
    is loaded on demand.
    """
    def __init__(self, model_name: str, optimize: bool = False, torch_compile: bool = False, dtype: str = "auto"):
        super().__init__()
        self.model_name = model_name
        self.optimize = optimize
        self.torch_compile = torch_compile
        self.dtype = dtype
        self.model = None  # This will hold the actual loaded voxcpm.VoxCPM instance
        # Dynamic VRAM estimate based on model name
        if "VoxCPM2" in model_name or "voxcpm2" in model_name.lower():
            self.size = int(8.0 * (1024**3))
        else:
            self.size = int(2.5 * (1024**3))


def load_model(model_name: str, optimize: bool = False, torch_compile: bool = False, dtype: str = "auto"):
    """
    Load a VoxCPM model, downloading it if necessary. Caches the loaded model instance.
    Uses the pip-installed voxcpm package for model loading and inference.
    """
    cache_key = f"{model_name}_opt{optimize}_compile{torch_compile}_dtype{dtype}"
    if cache_key in LOADED_MODELS_CACHE:
        logger.info(f"Using cached VoxCPM model instance: {cache_key}")
        return LOADED_MODELS_CACHE[cache_key]

    model_info = AVAILABLE_VOXCPM_MODELS.get(model_name)
    if not model_info:
        raise ValueError(f"Model '{model_name}' not found. Available models: {list(AVAILABLE_VOXCPM_MODELS.keys())}")

    voxcpm_path = None

    if model_info["type"] == "local":
        voxcpm_path = model_info["path"]
        logger.info(f"Loading local model from: {voxcpm_path}")
    elif model_info["type"] == "official":
        from huggingface_hub import snapshot_download
        base_tts_path = os.path.join(folder_paths.get_folder_paths("tts")[0])
        voxcpm_models_dir = os.path.join(base_tts_path, "VoxCPM")
        os.makedirs(voxcpm_models_dir, exist_ok=True)
        voxcpm_path = os.path.join(voxcpm_models_dir, model_name)

        has_bin = os.path.exists(os.path.join(voxcpm_path, "pytorch_model.bin"))
        has_safe = os.path.exists(os.path.join(voxcpm_path, "model.safetensors"))

        if not (has_bin or has_safe):
            logger.info(f"Downloading official VoxCPM model '{model_name}' from {model_info['repo_id']}...")
            snapshot_download(
                repo_id=model_info["repo_id"],
                local_dir=voxcpm_path,
                local_dir_use_symlinks=False,
            )

    if not voxcpm_path:
        raise RuntimeError(f"Could not determine path for model '{model_name}'")

    logger.info("Instantiating VoxCPM model...")

    # Use the bundled voxcpm package (src/voxcpm/) which supports both V1 and V2.
    # This ensures all ComfyUI patches (ProgressBar, torch.compile, etc.) are applied.
    import sys
    _src_dir = os.path.join(os.path.dirname(__file__), '..', 'src')
    _src_dir = os.path.normpath(_src_dir)
    if _src_dir not in sys.path:
        sys.path.insert(0, _src_dir)
    # Purge any previously imported voxcpm modules (from pip package)
    # so Python re-imports from our local src/ directory.
    for key in list(sys.modules.keys()):
        if key == 'voxcpm' or key.startswith('voxcpm.'):
            del sys.modules[key]

    from voxcpm import VoxCPM as VoxCPMPipeline
    from voxcpm.model.voxcpm import LoRAConfig

    # Build LoRA config (always enable so user can load LoRA at runtime)
    lora_config = LoRAConfig(
        enable_lm=True,
        enable_dit=True,
        enable_proj=False,
        r=32,
        alpha=16,
    )

    model_instance = VoxCPMPipeline(
        voxcpm_model_path=voxcpm_path,
        zipenhancer_model_path=None,
        enable_denoiser=False,
        optimize=optimize,
        lora_config=lora_config,
    )

    # Resolve effective dtype and convert if needed
    effective_dtype = _resolve_dtype(dtype)
    config_dtype = getattr(model_instance.tts_model.config, 'dtype', 'bfloat16')
    # Normalize to short form so "bfloat16" matches "bf16"
    config_dtype = config_dtype.replace("float", "f").replace("torch.", "")

    if effective_dtype == "auto":
        # CPU mode or no override — leave as-is
        pass
    elif config_dtype != effective_dtype:
        target_dtype = torch.bfloat16 if effective_dtype == "bf16" else torch.float16
        logger.info(f"Converting model from {config_dtype} to {effective_dtype}")
        # audio_vae must stay float32 — detach, convert the rest, then restore
        audio_vae_ref = model_instance.tts_model.audio_vae
        model_instance.tts_model.audio_vae = None
        model_instance.tts_model.to(target_dtype)
        model_instance.tts_model.audio_vae = audio_vae_ref.to(torch.float32)
        # Update config so internal inference casts match
        model_instance.tts_model.config.dtype = effective_dtype
        # Rebuild KV caches with the new dtype
        model_instance.tts_model.base_lm.setup_cache(
            1, model_instance.tts_model.config.max_length,
            model_instance.tts_model.device, target_dtype,
        )
        model_instance.tts_model.residual_lm.setup_cache(
            1, model_instance.tts_model.config.max_length,
            model_instance.tts_model.device, target_dtype,
        )
    elif effective_dtype == "bf16" and config_dtype == "bf16":
        # Native bf16 — ensure audio_vae is float32 as the model expects
        model_instance.tts_model.audio_vae.to(torch.float32)

    LOADED_MODELS_CACHE[cache_key] = model_instance
    return model_instance
