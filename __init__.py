import os
import logging
import warnings
import folder_paths

# Silence noisy PyTorch deprecation/inductor warnings
warnings.filterwarnings("ignore", message=".*weight_norm.*deprecated.*", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*Online softmax is disabled.*")
from .modules.model_info import AVAILABLE_VOXCPM_MODELS, MODEL_CONFIGS

# ---------------------------------------------------------------------------
# TorchCodec compatibility — newer torchaudio (2.9+) routes load/save through
# *_with_torchcodec helpers.  torchcodec may be missing, fail to import, or
# import successfully but be broken at runtime (missing FFmpeg DLLs, version
# mismatch, etc.).  Always replace the helpers with soundfile-based fallbacks
# so voxcpm2_nodes.py can call torchaudio.save/load without worrying about it.
# ---------------------------------------------------------------------------
try:
    import torchaudio
    if hasattr(torchaudio, 'load_with_torchcodec'):
        import torch as _torch
        import soundfile as _sf  # type: ignore

        def _fallback_load(path, *args, **kwargs):
            waveform, sr = _sf.read(str(path), dtype="float32", always_2d=True)
            tensor = _torch.from_numpy(waveform.T)
            return tensor, sr

        def _fallback_save(path, tensor, sample_rate, *args, **kwargs):
            data = tensor.cpu().numpy().T
            if data.ndim == 1:
                data = data[:, None]
            _sf.write(str(path), data, sample_rate)

        torchaudio.load_with_torchcodec = _fallback_load
        torchaudio.save_with_torchcodec = _fallback_save
except Exception:
    pass

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.propagate = False
if not logger.hasHandlers():
    handler = logging.StreamHandler()
    formatter = logging.Formatter(f"[ComfyUI-VoxCPM] %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)

VOXCPM_SUBDIR_NAME = "VoxCPM"

tts_path = os.path.join(folder_paths.models_dir, "tts")
os.makedirs(tts_path, exist_ok=True)

if "tts" not in folder_paths.folder_names_and_paths:
    folder_paths.folder_names_and_paths["tts"] = ([tts_path], folder_paths.supported_pt_extensions)
else:
    if tts_path not in folder_paths.folder_names_and_paths["tts"][0]:
        folder_paths.folder_names_and_paths["tts"][0].append(tts_path)

# Official models are already populated in model_info.py at import time.
# Scan for local models and add them.

voxcpm_search_paths = []
for tts_folder in folder_paths.get_folder_paths("tts"):
    potential_path = os.path.join(tts_folder, VOXCPM_SUBDIR_NAME)
    if os.path.isdir(potential_path) and potential_path not in voxcpm_search_paths:
        voxcpm_search_paths.append(potential_path)

for search_path in voxcpm_search_paths:
    if not os.path.isdir(search_path):
        continue
    for item in os.listdir(search_path):
        item_path = os.path.join(search_path, item)
        if os.path.isdir(item_path) and item not in AVAILABLE_VOXCPM_MODELS:
            config_exists = os.path.exists(os.path.join(item_path, "config.json"))
            weights_exist = os.path.exists(os.path.join(item_path, "pytorch_model.bin")) or os.path.exists(os.path.join(item_path, "model.safetensors"))

            if config_exists and weights_exist:
                AVAILABLE_VOXCPM_MODELS[item] = {
                    "type": "local",
                    "path": item_path
                }

from .voxcpm2_nodes import comfy_entrypoint

__all__ = ['comfy_entrypoint']