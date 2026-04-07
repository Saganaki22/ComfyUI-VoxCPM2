"""
ZipEnhancer Module - Audio Denoising Enhancer

Provides on-demand import ZipEnhancer functionality for audio denoising processing.
Caches the model in ComfyUI's models directory for offline use after first download.
"""

import os
import tempfile
from typing import Optional, Union
import torchaudio
import torch
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks


# Default ModelScope model ID
_DEFAULT_MODEL_ID = "iic/speech_zipenhancer_ans_multiloss_16k_base"


def _resolve_denoiser_path(model_path: str = None) -> str:
    """Resolve the denoiser model to a local path, downloading if needed.

    If model_path is already a local directory with files, return it as-is.
    Otherwise, download from ModelScope to ComfyUI's models/denoiser/ directory.
    """
    # If a local path is given and exists, use it directly
    if model_path and os.path.isdir(model_path) and os.listdir(model_path):
        return model_path

    model_id = model_path or _DEFAULT_MODEL_ID

    # Try to find ComfyUI's models directory
    try:
        import folder_paths
        denoiser_dir = os.path.join(folder_paths.models_dir, "denoiser", "zipenhancer")
    except ImportError:
        from pathlib import Path
        denoiser_dir = os.path.join(Path.home(), ".cache", "voxcpm", "denoiser", "zipenhancer")

    os.makedirs(denoiser_dir, exist_ok=True)

    # If already downloaded, use the cached version
    if os.listdir(denoiser_dir):
        return denoiser_dir

    # Download from ModelScope
    print(f"[ZipEnhancer] Downloading denoiser model '{model_id}' to {denoiser_dir}...", flush=True)
    try:
        from modelscope.hub.snapshot_download import snapshot_download as ms_download
        ms_download(model_id=model_id, cache_dir=denoiser_dir)
        # ModelScope downloads to a subdirectory — find it
        for item in os.listdir(denoiser_dir):
            item_path = os.path.join(denoiser_dir, item)
            if os.path.isdir(item_path):
                # Check if this looks like the model (has model files)
                has_files = any(
                    f.endswith(('.bin', '.pth', '.json', '.ckpt', '.safetensors'))
                    for f in os.listdir(item_path)
                )
                if has_files:
                    return item_path
        return denoiser_dir
    except Exception:
        # Fallback: let ModelScope handle caching itself
        print(f"[ZipEnhancer] Could not pre-download. Falling back to ModelScope auto-cache.", flush=True)
        return model_id


class ZipEnhancer:
    """ZipEnhancer Audio Denoising Enhancer"""
    def __init__(self, model_path: str = None):
        """
        Initialize ZipEnhancer
        Args:
            model_path: ModelScope model ID, or local path. If None, uses default
                        and caches in ComfyUI's models/denoiser/ directory.
        """
        self.model_path = model_path or _DEFAULT_MODEL_ID
        local_path = _resolve_denoiser_path(self.model_path)
        self._pipeline = pipeline(
                Tasks.acoustic_noise_suppression,
                model=local_path,
            )
        
    def _normalize_loudness(self, wav_path: str):
        """
        Audio loudness normalization
        
        Args:
            wav_path: Audio file path
        """
        audio, sr = torchaudio.load(wav_path)
        loudness = torchaudio.functional.loudness(audio, sr)
        normalized_audio = torchaudio.functional.gain(audio, -20-loudness)
        torchaudio.save(wav_path, normalized_audio, sr)
    
    def enhance(self, input_path: str, output_path: Optional[str] = None, 
                normalize_loudness: bool = True) -> str:
        """
        Audio denoising enhancement
        Args:
            input_path: Input audio file path
            output_path: Output audio file path (optional, creates temp file by default)
            normalize_loudness: Whether to perform loudness normalization
        Returns:
            str: Output audio file path
        Raises:
            RuntimeError: If pipeline is not initialized or processing fails
        """
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Input audio file does not exist: {input_path}")
        # Create temporary file if no output path is specified
        if output_path is None:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
                output_path = tmp_file.name
        try:
            # Perform denoising processing
            self._pipeline(input_path, output_path=output_path)
            # Loudness normalization
            if normalize_loudness:
                self._normalize_loudness(output_path)
            return output_path
        except Exception as e:
            # Clean up possibly created temporary files
            if output_path and os.path.exists(output_path):
                try:
                    os.unlink(output_path)
                except OSError:
                    pass
            raise RuntimeError(f"Audio denoising processing failed: {e}")