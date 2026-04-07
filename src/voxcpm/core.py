import os
import json
import re
import tempfile
import torch
import numpy as np
from typing import Generator, Optional
from huggingface_hub import snapshot_download
from .model.voxcpm import VoxCPMModel, LoRAConfig
from .model.voxcpm2 import VoxCPM2Model


class VoxCPM:
    def __init__(self,
            voxcpm_model_path : str,
            zipenhancer_model_path : str = None,
            enable_denoiser : bool = False,
            optimize: bool = True,
            lora_config: Optional[LoRAConfig] = None,
            lora_weights_path: Optional[str] = None,
        ):
        # If lora_weights_path is provided but no lora_config, create a default one
        if lora_weights_path is not None and lora_config is None:
            lora_config = LoRAConfig(
                enable_lm=True,
                enable_dit=True,
                enable_proj=False,
            )

        # Detect architecture from config.json
        config_path = os.path.join(voxcpm_model_path, "config.json")
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
        arch = config.get("architecture", "voxcpm").lower()

        if arch == "voxcpm2":
            self.tts_model = VoxCPM2Model.from_local(voxcpm_model_path, optimize=optimize, lora_config=lora_config)
        else:
            self.tts_model = VoxCPMModel.from_local(voxcpm_model_path, optimize=optimize, lora_config=lora_config)

        # Load LoRA weights if path is provided
        if lora_weights_path is not None:
            self.tts_model.load_lora_weights(lora_weights_path)

        self.text_normalizer = None
        self._is_v2 = isinstance(self.tts_model, VoxCPM2Model)

        # Denoiser handling: disabled by default for ComfyUI
        self.denoiser = None
        self._denoiser_model_path = zipenhancer_model_path  # None = auto-cache in ComfyUI models dir
        if enable_denoiser:
            self._init_denoiser()

    def _init_denoiser(self):
        if self.denoiser is not None:
            return True
        try:
            from .zipenhancer import ZipEnhancer
            self.denoiser = ZipEnhancer(self._denoiser_model_path)
            return True
        except ImportError:
            print("[VoxCPM] Warning: modelscope not installed. Denoiser disabled. Install with: pip install modelscope")
            return False

    @classmethod
    def from_pretrained(cls,
            hf_model_id: str = "openbmb/VoxCPM1.5",
            load_denoiser: bool = False,
            zipenhancer_model_id: str = None,
            cache_dir: str = None,
            local_files_only: bool = False,
            optimize: bool = True,
            lora_config: Optional[LoRAConfig] = None,
            lora_weights_path: Optional[str] = None,
            **kwargs,
        ):
        repo_id = hf_model_id
        if not repo_id:
            raise ValueError("You must provide hf_model_id")

        if os.path.isdir(repo_id):
            local_path = repo_id
        else:
            local_path = snapshot_download(
                repo_id=repo_id,
                cache_dir=cache_dir,
                local_files_only=local_files_only,
            )

        return cls(
            voxcpm_model_path=local_path,
            zipenhancer_model_path=zipenhancer_model_id if load_denoiser else None,
            enable_denoiser=load_denoiser,
            optimize=optimize,
            lora_config=lora_config,
            lora_weights_path=lora_weights_path,
            **kwargs,
        )

    def generate(self, *args, **kwargs) -> np.ndarray:
        return next(self._generate(*args, streaming=False, **kwargs))

    def generate_streaming(self, *args, **kwargs) -> Generator[np.ndarray, None, None]:
        return self._generate(*args, streaming=True, **kwargs)

    def _generate(self,
            text : str,
            prompt_wav_path : str = None,
            prompt_text : str = None,
            reference_wav_path : str = None,
            cfg_value : float = 2.0,
            inference_timesteps : int = 10,
            min_len : int = 2,
            max_len : int = 4096,
            normalize : bool = False,
            denoise : bool = False,
            retry_badcase : bool = True,
            retry_badcase_max_times : int = 3,
            retry_badcase_ratio_threshold : float = 6.0,
            streaming: bool = False,
        ) -> Generator[np.ndarray, None, None]:

        if not text.strip() or not isinstance(text, str):
            raise ValueError("target text must be a non-empty string")

        if prompt_wav_path is not None and not os.path.exists(prompt_wav_path):
            raise FileNotFoundError(f"prompt_wav_path does not exist: {prompt_wav_path}")

        if reference_wav_path is not None and not os.path.exists(reference_wav_path):
            raise FileNotFoundError(f"reference_wav_path does not exist: {reference_wav_path}")

        # reference_wav_path requires V2
        if reference_wav_path is not None and not self._is_v2:
            raise ValueError("reference_wav_path is only supported with VoxCPM2 models")

        # For V2 with both prompt and reference, prompt_text + prompt_wav_path must be paired
        if self._is_v2 and prompt_wav_path is not None and reference_wav_path is not None:
            if prompt_text is None or not prompt_text.strip():
                raise ValueError("prompt_text is required when using both prompt_wav_path and reference_wav_path")

        text = text.replace("\n", " ")
        text = re.sub(r'\s+', ' ', text)
        temp_files = []

        try:
            actual_prompt_path = prompt_wav_path
            actual_ref_path = reference_wav_path

            # Denoise if requested
            if denoise and self.denoiser is not None:
                if prompt_wav_path is not None:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                        temp_files.append(tmp.name)
                    self.denoiser.enhance(prompt_wav_path, output_path=temp_files[-1])
                    actual_prompt_path = temp_files[-1]
                if reference_wav_path is not None:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                        temp_files.append(tmp.name)
                    self.denoiser.enhance(reference_wav_path, output_path=temp_files[-1])
                    actual_ref_path = temp_files[-1]

            # Build prompt cache if any audio is provided
            if actual_prompt_path is not None or actual_ref_path is not None:
                if self._is_v2:
                    fixed_prompt_cache = self.tts_model.build_prompt_cache(
                        prompt_text=prompt_text,
                        prompt_wav_path=actual_prompt_path,
                        reference_wav_path=actual_ref_path,
                    )
                else:
                    fixed_prompt_cache = self.tts_model.build_prompt_cache(
                        prompt_text=prompt_text,
                        prompt_wav_path=actual_prompt_path,
                    )
            else:
                fixed_prompt_cache = None

            # Text Normalization
            if normalize:
                if self.text_normalizer is None:
                    try:
                        from .utils.text_normalize import TextNormalizer
                        self.text_normalizer = TextNormalizer()
                    except ImportError:
                        class MockNormalizer:
                            def normalize(self, t): return t
                        self.text_normalizer = MockNormalizer()
                text = self.text_normalizer.normalize(text)

            generate_result = self.tts_model._generate_with_prompt_cache(
                target_text=text,
                prompt_cache=fixed_prompt_cache,
                min_len=min_len,
                max_len=max_len,
                inference_timesteps=inference_timesteps,
                cfg_value=cfg_value,
                retry_badcase=retry_badcase,
                retry_badcase_max_times=retry_badcase_max_times,
                retry_badcase_ratio_threshold=retry_badcase_ratio_threshold,
                streaming=streaming,
            )

            for wav, _, _ in generate_result:
                yield wav.squeeze(0).cpu().numpy()

        finally:
            for tmp_path in temp_files:
                if tmp_path and os.path.exists(tmp_path):
                    try:
                        os.unlink(tmp_path)
                    except OSError:
                        pass

    # ------------------------------------------------------------------ #
    # LoRA Interface
    # ------------------------------------------------------------------ #
    def load_lora(self, lora_weights_path: str) -> tuple:
        if self.tts_model.lora_config is None:
            raise RuntimeError("Cannot load LoRA weights: model was not initialized with LoRA config.")
        return self.tts_model.load_lora_weights(lora_weights_path)

    def unload_lora(self):
        self.tts_model.reset_lora_weights()

    def set_lora_enabled(self, enabled: bool):
        self.tts_model.set_lora_enabled(enabled)

    def get_lora_state_dict(self) -> dict:
        return self.tts_model.get_lora_state_dict()

    @property
    def lora_enabled(self) -> bool:
        return self.tts_model.lora_config is not None
