import torch
import gc
import logging
import comfy.model_patcher
import comfy.model_management as model_management

logger = logging.getLogger(__name__)


def _detect_vbar():
    """Detect if ComfyUI's dynamic VRAM management (VBAR/aimdo) is available."""
    try:
        from comfy_aimdo.model_vbar import ModelVBAR
        return True, True
    except ImportError:
        pass
    try:
        import comfy_aimdo
        return False, True
    except ImportError:
        pass
    return False, False


class VoxCPMPatcher(comfy.model_patcher.ModelPatcher):
    """
    Custom ModelPatcher for managing VoxCPM models in ComfyUI.
    """
    def __init__(self, model, *args, **kwargs):
        super().__init__(model, *args, **kwargs)
        self.cache_key = getattr(model, 'model_name', 'VoxCPM_Unknown')
        self._vbar_active = False
        self._aimdo_auto = False

    @property
    def is_loaded(self) -> bool:
        return hasattr(self, 'model') and self.model is not None and hasattr(self.model, 'model') and self.model.model is not None

    def _check_vbar(self):
        if not self._vbar_active and not self._aimdo_auto:
            self._vbar_active, self._aimdo_auto = _detect_vbar()
            if self._vbar_active:
                logger.info("ComfyUI Dynamic VRAM (VBAR) detected")
            elif self._aimdo_auto:
                logger.info("ComfyUI Dynamic VRAM (aimdo auto) detected")

    def patch_model(self, device_to=None, *args, **kwargs):
        target_device = self.load_device if device_to is None else device_to

        if self.model.model is None:
            logger.info(f"Loading VoxCPM model '{self.model.model_name}' into RAM...")
            from .loader import load_model
            self.model.model = load_model(
                self.model.model_name,
                optimize=self.model.torch_compile,
                torch_compile=self.model.torch_compile,
                dtype=self.model.dtype,
            )

        # Move the tts_model (the actual nn.Module) to the target device
        self.model.model.tts_model.to(target_device)

        self._check_vbar()

        return super().patch_model(device_to=target_device, *args, **kwargs)

    def force_unload(self):
        """Fully unload the model from VRAM and RAM, clearing all caches.
        Always runs, even when VBAR/aimdo is active."""
        from .loader import LOADED_MODELS_CACHE

        if self.is_loaded:
            logger.info(f"Force offloading VoxCPM model '{self.cache_key}' from VRAM and RAM...")
            try:
                self.model.model.tts_model.to("cpu")
            except Exception:
                pass

            self.model.model = None

        # Clear the loader cache entry so the model is fully released
        cache_key = f"{self.model.model_name}_opt{self.model.torch_compile}_dtype{self.model.dtype}"
        if cache_key in LOADED_MODELS_CACHE:
            del LOADED_MODELS_CACHE[cache_key]

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        gc.collect()
        model_management.soft_empty_cache()

        logger.info("Force offload complete.")

    def unpatch_model(self, device_to=None, unpatch_weights=True, *args, **kwargs):
        """Gentle offload — skips manual CPU move when VBAR/aimdo handles it."""
        if unpatch_weights and self.is_loaded:
            if self._vbar_active or self._aimdo_auto:
                mode = "VBAR" if self._vbar_active else "aimdo auto"
                logger.info(f"{mode} active — letting ComfyUI handle offload")
            else:
                try:
                    self.model.model.tts_model.to(self.offload_device)
                except Exception:
                    pass

            self.model.model = None
            gc.collect()
            model_management.soft_empty_cache()

        return super().unpatch_model(device_to, unpatch_weights, *args, **kwargs)
