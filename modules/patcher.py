import math
import weakref
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


class _NoopFinalizer:
    """Drop-in replacement for weakref.finalize that never fires.

    ComfyUI's LoadedModel.model_unload() calls model_finalizer.detach()
    unconditionally, so we need an object with a detach() method.  Using a
    real weakref.finalize with cleanup_models as the callback is dangerous
    because gc.collect() inside free_memory can trigger it, which mutates
    current_loaded_models mid-iteration and crashes ComfyUI.
    """
    def detach(self):
        pass


class VoxCPMVBar:
    """VBar implementation for aimdo dynamic VRAM visualization.

    Reports per-page residency for the VoxCPM model.  Since VoxCPM
    is always fully on GPU or fully on CPU, all pages move together.
    """

    page_size: int = 32 * 1024 * 1024  # 32 MB per page
    offset: int = 0

    def __init__(self, model: object, device: torch.device):
        self._model = model
        self._device = torch.device(device.type)
        self._total_size = sum(
            p.numel() * p.element_size() for p in model.parameters()
        )
        self._total_pages = max(1, math.ceil(self._total_size / self.page_size))
        self._watermark: int = 0

    def loaded_size(self) -> int:
        """Bytes currently in VRAM."""
        try:
            param = next(self._model.parameters(), None)
            if param is None:
                return 0
            if torch.device(param.device.type) == self._device:
                return self._total_size
        except Exception:
            pass
        return 0

    def get_residency(self) -> list:
        """Per-page flags.  bit 1 = resident, bit 2 = pinned."""
        loaded = self.loaded_size()
        resident_pages = min(
            int(loaded // self.page_size), self._total_pages
        )
        return [1 if i < resident_pages else 0 for i in range(self._total_pages)]

    def get_watermark(self) -> int:
        """Current high-watermark."""
        current = self.loaded_size()
        self._watermark = max(self._watermark, current)
        return self._watermark

    def prioritize(self):
        """Reset watermark (triggered by wm button in viz panel)."""
        self._watermark = self.loaded_size()


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

    def is_dynamic(self):
        return False

    def _vbar_get(self):
        if not self.is_loaded:
            return None
        vbars = getattr(self.model.model.tts_model, "dynamic_vbars", {})
        if vbars:
            return next(iter(vbars.values()))
        return None

    def _check_vbar(self):
        if not self._vbar_active and not self._aimdo_auto:
            self._vbar_active, self._aimdo_auto = _detect_vbar()
            if self._vbar_active:
                logger.info("ComfyUI Dynamic VRAM (VBAR) detected")
            elif self._aimdo_auto:
                logger.info("ComfyUI Dynamic VRAM (aimdo auto) detected")

    def _register_with_comfy(self):
        """Register with ComfyUI's VRAM tracking and attach aimdo VBar."""
        if not self.is_loaded:
            return
        load_device = self.load_device
        if load_device.type == "cpu":
            return

        try:
            loaded = model_management.LoadedModel(self)
            tts_model = self.model.model.tts_model

            # Report model size to ComfyUI
            model_size = self.model_size()
            tts_model.model_loaded_weight_memory = model_size

            # Attach aimdo dynamic VBar for viz panel
            tts_model.dynamic_vbars = {
                load_device: VoxCPMVBar(tts_model, load_device),
            }

            loaded.real_model = weakref.ref(tts_model)
            # Use a no-op finalizer instead of weakref.finalize(tts_model,
            # cleanup_models).  The real callback fires cleanup_models() when
            # gc.collect() runs, which pops entries from
            # current_loaded_models mid-iteration inside free_memory() and
            # crashes ComfyUI with IndexError.
            loaded.model_finalizer = _NoopFinalizer()
            loaded.currently_used = True

            model_management.current_loaded_models.insert(0, loaded)
            logger.info(
                f"VoxCPM registered with ComfyUI VRAM management "
                f"({model_size / (1024 * 1024):.1f} MB)."
            )
        except Exception as e:
            logger.warning(f"Could not register with ComfyUI VRAM management: {e}")

    def _unregister_from_comfy(self):
        """Remove this patcher from ComfyUI's current_loaded_models."""
        try:
            model_management.current_loaded_models[:] = [
                lm for lm in model_management.current_loaded_models
                if lm.model is not self
            ]
        except Exception:
            pass

    def patch_model(self, device_to=None, *args, **kwargs):
        target_device = self.load_device if device_to is None else device_to

        if self.model.model is None:
            logger.info(f"Loading VoxCPM model '{self.model.model_name}' into RAM...")
            from .loader import load_model
            self.model.model = load_model(
                self.model.model_name,
                optimize=self.model.optimize,
                torch_compile=self.model.torch_compile,
                dtype=self.model.dtype,
            )

        # Move the tts_model (the actual nn.Module) to the target device
        self.model.model.tts_model.to(target_device)

        self._check_vbar()
        self._register_with_comfy()

        return super().patch_model(device_to=target_device, *args, **kwargs)

    def _unload_model_memory(self):
        """Free model from VRAM/RAM and clear caches.

        Does NOT touch current_loaded_models — safe to call during
        free_memory()'s index-based iteration.  free_memory handles
        list cleanup itself (pop by index after the loop).
        """
        from .loader import LOADED_MODELS_CACHE

        if self.is_loaded:
            logger.info(f"Unloading VoxCPM model '{self.cache_key}' from VRAM and RAM...")
            model_instance = self.model.model
            self.model.model = None
            if model_instance is not None:
                del model_instance

        cache_key = f"{self.model.model_name}_opt{self.model.optimize}_compile{self.model.torch_compile}_dtype{self.model.dtype}"
        if cache_key in LOADED_MODELS_CACHE:
            del LOADED_MODELS_CACHE[cache_key]

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        gc.collect()
        model_management.soft_empty_cache()

    def force_unload(self):
        """Fully unload the model from VRAM and RAM, clearing all caches.

        Always runs, even when VBAR/aimdo is active.  Safe to call from
        the generate node (not during free_memory iteration).
        """
        self._unload_model_memory()
        self._unregister_from_comfy()
        logger.info("Force offload complete.")

    def unpatch_model(self, device_to=None, unpatch_weights=True, *args, **kwargs):
        """Unload model when ComfyUI's free_memory iterates current_loaded_models.

        Called via LoadedModel.model_unload() → ModelPatcher.detach() →
        unpatch_model().  MUST NOT mutate current_loaded_models — free_memory
        saves indices in a first pass and uses them in a second pass; mutating
        the list between passes causes IndexError.
        """
        if unpatch_weights and self.is_loaded:
            self._unload_model_memory()

        return super().unpatch_model(device_to, unpatch_weights, *args, **kwargs)
