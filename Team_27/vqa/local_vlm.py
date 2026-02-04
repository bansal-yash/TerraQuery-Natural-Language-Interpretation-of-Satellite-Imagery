# local_vlm.py  (Fixed duplication)
import unsloth
from unsloth import FastVisionModel
from typing import Union, List, Optional, Dict, Any, Tuple
from PIL import Image
import os
import torch
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True
torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True

# from torch.backends.cuda import sdp_kernel

# # sdp_kernel.enable_flash(True)
# sdp_kernel.enable_math(True)
# sdp_kernel.enable_mem_efficient(True)
import json
import inspect
import traceback
import re # Added explicit import
import json as json_module
from contextlib import contextmanager, nullcontext
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration, TextStreamer
from pydantic import PrivateAttr
from peft import PeftModel
import time
from unsloth import FastVisionModel
import gc

# LangChain / LangGraph imports
try:
    from langchain_core.tools import tool, Tool, StructuredTool
    from langchain_core.language_models.chat_models import BaseChatModel
    from langchain_core.messages import BaseMessage, AIMessage, HumanMessage, SystemMessage
    try:
        from langchain_core.messages import ToolMessage
    except ImportError:
        ToolMessage = BaseMessage
    from langchain_core.outputs import ChatResult, ChatGeneration
    from langchain_core.utils.function_calling import convert_to_openai_tool
    from langgraph.prebuilt import create_react_agent
    LC_AVAILABLE = True
except Exception as e:
    print(f"[WARNING] LangChain/LangGraph not available: {e}")
    LC_AVAILABLE = False

import cv2
import geometric_utils

CAPTION_SYSTEM_PROMPT = (
    "You are an expert captioning assistant. Describe the provided images in ~50 words, focusing on the most salient context and relationships. "
    "In addition to your default behavior, always incorporate any specific details the user explicitly requests."
)

class LocalVLM:
    # Shared, lazily-initialized resources to avoid re-loading the 8B model per instance
    _shared_processor: Optional[AutoProcessor] = None
    _shared_model: Optional[Qwen3VLForConditionalGeneration] = None
    _shared_precision: Optional[str] = None
    _shared_attn_impl: Optional[str] = None
    _shared_device: Optional[str] = None
    _lora_loaded: bool = False
    _lora_path: Optional[str] = None
    _base_model_backup: Optional[Dict[str, torch.Tensor]] = None
    _base_adapter_path: Optional[str] = None
    _base_adapter_name: str = "__base_adapter__"
    _active_adapter_name: str = "__active_adapter__"
    _current_adapter: Optional[str] = None
    _perf_env_default: bool = bool(int(os.environ.get("LOCAL_VLM_PROFILE", "0")))
    _crops_context_cache: Dict[str, Any] = {
        "path": None,
        "mtime": None,
        "text": "",
        "total_crops": 0,
        "prompt_groups": 0,
    }

    @property
    def model(self) -> Qwen3VLForConditionalGeneration:
        return LocalVLM._shared_model

    @property
    def processor(self) -> AutoProcessor:
        return LocalVLM._shared_processor

    @property
    def last_perf(self) -> Optional[Dict[str, float]]:
        return getattr(self, "_last_perf", None)

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-VL-8B-Instruct",
        device: Optional[str] = None,
        system_prompt: Optional[str] = None,
        stream_thoughts: bool = True,
        precision: str = "bf16",
        attn_implementation: Optional[str] = "sdpa",
        prefer_unsloth: bool = False,
        base_adapter_path: Optional[str] = None,
        null_adapter_path: Optional[str] = None,
        enable_perf_logs: Optional[bool] = None,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.system_prompt = system_prompt or (
            "You are a visual reasoning assistant. Think step-by-step and explain logic clearly."
        )
        self.stream_thoughts = stream_thoughts
        self.precision = precision.lower()
        if self.precision not in {"bf16", "fp16"}:
            raise ValueError("precision must be either 'bf16' or 'fp16'")
        self.attn_implementation = attn_implementation
        self._prefer_unsloth = prefer_unsloth
        self._requested_base_adapter = base_adapter_path or null_adapter_path
        self._null_adapter_path = null_adapter_path or base_adapter_path
        self._device_is_cuda = self.device.startswith("cuda") and torch.cuda.is_available()
        self.torch_dtype = torch.bfloat16 if self.precision == "bf16" else torch.float16
        self._pixel_dtype = self.torch_dtype if self._device_is_cuda else torch.float32
        self._autocast_kwargs = {"device_type": "cuda", "dtype": self.torch_dtype} if self._device_is_cuda else None
        self._perf_enabled = LocalVLM._perf_env_default if enable_perf_logs is None else bool(enable_perf_logs)

        # Improve matmul performance on CPU and some GPUs
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass

        if LocalVLM._shared_processor is None or LocalVLM._shared_model is None:
            self._load_shared_resources(model_name)
            if self.stream_thoughts:
                print("[Init] Streaming mode enabled — reasoning tokens will print as they generate.\n")
        else:
            print(f"[Init] Reusing shared {model_name} on {self.device} — no reload.")
            if LocalVLM._shared_precision and LocalVLM._shared_precision != self.precision:
                print(f"[Init] ⚠ Requested precision {self.precision} differs from shared precision {LocalVLM._shared_precision}. Reusing loaded model.")
            if self.attn_implementation and LocalVLM._shared_attn_impl != self.attn_implementation:
                print(f"[Init] ⚠ Requested attention impl {self.attn_implementation} differs from shared model; currently {LocalVLM._shared_attn_impl}.")

        adapter_to_attach = self._requested_base_adapter
        if adapter_to_attach:
            self._maybe_attach_base_adapter(adapter_to_attach, set_active=True)

        # Print model generation config details (only once)
        if not hasattr(LocalVLM, '_config_printed'):
            print("\n[VLM Debug] Model DEFAULT generation config (before overrides):")
            print(f"  Model type: {self.model.config.model_type}")
            gen_config = self.model.generation_config
            print(f"  Default generation config:")
            print(f"    max_length: {gen_config.max_length}")
            print(f"    max_new_tokens: {gen_config.max_new_tokens}")
            print(f"    do_sample: {gen_config.do_sample} (will be overridden to False)")
            print(f"    temperature: {gen_config.temperature}")
            print(f"    top_p: {gen_config.top_p}")
            print(f"    top_k: {gen_config.top_k}")
            print(f"    repetition_penalty: {gen_config.repetition_penalty}")
            print(f"    eos_token_id: {gen_config.eos_token_id}")
            print(f"    pad_token_id: {gen_config.pad_token_id}")
            print(f"    bos_token_id: {gen_config.bos_token_id}")
            LocalVLM._config_printed = True

        if LocalVLM._base_adapter_path:
            LocalVLM._current_adapter = LocalVLM._base_adapter_name

        self._agent = None
        self._tools_registered = False
        self._suppress_debug_prints = False
        # Small cache for opened images by path to avoid repeated disk I/O
        self._image_cache: Dict[str, Image.Image] = {}
        self._last_perf: Optional[Dict[str, float]] = None

    def _load_shared_resources(self, model_name: str) -> None:
        device_desc = "cuda" if self._device_is_cuda else "cpu"
        print(f"[Init] Loading {model_name} on {device_desc} …")

        if not self._prefer_unsloth:
            try:
                LocalVLM._shared_processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
                model_kwargs: Dict[str, Any] = dict(
                    torch_dtype=self.torch_dtype if self._device_is_cuda else torch.float32,
                    device_map="auto" if self._device_is_cuda else None,
                    trust_remote_code=True,
                )
                if self._device_is_cuda and self.attn_implementation:
                    model_kwargs["attn_implementation"] = self.attn_implementation

                LocalVLM._shared_model = Qwen3VLForConditionalGeneration.from_pretrained(
                    model_name,
                    **model_kwargs,
                ).eval()
                LocalVLM._shared_precision = self.precision
                LocalVLM._shared_attn_impl = self.attn_implementation
                LocalVLM._shared_device = device_desc
                print(f"[Init] Model ready (Transformers, attn={self.attn_implementation}).")
                return
            except Exception as exc:
                print(f"[Init] Transformers load failed ({exc}); falling back to FastVisionModel.")

        LocalVLM._shared_model, LocalVLM._shared_processor = FastVisionModel.from_pretrained(
            model_name,
            load_in_4bit=False,
            device_map="auto" if self._device_is_cuda else None,
            trust_remote_code=True,
            use_safetensors=True,
            torch_dtype=self.torch_dtype if self._device_is_cuda else torch.float32,
        )
        LocalVLM._shared_model.eval()
        LocalVLM._shared_precision = self.precision
        LocalVLM._shared_attn_impl = None
        LocalVLM._shared_device = device_desc
        print("[Init] Model ready (FastVisionModel fallback, SDPA disabled).")

    def _maybe_attach_base_adapter(self, adapter_path: str, *, set_active: bool = False) -> None:
        if not os.path.isdir(adapter_path):
            raise FileNotFoundError(f"Base adapter directory not found: {adapter_path}")

        model = LocalVLM._shared_model
        if model is None:
            raise RuntimeError("Model must be initialized before attaching base adapter")

        if LocalVLM._base_adapter_path and LocalVLM._base_adapter_path != adapter_path:
            print(
                f"[Init] ⚠ Base adapter already set to {LocalVLM._base_adapter_path}; "
                f"cannot switch to {adapter_path} without reload"
            )
            return

        if LocalVLM._base_adapter_path == adapter_path and isinstance(model, PeftModel):
            if set_active:
                model.set_adapter(LocalVLM._base_adapter_name)
                LocalVLM._current_adapter = LocalVLM._base_adapter_name
                LocalVLM._lora_loaded = False
                LocalVLM._lora_path = None
            return

        try:
            if isinstance(model, PeftModel):
                if LocalVLM._base_adapter_name in getattr(model, "peft_config", {}):
                    if set_active:
                        model.set_adapter(LocalVLM._base_adapter_name)
                else:
                    model.load_adapter(adapter_path, adapter_name=LocalVLM._base_adapter_name)
                    if set_active:
                        model.set_adapter(LocalVLM._base_adapter_name)
                model.eval()
                LocalVLM._shared_model = model
            else:
                LocalVLM._shared_model = PeftModel.from_pretrained(
                    model,
                    adapter_path,
                    adapter_name=LocalVLM._base_adapter_name,
                    is_trainable=False,
                ).eval()
        except Exception as exc:
            raise RuntimeError(f"Failed to attach base adapter from {adapter_path}: {exc}") from exc

        LocalVLM._base_adapter_path = adapter_path
        if set_active:
            LocalVLM._lora_loaded = False
            LocalVLM._lora_path = None
            LocalVLM._current_adapter = LocalVLM._base_adapter_name
        print(f"[Init] Base adapter '{LocalVLM._base_adapter_name}' ready from {adapter_path}")

    def _infer_model_device(self) -> torch.device:
        model = self.model
        if model is None:
            raise RuntimeError("Model not initialized")
        if hasattr(model, "device") and model.device is not None:
            return torch.device(model.device)
        try:
            param = next(model.parameters())
            return param.device
        except StopIteration:
            return torch.device(self.device if self._device_is_cuda else "cpu")

    def _prepare_inputs(self, batch) -> Dict[str, Any]:
        device = self._infer_model_device()
        for key, value in list(batch.items()):
            if isinstance(value, torch.Tensor):
                non_blocking = self._device_is_cuda
                tensor = value.to(device=device, non_blocking=non_blocking)
                if key == "pixel_values":
                    tensor = tensor.to(dtype=self._pixel_dtype)
                    if tensor.dim() == 4:
                        tensor = tensor.to(memory_format=torch.channels_last)
                    else:
                        tensor = tensor.contiguous()
                batch[key] = tensor
        return batch

    def _autocast_context(self):
        if self._autocast_kwargs:
            return torch.autocast(**self._autocast_kwargs)
        return nullcontext()
    def _load_lora_via_adapter(self, adapter_path: str) -> Dict[str, Any]:
        model = LocalVLM._shared_model
        if not isinstance(model, PeftModel):
            raise RuntimeError("Adapter swap requested but model is not a PeftModel")

        adapter_name = LocalVLM._active_adapter_name
        start_time = time.time()
        initial_memory = self._gpu_memory_gb()

        if adapter_name in getattr(model, "peft_config", {}):
            if hasattr(model, "delete_adapter"):
                model.delete_adapter(adapter_name)
            else:
                model.peft_config.pop(adapter_name, None)

        try:
            model.load_adapter(adapter_path, adapter_name=adapter_name)
            model.set_adapter(adapter_name)
            model.eval()
        except Exception as exc:
            return {"status": "failed", "error": str(exc), "path": adapter_path}

        LocalVLM._shared_model = model
        LocalVLM._lora_loaded = True
        LocalVLM._lora_path = adapter_path
        LocalVLM._current_adapter = adapter_name

        final_memory = self._gpu_memory_gb()
        load_time = time.time() - start_time
        print(f"[VLM] ✓ LoRA adapter '{adapter_name}' loaded via set_adapter in {load_time:.2f}s")
        if self._perf_enabled:
            print(
                f"[VLM PERF] Adapter memory delta: {(final_memory - initial_memory):.2f} GB"
            )

        return {
            "status": "success",
            "path": adapter_path,
            "load_time": load_time,
            "initial_memory_gb": initial_memory,
            "final_memory_gb": final_memory,
            "memory_increase_gb": final_memory - initial_memory,
            "adapter_name": adapter_name,
        }


    def _can_swap_adapters(self) -> bool:
        return isinstance(LocalVLM._shared_model, PeftModel) and LocalVLM._base_adapter_path is not None

    def _gpu_memory_gb(self) -> float:
        if torch.cuda.is_available():
            try:
                torch.cuda.synchronize()

            except Exception:
                pass

            return torch.cuda.memory_allocated() / 1024**3

        return 0.0


    @contextmanager
    def _perf_section(self, label: str):
        if not self._perf_enabled:
            yield
            return
        start = time.perf_counter()
        try:
            yield
        finally:
            delta = (time.perf_counter() - start) * 1000
            print(f"[VLM PERF] {label}: {delta:.1f} ms")

    # ------------------ LoRA adapter management ------------------

    def load_lora(self, adapter_path: str) -> Dict[str, Any]:
        """
        Load LoRA adapter weights into the model.
        
        Args:
            adapter_path: Path to the LoRA adapter checkpoint directory
            
        Returns:
            Dict containing timing and memory info
        """
        if self._can_swap_adapters():
            return self._load_lora_via_adapter(adapter_path)

        if LocalVLM._lora_loaded:
            print(f"[VLM] LoRA adapter already loaded from {LocalVLM._lora_path}")
            if LocalVLM._lora_path == adapter_path:
                return {"status": "already_loaded", "path": adapter_path}
            else:
                print(f"[VLM] Different adapter requested, unloading current adapter first")
                null_path = self._null_adapter_path or LocalVLM._base_adapter_path
                self.unload_lora(null_path=null_path)
        
        start_time = time.time()
        
        # Get initial GPU memory
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            initial_memory = torch.cuda.memory_allocated() / 1024**3  # GB
        else:
            initial_memory = 0
        
        print(f"[VLM] Loading LoRA adapter from {adapter_path}...")
        print(f"[VLM] Saving base model state for restoration...")
        
        try:
            # CRITICAL: Save base model state_dict before loading adapter
            # This allows us to truly restore original weights during unload
            LocalVLM._base_model_backup = {
                k: v.cpu().clone() for k, v in LocalVLM._shared_model.state_dict().items()
            }
            
            # Load LoRA adapter using PeftModel
            LocalVLM._shared_model = PeftModel.from_pretrained(
                LocalVLM._shared_model,
                adapter_path,
                is_trainable=False  # Inference mode
            ).eval()
            
            LocalVLM._lora_loaded = True
            LocalVLM._lora_path = adapter_path
            
            load_time = time.time() - start_time
            
            # Get final GPU memory
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                final_memory = torch.cuda.memory_allocated() / 1024**3  # GB
                memory_increase = final_memory - initial_memory
            else:
                final_memory = 0
                memory_increase = 0
            
            print(f"[VLM] ✓ LoRA adapter loaded successfully in {load_time:.2f}s")
            print(f"[VLM] GPU Memory: {initial_memory:.2f}GB → {final_memory:.2f}GB (+{memory_increase:.2f}GB)")
            
            return {
                "status": "success",
                "path": adapter_path,
                "load_time": load_time,
                "initial_memory_gb": initial_memory,
                "final_memory_gb": final_memory,
                "memory_increase_gb": memory_increase
            }
            
        except Exception as e:
            print(f"[VLM] ✗ Failed to load LoRA adapter: {e}")
            traceback.print_exc()
            LocalVLM._lora_loaded = False
            LocalVLM._lora_path = None
            LocalVLM._base_model_backup = None
            return {
                "status": "failed",
                "path": adapter_path,
                "error": str(e)
            }
    
    def unload_lora(self, null_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Unload LoRA adapter and revert to ORIGINAL base model weights.
        
        Returns:
            Dict containing timing and memory info
        """
        if null_path and not self._null_adapter_path:
            self._null_adapter_path = null_path
        if not null_path and self._null_adapter_path:
            null_path = self._null_adapter_path

        if self._can_swap_adapters() and LocalVLM._lora_loaded:
            return self._unload_lora_via_adapter(null_path)

        if not LocalVLM._lora_loaded:
            print("[VLM] No LoRA adapter loaded, nothing to unload")
            return {"status": "no_adapter_loaded"}
        
        if LocalVLM._base_model_backup is None:
            print("[VLM] WARNING: No base model backup found! Cannot restore original weights.")
            return {"status": "failed", "error": "No base model backup available"}
        
        start_time = time.time()
        
        # Get initial GPU memory
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            initial_memory = torch.cuda.memory_allocated() / 1024**3  # GB
        else:
            initial_memory = 0
        
        print(f"[VLM] Unloading LoRA adapter from {LocalVLM._lora_path}...")
        print(f"[VLM] Restoring original base model weights...")
        
        try:
            # Unload adapters to restore original model structure
            # This removes LoRA layers and restores original linear layers
            base_model = LocalVLM._shared_model.unload()
            
            # Restore the original base model weights from backup
            # This ensures any weight modifications are completely reverted
            base_model.load_state_dict(
                {k: v.to(base_model.device) for k, v in LocalVLM._base_model_backup.items()},
                strict=True
            )
            
            # Update shared model reference
            LocalVLM._shared_model = base_model
            
            # Clean up backup
            del LocalVLM._base_model_backup
            LocalVLM._base_model_backup = None
            torch.cuda.empty_cache()
            gc.collect()
            
            # Reset LoRA state
            LocalVLM._lora_loaded = False
            adapter_path = LocalVLM._lora_path
            LocalVLM._lora_path = None
            
            unload_time = time.time() - start_time
            # Get final GPU memory
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
                gc.collect()
                final_memory = torch.cuda.memory_allocated() / 1024**3  # GB
                memory_decrease = initial_memory - final_memory
            else:
                final_memory = 0
                memory_decrease = 0
            
            print(f"[VLM] ✓ LoRA adapter unloaded and base weights restored in {unload_time:.2f}s")
            print(f"[VLM] GPU Memory: {initial_memory:.2f}GB → {final_memory:.2f}GB (-{memory_decrease:.2f}GB)")
            
            return {
                "status": "success",
                "path": adapter_path,
                "unload_time": unload_time,
                "initial_memory_gb": initial_memory,
                "final_memory_gb": final_memory,
                "memory_decrease_gb": memory_decrease
            }
            
        except Exception as e:
            print(f"[VLM] ✗ Failed to unload LoRA adapter: {e}")
            traceback.print_exc()
            return {
                "status": "failed",
                "error": str(e)
            }

    def _unload_lora_via_adapter(self, null_path: Optional[str] = None) -> Dict[str, Any]:
        model = LocalVLM._shared_model
        if not isinstance(model, PeftModel):
            return {"status": "failed", "error": "Model is not a PeftModel"}

        if LocalVLM._base_adapter_path is None and null_path:
            try:
                self._maybe_attach_base_adapter(null_path, set_active=False)
            except Exception as exc:
                return {"status": "failed", "error": f"Failed to attach null adapter: {exc}"}

        if LocalVLM._base_adapter_path is None:
            return {"status": "failed", "error": "No base adapter available for swap"}

        if LocalVLM._base_adapter_name not in getattr(model, "peft_config", {}):
            try:
                model.load_adapter(
                    LocalVLM._base_adapter_path,
                    adapter_name=LocalVLM._base_adapter_name,
                )
            except Exception as exc:
                return {"status": "failed", "error": f"Failed to load base adapter: {exc}"}

        adapter_name = LocalVLM._active_adapter_name
        start_time = time.time()
        initial_memory = self._gpu_memory_gb()

        if adapter_name in getattr(model, "peft_config", {}):
            if hasattr(model, "set_adapter"):
                model.set_adapter(LocalVLM._base_adapter_name)
            if hasattr(model, "delete_adapter"):
                model.delete_adapter(adapter_name)
            else:
                model.peft_config.pop(adapter_name, None)

        LocalVLM._lora_loaded = False
        LocalVLM._lora_path = None
        LocalVLM._current_adapter = LocalVLM._base_adapter_name

        if hasattr(model, "eval"):
            model.eval()

        final_memory = self._gpu_memory_gb()
        unload_time = time.time() - start_time
        print(f"[VLM] ✓ Adapter '{adapter_name}' disabled; reverted to base adapter")

        return {
            "status": "success",
            "unload_time": unload_time,
            "initial_memory_gb": initial_memory,
            "final_memory_gb": final_memory,
            "memory_decrease_gb": initial_memory - final_memory,
        }

    # ------------------ internal helpers ------------------

    def _img_item(self, image_or_path: Union[str, Image.Image]):
        if isinstance(image_or_path, str):
            if os.path.exists(image_or_path):
                try:
                    # Cache opened images by path to reduce repeated disk I/O
                    img = self._image_cache.get(image_or_path)
                    if img is None:
                        img = Image.open(image_or_path)
                        # PIL lazy loads; force load to cache pixels now to avoid repeated file hits
                        img.load()
                        self._image_cache[image_or_path] = img
                    if img.mode != 'RGB':
                        img = img.convert("RGB")
                    return {"type": "image", "image": img}
                except Exception as e:
                    print(f"[local_vlm] Warning: failed to open image '{image_or_path}': {e}.")
                    return {"type": "image", "image": image_or_path}
            else:
                return {"type": "image", "image": image_or_path}

        if isinstance(image_or_path, Image.Image):
            return {"type": "image", "image": image_or_path}
        return {"type": "image", "image": image_or_path}

    @staticmethod
    def _format_crops_context(manifest: Dict[str, Any]) -> Tuple[str, Dict[str, int]]:
        prompts_data = manifest.get("prompts", []) if isinstance(manifest, dict) else []
        if not prompts_data:
            return "", {"total_crops": 0, "prompt_groups": 0}

        lines = ["\n\nDETECTED OBJECTS INFORMATION:\n", "Image 0 is the full base image.\n"]
        total_crops = 0
        for prompt_info in prompts_data:
            prompt = prompt_info.get("prompt", "unknown")
            crops = prompt_info.get("crops", []) or []
            if not crops:
                continue
            lines.append(f"\n=== Objects detected as '{prompt}': ===\n")
            for crop in crops:
                idx = crop.get('index', 'N/A')
                bbox_px = crop.get('bbox_pixels', [])
                props = crop.get('properties', {}) or {}
                lines.append(f"\nCrop {idx} (Image {idx + 1}):\n")
                if bbox_px and len(bbox_px) == 4:
                    lines.append(
                        f"  Location in base image: bbox=[{bbox_px[0]},{bbox_px[1]},{bbox_px[2]},{bbox_px[3]}]px\n"
                    )
                if props:
                    lines.append(f"  Area: {props.get('area', 'N/A'):.1f}px²\n")
                    lines.append(
                        f"  Centroid: ({props.get('centroid_x', 'N/A'):.1f},{props.get('centroid_y', 'N/A'):.1f})\n"
                    )
                    lines.append(f"  Orientation: {props.get('orientation', 'N/A'):.1f}°\n")
                    lines.append(f"  Aspect ratio: {props.get('aspect_ratio', 'N/A'):.2f}\n")
                    lines.append(
                        f"  Size: {props.get('bbox_width', 'N/A'):.0f}x{props.get('bbox_height', 'N/A'):.0f}px\n"
                    )
            total_crops += len(crops)

        context = "".join(lines)
        return context, {"total_crops": total_crops, "prompt_groups": len(prompts_data)}

    def _get_crops_context(self) -> Tuple[str, Dict[str, int]]:
        manifest_path = os.path.join("vqa_outputs_fresh", "crops_manifest.json")
        cache = LocalVLM._crops_context_cache
        try:
            if not os.path.exists(manifest_path):
                return "", {"total_crops": 0, "prompt_groups": 0}
            mtime = os.path.getmtime(manifest_path)
            if cache["path"] == manifest_path and cache["mtime"] == mtime:
                return cache["text"], {
                    "total_crops": cache["total_crops"],
                    "prompt_groups": cache["prompt_groups"],
                }
            with open(manifest_path, 'r') as f:
                manifest = json.load(f)
            context, stats = self._format_crops_context(manifest)
            LocalVLM._crops_context_cache = {
                "path": manifest_path,
                "mtime": mtime,
                "text": context,
                "total_crops": stats["total_crops"],
                "prompt_groups": stats["prompt_groups"],
            }
            return context, stats
        except Exception as exc:
            if self._perf_enabled:
                print(f"[VLM PERF] Crops manifest cache failed: {exc}")
            return "", {"total_crops": 0, "prompt_groups": 0}

    def _apply_chat(self, messages, tools=None):
        formatted_tools = None
        if tools and LC_AVAILABLE:
            try:
                formatted_tools = [convert_to_openai_tool(t) for t in tools]
                # Debug print removed to reduce clutter
            except Exception as e:
                print(f"[LocalVLM] Warning: failed to convert tools: {e}")

        with self._perf_section("apply_chat_template"):
            batch = self.processor.apply_chat_template(
                messages,
                tools=formatted_tools,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
            )
        return self._prepare_inputs(batch)

    def _trim_generated_ids(self, outputs, inputs):
        cut = inputs.input_ids.size(1)
        return [outputs[i, cut:] for i in range(outputs.size(0))]

    def _decode_new_tokens(self, trimmed_sequences):
        decoded = self.processor.batch_decode(trimmed_sequences, skip_special_tokens=True)
        return decoded[0] if decoded else ""

    # ------------------ original methods ------------------

    def caption_image(self, image_paths: Union[str, List[str]], user_prompt: Optional[str] = None) -> str:
        if isinstance(image_paths, str):
            image_paths = [image_paths]

        instruction = CAPTION_SYSTEM_PROMPT
        if user_prompt:
            instruction = f"{instruction}\n\nUser request: {user_prompt.strip()}"

        messages = [
            {
                "role": "user",
                "content": [
                    *[self._img_item(p) for p in image_paths],
                    {"type": "text", "text": instruction},
                ],
            }
        ]

        print(f"[Reasoning] 🧩 Describing {len(image_paths)} image(s)…")
        inputs = self._apply_chat(messages)
        streamer = TextStreamer(self.processor.tokenizer, skip_prompt=True, skip_special_tokens=True)
        # Use inference_mode and KV cache for faster generation
        with torch.inference_mode():
            with self._autocast_context():
                outputs = self.model.generate(
                    **inputs,
                    streamer=streamer,
                    max_new_tokens=128,
                    use_cache=True,
                    do_sample=False,
                    repetition_penalty=1.1,
                    eos_token_id=self.processor.tokenizer.eos_token_id,
                )
                trimmed = self._trim_generated_ids(outputs, inputs)
                return self._decode_new_tokens(trimmed)

    def answer_question(self, image_paths: Union[str, List[str]], question: str, max_length: int = 3200, tools: Optional[List[Any]] = None) -> str:
        if isinstance(image_paths, str):
            image_paths = [image_paths]

        crops_context, _ = self._get_crops_context()

        # Prepend crops context to question
        full_question_with_context = crops_context + "\n" + question

        if tools:
            messages = []
            if self.system_prompt:
                messages.append({"role": "system", "content": [{"type": "text", "text": self.system_prompt}]})
            messages.append({
                "role": "user",
                "content": [
                    *[self._img_item(p) for p in image_paths],
                    {"type": "text", "text": full_question_with_context},
                ],
            })
        else:
            full_question = f"{self.system_prompt}\n\n{full_question_with_context}"
            messages = [
                {
                    "role": "user",
                    "content": [
                        *[self._img_item(p) for p in image_paths],
                        {"type": "text", "text": full_question},
                    ],
                }
            ]

        # Only print "Question received" if explicitly NOT suppressed
        if not self._suppress_debug_prints:
            print(f"\n[VLM Reasoning] 🧠 Question received")
            print(f"[VLM Reasoning] 🖼️ Processing {len(image_paths)} image(s).")
            print(f"[VLM Reasoning] 💭 Beginning reasoning stream...\n")

        inputs = self._apply_chat(messages, tools=tools)

        # Logic: Stream if enabled globally AND not suppressed for this run
        should_stream = self.stream_thoughts and not self._suppress_debug_prints
        
        # Print actual generation parameters being used (only first time)
        if not hasattr(LocalVLM, '_gen_params_printed'):
            print("\n[VLM Debug] Actual generation parameters used:")
            print(f"    max_new_tokens: {max_length}")
            print(f"    do_sample: False (overriding model default)")
            print(f"    use_cache: True")
            print(f"    eos_token_id: {self.processor.tokenizer.eos_token_id}")
            LocalVLM._gen_params_printed = True

        gen_kwargs = dict(
            max_new_tokens=max_length,
            do_sample=False,
            use_cache=True,
            repetition_penalty=1.1,
            eos_token_id=self.processor.tokenizer.eos_token_id,
        )
        if should_stream:
            gen_kwargs["streamer"] = TextStreamer(
                self.processor.tokenizer, skip_prompt=True, skip_special_tokens=True
            )

        with torch.inference_mode():
            start = time.perf_counter()
            with self._autocast_context():
                outputs = self.model.generate(**inputs, **gen_kwargs)
            latency = time.perf_counter() - start

        trimmed = self._trim_generated_ids(outputs, inputs)
        answer = self._decode_new_tokens(trimmed)

        token_count = sum(seq.size(0) for seq in trimmed if isinstance(seq, torch.Tensor))
        throughput = token_count / latency if latency > 0 else float("inf")

        if self._perf_enabled:
            print(
                f"[VLM PERF] generation latency: {latency * 1000:.1f} ms | new tokens: {token_count} | throughput: {throughput:.1f} tok/s"
            )

        self._last_perf = {
            "latency_s": latency,
            "new_tokens": float(token_count),
            "throughput_tps": float(throughput),
        }
        
        # Print output based on streaming mode
        if should_stream:
            # Streaming already printed tokens, just print completion
            print(f"\n[VLM Result] ✅ Reasoning complete.")
        else:
            # No streaming - print the complete answer now
            if not self._suppress_debug_prints:
                print(answer)
                print(f"\n[VLM Result] ✅ Reasoning complete.")
            
        return answer

    # ------------------ Agent + tools integration ------------------

    def _register_geometric_tools(self):
        if not LC_AVAILABLE:
            raise RuntimeError("LangChain core APIs not available.")

        if self._tools_registered:
            return

        tools = []
        
        # --- Helper: Load crops manifest and get mask path by crop index ---
        def get_mask_path_by_index(crop_index: int) -> str:
            """Read crops manifest and return the mask path for a given crop index."""
            try:
                manifest_path = os.path.join("vqa_outputs_fresh", "crops_manifest.json")
                if not os.path.exists(manifest_path):
                    return None
                with open(manifest_path, 'r') as f:
                    manifest = json.load(f)
                
                # Search through prompts structure
                prompts_data = manifest.get("prompts", [])
                for prompt_info in prompts_data:
                    crops = prompt_info.get("crops", [])
                    for crop in crops:
                        if crop.get("index") == crop_index:
                            return crop.get("mask_path")
                
                return None
            except Exception as e:
                print(f"[TOOL ERROR] Failed to read crops manifest: {e}")
                return None
        
        def wrap_read_mask_by_index(crop_index: int):
            """Read mask file for a crop index."""
            mask_path = get_mask_path_by_index(crop_index)
            if mask_path is None:
                return {"error": f"mask not found for crop {crop_index}"}
            m = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            return m > 0 if m is not None else {"error": f"mask file not readable: {mask_path}"}

        # --- Tool Definitions (VLM passes crop indices, tools handle file I/O) ---
        def t_compute_min_distance_between_crops(crop1_index: int, crop2_index: int) -> Dict[str, Any]:
            """Calculate distance between two crops using their indices."""
            print(f"\n[TOOL CALL] distance(crop {crop1_index}, crop {crop2_index})")
            m1, m2 = wrap_read_mask_by_index(crop1_index), wrap_read_mask_by_index(crop2_index)
            if isinstance(m1, dict): return m1
            if isinstance(m2, dict): return m2
            res = {"distance": float(geometric_utils.compute_min_distance_between_masks(m1, m2))}
            print(f"[TOOL RESULT] Distance between crop {crop1_index} and crop {crop2_index}: {res['distance']:.2f} pixels")
            return res

        def t_get_relative_position(crop1_index: int, crop2_index: int) -> Dict[str, Any]:
            """Get relative position of crop2 with respect to crop1."""
            print(f"\n[TOOL CALL] relative_position(crop {crop1_index}, crop {crop2_index})")
            m1, m2 = wrap_read_mask_by_index(crop1_index), wrap_read_mask_by_index(crop2_index)
            if isinstance(m1, dict): return m1
            if isinstance(m2, dict): return m2
            res = {"position": geometric_utils.get_relative_position(m1, m2)}
            print(f"[TOOL RESULT] Crop {crop2_index} is {res['position']} of crop {crop1_index}")
            return res

        def t_calculate(expression: str) -> Dict[str, Any]:
            """Safely evaluate a mathematical expression OR rank/sort values.
            
            For arithmetic: Supports +, -, *, /, //, %, ** (power), and parentheses.
            Example: '(1+2+3+4)/3' returns 3.333...
            
            For ranking: Use syntax RANK[value1, value2, ...] or RANK_DESC[value1, value2, ...]
            - RANK[...] sorts ascending (smallest to largest)
            - RANK_DESC[...] sorts descending (largest to smallest)
            Example: 'RANK[45.2, 12.3, 78.9]' returns ranked indices and values
            Example: 'RANK_DESC[45.2, 12.3, 78.9]' returns ranked indices and values (descending)
            """
            print(f"\n[TOOL CALL] calculate('{expression}')")
            try:
                import re
                expr = expression.strip()
                
                # Check for ranking syntax
                rank_match = re.match(r'^RANK(_DESC)?\s*\[\s*([0-9.,\s]+)\s*\]$', expr, re.IGNORECASE)
                if rank_match:
                    descending = rank_match.group(1) is not None
                    values_str = rank_match.group(2)
                    
                    # Parse values
                    try:
                        values = [float(v.strip()) for v in values_str.split(',') if v.strip()]
                    except ValueError as e:
                        return {"error": f"Invalid number format in ranking: {str(e)}"}
                    
                    if len(values) == 0:
                        return {"error": "No values provided for ranking"}
                    
                    # Create list of (original_index, value) tuples
                    indexed_values = list(enumerate(values))
                    
                    # Sort by value
                    sorted_values = sorted(indexed_values, key=lambda x: x[1], reverse=descending)
                    
                    # Extract rankings
                    ranked_indices = [idx for idx, val in sorted_values]
                    ranked_values = [val for idx, val in sorted_values]
                    
                    order_type = "descending (largest to smallest)" if descending else "ascending (smallest to largest)"
                    print(f"[TOOL RESULT] Ranked {order_type}: {ranked_values}")
                    print(f"[TOOL RESULT] Original indices: {ranked_indices}")
                    
                    return {
                        "operation": "rank",
                        "order": "descending" if descending else "ascending",
                        "ranked_values": ranked_values,
                        "ranked_indices": ranked_indices,
                        "input_values": values,
                        "description": f"Values ranked {order_type}"
                    }
                
                # Standard arithmetic evaluation
                # Allowed characters: digits, operators, parentheses, decimal point
                if not re.match(r'^[0-9+\-*/%().\s**]+$', expr):
                    return {"error": "Expression contains invalid characters. Only numbers and operators (+, -, *, /, %, **, parentheses) allowed, or use RANK[...]/RANK_DESC[...] for ranking."}
                
                # Use eval with restricted namespace (no built-in functions accessible)
                result = eval(expr, {"__builtins__": {}}, {})
                
                # Ensure result is a number
                if not isinstance(result, (int, float)):
                    return {"error": f"Expression did not evaluate to a number: {type(result).__name__}"}
                
                print(f"[TOOL RESULT] {expression} = {result}")
                return {"result": float(result), "expression": expression}
                
            except ZeroDivisionError:
                error_msg = "Division by zero error"
                print(f"[TOOL ERROR] {error_msg}")
                return {"error": error_msg}
            except SyntaxError as e:
                error_msg = f"Invalid syntax in expression: {str(e)}"
                print(f"[TOOL ERROR] {error_msg}")
                return {"error": error_msg}
            except Exception as e:
                error_msg = f"Calculation error: {str(e)}"
                print(f"[TOOL ERROR] {error_msg}")
                return {"error": error_msg}

        # Register tools (VLM only needs to know crop indices)
        tools.append(StructuredTool.from_function(
            t_compute_min_distance_between_crops, 
            name="compute_min_distance_between_crops", 
            description="Calculate the pixel distance between two crops. Args: crop1_index (int), crop2_index (int). Returns: distance in pixels."
        ))
        tools.append(StructuredTool.from_function(
            t_get_relative_position, 
            name="get_relative_position", 
            description="Get the relative position of crop2 with respect to crop1. Args: crop1_index (int), crop2_index (int). Returns: position (e.g., 'left', 'right', 'above', 'below')."
        ))
        tools.append(StructuredTool.from_function(
            t_calculate,
            name="calculate",
            description="Safely evaluate mathematical expressions OR rank/sort values. For arithmetic: supports +, -, *, /, %, ** (power), parentheses. Example: '(1+2+3)/3'. For ranking: use RANK[val1,val2,...] for ascending or RANK_DESC[val1,val2,...] for descending. Returns result (float) for arithmetic, or ranked_indices and ranked_values for ranking."
        ))

        self._tools_registered = True
        self._lc_tools = tools

    
    class _AdapterChatModel(BaseChatModel):
        _parent: Any = PrivateAttr()
        _image_paths: List[str] = PrivateAttr(default_factory=list)
        _tools: List[Any] = PrivateAttr(default_factory=list)

        def __init__(self, parent: Any, image_paths: Optional[List[str]] = None, **kwargs):
            super().__init__(**kwargs)
            self._parent = parent
            self._image_paths = image_paths or []

        @property
        def _llm_type(self) -> str: return "local_vlm"
        
        @property
        def _identifying_params(self) -> Dict[str, Any]: return {"adapter": "LocalVLMAdapter"}

        def bind_tools(self, tools: List[Any], **kwargs: Any):
            self._tools = tools
            return self

        def _generate(self, messages: List[BaseMessage], **kwargs) -> ChatResult:
            print(f"\n[ADAPTER DEBUG] _generate invoked with {len(messages)} messages")

            # 1. Prepare Text (skip system messages, flag tool observations)
            text_pieces = []
            for idx, m in enumerate(messages):
                msg_type = m.__class__.__name__
                content = getattr(m, "content", "")
                    
                if isinstance(m, SystemMessage):
                    print(f"  [ADAPTER DEBUG] Message {idx}: SystemMessage skipped")
                    continue

                if isinstance(m, ToolMessage) or getattr(m, "type", "") == "tool":
                    print(f"  [ADAPTER DEBUG] Message {idx}: Tool observation ingested")
                    content = f"\nOBSERVATION: {content}\n"
                else:
                    print(f"  [ADAPTER DEBUG] Message {idx}: {msg_type}")

                if isinstance(content, list):
                    for c in content:
                        if isinstance(c, dict) and c.get("type") == "text": text_pieces.append(c.get("text", ""))
                        elif isinstance(c, str): text_pieces.append(c)
                elif isinstance(content, str):
                    text_pieces.append(content)

            prompt_text = "\n".join([p for p in text_pieces if p]).strip()
            print(f"[ADAPTER DEBUG] Prompt length heading into VLM: {len(prompt_text)} chars")

            # 2. Call VLM (Streaming happens here)
            try:
                out_text = self._parent.answer_question(
                    self._image_paths,
                    prompt_text,
                    max_length=3200,
                    tools=self._tools,
                )
            except Exception as e:
                out_text = f"[VLM Error: {e}]"

            # 3. Parse Tool Calls from fresh output only
            print(f"[ADAPTER DEBUG] Raw model output length: {len(out_text)} chars")
            tool_calls = []

            if "[FINAL ANSWER]" in out_text:
                print(f"[ADAPTER DEBUG] Detected [FINAL ANSWER]; halting agent loop")
                msg = AIMessage(content=out_text, tool_calls=[])
                return ChatResult(generations=[ChatGeneration(message=msg)])

            latest_response = out_text
            if "[THINKING]" in out_text:
                parts = out_text.split("[THINKING]")
                latest_response = parts[-1]
                print(f"[ADAPTER DEBUG] Latest assistant chunk length: {len(latest_response)} chars")

            if "<tool_call>" in latest_response:
                print(f"[ADAPTER DEBUG] Parsing <tool_call> tags")
                matches = re.findall(r'<tool_call>(.*?)</tool_call>', latest_response, re.DOTALL)
                for idx, match in enumerate(matches):
                    try:
                        data = json_module.loads(match)
                        args = data.get("arguments")
                        if isinstance(args, str):
                            args = json_module.loads(args)
                        tool_calls.append({
                            "name": data.get("name"),
                            "args": args,
                            "id": f"call_{idx}",
                            "type": "tool_call"
                        })
                    except Exception as err:
                        print(f"[ADAPTER DEBUG] Failed to parse <tool_call>: {err}")

            if not tool_calls:
                print(f"[ADAPTER DEBUG] Checking ACTION/ACTION INPUT fallback")
                matches = re.findall(r'ACTION:\s*(\w+)\s*\nACTION INPUT:\s*(\{[^}]+\})', latest_response, re.DOTALL)
                for idx, (name, args_str) in enumerate(matches):
                    try:
                        tool_calls.append({
                            "name": name,
                            "args": json_module.loads(args_str),
                            "id": f"call_{idx}",
                            "type": "tool_call"
                        })
                    except Exception as err:
                        print(f"[ADAPTER DEBUG] Failed to parse ACTION pair: {err}")

            if tool_calls:
                for tc in tool_calls:
                    print(f"[ADAPTER DEBUG] Scheduling tool call -> {tc['name']} with args {tc['args']}")
            else:
                print(f"[ADAPTER DEBUG] No tool calls detected in latest response")

            msg = AIMessage(content=out_text, tool_calls=tool_calls)
            return ChatResult(generations=[ChatGeneration(message=msg)])

    def run_agent(self, image_paths: Union[str, List[str]], question: str, max_iterations: int = 5, show_thinking_once: bool = True):
        if not LC_AVAILABLE:
            raise RuntimeError("LangChain not available.")
        if isinstance(image_paths, str):
            image_paths = [image_paths]

        crops_context, stats = self._get_crops_context()
        if stats.get("total_crops"):
            print(
                f"[AGENT DEBUG] Loaded crops manifest with {stats['total_crops']} crops across {stats['prompt_groups']} prompts"
            )

        # Prepend crops context to question (VLM only sees crop info, tools handle mask I/O)
        full_question = crops_context + "\n" + question

        prev_suppress = self._suppress_debug_prints
        self._suppress_debug_prints = False
        try:
            print("\n[AGENT DEBUG] Registering geometric tools…")
            self._register_geometric_tools()
            print(f"[AGENT DEBUG] {len(self._lc_tools)} tools ready")

            adapter_model = LocalVLM._AdapterChatModel(self, image_paths=image_paths)
            print("[AGENT DEBUG] Creating ReAct agent")
            agent = create_react_agent(model=adapter_model, tools=self._lc_tools)

            messages: List[BaseMessage] = []
            if self.system_prompt:
                messages.append(SystemMessage(content=self.system_prompt))
                print(f"[AGENT DEBUG] Injected system prompt ({len(self.system_prompt)} chars)")

            messages.append(HumanMessage(content=full_question))
            print(f"[AGENT DEBUG] Injected user question with manifests ({len(full_question)} chars)")

            initial_state = {"messages": messages}
            print(f"[AGENT DEBUG] Invoking agent with recursion limit {max_iterations + 5}")
            final_state = agent.invoke(initial_state, config={"recursion_limit": max_iterations + 5})

            messages = final_state.get("messages", [])
            for msg in reversed(messages):
                if isinstance(msg, AIMessage):
                    content = msg.content
                    if "[FINAL ANSWER]" in content:
                        parts = content.split("[FINAL ANSWER]")
                        final_answer = parts[-1].strip()
                        print(f"\n[FINAL ANSWER]\n{final_answer}\n")
                        return f"[FINAL ANSWER]\n{final_answer}"
                    print(content)
                    return content

            print("[AGENT DEBUG] No AIMessage found in final state; returning raw state")
            return str(final_state)
        except Exception as e:
            traceback.print_exc()
            return f"Error: {e}"
        finally:
            self._suppress_debug_prints = prev_suppress