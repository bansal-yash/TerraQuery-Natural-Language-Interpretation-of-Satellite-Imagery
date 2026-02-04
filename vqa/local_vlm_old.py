# local_vlm.py  (Fixed duplication)
from typing import Union, List, Optional, Dict, Any
from PIL import Image
import os
import torch
import json
import inspect
import traceback
import re # Added explicit import

from transformers import AutoProcessor, Qwen3VLForConditionalGeneration, TextStreamer
from pydantic import PrivateAttr

# LangChain / LangGraph imports
try:
    from langchain_core.tools import tool, Tool, StructuredTool
    from langchain_core.language_models.chat_models import BaseChatModel
    from langchain_core.messages import BaseMessage, AIMessage, HumanMessage, SystemMessage
    from langchain_core.outputs import ChatResult, ChatGeneration
    from langchain_core.utils.function_calling import convert_to_openai_tool
    from langgraph.prebuilt import create_react_agent
    LC_AVAILABLE = True
except Exception as e:
    print(f"[WARNING] LangChain/LangGraph not available: {e}")
    LC_AVAILABLE = False

import cv2
import geometric_utils
import create_agent_entry

class LocalVLM:
    # Shared, lazily-initialized resources to avoid re-loading the 8B model per instance
    _shared_processor: Optional[AutoProcessor] = None
    _shared_model: Optional[Qwen3VLForConditionalGeneration] = None

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-VL-8B-Instruct",
        device: Optional[str] = None,
        system_prompt: Optional[str] = None,
        stream_thoughts: bool = True,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.system_prompt = system_prompt or (
            "You are a visual reasoning assistant. Think step-by-step and explain logic clearly."
        )
        self.stream_thoughts = stream_thoughts

        # Improve matmul performance on CPU and some GPUs
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass

        # Lazily load and share heavy resources across instances
        if LocalVLM._shared_processor is None or LocalVLM._shared_model is None:
            print(f"[Init] Loading {model_name} on {self.device} …")
            LocalVLM._shared_processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
            LocalVLM._shared_model = Qwen3VLForConditionalGeneration.from_pretrained(
                model_name,
                dtype=torch.float16 if self.device.startswith("cuda") else torch.float32,
                device_map="auto" if self.device.startswith("cuda") else None,
                trust_remote_code=True,
            ).eval()
            print("[Init] Model ready.")
            if self.stream_thoughts:
                print("[Init] Streaming mode enabled — reasoning tokens will print as they generate.\n")
        else:
            # Fast path: reuse already-loaded model and processor
            print(f"[Init] Reusing shared {model_name} on {self.device} — no reload.")

        self.processor = LocalVLM._shared_processor
        self.model = LocalVLM._shared_model

        self._agent = None
        self._tools_registered = False
        self._suppress_debug_prints = False
        # Small cache for opened images by path to avoid repeated disk I/O
        self._image_cache: Dict[str, Image.Image] = {}

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

    def _apply_chat(self, messages, tools=None):
        formatted_tools = None
        if tools and LC_AVAILABLE:
            try:
                formatted_tools = [convert_to_openai_tool(t) for t in tools]
                # Debug print removed to reduce clutter
            except Exception as e:
                print(f"[LocalVLM] Warning: failed to convert tools: {e}")

        return self.processor.apply_chat_template(
            messages,
            tools=formatted_tools,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.device)

    def _decode_new_tokens(self, outputs, inputs):
        trimmed = [outputs[i, inputs.input_ids.size(1):] for i in range(outputs.size(0))]
        return self.processor.batch_decode(trimmed, skip_special_tokens=True)[0]

    # ------------------ original methods ------------------

    def caption_image(self, image_paths: Union[str, List[str]]) -> str:
        if isinstance(image_paths, str):
            image_paths = [image_paths]

        messages = [
            {
                "role": "user",
                "content": [
                    *[self._img_item(p) for p in image_paths],
                    {"type": "text", "text": "Describe these images briefly and coherently."},
                ],
            }
        ]

        print(f"[Reasoning] 🧩 Describing {len(image_paths)} image(s)…")
        inputs = self._apply_chat(messages)
        streamer = TextStreamer(self.processor.tokenizer, skip_prompt=True, skip_special_tokens=True)
        # Use inference_mode and KV cache for faster generation
        with torch.inference_mode():
            outputs = self.model.generate(
                **inputs,
                streamer=streamer,
                max_new_tokens=128,
                use_cache=True,
                do_sample=False,
                eos_token_id=self.processor.tokenizer.eos_token_id,
            )
        return self._decode_new_tokens(outputs, inputs)

    def answer_question(self, image_paths: Union[str, List[str]], question: str, max_length: int = 1600, tools: Optional[List[Any]] = None) -> str:
        if isinstance(image_paths, str):
            image_paths = [image_paths]

        if tools:
            messages = []
            if self.system_prompt:
                messages.append({"role": "system", "content": [{"type": "text", "text": self.system_prompt}]})
            messages.append({
                "role": "user",
                "content": [
                    *[self._img_item(p) for p in image_paths],
                    {"type": "text", "text": question},
                ],
            })
        else:
            full_question = f"{self.system_prompt}\n\n{question}"
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
        
        if should_stream:
            streamer = TextStreamer(self.processor.tokenizer, skip_prompt=True, skip_special_tokens=True)
            # For streaming, avoid aggressive early stops; rely on model's EOS.
            with torch.inference_mode():
                outputs = self.model.generate(
                    **inputs,
                    streamer=streamer,
                    max_new_tokens=max_length,
                    do_sample=False,
                    use_cache=True,
                    eos_token_id=self.processor.tokenizer.eos_token_id,
                )
        else:
            # Non-streaming path (agent loop); conservative token budget.
            with torch.inference_mode():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_length,
                    do_sample=False,
                    use_cache=True,
                    eos_token_id=self.processor.tokenizer.eos_token_id,
                )

        answer = self._decode_new_tokens(outputs, inputs)
        
        # Only print completion if streaming was enabled
        if should_stream:
            print(f"\n[VLM Result] ✅ Reasoning complete.")
            
        return answer

    # ------------------ Agent + tools integration ------------------

    def _register_geometric_tools(self):
        if not LC_AVAILABLE:
            raise RuntimeError("LangChain core APIs not available.")

        if self._tools_registered:
            return

        tools = []
        
        # --- Tool Definitions ---
        def wrap_read_mask(path: str):
            m = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            return m > 0 if m is not None else {"error": f"mask not found: {path}"}

        def t_compute_mask_properties(mask_path: str) -> Dict[str, Any]:
            print(f"\n[TOOL CALL] compute_mask_properties('{mask_path}')")
            mask = wrap_read_mask(mask_path)
            if isinstance(mask, dict): return mask
            res = geometric_utils.compute_mask_properties(mask)
            print(f"[TOOL RESULT] Success: {res}")
            return res

        def t_compute_min_distance_between_masks(mask1_path: str, mask2_path: str) -> Dict[str, Any]:
            print(f"\n[TOOL CALL] distance('{mask1_path}', '{mask2_path}')")
            m1, m2 = wrap_read_mask(mask1_path), wrap_read_mask(mask2_path)
            if isinstance(m1, dict) or isinstance(m2, dict): return {"error": "mask not found"}
            res = {"distance": float(geometric_utils.compute_min_distance_between_masks(m1, m2))}
            print(f"[TOOL RESULT] Success: {res}")
            return res

        def t_compute_mask_overlap(mask1_path: str, mask2_path: str) -> Dict[str, Any]:
            print(f"\n[TOOL CALL] overlap('{mask1_path}', '{mask2_path}')")
            m1, m2 = wrap_read_mask(mask1_path), wrap_read_mask(mask2_path)
            if isinstance(m1, dict) or isinstance(m2, dict): return {"error": "mask not found"}
            res = geometric_utils.compute_mask_overlap(m1, m2)
            print(f"[TOOL RESULT] Success: {res}")
            return res

        def t_get_relative_position(mask1_path: str, mask2_path: str) -> Dict[str, Any]:
            print(f"\n[TOOL CALL] relative_pos('{mask1_path}', '{mask2_path}')")
            m1, m2 = wrap_read_mask(mask1_path), wrap_read_mask(mask2_path)
            if isinstance(m1, dict) or isinstance(m2, dict): return {"error": "mask not found"}
            res = {"position": geometric_utils.get_relative_position(m1, m2)}
            print(f"[TOOL RESULT] Success: {res}")
            return res

        def t_compute_total_area(mask_paths: List[str]) -> Dict[str, Any]:
            print(f"\n[TOOL CALL] total_area({mask_paths})")
            masks = [wrap_read_mask(p) for p in mask_paths]
            if any(isinstance(m, dict) for m in masks): return {"error": "mask not found"}
            res = {"total_area": float(geometric_utils.compute_total_area(masks))}
            print(f"[TOOL RESULT] Success: {res}")
            return res

        def t_get_image_dimensions(image_path: str) -> Dict[str, Any]:
            print(f"\n[TOOL CALL] get_dims('{image_path}')")
            if not os.path.exists(image_path): return {"error": "file not found"}
            img = cv2.imread(image_path)
            if img is None: return {"error": "failed to read"}
            h, w = img.shape[:2]
            res = {"width": w, "height": h, "area": w*h}
            print(f"[TOOL RESULT] Success: {res}")
            return res

        def t_get_mask_for_crop(crop_index: int) -> Dict[str, Any]:
            print(f"\n[TOOL CALL] get_mask_for_crop({crop_index})")
            # Prefer manifest if present for robust mapping
            manifest_path = os.path.join("vqa_outputs", "masks_manifest.json")
            resolved = None
            try:
                if os.path.exists(manifest_path):
                    with open(manifest_path, 'r') as f:
                        manifest = json.load(f)
                    items = manifest.get("items", [])
                    if 0 <= crop_index < len(items):
                        resolved = items[crop_index].get("mask_path")
            except Exception as e:
                print(f"[TOOL WARN] Failed to read manifest: {e}")
                resolved = None

            path = resolved or f"vqa_outputs/mask_{crop_index}.png"
            if not os.path.exists(path):
                return {"error": "not found"}
            res = {"crop_index": crop_index, "mask_path": path, "exists": True}
            print(f"[TOOL RESULT] Success: {res}")
            return res

        def t_analyze_crop_mask(crop_index: int) -> Dict[str, Any]:
            print(f"\n[TOOL CALL] analyze_crop_mask({crop_index})")
            # Resolve via manifest if available
            manifest_path = os.path.join("vqa_outputs", "masks_manifest.json")
            resolved = None
            try:
                if os.path.exists(manifest_path):
                    with open(manifest_path, 'r') as f:
                        manifest = json.load(f)
                    items = manifest.get("items", [])
                    if 0 <= crop_index < len(items):
                        resolved = items[crop_index].get("mask_path")
            except Exception as e:
                print(f"[TOOL WARN] Failed to read manifest: {e}")
                resolved = None

            path = resolved or f"vqa_outputs/mask_{crop_index}.png"
            if not os.path.exists(path):
                return {"error": "not found"}
            mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                return {"error": "failed read"}
            res = {"crop_index": crop_index, "mask_path": path, **geometric_utils.compute_mask_properties(mask > 0)}
            print(f"[TOOL RESULT] Success: {res}")
            return res

        # Register
        tools.append(StructuredTool.from_function(t_compute_mask_properties, name="compute_mask_properties", description="Compute geometry for mask."))
        tools.append(StructuredTool.from_function(t_compute_min_distance_between_masks, name="compute_min_distance_between_masks", description="Distance between two masks."))
        tools.append(StructuredTool.from_function(t_compute_mask_overlap, name="compute_mask_overlap", description="Overlap/IoU between masks."))
        tools.append(StructuredTool.from_function(t_get_relative_position, name="get_relative_position", description="Relative position (left/right/etc)."))
        tools.append(StructuredTool.from_function(t_compute_total_area, name="compute_total_area", description="Total area of multiple masks."))
        tools.append(StructuredTool.from_function(t_get_image_dimensions, name="get_image_dimensions", description="Image dimensions."))
        tools.append(StructuredTool.from_function(t_get_mask_for_crop, name="get_mask_for_crop", description="Get mask path for crop index."))
        tools.append(StructuredTool.from_function(t_analyze_crop_mask, name="analyze_crop_mask", description="Analyze crop mask properties."))

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
            # 1. Prepare Text
            text_pieces = []
            for m in messages:
                content = getattr(m, "content", "")
                if isinstance(content, list):
                    for c in content:
                        if isinstance(c, dict) and c.get("type") == "text": text_pieces.append(c.get("text", ""))
                        elif isinstance(c, str): text_pieces.append(c)
                elif isinstance(content, str): text_pieces.append(content)
            prompt_text = "\n".join([p for p in text_pieces if p]).strip()

            # 2. Call VLM (Streaming happens here)
            try:
                # Increase max_length to avoid truncating long, tool-rich outputs
                out_text = self._parent.answer_question(
                    self._image_paths,
                    prompt_text,
                    max_length=1600,
                    tools=self._tools,
                )
            except Exception as e:
                out_text = f"[VLM Error: {e}]"

            # FIX: REMOVED THE DUPLICATE PRINT HERE
            # The streamer in answer_question has already shown the content.

            # 3. Parse Tool Calls
            # CRITICAL FIX: Only parse from the NEW output (out_text), not from conversation history.
            # The agent loop re-parses the full message chain, so we must avoid re-extracting old tool calls.
            tool_calls = []
            import json as json_module
            
            # TERMINATION CHECK: If [FINAL ANSWER] is present, return immediately with NO tool calls
            # This signals to LangGraph that the agent is done and should stop looping.
            if "[FINAL ANSWER]" in out_text:
                msg = AIMessage(content=out_text, tool_calls=[])
                return ChatResult(generations=[ChatGeneration(message=msg)])
            
            # Split by [THINKING] to isolate only the latest model response
            # (ignore anything before the last assistant turn in out_text)
            latest_response = out_text
            if "[THINKING]" in out_text:
                # Extract only the content after the last [THINKING] marker
                parts = out_text.split("[THINKING]")
                latest_response = parts[-1]
            
            # A. Native Qwen Tags
            if "<tool_call>" in latest_response:
                matches = re.findall(r'<tool_call>(.*?)</tool_call>', latest_response, re.DOTALL)
                for idx, match in enumerate(matches):
                    try:
                        data = json_module.loads(match)
                        tool_calls.append({
                            "name": data.get("name"),
                            "args": data.get("arguments") if isinstance(data.get("arguments"), dict) else json_module.loads(data.get("arguments")),
                            "id": f"call_{idx}",
                            "type": "tool_call"
                        })
                    except: pass

            # B. Legacy ACTION pattern (Fallback) - only in latest response
            if not tool_calls:
                matches = re.findall(r'ACTION:\s*(\w+)\s*\nACTION INPUT:\s*(\{[^}]+\})', latest_response, re.DOTALL)
                for idx, (name, args_str) in enumerate(matches):
                    try:
                        tool_calls.append({
                            "name": name, 
                            "args": json_module.loads(args_str), 
                            "id": f"call_{idx}", 
                            "type": "tool_call"
                        })
                    except: pass

            msg = AIMessage(content=out_text, tool_calls=tool_calls)
            return ChatResult(generations=[ChatGeneration(message=msg)])

    def run_agent(self, image_paths: Union[str, List[str]], question: str, max_iterations: int = 5, show_thinking_once: bool = True):
        if not LC_AVAILABLE: raise RuntimeError("LangChain not available.")
        if isinstance(image_paths, str): image_paths = [image_paths]
        
        # We removed the "show_thinking_once" pre-pass because it forced the model to generate 
        # without tools first, leading to hallucinated [FINAL ANSWER]s before the agent loop 
        # could actually run with tools. Now we just run the agent directly.
        # Streaming is handled by self.answer_question called within the adapter.

        try:
            self._register_geometric_tools()
            adapter_model = LocalVLM._AdapterChatModel(self, image_paths=image_paths)
            agent = create_react_agent(model=adapter_model, tools=self._lc_tools)
            
            initial_state = {"messages": [HumanMessage(content=question)]}
            final_state = agent.invoke(initial_state, config={"recursion_limit": max_iterations + 5})
            
            messages = final_state.get("messages", [])
            for msg in reversed(messages):
                if isinstance(msg, AIMessage):
                    content = msg.content
                    # Print exactly once here for the final consolidated content
                    # (no streaming occurred during iterations).
                    if "[FINAL ANSWER]" in content:
                        parts = content.split("[FINAL ANSWER]")
                        final_answer = parts[-1].strip()
                        print(f"\n[FINAL ANSWER]\n{final_answer}\n")
                        return f"[FINAL ANSWER]\n{final_answer}"
                    print(content)
                    return content
            return str(final_state)
        except Exception as e:
            traceback.print_exc()
            return f"Error: {e}"