"""API-based VLM client for visual question answering.

This module replaces the local transformer-based VLM with an HTTP client
that calls the Qwen API endpoints. This eliminates the need for torch/transformers
dependencies in the VQA orchestration layer.
"""

import os
import re
import json
import traceback
from typing import Union, List, Optional, Dict, Any
from pathlib import Path
import requests
import cv2

# LangChain / LangGraph imports
try:
    from langchain_core.tools import StructuredTool
    from langchain_core.language_models.chat_models import BaseChatModel
    from langchain_core.messages import BaseMessage, AIMessage, HumanMessage, SystemMessage
    from langchain_core.outputs import ChatResult, ChatGeneration
    from langgraph.prebuilt import create_react_agent
    from pydantic import PrivateAttr
    LC_AVAILABLE = True
except Exception as e:
    print(f"[WARNING] LangChain/LangGraph not available: {e}")
    LC_AVAILABLE = False

# Import geometric utilities for tools
try:
    import geometric_utils
    GEOM_AVAILABLE = True
except Exception as e:
    print(f"[WARNING] geometric_utils not available: {e}")
    GEOM_AVAILABLE = False


class LocalVLM:
    """API-based VLM client that communicates with Qwen inference server."""
    
    def __init__(
        self,
        api_base_url: str = None,
        device: Optional[str] = None,  # Kept for backward compatibility, not used
        system_prompt: Optional[str] = None,
        stream_thoughts: bool = True,
        timeout: int = 120,
    ):
        """Initialize VLM API client.
        
        Args:
            api_base_url: Base URL for Qwen API (default: env QWEN_API_URL or localhost:8001)
            device: Ignored, kept for backward compatibility
            system_prompt: Default system prompt for queries
            stream_thoughts: If True, prints reasoning progress (informational only)
            timeout: Request timeout in seconds
        """
        self.api_base_url = (
            api_base_url 
            or os.environ.get("QWEN_API_URL", "http://localhost:8001")
        ).rstrip("/")
        
        self.system_prompt = system_prompt or (
            "You are a visual reasoning assistant. Think step-by-step and explain logic clearly."
        )
        self.stream_thoughts = stream_thoughts
        self.timeout = timeout
        
        self._agent = None
        self._tools_registered = False
        self._suppress_debug_prints = False
        
        print(f"[LocalVLM] Initialized API client pointing to {self.api_base_url}")
        print(f"[LocalVLM] Streaming mode: {stream_thoughts}")

    # ------------------ API request helpers ------------------
    
    def _call_api(
        self, 
        endpoint: str, 
        image_paths: List[str], 
        user_prompt: str,
        system_prompt: Optional[str] = None,
        max_new_tokens: int = 512,
    ) -> str:
        """Make API call to general_inference endpoint.
        
        Args:
            endpoint: API endpoint (e.g., "/general_inference")
            image_paths: List of absolute paths to images
            user_prompt: The question/instruction text
            system_prompt: Optional system prompt override
            max_new_tokens: Max tokens to generate
            
        Returns:
            Response text from the model
        """
        url = f"{self.api_base_url}{endpoint}"
        
        # Prepare files for multipart upload
        files = []
        for idx, img_path in enumerate(image_paths):
            if not os.path.exists(img_path):
                raise FileNotFoundError(f"Image not found: {img_path}")
            with open(img_path, 'rb') as f:
                img_data = f.read()
            files.append(("images", (f"image_{idx}.jpg", img_data, "image/jpeg")))
        
        # Prepare form data
        data = {
            "user_prompt": user_prompt,
            "max_new_tokens": max_new_tokens,
        }
        if system_prompt:
            data["system_prompt"] = system_prompt
        
        try:
            response = requests.post(url, data=data, files=files, timeout=self.timeout)
            response.raise_for_status()
            result = response.json()
            return result.get("response", "")
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"API request failed: {e}")

    # ------------------ Public methods (compatible with old interface) ------------------

    def caption_image(self, image_paths: Union[str, List[str]]) -> str:
        """Generate captions for images using the API.
        
        Args:
            image_paths: Single path or list of paths to images
            
        Returns:
            Generated caption text
        """
        if isinstance(image_paths, str):
            image_paths = [image_paths]
        
        if not self._suppress_debug_prints:
            print(f"[Reasoning] 🧩 Describing {len(image_paths)} image(s)…")
        
        prompt = "Describe these images briefly and coherently."
        response = self._call_api(
            "/general_inference",
            image_paths,
            prompt,
            max_new_tokens=128,
        )
        
        return response

    def answer_question(
        self, 
        image_paths: Union[str, List[str]], 
        question: str, 
        max_length: int = 1600,
        tools: Optional[List[Any]] = None,
    ) -> str:
        """Answer a question about images using the API.
        
        Args:
            image_paths: Single path or list of paths to images
            question: The question to answer
            max_length: Maximum tokens to generate
            tools: Optional tools list (for LangChain compatibility, not used in API call)
            
        Returns:
            Generated answer text
        """
        if isinstance(image_paths, str):
            image_paths = [image_paths]
        
        # Only print if not suppressed
        if not self._suppress_debug_prints:
            print(f"\n[VLM Reasoning] 🧠 Question received")
            print(f"[VLM Reasoning] 🖼️ Processing {len(image_paths)} image(s).")
            if self.stream_thoughts:
                print(f"[VLM Reasoning] 💭 Sending request to API...\n")
        
        # Prepare prompt with system context
        full_prompt = question
        
        # Call API
        response = self._call_api(
            "/general_inference",
            image_paths,
            full_prompt,
            system_prompt=self.system_prompt,
            max_new_tokens=max_length,
        )
        
        if not self._suppress_debug_prints and self.stream_thoughts:
            print(f"\n[VLM Result] ✅ Response received.")
        
        return response

    # ------------------ Agent + tools integration ------------------

    def _register_geometric_tools(self):
        """Register geometric measurement tools for agent use."""
        if not LC_AVAILABLE:
            raise RuntimeError("LangChain core APIs not available.")
        
        if not GEOM_AVAILABLE:
            raise RuntimeError("geometric_utils not available.")
        
        if self._tools_registered:
            return

        tools = []
        
        # --- Helper to read masks ---
        def wrap_read_mask(path: str):
            m = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            return m > 0 if m is not None else {"error": f"mask not found: {path}"}

        # --- Tool Definitions ---
        def t_compute_mask_properties(mask_path: str) -> Dict[str, Any]:
            print(f"\n[TOOL CALL] compute_mask_properties('{mask_path}')")
            mask = wrap_read_mask(mask_path)
            if isinstance(mask, dict): 
                return mask
            res = geometric_utils.compute_mask_properties(mask)
            print(f"[TOOL RESULT] Success: {res}")
            return res

        def t_compute_min_distance_between_masks(mask1_path: str, mask2_path: str) -> Dict[str, Any]:
            print(f"\n[TOOL CALL] distance('{mask1_path}', '{mask2_path}')")
            m1, m2 = wrap_read_mask(mask1_path), wrap_read_mask(mask2_path)
            if isinstance(m1, dict) or isinstance(m2, dict): 
                return {"error": "mask not found"}
            res = {"distance": float(geometric_utils.compute_min_distance_between_masks(m1, m2))}
            print(f"[TOOL RESULT] Success: {res}")
            return res

        def t_compute_mask_overlap(mask1_path: str, mask2_path: str) -> Dict[str, Any]:
            print(f"\n[TOOL CALL] overlap('{mask1_path}', '{mask2_path}')")
            m1, m2 = wrap_read_mask(mask1_path), wrap_read_mask(mask2_path)
            if isinstance(m1, dict) or isinstance(m2, dict): 
                return {"error": "mask not found"}
            res = geometric_utils.compute_mask_overlap(m1, m2)
            print(f"[TOOL RESULT] Success: {res}")
            return res

        def t_get_relative_position(mask1_path: str, mask2_path: str) -> Dict[str, Any]:
            print(f"\n[TOOL CALL] relative_pos('{mask1_path}', '{mask2_path}')")
            m1, m2 = wrap_read_mask(mask1_path), wrap_read_mask(mask2_path)
            if isinstance(m1, dict) or isinstance(m2, dict): 
                return {"error": "mask not found"}
            res = {"position": geometric_utils.get_relative_position(m1, m2)}
            print(f"[TOOL RESULT] Success: {res}")
            return res

        def t_compute_total_area(mask_paths: List[str]) -> Dict[str, Any]:
            print(f"\n[TOOL CALL] total_area({mask_paths})")
            masks = [wrap_read_mask(p) for p in mask_paths]
            if any(isinstance(m, dict) for m in masks): 
                return {"error": "mask not found"}
            res = {"total_area": float(geometric_utils.compute_total_area(masks))}
            print(f"[TOOL RESULT] Success: {res}")
            return res

        def t_get_image_dimensions(image_path: str) -> Dict[str, Any]:
            print(f"\n[TOOL CALL] get_dims('{image_path}')")
            if not os.path.exists(image_path): 
                return {"error": "file not found"}
            img = cv2.imread(image_path)
            if img is None: 
                return {"error": "failed to read"}
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
            res = {
                "crop_index": crop_index, 
                "mask_path": path, 
                **geometric_utils.compute_mask_properties(mask > 0)
            }
            print(f"[TOOL RESULT] Success: {res}")
            return res

        # Register all tools
        tools.append(StructuredTool.from_function(
            t_compute_mask_properties, 
            name="compute_mask_properties", 
            description="Compute geometry for mask."
        ))
        tools.append(StructuredTool.from_function(
            t_compute_min_distance_between_masks, 
            name="compute_min_distance_between_masks", 
            description="Distance between two masks."
        ))
        tools.append(StructuredTool.from_function(
            t_compute_mask_overlap, 
            name="compute_mask_overlap", 
            description="Overlap/IoU between masks."
        ))
        tools.append(StructuredTool.from_function(
            t_get_relative_position, 
            name="get_relative_position", 
            description="Relative position (left/right/etc)."
        ))
        tools.append(StructuredTool.from_function(
            t_compute_total_area, 
            name="compute_total_area", 
            description="Total area of multiple masks."
        ))
        tools.append(StructuredTool.from_function(
            t_get_image_dimensions, 
            name="get_image_dimensions", 
            description="Image dimensions."
        ))
        tools.append(StructuredTool.from_function(
            t_get_mask_for_crop, 
            name="get_mask_for_crop", 
            description="Get mask path for crop index."
        ))
        tools.append(StructuredTool.from_function(
            t_analyze_crop_mask, 
            name="analyze_crop_mask", 
            description="Analyze crop mask properties."
        ))

        self._tools_registered = True
        self._lc_tools = tools

    class _AdapterChatModel(BaseChatModel):
        """LangChain adapter that wraps the API-based VLM."""
        _parent: Any = PrivateAttr()
        _image_paths: List[str] = PrivateAttr(default_factory=list)
        _tools: List[Any] = PrivateAttr(default_factory=list)

        def __init__(self, parent: Any, image_paths: Optional[List[str]] = None, **kwargs):
            super().__init__(**kwargs)
            self._parent = parent
            self._image_paths = image_paths or []

        @property
        def _llm_type(self) -> str: 
            return "local_vlm_api"
        
        @property
        def _identifying_params(self) -> Dict[str, Any]: 
            return {"adapter": "LocalVLMAPIAdapter"}

        def bind_tools(self, tools: List[Any], **kwargs: Any):
            self._tools = tools
            return self

        def _generate(self, messages: List[BaseMessage], **kwargs) -> ChatResult:
            # 1. Prepare Text from message history
            text_pieces = []
            for m in messages:
                content = getattr(m, "content", "")
                if isinstance(content, list):
                    for c in content:
                        if isinstance(c, dict) and c.get("type") == "text":
                            text_pieces.append(c.get("text", ""))
                elif isinstance(content, str): 
                    text_pieces.append(content)
            prompt_text = "\n".join([p for p in text_pieces if p]).strip()

            # 2. Call API-based VLM
            try:
                out_text = self._parent.answer_question(
                    self._image_paths,
                    prompt_text,
                    max_length=1600,
                    tools=self._tools,
                )
            except Exception as e:
                out_text = f"[VLM Error: {e}]"

            # 3. Parse Tool Calls from the NEW output only
            tool_calls = []
            
            # TERMINATION CHECK: If [FINAL ANSWER] is present, stop immediately
            if "[FINAL ANSWER]" in out_text:
                msg = AIMessage(content=out_text, tool_calls=[])
                return ChatResult(generations=[ChatGeneration(message=msg)])
            
            # Extract only the latest response after [THINKING]
            latest_response = out_text
            if "[THINKING]" in out_text:
                parts = out_text.split("[THINKING]")
                latest_response = parts[-1]
            
            # A. Native Qwen Tags
            if "<tool_call>" in latest_response:
                matches = re.findall(r'<tool_call>(.*?)</tool_call>', latest_response, re.DOTALL)
                for idx, match in enumerate(matches):
                    try:
                        parsed = json.loads(match.strip())
                        tool_calls.append({
                            "id": f"call_{idx}",
                            "name": parsed.get("name", ""),
                            "args": parsed.get("arguments", {}),
                            "type": "tool_call",
                        })
                    except: 
                        pass

            # B. Legacy ACTION pattern (Fallback)
            if not tool_calls:
                matches = re.findall(
                    r'ACTION:\s*(\w+)\s*\nACTION INPUT:\s*(\{[^}]+\})', 
                    latest_response, 
                    re.DOTALL
                )
                for idx, (name, args_str) in enumerate(matches):
                    try:
                        args = json.loads(args_str.strip())
                        tool_calls.append({
                            "id": f"call_{idx}",
                            "name": name,
                            "args": args,
                            "type": "tool_call",
                        })
                    except: 
                        pass

            msg = AIMessage(content=out_text, tool_calls=tool_calls)
            return ChatResult(generations=[ChatGeneration(message=msg)])

    def run_agent(
        self, 
        image_paths: Union[str, List[str]], 
        question: str, 
        max_iterations: int = 5,
        show_thinking_once: bool = True,
    ):
        """Run agent-based VQA with tool support.
        
        Args:
            image_paths: Single path or list of paths to images
            question: The question to answer
            max_iterations: Maximum agent iterations
            show_thinking_once: Unused (kept for compatibility)
            
        Returns:
            Final state from agent execution
        """
        if not LC_AVAILABLE: 
            raise RuntimeError("LangChain not available.")
        if isinstance(image_paths, str): 
            image_paths = [image_paths]
        
        try:
            self._register_geometric_tools()
            adapter_model = LocalVLM._AdapterChatModel(self, image_paths=image_paths)
            agent = create_react_agent(model=adapter_model, tools=self._lc_tools)
            
            initial_state = {"messages": [HumanMessage(content=question)]}
            final_state = agent.invoke(initial_state, config={"recursion_limit": max_iterations + 5})
            
            return str(final_state)
        except Exception as e:
            traceback.print_exc()
            return f"Error: {e}"
