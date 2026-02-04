# vlm_adapter.py

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # only used by static type checkers; not executed at runtime
    from local_vlm import LocalVLM

class LocalVLMAdapter:
    """
    Small adapter to let LangGraph call LocalVLM as a model.
    The agent will call adapter.generate(prompt, images=...).
    """

    def __init__(self):
        from typing import List, Dict
        from local_vlm import LocalVLM
        self.vlm = vlm

    def generate(self, prompt: str, image_paths: list[str], max_length: int = 1024, stream: bool = False) -> str:
        """
        Synchronous interface: returns generated text.
        LangGraph typically expects a callable that returns model strings.
        """
        # We send the system prompt + prompt to the VLM as a single question.
        # You can choose to inject system prompt elsewhere as required.
        full_question = prompt
        # LocalVLM.answer_question expects images, question -> returns text
        # Turn streaming off for structured tool selection phases; set stream False here.
        prev_stream = self.vlm.stream_thoughts
        try:
            self.vlm.stream_thoughts = stream
            answer = self.vlm.answer_question(image_paths, full_question, max_length=max_length)
        finally:
            self.vlm.stream_thoughts = prev_stream
        return answer

