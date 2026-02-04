"""
Local LLM wrapper for extracting visual object classes from user queries.
Uses a small, instruction-tuned local model (Qwen2.5-1.5B-Instruct or MiniCPM-2B).
"""

from typing import List
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import re


class LocalLLM:
    def __init__(self, model_name: str = None, device: str = None):
        """Initialize a small local LLM for query parsing."""
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Pick a small reliable model if not specified
        self.model_name = model_name or self._select_model()
        print(f"[LocalLLM] Loading {self.model_name} on {self.device}...")

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map="auto" if self.device == "cuda" else None,
            trust_remote_code=True,
        ).eval()

        if self.device == "cpu":
            self.model = self.model.to(self.device)

        print("[LocalLLM] Model loaded successfully.")

    def _select_model(self) -> str:
        """Choose a small but capable model automatically."""
        # These are small, instruction-tuned and lightweight
        candidates = [
            "Qwen/Qwen2.5-1.5B-Instruct",
            "openbmb/MiniCPM-2B-sft-bf16",
            "microsoft/Phi-2",  # fallback
        ]
        for name in candidates:
            try:
                AutoTokenizer.from_pretrained(name, trust_remote_code=True)
                return name
            except Exception:
                continue
        raise RuntimeError("No suitable small model found. Please install one manually.")

    def extract_classes_from_query(self, query: str, max_classes: int = 10) -> List[str]:
        """Extract object classes to search for based on the user query."""
        prompt = f"""You are a precise vision-language assistant.
        Given a user's question, list only the object categories (with color or attribute modifiers if mentioned)
        that must be visually detected to answer the question.

        Example:
        Question: count all red and yellow buses
        Output: red buses, yellow buses

        Question: how many people are wearing blue shirts
        Output: people with blue shirts

        Question: are there any green trees and white cars
        Output: green trees, white cars

        Now, extract the visual object classes from this question:
        Question: {query}
        Output:"""

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=64,
                temperature=0.2,
                top_p=0.9,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
                use_cache=True,
            )

        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = response[len(prompt):].strip()
        print(f"[LocalLLM] Raw response: {response}")

        classes = self._parse_classes(response, max_classes)
        print(f"[LocalLLM] Extracted classes: {classes}")
        return classes

    def _parse_classes(self, response: str, max_classes: int) -> List[str]:
        """Parse model output into a clean list of class strings."""
        response = re.sub(r'^(Output:|Answer:|Classes:)', '', response, flags=re.IGNORECASE).strip()
        # Stop at first "Question:" in case of drift
        response = response.split("Question:")[0].strip()

        # Split by common delimiters
        if "," in response:
            items = [c.strip() for c in response.split(",")]
        elif ";" in response:
            items = [c.strip() for c in response.split(";")]
        elif "\n" in response:
            items = [c.strip() for c in response.split("\n")]
        else:
            items = [response.strip()]

        cleaned = []
        for c in items:
            c = re.sub(r'^\d+[\.\)]\s*', '', c)
            c = c.strip('"\' ')
            c = re.sub(r'^(Output|Answer|Object[s]?|Classes?)[:\-\s]*', '', c, flags=re.IGNORECASE)
            c = ' '.join(c.split())
            if c and len(c) > 1:
                cleaned.append(c)

        return cleaned[:max_classes]


if __name__ == "__main__":
    llm = LocalLLM()
    test_queries = [
        "count all red and yellow buses",
        "how many cars are there?",
        "is there a person in the image?",
        "what is the color of the largest building?",
        "find all people wearing black jackets",
    ]
    for query in test_queries:
        print(f"\n{'='*60}\nQuery: {query}\n{'='*60}")
        classes = llm.extract_classes_from_query(query)
        print("Classes:", classes)
