"""Small Groq client used by the orchestrator.

This client is intentionally permissive in parsing responses because
Groq's response shape may vary across versions. It reads the API key
from the environment variable `GROQ`. You can override the endpoint by
setting `GROQ_API_URL` in the environment.

Note: This implementation uses the standard requests library and sends
a minimal completion-like payload. If your Groq account requires a
different endpoint or payload shape, set GROQ_API_URL accordingly.
"""
import os
import json
from typing import Optional
import requests

GROQ_API_KEY = os.environ.get("GROQ")
GROQ_API_URL = os.environ.get("GROQ_API_URL", "https://api.groq.ai/v1/completions")


def generate(prompt: str, model: str = "gpt-oss-20b", max_tokens: int = 256, temperature: float = 0.2) -> str:
    """Generate text using Groq API.

    This function tries to be robust to a few common response formats.
    It returns the generated string on success or raises a RuntimeError
    with helpful details on failure.
    """
    if not GROQ_API_KEY:
        raise RuntimeError("GROQ API key not found in environment variable GROQ")

    payload = {
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json",
    }

    resp = requests.post(GROQ_API_URL, headers=headers, data=json.dumps(payload), timeout=30)
    try:
        resp.raise_for_status()
    except Exception as exc:
        raise RuntimeError(f"Groq API request failed: {exc} - {resp.text}")

    data = resp.json()

    # Try a few common shapes
    # 1) { choices: [ { message: { content: "..." } } ] }
    try:
        return data["choices"][0]["message"]["content"].strip()
    except Exception:
        pass

    # 2) { choices: [ { text: "..." } ] }
    try:
        return data["choices"][0]["text"].strip()
    except Exception:
        pass

    # 3) direct field 'output' or 'generated'
    if isinstance(data.get("output"), list) and len(data["output"]) > 0:
        try:
            # sometimes nested
            return data["output"][0].get("content", "").strip() or str(data["output"])[:1000]
        except Exception:
            pass

    # fallback: pretty-print the JSON (not ideal, but informative)
    raise RuntimeError(f"Unable to parse Groq response: {json.dumps(data)[:2000]}")
