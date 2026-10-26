from __future__ import annotations

import base64
import os
import time
from typing import Any, Dict, List, Optional

import requests

DEFAULT_HOST = "http://localhost:11434"


class OllamaError(Exception):
    """Raised when Ollama can't be reached or returns an error."""


class OllamaClient:
    """Minimal client for a local Ollama server."""

    def __init__(self, host: Optional[str] = None, timeout: float = 600.0):
        host = (host or os.getenv("OLLAMA_HOST", DEFAULT_HOST)).rstrip("/")
        self.host = host if host.startswith("http") else "http://" + host
        self.timeout = timeout

    def chat(
        self,
        model: str,
        messages: List[Dict[str, Any]],
        format: Optional[Dict[str, Any]] = None,
        options: Optional[Dict[str, Any]] = None,
        think: bool = False,
    ) -> str:
        payload: Dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": False,
            "think": think,
            "options": options or {},
        }
        if format is not None:
            payload["format"] = format
        return self._post("/api/chat", payload)["message"]["content"]

    def embed(self, model: str, texts: List[str]) -> List[List[float]]:
        """Embedding vectors for `texts`, in the same order."""
        return self._post("/api/embed", {"model": model, "input": texts})["embeddings"]

    def describe_image(self, model: str, image_png: bytes, prompt: str) -> str:
        """Ask a vision model about one image."""
        message = {
            "role": "user",
            "content": prompt,
            "images": [base64.b64encode(image_png).decode("ascii")],
        }
        return self.chat(model, [message], options={"temperature": 0.2}).strip()

    def loaded_models(self) -> List[str]:
        try:
            running = requests.get(f"{self.host}/api/ps", timeout=10).json().get("models", [])
        except (requests.RequestException, ValueError):
            return []
        return [n for n in (m.get("name") or m.get("model") for m in running) if n]

    def unload(self, model: str, wait: float = 15.0) -> None:
        """Free the memory a model is using, waiting until it's gone. Errors are ignored."""
        try:
            requests.post(
                f"{self.host}/api/generate", json={"model": model, "keep_alive": 0}, timeout=30
            )
        except requests.RequestException:
            return
        # Ollama frees the memory shortly after replying.
        deadline = time.monotonic() + wait
        while model in self.loaded_models() and time.monotonic() < deadline:
            time.sleep(0.25)

    def unload_all(self) -> List[str]:
        """Unload every loaded model. Returns the names unloaded."""
        names = self.loaded_models()
        for name in names:
            self.unload(name)
        return names

    def _post(self, path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        model = payload["model"]
        response = self._send(path, payload)
        # Several models loaded at once can run out of GPU memory ("Compute error"), which can
        # also leave the failing model broken. Unload everything and try once more from fresh.
        if response.status_code >= 500 and self.unload_all():
            response = self._send(path, payload)

        if response.status_code == 404:
            raise OllamaError(f"Model '{model}' not found. Install it with `ollama pull {model}`.")
        if response.status_code >= 400:
            raise OllamaError(f"Ollama error {response.status_code}: {response.text}")
        return response.json()

    def _send(self, path: str, payload: Dict[str, Any]) -> requests.Response:
        try:
            return requests.post(f"{self.host}{path}", json=payload, timeout=self.timeout)
        except requests.ConnectionError:
            raise OllamaError(
                f"Can't reach Ollama at {self.host}. Start it with `ollama serve`."
            ) from None
