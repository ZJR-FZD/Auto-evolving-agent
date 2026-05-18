"""Runtime configuration for search-proxy."""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass
class Settings:
    host: str = os.getenv("HOST", "127.0.0.1")
    port: int = int(os.getenv("PORT", "8090"))

    # Upstream API keys (must be set on the *CPU host*, not on the GPU host).
    serper_api_key: str = os.getenv("SERPER_API_KEY", "")
    jina_api_key: str = os.getenv("JINA_API_KEY", "")

    # Optional shared secret. If set, every request must carry
    # `Authorization: Bearer <token>`.
    api_token: str = os.getenv("PROXY_API_TOKEN", "")

    # Image upload backend used when the GPU side sends a local file
    # via /upload_image. Currently only `0x0` is supported.
    image_uploader: str = os.getenv("IMAGE_UPLOADER", "0x0")

    # Default HTTP timeouts (seconds) for outbound calls.
    serper_timeout: float = float(os.getenv("SERPER_TIMEOUT", "30"))
    jina_timeout: float = float(os.getenv("JINA_TIMEOUT", "45"))
    upload_timeout: float = float(os.getenv("UPLOAD_TIMEOUT", "60"))


settings = Settings()
