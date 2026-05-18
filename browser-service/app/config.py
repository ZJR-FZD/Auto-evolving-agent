"""服务全局配置，通过环境变量覆盖。"""

from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


def _bool(v: str | None, default: bool = False) -> bool:
    if v is None:
        return default
    return v.strip().lower() in {"1", "true", "yes", "y", "on"}


@dataclass
class Settings:
    host: str = os.getenv("HOST", "0.0.0.0")
    port: int = int(os.getenv("PORT", "8080"))

    headless: bool = _bool(os.getenv("HEADLESS"), True)
    browser_cdp_port: int = int(os.getenv("BROWSER_CDP_PORT", "9222"))
    viewport_width: int = int(os.getenv("DEFAULT_VIEWPORT_WIDTH", "1280"))
    viewport_height: int = int(os.getenv("DEFAULT_VIEWPORT_HEIGHT", "800"))

    max_sessions: int = int(os.getenv("MAX_SESSIONS", "10"))
    session_idle_timeout: int = int(os.getenv("SESSION_IDLE_TIMEOUT", "1800"))

    api_token: str = os.getenv("API_TOKEN", "").strip()


settings = Settings()
