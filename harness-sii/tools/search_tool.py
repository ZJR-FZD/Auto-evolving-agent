"""
Web search tool — runs on the GPU host (no internet) by default.

Two operating modes are supported, picked automatically at import time:

1. **Proxy mode** (preferred when GPU host has no internet)
   Set ``SEARCH_PROXY_URL`` to the URL of the *search-proxy* FastAPI
   service running on a CPU host that DOES have internet access. With
   VS Code (or plain ``ssh -L``) port forwarding this can be a localhost
   address such as ``http://127.0.0.1:8090``. The GPU host then needs
   neither outbound internet nor any third-party API keys — keys live
   on the CPU host with the proxy.

2. **Direct mode** (legacy / local-dev)
   When ``SEARCH_PROXY_URL`` is empty, the tool talks directly to
   Serper / Jina from this process, which requires both internet
   access AND the original API keys (`SERPER_API_KEY`, `JINA_API_KEY`).

The two public tool functions keep identical signatures and return shapes
so the LLM tool schema does not change between modes::

    search_text(query, top_k=5, fetch=True, max_chars=500) -> list[dict]
    search_image(image_url, top_k=5, fetch=True, max_chars=500) -> list[dict]

Each result dict::

    {"rank": int, "title": str, "url": str, "snippet": str, "content"?: str}
"""

from __future__ import annotations

import logging
import os
from typing import Optional

import requests

logger = logging.getLogger("harness.tools.search")


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
# Proxy mode (recommended for the air-gapped GPU host).
SEARCH_PROXY_URL    = os.getenv("SEARCH_PROXY_URL", "https://nat2-notebook-inspire.sii.edu.cn/ws-7c23bd1d-9bae-4238-803a-737a35480e18/project-39fbffc7-dcca-4fb4-b43a-2f69f72f7e52/user-d7980cee-ce7d-4b08-a7c6-362b3b416af7/vscode/c2020a53-a173-4691-9875-61f34a35a051/e7c3ebcb-1fbb-48ad-8d41-1e770ada0c00/proxy/1227/").rstrip("/")
SEARCH_PROXY_TOKEN  = os.getenv("SEARCH_PROXY_TOKEN", "") or os.getenv(
    "PROXY_API_TOKEN", ""
)
PROXY_HTTP_TIMEOUT  = float(os.getenv("SEARCH_PROXY_TIMEOUT", "120"))

# When SEARCH_PROXY_URL points at a 3rd-party tunnel (vscode.dev / Codespaces
# preview / cloudflared / ngrok), SSL verification or extra headers may be
# needed. These knobs let you control that without code changes.
#   * SEARCH_PROXY_VERIFY_SSL=false  -> skip TLS cert verify (cloudflared etc.)
#   * SEARCH_PROXY_EXTRA_HEADERS     -> JSON, e.g. '{"Cookie":"vscode-tkn=..."}'
SEARCH_PROXY_VERIFY_SSL = os.getenv("SEARCH_PROXY_VERIFY_SSL", "true").lower() not in (
    "0", "false", "no"
)
try:
    import json as _json
    SEARCH_PROXY_EXTRA_HEADERS: dict = _json.loads(
        os.getenv("SEARCH_PROXY_EXTRA_HEADERS", "") or "{}"
    )
    if not isinstance(SEARCH_PROXY_EXTRA_HEADERS, dict):
        SEARCH_PROXY_EXTRA_HEADERS = {}
except Exception:  # noqa: BLE001
    SEARCH_PROXY_EXTRA_HEADERS = {}

# Direct mode (only used when SEARCH_PROXY_URL is empty).
SERPER_API_KEY      = os.getenv("SERPER_API_KEY", "a42e8b4adb370b5c866a2c6feb870641691c5901")
JINA_API_KEY        = os.getenv("JINA_API_KEY", "jina_27b632dc368a4d878d77a086367a1493HIydGZpZQJhxBWjflZBGmr89R44M")

SERPER_SEARCH_URL   = "https://google.serper.dev/search"
SERPER_LENS_URL     = "https://google.serper.dev/lens"
JINA_READER_BASE    = "https://r.jina.ai/"

DEFAULT_TIMEOUT     = 30
JINA_TIMEOUT        = 45


def _proxy_enabled() -> bool:
    return bool(SEARCH_PROXY_URL)


def _proxy_headers(json_body: bool = True) -> dict:
    h: dict = {}
    if json_body:
        h["Content-Type"] = "application/json"
    if SEARCH_PROXY_TOKEN:
        h["Authorization"] = f"Bearer {SEARCH_PROXY_TOKEN}"
    # Extra headers (e.g. tunnel auth cookies) take precedence so users
    # can override Content-Type / Authorization if their tunnel needs it.
    if SEARCH_PROXY_EXTRA_HEADERS:
        h.update(SEARCH_PROXY_EXTRA_HEADERS)
    return h


# ---------------------------------------------------------------------------
# Proxy-mode helpers
# ---------------------------------------------------------------------------
def _proxy_post(path: str, payload: dict, timeout: float = PROXY_HTTP_TIMEOUT) -> dict:
    url = f"{SEARCH_PROXY_URL}{path}"
    resp = requests.post(
        url,
        json=payload,
        headers=_proxy_headers(json_body=True),
        timeout=timeout,
        verify=SEARCH_PROXY_VERIFY_SSL,
    )
    resp.raise_for_status()
    return resp.json()


def _proxy_search(path: str, payload: dict) -> list[dict]:
    """POST /search/text or /search/image; normalize to the legacy result shape."""
    data = _proxy_post(path, payload)
    if not data.get("ok", False):
        err = data.get("error", "unknown proxy error")
        logger.warning("search-proxy %s failed: %s", path, err)
        return [{"rank": 1, "title": "", "url": "", "snippet": f"[proxy-error] {err}"}]

    out: list[dict] = []
    for hit in data.get("results", []) or []:
        entry = {
            "rank":    hit.get("rank", len(out) + 1),
            "title":   hit.get("title", ""),
            "url":     hit.get("url", ""),
            "snippet": hit.get("snippet", ""),
        }
        if hit.get("content") is not None:
            entry["content"] = hit["content"]
        out.append(entry)
    return out


# ---------------------------------------------------------------------------
# Direct-mode helpers (used only when SEARCH_PROXY_URL is empty)
# ---------------------------------------------------------------------------
def _require_serper_key() -> str:
    if not SERPER_API_KEY:
        raise RuntimeError(
            "SERPER_API_KEY not set. Either export it, or point "
            "SEARCH_PROXY_URL at a running search-proxy instance."
        )
    return SERPER_API_KEY


def _serper_post(url: str, payload: dict) -> dict:
    headers = {
        "X-API-KEY":    _require_serper_key(),
        "Content-Type": "application/json",
    }
    resp = requests.post(url, json=payload, headers=headers, timeout=DEFAULT_TIMEOUT)
    resp.raise_for_status()
    return resp.json()


def _jina_fetch(url: str, max_chars: int) -> str:
    if not url:
        return ""
    reader_url = JINA_READER_BASE + url
    headers = {"Accept": "text/plain"}
    if JINA_API_KEY:
        headers["Authorization"] = f"Bearer {JINA_API_KEY}"
    try:
        resp = requests.get(reader_url, headers=headers, timeout=JINA_TIMEOUT)
        resp.raise_for_status()
        text = resp.text or ""
        if max_chars and len(text) > max_chars:
            text = text[:max_chars] + f"\n\n...[truncated at {max_chars} chars]"
        return text
    except Exception as exc:  # noqa: BLE001
        logger.warning("Jina fetch failed for %s: %s", url, exc)
        return f"[jina-error] {type(exc).__name__}: {exc}"


def _require_image_url(image_url: Optional[str]) -> str:
    """Validate that ``image_url`` is a public http(s) URL.

    Local file paths are no longer accepted: uploading user images to a
    third-party host on the fly is unsafe and has been removed. Callers
    must supply an online URL that points to the same image they want
    reverse-searched.
    """
    if not image_url or not str(image_url).strip():
        raise ValueError("search_image requires a non-empty image URL.")
    s = str(image_url).strip()
    if not (s.startswith("http://") or s.startswith("https://")):
        raise ValueError(
            f"search_image: {s!r} is not an http(s) URL. "
            "Local image upload has been disabled for safety - please pass "
            "a publicly reachable URL pointing to the image."
        )
    return s


# ---------------------------------------------------------------------------
# Public tools
# ---------------------------------------------------------------------------
def search_text(
    query: str,
    top_k: int = 3,
    fetch: bool = True,
    max_chars: int = 500,
) -> list[dict]:
    """Text search on Google (via Serper) optionally enriched with full-text via Jina.

    Returns ``list[dict]`` with keys ``{rank, title, url, snippet, content?}``.
    """
    if not query or not query.strip():
        return []
    top_k = max(1, min(int(top_k), 10))
    if _proxy_enabled():
        logger.info("search_text(proxy) q=%r top_k=%d fetch=%s",
                    query, top_k, fetch)
        return _proxy_search(
            "/search/text",
            {
                "query":     query,
                "top_k":     top_k,
                "fetch":     bool(fetch),
                "max_chars": int(max_chars),
            },
        )

    # Direct mode
    logger.info("search_text(direct) q=%r top_k=%d fetch=%s",
                query, top_k, fetch)
    payload = {"q": query, "num": top_k}
    data = _serper_post(SERPER_SEARCH_URL, payload)
    organic = data.get("organic", []) or []

    results: list[dict] = []
    for rank, item in enumerate(organic[:top_k], start=1):
        url = item.get("link") or ""
        entry = {
            "rank":    rank,
            "title":   item.get("title", ""),
            "url":     url,
            "snippet": item.get("snippet", ""),
        }
        if fetch and url:
            entry["content"] = _jina_fetch(url, max_chars)
        results.append(entry)
    return results


def search_image(
    image: Optional[str] = None,
    top_k: int = 1,
    fetch: bool = True,
    max_chars: int = 500,
    *,
    image_url: Optional[str] = None,
) -> list[dict]:
    """Reverse image search via Google Lens (Serper /lens).

    ``image``/``image_url`` must be an http(s) URL. Local file upload is
    intentionally disabled because concurrent upload-and-search can produce
    transient public links that fail before Serper/Lens can fetch them.
    """
    if image is not None and image_url is not None:
        image_s = str(image).strip()
        image_url_s = str(image_url).strip()
        if image_s != image_url_s:
            raise ValueError(
                "search_image received both `image` and `image_url` with different values."
            )
    image_url = _require_image_url(image_url if image_url is not None else image)
    top_k = max(1, min(int(top_k), 10))

    if _proxy_enabled():
        logger.info("search_image(proxy) image_url=%s top_k=%d fetch=%s",
                    image_url, top_k, fetch)
        return _proxy_search(
            "/search/image",
            {
                "image_url": image_url,
                "top_k":     top_k,
                "fetch":     bool(fetch),
                "max_chars": int(max_chars),
            },
        )

    # Direct mode
    logger.info("search_image(direct) image_url=%s top_k=%d fetch=%s",
                image_url, top_k, fetch)
    payload = {"url": image_url}
    data = _serper_post(SERPER_LENS_URL, payload)
    items = data.get("organic") or data.get("visual_matches") or []

    results: list[dict] = []
    for rank, item in enumerate(items[:top_k], start=1):
        url = item.get("link") or item.get("url") or ""
        entry = {
            "rank":    rank,
            "title":   item.get("title", ""),
            "url":     url,
            "snippet": item.get("snippet", "") or item.get("source", ""),
        }
        if fetch and url:
            entry["content"] = _jina_fetch(url, max_chars)
        results.append(entry)
    return results


# ---------------------------------------------------------------------------
# CLI smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse
    import json

    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_text = sub.add_parser("text")
    p_text.add_argument("query")
    p_text.add_argument("--top-k", type=int, default=3)
    p_text.add_argument("--no-fetch", action="store_true")

    p_img = sub.add_parser("image")
    p_img.add_argument("image_url", help="http(s) image URL")
    p_img.add_argument("--top-k", type=int, default=3)
    p_img.add_argument("--no-fetch", action="store_true")

    args = ap.parse_args()

    print(f"[mode] {'proxy via ' + SEARCH_PROXY_URL if _proxy_enabled() else 'direct'}")

    if args.cmd == "text":
        out = search_text(args.query, top_k=args.top_k, fetch=not args.no_fetch)
    else:
        out = search_image(image_url=args.image_url, top_k=args.top_k, fetch=not args.no_fetch)

    print(json.dumps(out, ensure_ascii=False, indent=2)[:2000])
