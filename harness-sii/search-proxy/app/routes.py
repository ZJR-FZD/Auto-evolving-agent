"""HTTP routes for the search-proxy service."""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, File, Form, Header, HTTPException, UploadFile

from . import upstream
from .config import settings
from .schemas import (
    FetchReq,
    FetchResp,
    HealthResp,
    SearchHit,
    SearchImageReq,
    SearchResp,
    SearchTextReq,
    UploadResp,
)

logger = logging.getLogger("search-proxy.routes")
router = APIRouter()


# ---------- auth ----------
async def auth_dep(authorization: str | None = Header(default=None)) -> None:
    if not settings.api_token:
        return
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(401, "missing bearer token")
    token = authorization.removeprefix("Bearer ").strip()
    if token != settings.api_token:
        raise HTTPException(401, "invalid token")


# ---------- health ----------
@router.get("/health", response_model=HealthResp)
async def health() -> HealthResp:
    return HealthResp(
        status="ok",
        serper_configured=bool(settings.serper_api_key),
        jina_configured=bool(settings.jina_api_key),
    )


# ---------- search ----------
@router.post(
    "/search/text", response_model=SearchResp, dependencies=[Depends(auth_dep)]
)
async def search_text(req: SearchTextReq) -> SearchResp:
    try:
        organic = upstream.serper_search(req.query, req.top_k)
    except Exception as exc:  # noqa: BLE001
        return SearchResp(ok=False, error=f"{type(exc).__name__}: {exc}")

    hits: list[SearchHit] = []
    for rank, item in enumerate(organic[: req.top_k], start=1):
        url = item.get("link") or ""
        hit = SearchHit(
            rank=rank,
            title=item.get("title", ""),
            url=url,
            snippet=item.get("snippet", ""),
        )
        if req.fetch and url:
            try:
                content, _ = upstream.jina_fetch(url, req.max_chars)
                hit.content = content
            except Exception as exc:  # noqa: BLE001
                hit.content = f"[jina-error] {type(exc).__name__}: {exc}"
        hits.append(hit)
    return SearchResp(ok=True, results=hits)


@router.post(
    "/search/image", response_model=SearchResp, dependencies=[Depends(auth_dep)]
)
async def search_image(req: SearchImageReq) -> SearchResp:
    try:
        items = upstream.serper_lens(req.image_url, req.top_k)
    except Exception as exc:  # noqa: BLE001
        return SearchResp(ok=False, error=f"{type(exc).__name__}: {exc}")

    hits: list[SearchHit] = []
    for rank, item in enumerate(items[: req.top_k], start=1):
        url = item.get("link") or item.get("url") or ""
        hit = SearchHit(
            rank=rank,
            title=item.get("title", ""),
            url=url,
            snippet=item.get("snippet", "") or item.get("source", ""),
        )
        if req.fetch and url:
            try:
                content, _ = upstream.jina_fetch(url, req.max_chars)
                hit.content = content
            except Exception as exc:  # noqa: BLE001
                hit.content = f"[jina-error] {type(exc).__name__}: {exc}"
        hits.append(hit)
    return SearchResp(ok=True, results=hits)


@router.post("/fetch", response_model=FetchResp, dependencies=[Depends(auth_dep)])
async def fetch_url(req: FetchReq) -> FetchResp:
    """Standalone Jina fetch — handy for manual debugging from the GPU side."""
    try:
        content, truncated = upstream.jina_fetch(req.url, req.max_chars)
        return FetchResp(ok=True, url=req.url, content=content, truncated=truncated)
    except Exception as exc:  # noqa: BLE001
        return FetchResp(
            ok=False, url=req.url, error=f"{type(exc).__name__}: {exc}"
        )


# ---------- image upload ----------
@router.post(
    "/upload_image", response_model=UploadResp, dependencies=[Depends(auth_dep)]
)
async def upload_image(file: UploadFile = File(...), filename: str | None = Form(default=None)) -> UploadResp:
    """Upload a local image (sent from GPU side) to a public host (e.g. 0x0.st)
    and return the resulting public URL — used as input to /search/image."""
    try:
        data = await file.read()
        if not data:
            return UploadResp(ok=False, error="empty file")
        public_url = upstream.upload_image(
            data, filename or file.filename or "image.bin"
        )
        return UploadResp(ok=True, url=public_url)
    except Exception as exc:  # noqa: BLE001
        return UploadResp(ok=False, error=f"{type(exc).__name__}: {exc}")
