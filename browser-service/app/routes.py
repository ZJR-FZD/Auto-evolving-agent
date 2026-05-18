"""HTTP 路由。"""

from __future__ import annotations

import base64
import logging
from typing import Any

from fastapi import APIRouter, Depends, Header, HTTPException

from .browser import manager
from .config import settings
from .schemas import (
    BaseResp,
    CDPInfoResp,
    ClickReq,
    CloseTabReq,
    CreateSessionResp,
    EvalReq,
    EvalResp,
    GetHtmlReq,
    GetHtmlResp,
    GetTextReq,
    GetTextResp,
    ListSessionsResp,
    ListTabsResp,
    NavigateReq,
    NavigateResp,
    NewTabReq,
    NewTabResp,
    ScreenshotReq,
    ScreenshotResp,
    ScrollReq,
    SessionInfo,
    TitleResp,
    TypeReq,
)

logger = logging.getLogger(__name__)
router = APIRouter()


# ---------- 鉴权 ----------
async def auth_dep(authorization: str | None = Header(default=None)) -> None:
    if not settings.api_token:
        return
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(401, "missing bearer token")
    token = authorization.removeprefix("Bearer ").strip()
    if token != settings.api_token:
        raise HTTPException(401, "invalid token")


# ---------- 健康检查 ----------
@router.get("/health")
async def health() -> dict:
    return {
        "status": "ok",
        "browser_running": manager.cdp_http_endpoint != "",
        "sessions": len(manager.list_sessions()),
    }


# ---------- Session ----------
@router.post(
    "/session/create",
    response_model=CreateSessionResp,
    dependencies=[Depends(auth_dep)],
)
async def create_session() -> CreateSessionResp:
    sess = await manager.create_session()
    return CreateSessionResp(
        session_id=sess.session_id,
        tab_id=sess.active_tab_id or "",
        cdp_url=manager.cdp_http_endpoint,
    )


@router.get(
    "/session/list",
    response_model=ListSessionsResp,
    dependencies=[Depends(auth_dep)],
)
async def list_sessions() -> ListSessionsResp:
    out = []
    for s in manager.list_sessions():
        out.append(
            SessionInfo(
                session_id=s.session_id,
                cdp_url=manager.cdp_http_endpoint,
                tabs=list(s.tabs.keys()),
                active_tab=s.active_tab_id or "",
            )
        )
    return ListSessionsResp(sessions=out)


@router.delete(
    "/session/{session_id}",
    response_model=BaseResp,
    dependencies=[Depends(auth_dep)],
)
async def delete_session(session_id: str) -> BaseResp:
    await manager.close_session(session_id)
    return BaseResp()


# ---------- Tab ----------
@router.post(
    "/tab/new", response_model=NewTabResp, dependencies=[Depends(auth_dep)]
)
async def new_tab(req: NewTabReq) -> NewTabResp:
    sess, tab = await manager.new_tab(req.session_id, req.url)
    return NewTabResp(session_id=sess.session_id, tab_id=tab.tab_id)


@router.post(
    "/tab/close", response_model=BaseResp, dependencies=[Depends(auth_dep)]
)
async def close_tab(req: CloseTabReq) -> BaseResp:
    if not req.session_id or not req.tab_id:
        raise HTTPException(400, "session_id and tab_id required")
    await manager.close_tab(req.session_id, req.tab_id)
    return BaseResp()


@router.get(
    "/tab/list/{session_id}",
    response_model=ListTabsResp,
    dependencies=[Depends(auth_dep)],
)
async def list_tabs(session_id: str) -> ListTabsResp:
    try:
        sess, _ = manager.get_page(session_id, None)
    except KeyError as e:
        raise HTTPException(404, str(e)) from e
    tabs = []
    for tid, t in sess.tabs.items():
        try:
            tabs.append(
                {"tab_id": tid, "url": t.page.url, "title": await t.page.title()}
            )
        except Exception:
            tabs.append({"tab_id": tid, "url": "", "title": ""})
    return ListTabsResp(
        session_id=sess.session_id,
        active_tab=sess.active_tab_id or "",
        tabs=tabs,
    )


# ---------- 浏览器操作 ----------
def _resolve(req: Any):
    try:
        return manager.get_page(
            getattr(req, "session_id", None), getattr(req, "tab_id", None)
        )
    except KeyError as e:
        raise HTTPException(404, str(e)) from e


@router.post(
    "/browser/navigate",
    response_model=NavigateResp,
    dependencies=[Depends(auth_dep)],
)
async def navigate(req: NavigateReq) -> NavigateResp:
    if not req.session_id:
        sess = await manager.create_session()
        tab = sess.tabs[sess.active_tab_id]  # type: ignore[index]
    else:
        sess, tab = _resolve(req)
    await tab.page.goto(req.url, wait_until=req.wait_until, timeout=req.timeout_ms)
    return NavigateResp(url=tab.page.url, title=await tab.page.title())


@router.post(
    "/browser/get_text",
    response_model=GetTextResp,
    dependencies=[Depends(auth_dep)],
)
async def get_text(req: GetTextReq) -> GetTextResp:
    _, tab = _resolve(req)
    if req.selector:
        loc = tab.page.locator(req.selector).first
        text = await loc.inner_text()
    else:
        text = await tab.page.evaluate("() => document.body.innerText")
    return GetTextResp(text=text)


@router.post(
    "/browser/get_html",
    response_model=GetHtmlResp,
    dependencies=[Depends(auth_dep)],
)
async def get_html(req: GetHtmlReq) -> GetHtmlResp:
    _, tab = _resolve(req)
    if req.selector:
        loc = tab.page.locator(req.selector).first
        html = await loc.evaluate("el => el.outerHTML")
    else:
        html = await tab.page.content()
    return GetHtmlResp(html=html)


@router.post(
    "/browser/screenshot",
    response_model=ScreenshotResp,
    dependencies=[Depends(auth_dep)],
)
async def screenshot(req: ScreenshotReq) -> ScreenshotResp:
    _, tab = _resolve(req)
    fmt = req.image_format.lower()
    if fmt not in {"png", "jpeg"}:
        raise HTTPException(400, "image_format must be png|jpeg")

    if req.selector:
        loc = tab.page.locator(req.selector).first
        img_bytes = await loc.screenshot(type=fmt)  # type: ignore[arg-type]
    else:
        img_bytes = await tab.page.screenshot(
            type=fmt,  # type: ignore[arg-type]
            full_page=req.full_page,
        )
    return ScreenshotResp(
        image_base64=base64.b64encode(img_bytes).decode("utf-8"),
        image_format=fmt,
    )


@router.post(
    "/browser/click", response_model=BaseResp, dependencies=[Depends(auth_dep)]
)
async def click(req: ClickReq) -> BaseResp:
    _, tab = _resolve(req)
    await tab.page.locator(req.selector).first.click(timeout=req.timeout_ms)
    return BaseResp()


@router.post(
    "/browser/type", response_model=BaseResp, dependencies=[Depends(auth_dep)]
)
async def type_text(req: TypeReq) -> BaseResp:
    _, tab = _resolve(req)
    loc = tab.page.locator(req.selector).first
    if req.clear:
        await loc.fill("")
    if req.delay_ms > 0:
        await loc.type(req.text, delay=req.delay_ms)
    else:
        await loc.fill(req.text)
    if req.press_enter:
        await loc.press("Enter")
    return BaseResp()


@router.post(
    "/browser/scroll", response_model=BaseResp, dependencies=[Depends(auth_dep)]
)
async def scroll(req: ScrollReq) -> BaseResp:
    _, tab = _resolve(req)
    direction = req.direction.lower()
    if direction == "down":
        await tab.page.evaluate(f"window.scrollBy(0, {req.pixels})")
    elif direction == "up":
        await tab.page.evaluate(f"window.scrollBy(0, -{req.pixels})")
    elif direction == "top":
        await tab.page.evaluate("window.scrollTo(0, 0)")
    elif direction == "bottom":
        await tab.page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
    else:
        raise HTTPException(400, "direction must be up|down|top|bottom")
    return BaseResp()


@router.post(
    "/browser/eval", response_model=EvalResp, dependencies=[Depends(auth_dep)]
)
async def evaluate_js(req: EvalReq) -> EvalResp:
    _, tab = _resolve(req)
    try:
        result = await tab.page.evaluate(req.script)
    except Exception as e:
        raise HTTPException(400, f"eval error: {e}") from e
    # 保证 JSON 可序列化
    try:
        import json

        json.dumps(result)
    except (TypeError, ValueError):
        result = repr(result)
    return EvalResp(result=result)


@router.get(
    "/browser/title", response_model=TitleResp, dependencies=[Depends(auth_dep)]
)
async def get_title(
    session_id: str | None = None, tab_id: str | None = None
) -> TitleResp:
    try:
        _, tab = manager.get_page(session_id, tab_id)
    except KeyError as e:
        raise HTTPException(404, str(e)) from e
    return TitleResp(title=await tab.page.title(), url=tab.page.url)


@router.get(
    "/browser/cdp_url",
    response_model=CDPInfoResp,
    dependencies=[Depends(auth_dep)],
)
async def cdp_url() -> CDPInfoResp:
    if not manager.cdp_http_endpoint:
        await manager.start()
    return CDPInfoResp(
        cdp_url=manager.cdp_http_endpoint,
        browser_version=await manager.browser_version(),
    )
