"""Application factory: ``uvicorn blackthorn.app:create_app --factory``."""

from __future__ import annotations

import logging
import os
import re
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Optional

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse, Response
from starlette.datastructures import MutableHeaders

from . import __version__
from .auth import AuthMiddleware, router as auth_router
from .api import Services, router
from .config import Settings, get_settings
from .codespace_channel import CodespaceChannel
from .codespace_computer import CodespaceComputer
from .codespaces import CodespaceAPI, CodespaceError
from .llm import new_http_client
from .route import RouteResolver
from .runs import RunManager
from .sql import D1Executor, SqliteExecutor
from .store import ChatStore
from .tools import default_registry
from .tools.web import TavilyKeys

log = logging.getLogger("blackthorn")
STATIC_DIR = Path(__file__).resolve().parent / "static"
HASHED = re.compile(r"-[A-Za-z0-9_-]{8,}\.[a-z0-9]+$")

CSP = ("default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; img-src 'self' data:; "
       "font-src 'self' data:; connect-src 'self'; object-src 'none'; base-uri 'self'; frame-ancestors 'none'; form-action 'self'")


def build_services(settings: Settings, *, executor: Any = None, http: Any = None, static_dir: Optional[Path] = None) -> Services:
    backend = settings.resolved_store()
    if executor is None:
        if backend == "d1":
            executor = D1Executor(os.environ.get("CLOUDFLARE_ACCOUNT_ID", ""), os.environ.get("CLOUDFLARE_D1_DATABASE_ID", ""),
                                  os.environ.get("CLOUDFLARE_API_TOKEN", ""))
        else:
            executor = SqliteExecutor(settings.sqlite_path)
    store = ChatStore(executor, source=settings.session_source, backend=backend)
    http = http or new_http_client()
    resolver = RouteResolver(store)
    runs = RunManager(ttl=settings.run_ttl_s, max_active=settings.max_active_runs)
    # ---------------------------------------------------------------- the computer
    # GitHub Codespaces is the agent's computer: a machine that is created on demand, keeps its disk
    # across stop/start, and is reached through a channel its own agent opens outwards. The token is
    # the private drive PAT that already lives in the environment; without it the app still runs,
    # it simply has no computer and says so.
    codespaces = None
    computer = None
    channel = None
    repo = (os.environ.get("CODESPACE_REPO") or "jadeniran21/Computer-drive").strip()
    cs_token = (os.environ.get("COMPUTER_DRIVE_GITHUB_PAT") or os.environ.get("CODESPACE_GITHUB_TOKEN") or "").strip()
    if cs_token:
        channel = CodespaceChannel(store)
        api = CodespaceAPI(http, cs_token, repo=repo, owner=(os.environ.get("CODESPACE_OWNER") or "").strip(),
                           machine=(os.environ.get("CODESPACE_MACHINE") or "").strip(),
                           idle_timeout_minutes=int(os.environ.get("CODESPACE_IDLE_MINUTES") or 30),
                           retention_minutes=int(os.environ.get("CODESPACE_RETENTION_MINUTES") or 4320))
        computer = CodespaceComputer(
            http, api, channel=channel, store=store,
            idle_stop_s=int(os.environ.get("CODESPACE_IDLE_STOP_SECONDS") or 300),
            busy=lambda: bool(runs.active_count() if hasattr(runs, "active_count") else False))
        codespaces = computer
        log.info("computer: GitHub Codespaces (%s)", repo)
    else:
        from .codespace_computer import UnconfiguredComputer
        computer = UnconfiguredComputer()
        log.warning("computer: no Codespaces token configured; computer tools will report it unavailable")

    # Skills + tools live in the GitHub drive and are restored onto every computer and
    # lab VM, so a replaced microVM never loses its capability catalog.
    capabilities = None
    try:
        from .capabilities import from_env as _capabilities_from_env
        capabilities = _capabilities_from_env(http, store)
    except Exception as exc:  # noqa: BLE001 - discovery must not stop the app
        log.error("capability catalog unavailable: %s", type(exc).__name__)
    if capabilities is not None and hasattr(computer, "capabilities"):
        computer.capabilities = capabilities
    # The malware lab is a separate, on-demand microVM with its OWN provider account.
    # It is constructed here but creates nothing until the agent asks for it.
    lab = None
    try:
        from .keys_role import lab_key as _lab_key
        _dedicated = _lab_key(os.environ)
    except Exception:  # noqa: BLE001
        _dedicated = ""
    if _dedicated:
        from .malware_lab import from_env as _lab_from_env
        try:
            lab = _lab_from_env(http, store, _dedicated, capabilities=capabilities)
        except Exception as exc:  # noqa: BLE001 - a bad lab config must not stop the app
            log.error("malware lab unavailable: %s", type(exc).__name__)
            lab = None
    # ---------------------------------------------------------------- the Kaggle GPU
    # A second, on-demand accelerator that serves a local model. It costs a limited free quota, so it
    # is off unless asked for, and its endpoint is handed to the router so Qwen becomes selectable the
    # moment its kernel registers.
    gpu = None
    try:
        from .kaggle_gpu import KaggleGPU
        gpu = KaggleGPU(
            store, http, runs=runs,
            username=(os.environ.get("KAGGLE_USERNAME") or "").strip(),
            key=(os.environ.get("KAGGLE_KEY") or "").strip(),
            dataset=(os.environ.get("KAGGLE_MODEL_DATASET") or "").strip(),
            serve_slug=(os.environ.get("KAGGLE_SERVE_SLUG") or "blackthorn-qwen-serve").strip(),
            install_slug=(os.environ.get("KAGGLE_INSTALL_SLUG") or "blackthorn-qwen-install").strip(),
            public_url=(os.environ.get("BLACKTHORN_PUBLIC_URL") or "").strip(),
            kernel_token=(os.environ.get("BLACKTHORN_GPU_TOKEN") or "").strip(),
            hf_token=(os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_API_KEY") or "").strip())
        log.info("gpu: Kaggle GPU provider (%s)", gpu.username or "unconfigured")
    except Exception as exc:  # noqa: BLE001 - the app runs without a GPU
        log.error("gpu unavailable: %s", type(exc).__name__)
        gpu = None
    if gpu is not None:
        # The catalogue resolves the hosted model against the GPU's live endpoint while it serves.
        resolver._live = gpu
    return Services(settings=settings, store=store, resolver=resolver, registry=default_registry(), http=http,
                    tavily=TavilyKeys(store), computer=computer, runs=runs,
                    codespaces=codespaces, static_dir=static_dir or STATIC_DIR, channel=channel,
                    lab=lab, capabilities=capabilities, gpu=gpu)



class SecurityHeadersMiddleware:
    """Adds the security headers without buffering or wrapping the response body."""

    def __init__(self, app: Any) -> None:
        self.app = app

    async def __call__(self, scope: Any, receive: Any, send: Any) -> None:
        if scope.get("type") != "http":
            await self.app(scope, receive, send)
            return
        path = scope.get("path", "") or ""

        async def send_with_headers(message: Any) -> None:
            if message.get("type") == "http.response.start":
                headers = MutableHeaders(scope=message)
                headers.setdefault("X-Content-Type-Options", "nosniff")
                headers.setdefault("Referrer-Policy", "same-origin")
                headers.setdefault("X-Frame-Options", "DENY")
                if not path.startswith("/api/"):
                    headers.setdefault("Content-Security-Policy", CSP)
                else:
                    headers.setdefault("Cache-Control", "no-store")
            await send(message)

        await self.app(scope, receive, send_with_headers)


def create_app(settings: Optional[Settings] = None, *, services: Optional[Services] = None) -> FastAPI:
    settings = settings or get_settings()
    bt = services or build_services(settings)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        try:
            await bt.store.ensure_schema()
            swept = await bt.store.sweep_interrupted()
            if swept:
                log.warning("marked %s unfinished response(s) from a previous process as interrupted", swept)
        except Exception as exc:  # the UI reports storage trouble per request; the process must still come up
            log.error("storage not ready at startup: %s", exc)
        from .version import check_manifest
        integrity = check_manifest()
        if integrity.get("drift"):
            log.error("INTEGRITY: deployed files differ from the committed manifest: %s", integrity["drift"])
        from .version import overlay_status
        if overlay_status().get("active"):
            log.error("INTEGRITY: a code overlay was extracted over the checkout at build time; running code may differ from Git")
        gpu_task = None
        if bt.gpu is not None:
            import asyncio

            async def _gpu_power_loop() -> None:
                """Adopt a GPU that is still serving, and switch it off when it is genuinely idle.

                This runs on the server, so closing the browser or losing the connection cannot keep
                a GPU (and its quota) alive; and a stop is never ordered while a task is running.
                """
                try:
                    await bt.gpu.rehydrate()
                except Exception as exc:  # noqa: BLE001
                    log.warning("gpu state not restored: %s", type(exc).__name__)
                while True:
                    await asyncio.sleep(60)
                    try:
                        await bt.gpu.check_idle()
                    except Exception as exc:  # noqa: BLE001 - the loop must survive anything
                        log.warning("gpu idle check failed: %s", type(exc).__name__)

            gpu_task = asyncio.create_task(_gpu_power_loop())
        power_task = None
        if bt.codespaces is not None:
            import asyncio

            async def _codespace_power_loop() -> None:
                """Automatic power management: adopt a running machine, stop it when genuinely idle.

                ``maybe_stop`` refuses while a run is executing, while a tool call is in flight and
                while the idle window has not elapsed, so a task is never cut off mid-command. The
                task starts the computer on demand instead (any computer tool call does), so this
                loop only ever has to switch it off — and, if it was left running by a previous
                process, to notice and adopt it.
                """
                interval = 30.0
                try:
                    interval = float(os.environ.get("CODESPACE_POWER_INTERVAL_SECONDS") or 30)
                except ValueError:
                    interval = 30.0
                while True:
                    try:
                        # First make sure we know which machine is ours: a redeploy forgets it
                        # while GitHub keeps the codespace (and its meter) running.
                        adopted = await bt.codespaces.adopt()
                        if adopted.get("adopted"):
                            log.info("adopted running codespace %s", adopted.get("name"))
                        result = await bt.codespaces.maybe_stop()
                        if result.get("stopped"):
                            log.info("codespace stopped automatically (%s idle)", result.get("reason"))
                    except asyncio.CancelledError:
                        raise
                    except Exception as exc:  # noqa: BLE001 - the loop must outlive any single failure
                        log.error("codespace power check failed: %s", type(exc).__name__)
                    await asyncio.sleep(interval)

            power_task = asyncio.create_task(_codespace_power_loop())
        try:
            yield
        finally:
            for task in (power_task, gpu_task):
                if task is None:
                    continue
                task.cancel()
                try:
                    await task
                except __import__("asyncio").CancelledError:
                    pass
        await bt.runs.shutdown()
        await bt.http.aclose()
        await bt.store.db.aclose()

    app = FastAPI(title="Blackthorn", version=__version__, lifespan=lifespan, docs_url=None, redoc_url=None, openapi_url=None)
    app.state.bt = bt
    app.state.settings = settings
    app.include_router(router)
    app.include_router(auth_router)

    # Pure ASGI, not @app.middleware("http"): BaseHTTPMiddleware wraps every response in its own
    # stream and does not propagate a client disconnect into a streaming body, which left
    # generators running with nobody reading them. Headers are added on the start message only.
    app.add_middleware(SecurityHeadersMiddleware)
    app.add_middleware(AuthMiddleware)

    def _file(path: Path, *, immutable: bool) -> Response:
        cache = "public, max-age=31536000, immutable" if immutable else "no-cache"
        return FileResponse(path, headers={"Cache-Control": cache})

    @app.api_route("/assets/{name:path}", methods=["GET", "HEAD"])
    async def assets(name: str):
        root = (bt.static_dir / "assets").resolve()
        target = (root / name).resolve()
        if root not in target.parents or not target.is_file():
            return JSONResponse({"detail": {"code": "not_found", "message": "asset not found"}}, status_code=404)
        # Only content-hashed names may be cached forever: a same-name file can then never go stale in a browser.
        return _file(target, immutable=bool(HASHED.search(target.name)))

    @app.api_route("/{full_path:path}", methods=["GET", "HEAD"])
    async def spa(full_path: str):
        if full_path.startswith("api/"):
            return JSONResponse({"detail": {"code": "not_found", "message": "unknown API route"}}, status_code=404)
        root = bt.static_dir.resolve()
        if full_path:
            target = (root / full_path).resolve()
            if root in target.parents and target.is_file():
                return _file(target, immutable=False)
        index = root / "index.html"
        if not index.is_file():
            return JSONResponse({"detail": {"code": "ui_missing", "message": "The web UI build is missing from this deployment."}}, status_code=503)
        return FileResponse(index, headers={"Cache-Control": "no-store"})

    return app
