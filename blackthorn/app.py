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

from . import __version__
from .api import Services, router
from .config import Settings, get_settings
from .gpu import GpuService
from .llm import new_http_client
from .route import RouteResolver
from .runs import RunManager
from .sql import D1Executor, SqliteExecutor
from .store import ChatStore
from .tools import default_registry
from .tools.computer import ComputerClient
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
    return Services(settings=settings, store=store, resolver=resolver, registry=default_registry(), http=http,
                    tavily=TavilyKeys(store), computer=ComputerClient(resolver, http),
                    runs=RunManager(ttl=settings.run_ttl_s, max_active=settings.max_active_runs),
                    gpu=GpuService(store), static_dir=static_dir or STATIC_DIR)


def create_app(settings: Optional[Settings] = None, *, services: Optional[Services] = None, gpu_daemon: Optional[bool] = None) -> FastAPI:
    settings = settings or get_settings()
    bt = services or build_services(settings)
    start_daemon = (os.environ.get("BLACKTHORN_GPU_DAEMON", "1") == "1" and settings.resolved_store() == "d1") if gpu_daemon is None else gpu_daemon

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
        if start_daemon:
            try:
                import cloudflare_d1_client
                cloudflare_d1_client.start_permanent_gpu_daemon()
            except Exception as exc:
                log.error("GPU watchdog could not start: %s", exc)
        yield
        await bt.runs.shutdown()
        await bt.http.aclose()
        await bt.store.db.aclose()

    app = FastAPI(title="Blackthorn", version=__version__, lifespan=lifespan, docs_url=None, redoc_url=None, openapi_url=None)
    app.state.bt = bt
    app.state.settings = settings
    app.include_router(router)

    @app.middleware("http")
    async def headers(request: Request, call_next):
        response = await call_next(request)
        response.headers.setdefault("X-Content-Type-Options", "nosniff")
        response.headers.setdefault("Referrer-Policy", "same-origin")
        response.headers.setdefault("X-Frame-Options", "DENY")
        if not request.url.path.startswith("/api/"):
            response.headers.setdefault("Content-Security-Policy", CSP)
        if request.url.path.startswith("/api/"):
            response.headers.setdefault("Cache-Control", "no-store")
        return response

    def _file(path: Path, *, immutable: bool) -> Response:
        cache = "public, max-age=31536000, immutable" if immutable else "no-cache"
        return FileResponse(path, headers={"Cache-Control": cache})

    @app.get("/assets/{name:path}")
    async def assets(name: str):
        root = (bt.static_dir / "assets").resolve()
        target = (root / name).resolve()
        if root not in target.parents or not target.is_file():
            return JSONResponse({"detail": {"code": "not_found", "message": "asset not found"}}, status_code=404)
        # Only content-hashed names may be cached forever: a same-name file can then never go stale in a browser.
        return _file(target, immutable=bool(HASHED.search(target.name)))

    @app.get("/{full_path:path}")
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
