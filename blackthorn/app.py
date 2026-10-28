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
    fleet = None
    try:
        from .fleet import FleetManager, accounts_from_env
        accounts = accounts_from_env(os.environ)
        if accounts:
            fleet = FleetManager(accounts)
    except Exception as exc:  # noqa: BLE001 - the studio must run even without fleet accounts
        log.warning("fleet not configured: %s", exc)
    # The Panda route is opt-in. Never fall back to Kaggle when a configured
    # PandaStack sandbox fails; its filesystem is a different computer.
    panda_key = os.environ.get("PANDASTACK_API_KEY", "").strip()
    panda_id = os.environ.get("PANDASTACK_SANDBOX_ID", "").strip()
    drive_key = os.environ.get("COMPUTER_DRIVE_GITHUB_PAT", "").strip()
    if bool(panda_key) != bool(drive_key):
        raise ValueError("PandaStack computer requires both private provider and drive credentials")
    key_pool = None
    lifecycle = None
    if panda_key:
        from .panda_computer import PandaComputerClient
        from .panda_keys import PandaKeyPool
        from .lifecycle import SandboxLifecycle, DEFAULT_WINDOW_S
        try:
            key_pool = PandaKeyPool.from_env(http, store)
        except Exception as exc:  # noqa: BLE001 - a bad pool must not stop the app
            log.error("PandaStack key pool unavailable: %s", type(exc).__name__)
            key_pool = None
        computer = PandaComputerClient(http, api_key=key_pool or panda_key, sandbox_id=panda_id,
                                       github_token=drive_key, store=store, key_pool=key_pool)
        try:
            window = float(os.environ.get("PANDASTACK_HANDOVER_WINDOW_SECONDS") or DEFAULT_WINDOW_S)
        except ValueError:
            window = DEFAULT_WINDOW_S
        lifecycle = SandboxLifecycle(store, computer, window_s=window)
    else:
        computer = ComputerClient(resolver, http, store=store)
    # The malware lab is a separate, on-demand microVM. It is constructed here but
    # creates nothing until the agent asks for it.
    lab = None
    if panda_key:
        from .malware_lab import from_env as _lab_from_env
        try:
            lab = _lab_from_env(http, store, key_pool or panda_key)
        except Exception as exc:  # noqa: BLE001 - a bad lab config must not stop the app
            log.error("malware lab unavailable: %s", type(exc).__name__)
            lab = None
    return Services(settings=settings, store=store, resolver=resolver, registry=default_registry(), http=http,
                    tavily=TavilyKeys(store), computer=computer,
                    runs=RunManager(ttl=settings.run_ttl_s, max_active=settings.max_active_runs),
                    gpu=GpuService(store), static_dir=static_dir or STATIC_DIR, fleet=fleet,
                    key_pool=key_pool, lifecycle=lifecycle, lab=lab)



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
        from .version import overlay_status
        if overlay_status().get("active"):
            log.error("INTEGRITY: a code overlay was extracted over the checkout at build time; running code may differ from Git")
        if start_daemon:
            try:
                import cloudflare_d1_client
                cloudflare_d1_client.start_permanent_gpu_daemon()
            except Exception as exc:
                log.error("GPU watchdog could not start: %s", exc)
        if bt.fleet is not None:
            import threading

            def _fleet_watchdog() -> None:
                roles = ("model",) if os.environ.get("PANDASTACK_API_KEY") else ("model", "computer")
                active = {role: bt.fleet.state[role].active for role in roles}
                while True:
                    try:
                        for role in roles:
                            before = active[role]
                            st = bt.fleet.decide(role)
                            if st.active != before:
                                active[role] = st.active
                                log.warning("fleet failover: %s %s -> %s (%s)", role, before, st.active, st.reason)
                                try:
                                    import cloudflare_d1_client as _d1
                                    # A watchdog on Render startup sees "primary" as its
                                    # initial slot. Invalid primary credentials must not
                                    # resurrect a user-stopped backup computer/model.
                                    if _d1._role_intended_online(role) and not _d1._role_boot_in_progress(role):
                                        _d1.failover_fleet_role(role, before, st.active)
                                    else:
                                        log.info("fleet %s switched standby slot without booting an off role", role)
                                except Exception:  # noqa: BLE001
                                    log.exception("failover execution failed for %s", role)
                    except Exception:  # noqa: BLE001 - a watchdog must survive anything
                        log.exception("fleet watchdog tick failed")
                    import time as _t
                    _t.sleep(60)

            threading.Thread(target=_fleet_watchdog, daemon=True, name="fleet-watchdog").start()
        panda_watchdog = None
        if os.environ.get("PANDASTACK_API_KEY"):
            import asyncio
            from .panda_control import PandaControl, INTENT, SANDBOX
            from .route import BOOTING

            async def _panda_power_watchdog():
                while True:
                    try:
                        intent = await bt.store.get_setting(INTENT)
                        row = await bt.store.gpu_row()
                        state = str(row.get("status") or "").upper()
                        controller = PandaControl(bt.store, bt.http)
                        if intent == "on" and state in {"OFF", "OFFLINE", "GPU_STOPPED_SAVING_QUOTA", "STOPPED", "BOOT_FAILED", "GPU_UNAVAILABLE"}:
                            log.warning("model GPU is off; stopping linked PandaStack computer")
                            await controller.stop()
                        elif intent == "off" and state in (BOOTING | {"MODEL_READY_COLD", "MODEL_READY_AND_WARMED", "HEARTBEAT_ONLINE", "ONLINE"}):
                            # A failed model boot may be restarted by the fleet watchdog
                            # after we hibernated the computer. Wake an existing VM once;
                            # never rotate expired sandboxes merely to evade tier TTL.
                            if await bt.store.get_setting(SANDBOX):
                                owned = await controller._owned()
                                if len(owned) == 1 and str(owned[0].get("status") or "").lower() == "hibernated":
                                    await controller.start()
                    except asyncio.CancelledError:
                        raise
                    except Exception as exc:
                        log.error("PandaStack power reconciliation failed: %s", type(exc).__name__)
                    await asyncio.sleep(15)

            panda_watchdog = asyncio.create_task(_panda_power_watchdog())
        lifecycle_task = None
        rotation_task = None
        if bt.lifecycle is not None:
            import asyncio
            from .lifecycle import lifecycle_daemon

            async def _activity() -> bool:
                # True while any run is streaming: an active task must not be cut by a TTL reap.
                try:
                    return bool(bt.runs.active_count()) if hasattr(bt.runs, "active_count") else bool(bt.runs._active)
                except Exception:  # noqa: BLE001
                    return False

            async def _quota_rotation():
                pool = bt.key_pool
                while True:
                    try:
                        state = await pool.ensure_active()
                        if not state.valid:
                            log.error("PandaStack: no usable API key remains; affected operations are paused")
                    except asyncio.CancelledError:
                        raise
                    except Exception as exc:  # noqa: BLE001
                        log.error("PandaStack key rotation check failed: %s", type(exc).__name__)
                    await asyncio.sleep(60)

            try:
                interval = float(os.environ.get("PANDASTACK_LIFECYCLE_INTERVAL_SECONDS") or 60)
            except ValueError:
                interval = 60.0
            lifecycle_task = asyncio.create_task(
                lifecycle_daemon(bt.store, bt.computer, interval=interval,
                                 window_s=bt.lifecycle.window_s, activity=_activity))
            if bt.key_pool is not None:
                rotation_task = asyncio.create_task(_quota_rotation())
        try:
            yield
        finally:
            for task in (locals().get("lifecycle_task"), locals().get("rotation_task")):
                if task is not None:
                    task.cancel()
                    try:
                        await task
                    except __import__("asyncio").CancelledError:
                        pass
            if panda_watchdog is not None:
                panda_watchdog.cancel()
                try:
                    await panda_watchdog
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
    # stream and does not propagate a client disconnect into a streaming body, which left GPU relay
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
