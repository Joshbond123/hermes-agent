"""An Inrok-reached GPU must not be reported offline just because the Render relay hub is idle."""

from types import SimpleNamespace

from blackthorn.api import _relay_overlay
from cloudflare_d1_client import public_gpu_status, tunnel_transport


class _Hub:
    def __init__(self, alive):
        self._alive = alive

    def health(self):
        return {"alive": self._alive, "stalled": False}


def _req(hubs):
    return SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(services=SimpleNamespace(relay_hubs=hubs))))


def test_transport_classification_never_exposes_the_url():
    assert tunnel_transport("https://blackthorn.share.inrok.in") == "inrok"
    assert tunnel_transport("https://blackthorn-computer.share.inrok.in/") == "inrok"
    assert tunnel_transport("https://x.example.com/gpu-relay") == "relay"
    assert tunnel_transport("https://share.inrok.in.evil.com") == "relay"
    assert tunnel_transport("") == ""
    out = public_gpu_status({"active": True, "tunnel_url": "https://blackthorn.share.inrok.in", "api_key": "k"})
    assert out["transport"] == "inrok" and "tunnel_url" not in out and "api_key" not in out


def test_inrok_gpu_stays_online_while_relay_hub_is_idle(monkeypatch):
    import blackthorn.api as api
    monkeypatch.setattr(api, "svc", lambda request: request.app.state.services)
    out = {"online": True, "display_status": "Kaggle Ready", "transport": "inrok"}
    _relay_overlay(_req({"model": _Hub(alive=False)}), out)
    assert out["online"] is True and out["display_status"] == "Kaggle Ready"


def test_relay_gpu_still_goes_offline_when_relay_is_idle(monkeypatch):
    import blackthorn.api as api
    monkeypatch.setattr(api, "svc", lambda request: request.app.state.services)
    out = {"online": True, "display_status": "Kaggle Ready", "transport": "relay"}
    _relay_overlay(_req({"model": _Hub(alive=False)}), out)
    assert out["online"] is False and "relay" in out["display_status"].lower()
