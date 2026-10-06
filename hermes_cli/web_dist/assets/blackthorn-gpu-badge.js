(function () {
  function findGpuButton() {
    var buttons = Array.from(document.querySelectorAll("button"));
    return buttons.find(function (b) {
      return /GPU\s*(ON|OFF|\d+%)/i.test(b.textContent || "");
    }) || null;
  }

  function paintGpu(st) {
    if (!st) return;
    var status = String(st.status || "").toUpperCase();
    var online = ["ONLINE", "MODEL_READY_AND_WARMED", "HEARTBEAT_ONLINE", "MODEL_READY", "MODEL_READY_COLD", "TUNNEL_ONLINE"];
    var active = !!(st.active || online.indexOf(status) >= 0);
    var booting = !!st.booting;
    var pct = st.progress_pct || 0;
    var label = active ? "GPU ON" : (booting ? ("GPU " + pct + "%") : "GPU OFF");
    document.documentElement.setAttribute("data-gpu", active ? "on" : (booting ? "boot" : "off"));
    var btn = findGpuButton();
    if (btn) {
      btn.classList.toggle("bt-gpu-live", active);
      btn.classList.toggle("bt-gpu-off", !active && !booting);
      var spans = btn.querySelectorAll("span");
      var textSpan = null;
      spans.forEach(function (s) {
        if (/GPU/i.test(s.textContent || "")) textSpan = s;
      });
      if (textSpan) textSpan.textContent = label;
      btn.title = [st.display_status || st.status || "", st.gpu_info || "", st.tunnel_url || "", st.model || ""]
        .filter(Boolean).join(" | ");
    }
    try {
      if (active && st.tunnel_url) {
        localStorage.setItem("blackthorn_gpu_last", JSON.stringify({
          status: st.status || "ONLINE",
          active: true,
          tunnel_url: st.tunnel_url,
          display_status: st.display_status || "Kaggle Ready",
          progress_pct: 100,
          ts: Date.now(),
          gpu_info: st.gpu_info || "",
          model: st.model || ""
        }));
      }
    } catch (e) {}
  }

  function pollGpu() {
    return fetch("/api/kaggle-gpu/status", { credentials: "include" })
      .then(function (r) { return r.ok ? r.json() : null; })
      .then(function (st) { if (st) paintGpu(st); })
      .catch(function () {
        try {
          var raw = localStorage.getItem("blackthorn_gpu_last");
          if (raw) {
            var p = JSON.parse(raw);
            if (p && p.active && Date.now() - (p.ts || 0) < 3600000) paintGpu(p);
          }
        } catch (e) {}
      });
  }

  try {
    var raw0 = localStorage.getItem("blackthorn_gpu_last");
    if (raw0) {
      var p0 = JSON.parse(raw0);
      if (p0 && p0.active && Date.now() - (p0.ts || 0) < 3600000) paintGpu(p0);
    }
  } catch (e) {}

  var n = 0;
  function tick() {
    pollGpu().then(function () {
      n += 1;
      setTimeout(tick, n < 15 ? 2000 : 8000);
    });
  }
  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", tick);
  } else {
    tick();
  }
})();
