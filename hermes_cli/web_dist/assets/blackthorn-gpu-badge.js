(function () {
  "use strict";

  /* ---- GPU badge painter ---- */
  function findGpuButton() {
    return Array.from(document.querySelectorAll("button")).find(function (b) {
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
      spans.forEach(function (s) {
        if (/GPU/i.test(s.textContent || "")) s.textContent = label;
      });
      btn.title = [st.display_status || st.status || "", st.gpu_info || "", st.tunnel_url || "", st.model || ""]
        .filter(Boolean).join(" | ");
    }
    paintQuota(st);
  }

  /* ---- Always-visible weekly quota chip ---- */
  function paintQuota(st) {
    var q = (st && st.quota) || null;
    var host = document.getElementById("bt-quota-chip");
    if (!host) {
      host = document.createElement("span");
      host.id = "bt-quota-chip";
      host.title = "Kaggle GPU weekly quota (always visible)";
      var btn = findGpuButton();
      if (btn && btn.parentElement) {
        btn.parentElement.insertBefore(host, btn);
      } else {
        return;
      }
    }
    if (!q) {
      host.textContent = "Quota …";
      return;
    }
    var used = (typeof q.used_hours === "number") ? q.used_hours : 0;
    var total = (typeof q.total_hours === "number") ? q.total_hours : 30;
    var pct = (typeof q.used_pct === "number") ? q.used_pct : 0;
    var color = pct > 85 ? "#ef4444" : (pct > 60 ? "#f59e0b" : "#10b981");
    host.innerHTML =
      '<span>GPU ' + used.toFixed(1) + "h/" + total.toFixed(0) + "h</span>" +
      '<span class="bt-q-bar"><i style="width:' + Math.max(2, Math.min(100, pct)) + "%;background:" + color + '"></i></span>';
  }

  function pollGpu() {
    return fetch("/api/kaggle-gpu/status", { credentials: "include" })
      .then(function (r) { return r.ok ? r.json() : null; })
      .then(function (st) { if (st) paintGpu(st); })
      .catch(function () {});
  }

  /* ---- Sidebar toggle fix ----
     The stock header button only sets open=true (never toggles).
     We intercept clicks on the nav-open control and toggle body.bt-nav-open. */
  function wireSidebar() {
    document.addEventListener("click", function (ev) {
      var t = ev.target;
      if (!t || !t.closest) return;
      var btn = t.closest('button[aria-controls="app-sidebar"]');
      if (btn) {
        // Let React open state run, then force body class toggle for CSS drawer
        setTimeout(function () {
          var expanded = btn.getAttribute("aria-expanded") === "true";
          document.body.classList.toggle("bt-nav-open", expanded);
        }, 0);
        return;
      }
      // Backdrop click closes
      if (document.body.classList.contains("bt-nav-open")) {
        var aside = document.getElementById("app-sidebar");
        if (aside && !aside.contains(t) && !t.closest('button[aria-controls="app-sidebar"]')) {
          document.body.classList.remove("bt-nav-open");
          // try to click the close path by setting aria
          var openBtn = document.querySelector('button[aria-controls="app-sidebar"]');
          if (openBtn && openBtn.getAttribute("aria-expanded") === "true") {
            openBtn.click();
          }
        }
      }
    }, true);

    // Escape closes
    document.addEventListener("keydown", function (ev) {
      if (ev.key === "Escape" && document.body.classList.contains("bt-nav-open")) {
        document.body.classList.remove("bt-nav-open");
      }
    });
  }

  /* ---- Strip residual avatar nodes that CSS alone may miss ---- */
  function stripAvatars() {
    document.querySelectorAll('[class*="message"] [class*="rounded-full"].w-8, [class*="message"] [class*="rounded-full"].h-8').forEach(function (el) {
      if (el.tagName !== "BUTTON") el.style.display = "none";
    });
  }

  var n = 0;
  function tick() {
    pollGpu().then(function () {
      stripAvatars();
      n += 1;
      setTimeout(tick, n < 12 ? 2500 : 10000);
    });
  }

  function boot() {
    wireSidebar();
    tick();
    var mo = new MutationObserver(function () { stripAvatars(); });
    mo.observe(document.documentElement, { childList: true, subtree: true });
  }

  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", boot);
  else boot();
})();
