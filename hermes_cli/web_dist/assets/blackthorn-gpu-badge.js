(function () {
  "use strict";
  // blackthorn-ui v2026-10-07c gpu-panel

  var TOKEN = (window.__HERMES_SESSION_TOKEN__ || "");
  var MOCK_TITLES = [
    "reading your request", "understanding your request", "analyzing your request",
    "processing your request", "thinking", "thinking...", "working", "working…",
    "working on it…", "working on it", "worked for", "starting"
  ];

  function authHeaders() {
    var h = { "Content-Type": "application/json" };
    if (TOKEN) h["Authorization"] = "Bearer " + TOKEN;
    return h;
  }

  function svg(pathD, size) {
    size = size || 16;
    return '<svg xmlns="http://www.w3.org/2000/svg" width="' + size + '" height="' + size +
      '" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">' +
      pathD + "</svg>";
  }
  var ICONS = {
    menu: svg('<circle cx="12" cy="5" r="1"/><circle cx="12" cy="12" r="1"/><circle cx="12" cy="19" r="1"/>', 16),
    archive: svg('<polyline points="21 8 21 21 3 21 3 8"/><rect x="1" y="3" width="22" height="5"/><line x1="10" y1="12" x2="14" y2="12"/>', 16),
    trash: svg('<polyline points="3 6 5 6 21 6"/><path d="M19 6l-1 14a2 2 0 0 1-2 2H8a2 2 0 0 1-2-2L5 6"/><path d="M10 11v6"/><path d="M14 11v6"/><path d="M9 6V4a1 1 0 0 1 1-1h4a1 1 0 0 1 1 1v2"/>', 16),
    close: svg('<line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/>', 16),
    hist: svg('<line x1="8" y1="6" x2="21" y2="6"/><line x1="8" y1="12" x2="21" y2="12"/><line x1="8" y1="18" x2="21" y2="18"/><line x1="3" y1="6" x2="3.01" y2="6"/><line x1="3" y1="12" x2="3.01" y2="12"/><line x1="3" y1="18" x2="3.01" y2="18"/>', 16),
    plus: svg('<line x1="12" y1="5" x2="12" y2="19"/><line x1="5" y1="12" x2="19" y2="12"/>', 14),
    prompt: svg('<path d="M12 20h9"/><path d="M16.5 3.5a2.12 2.12 0 0 1 3 3L7 19l-4 1 1-4Z"/>', 14)
  };

  function syncChatMode() {
    var path = (location.pathname || "").replace(/\/+$/, "") || "/";
    var isChat = path === "/chat" || path.indexOf("/chat/") === 0 || path.indexOf("/studio") === 0;
    document.body.classList.toggle("bt-chat-mode", isChat);
    document.documentElement.classList.toggle("bt-chat-mode", isChat);
  }

  function isMockTitle(title) {
    var t = String(title || "").trim().toLowerCase();
    if (!t) return true;
    for (var i = 0; i < MOCK_TITLES.length; i++) {
      if (t === MOCK_TITLES[i] || t.indexOf(MOCK_TITLES[i]) === 0) return true;
    }
    return false;
  }


  /** Upgrade plain <pre><code> into professional blocks + highlight.js */
  function enhanceCodeBlocks(root) {
    root = root || document;
    var blocks = root.querySelectorAll("pre > code");
    for (var i = 0; i < blocks.length; i++) {
      var code = blocks[i];
      var pre = code.parentElement;
      if (!pre || pre.closest(".bt-code-block")) continue;
      if (pre.dataset.btEnhanced === "1") continue;
      pre.dataset.btEnhanced = "1";
      var lang = "";
      var cls = code.className || "";
      var m = cls.match(/language-([\w+-]+)/) || cls.match(/lang-([\w+-]+)/);
      if (m) lang = m[1];
      // Try highlight
      try {
        if (window.hljs) {
          if (lang && window.hljs.getLanguage && window.hljs.getLanguage(lang)) {
            code.innerHTML = window.hljs.highlight(code.textContent, { language: lang, ignoreIllegals: true }).value;
            code.classList.add("hljs", "language-" + lang);
          } else {
            var r = window.hljs.highlightAuto(code.textContent);
            code.innerHTML = r.value;
            code.classList.add("hljs");
            if (r.language) lang = r.language;
          }
        }
      } catch (e) {}
      // Wrap
      var wrap = document.createElement("div");
      wrap.className = "bt-code-block";
      var header = document.createElement("div");
      header.className = "bt-code-header";
      var label = document.createElement("span");
      label.textContent = (lang || "text").toLowerCase();
      var copyBtn = document.createElement("button");
      copyBtn.type = "button";
      copyBtn.textContent = "Copy";
      copyBtn.addEventListener("click", function (txt) {
        return function () {
          try { navigator.clipboard.writeText(txt); copyBtn.textContent = "Copied"; setTimeout(function(){ copyBtn.textContent = "Copy"; }, 1200); } catch (e) {}
        };
      }(code.textContent));
      header.appendChild(label);
      header.appendChild(copyBtn);
      var body = document.createElement("div");
      body.className = "bt-code-body";
      pre.parentNode.insertBefore(wrap, pre);
      wrap.appendChild(header);
      body.appendChild(pre);
      wrap.appendChild(body);
    }
  }

  /** Hide expanded thinking / mock activity rows that still slip through */
  function collapseThinking() {
    // Mark and collapse thinking/reasoning activity panels (Replit-style).
    // Never leave private chain-of-thought expanded by default.
    var panels = document.querySelectorAll(
      ".mb-2.min-w-0.overflow-hidden.rounded-xl.border, [class*='activity'], details, [data-kind]"
    );
    for (var i = 0; i < panels.length; i++) {
      var panel = panels[i];
      var text = (panel.textContent || "").toLowerCase();
      var titleEl = panel.querySelector("button span, summary, [data-title], .bt-activity-header");
      var title = titleEl ? (titleEl.textContent || "").toLowerCase() : text.slice(0, 80);
      var isThink = title.indexOf("reasoning") >= 0 || title.indexOf("thinking") >= 0
        || title.indexOf("thought") >= 0 || title.indexOf("chain of thought") >= 0;
      var isTool = panel.getAttribute("data-kind") === "tool"
        || title.indexOf("running") >= 0 || title.indexOf("tool") >= 0
        || title.indexOf("command") >= 0 || title.indexOf("search") >= 0;
      if (isThink) {
        panel.setAttribute("data-bt-activity", "thinking");
        if (!panel.getAttribute("data-bt-open")) panel.setAttribute("data-bt-open", "0");
        // Collapse <details>
        if (panel.tagName === "DETAILS") panel.open = false;
        // Click-to-expand header if expanded body visible
        var btn = panel.querySelector("button");
        var body = panel.querySelector("[class*='content'], [class*='body'], pre, code");
        if (btn && body && !panel.__btWired) {
          panel.__btWired = true;
          btn.addEventListener("click", function (p) {
            return function () {
              var open = p.getAttribute("data-bt-open") === "1";
              p.setAttribute("data-bt-open", open ? "0" : "1");
            };
          }(panel));
        }
        // Hide long monologue blocks that look like CoT dumps
        var blocks = panel.querySelectorAll("p, div, pre");
        for (var j = 0; j < blocks.length; j++) {
          var bt = (blocks[j].textContent || "").trim();
          if (bt.length > 400 && /i need to|let me think|my reasoning|chain-of-thought/i.test(bt)) {
            blocks[j].classList.add("bt-cot-hidden");
          }
        }
      } else if (isTool) {
        panel.setAttribute("data-bt-activity", "tool");
        if (!panel.getAttribute("data-bt-open")) panel.setAttribute("data-bt-open", "0");
      }
    }
  }

  function stripMockActivity() {
    document.querySelectorAll("[data-kind], [class*='activity'], [class*='Activity']").forEach(function (el) {
      var title = (el.getAttribute("data-title") || el.textContent || "").trim();
      // Only hide pure mock status lines — never hide real tool rows with code
      if (isMockTitle(title) && !el.querySelector("pre, code") && !(el.getAttribute("data-kind") === "tool")) {
        el.setAttribute("data-bt-mock-activity", "1");
        el.style.display = "none";
      }
    });
    // Hide empty "Working…" status chips in stream UI
    document.querySelectorAll("div, span").forEach(function (el) {
      if (el.children.length) return;
      var t = (el.textContent || "").trim().toLowerCase();
      if (t === "working…" || t === "working..." || t === "working on it…" || t === "working") {
        el.style.display = "none";
      }
    });
  }

  /* ---- GPU badge / quota ---- */
  function findGpuButton() {
    return Array.from(document.querySelectorAll("button")).find(function (b) {
      return /GPU\s*(ON|OFF|\d+%)/i.test(b.textContent || "") ||
        (b.getAttribute("aria-label") || "").toLowerCase().indexOf("kaggle gpu") >= 0;
    }) || null;
  }
  function paintGpu(st) {
    if (!st) return;
    _lastGpuStatus = st;
    var status = String(st.status || "").toUpperCase();
    var online = ["ONLINE", "MODEL_READY_AND_WARMED", "HEARTBEAT_ONLINE", "MODEL_READY", "MODEL_READY_COLD", "TUNNEL_ONLINE"];
    var active = !!(st.active || online.indexOf(status) >= 0);
    var booting = !!st.booting || /BOOTING|STARTING|DOWNLOAD|WARMING|ALLOCAT/i.test(status);
    if (/ERROR|TUNNEL_ERROR|FAILED|DEAD/i.test(status)) { active = false; booting = false; }
    var pct = st.progress_pct || 0;
    var label = active ? "GPU ON" : (booting ? ("GPU " + pct + "%") : "GPU OFF");
    document.documentElement.setAttribute("data-gpu", active ? "on" : (booting ? "boot" : "off"));
    var btn = findGpuButton();
    if (btn) {
      btn.classList.add("bt-gpu-btn");
      btn.classList.toggle("bt-gpu-live", active);
      btn.classList.toggle("bt-gpu-off", !active && !booting);
      Array.from(btn.querySelectorAll("span")).forEach(function (s) {
        if (/GPU/i.test(s.textContent || "")) s.textContent = label;
      });
      btn.title = [st.display_status || st.status || "", st.gpu_info || ""].filter(Boolean).join(" | ");
    }
    var q = st.quota;
    var host = document.getElementById("bt-quota-chip");
    if (!host) {
      host = document.createElement("span");
      host.id = "bt-quota-chip";
      host.title = "Kaggle GPU weekly quota";
      if (btn && btn.parentElement) btn.parentElement.insertBefore(host, btn);
      else return;
    }
    if (!q) { host.textContent = "Quota …"; return; }
    var used = typeof q.used_hours === "number" ? q.used_hours : 0;
    var total = typeof q.total_hours === "number" ? q.total_hours : 30;
    var pctU = typeof q.used_pct === "number" ? q.used_pct : 0;
    var color = pctU > 85 ? "#ef4444" : (pctU > 60 ? "#f59e0b" : "#10b981");
    host.innerHTML = "<span>GPU " + used.toFixed(1) + "h/" + total.toFixed(0) + "h</span>" +
      '<span class="bt-q-bar"><i style="width:' + Math.max(2, Math.min(100, pctU)) + "%;background:" + color + '"></i></span>';
  
    paintQuota(st);
    refreshGpuPanel(st);
  }
  var _lastGpuStatus = null;

  function pollGpu() {
    return fetch("/api/kaggle-gpu/status", { credentials: "include" })
      .then(function (r) { return r.ok ? r.json() : null; })
      .then(function (st) { if (st) paintGpu(st); })
      .catch(function () {});
  }


  var _gpuBusy = false;
  var _gpuPollTimer = null;

  
  var _lastGpuStatus = null;

  function escHtml(s) {
    return String(s || "").replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
  }

  function paintQuota(st) {
    var q = (st && st.quota) || null;
    var host = document.getElementById("bt-quota-chip");
    if (!host) {
      host = document.createElement("span");
      host.id = "bt-quota-chip";
      host.title = "Kaggle GPU weekly quota";
      var btn = findGpuButton();
      if (btn && btn.parentElement) btn.parentElement.insertBefore(host, btn);
      else return;
    }
    if (!q) { host.textContent = "Quota …"; return; }
    var used = (typeof q.used_hours === "number") ? q.used_hours : 0;
    var total = (typeof q.total_hours === "number") ? q.total_hours : 30;
    var pct = (typeof q.used_pct === "number") ? q.used_pct : 0;
    var color = pct > 85 ? "#ef4444" : (pct > 60 ? "#f59e0b" : "#10b981");
    host.innerHTML =
      "<span>GPU " + used.toFixed(1) + "h/" + total.toFixed(0) + "h</span>" +
      '<span class="bt-q-bar"><i style="width:' + Math.max(2, Math.min(100, pct)) + "%;background:" + color + '"></i></span>';
  }

  function ensureGpuPanel() {
    if (document.getElementById("bt-gpu-panel")) return;
    var panel = document.createElement("div");
    panel.id = "bt-gpu-panel";
    panel.setAttribute("role", "dialog");
    panel.setAttribute("aria-label", "GPU status");
    panel.innerHTML =
      '<div class="bt-gpu-panel-card">' +
      '  <header class="bt-gpu-panel-head">' +
      '    <div><strong>Kaggle GPU</strong><span class="bt-gpu-panel-sub" id="bt-gpu-panel-sub">Status</span></div>' +
      '    <button type="button" class="bt-menu-btn" id="bt-gpu-panel-close" aria-label="Close">×</button>' +
      '  </header>' +
      '  <div class="bt-gpu-panel-body" id="bt-gpu-panel-body"><p class="bt-gpu-note">Loading status…</p></div>' +
      '  <div class="bt-gpu-panel-actions">' +
      '    <button type="button" class="bt-hdr-btn" id="bt-gpu-panel-refresh">Refresh</button>' +
      '    <button type="button" class="bt-hdr-btn bt-primary" id="bt-gpu-panel-toggle">Turn on</button>' +
      '  </div>' +
      '</div>';
    document.body.appendChild(panel);
    document.getElementById("bt-gpu-panel-close").addEventListener("click", function () { closeGpuPanel(); });
    panel.addEventListener("click", function (e) { if (e.target === panel) closeGpuPanel(); });
    document.getElementById("bt-gpu-panel-refresh").addEventListener("click", function () { pollGpu(); });
    document.getElementById("bt-gpu-panel-toggle").addEventListener("click", function () { runGpuToggle(); });
    document.addEventListener("keydown", function (ev) {
      if (ev.key === "Escape") closeGpuPanel();
    });
  }

  function closeGpuPanel() {
    var panel = document.getElementById("bt-gpu-panel");
    if (panel) panel.classList.remove("bt-open");
  }

  function openGpuPanel() {
    ensureGpuPanel();
    var panel = document.getElementById("bt-gpu-panel");
    panel.classList.add("bt-open");
    if (_lastGpuStatus) refreshGpuPanel(_lastGpuStatus);
    pollGpu();
  }

  function refreshGpuPanel(st) {
    var body = document.getElementById("bt-gpu-panel-body");
    var sub = document.getElementById("bt-gpu-panel-sub");
    var toggle = document.getElementById("bt-gpu-panel-toggle");
    if (!body || !st) return;
    var status = String(st.status || "").toUpperCase();
    var online = ["ONLINE","MODEL_READY_AND_WARMED","HEARTBEAT_ONLINE","MODEL_READY","MODEL_READY_COLD","TUNNEL_ONLINE"];
    var active = !!(st.active || online.indexOf(status) >= 0);
    var booting = !!st.booting || /BOOTING|STARTING|DOWNLOAD|WARMING|ALLOCAT/i.test(status);
    var q = st.quota || {};
    var used = (typeof q.used_hours === "number") ? q.used_hours : 0;
    var total = (typeof q.total_hours === "number") ? q.total_hours : 30;
    var pct = (typeof q.used_pct === "number") ? q.used_pct : 0;
    var remain = (typeof q.remaining_hours === "number") ? q.remaining_hours : (total - used);
    var barColor = pct > 85 ? "#ef4444" : (pct > 60 ? "#f59e0b" : "#10b981");
    if (sub) sub.textContent = st.display_status || st.status || "—";
    body.innerHTML =
      '<div class="bt-gpu-row"><span>State</span><strong>' + (active ? "Online" : (booting ? "Starting" : "Offline")) + "</strong></div>" +
      '<div class="bt-gpu-row"><span>Status</span><strong>' + escHtml(st.display_status || st.status || "—") + "</strong></div>" +
      ((booting || st.progress_step) ? ('<div class="bt-gpu-row"><span>Progress</span><strong>' + (st.progress_pct || 0) + "% — " + escHtml(st.progress_step || "") + "</strong></div>") : "") +
      '<div class="bt-gpu-row"><span>Model</span><strong>' + escHtml(st.model || "—") + "</strong></div>" +
      '<div class="bt-gpu-row"><span>Hardware</span><strong>' + escHtml(st.gpu_info || "—") + "</strong></div>" +
      '<div class="bt-gpu-row"><span>Kernel</span><strong>' + escHtml(st.kaggle_kernel || "—") + "</strong></div>" +
      (st.tunnel_url ? ('<div class="bt-gpu-row"><span>Tunnel</span><strong class="bt-gpu-mono">' + escHtml(st.tunnel_url) + "</strong></div>") : "") +
      '<div class="bt-gpu-quota">' +
      '  <div class="bt-gpu-quota-top"><span>Weekly quota</span><span>' + used.toFixed(1) + "h / " + total.toFixed(0) + "h · " + remain.toFixed(1) + "h left</span></div>" +
      '  <div class="bt-q-bar bt-q-bar-lg"><i style="width:' + Math.max(2, Math.min(100, pct)) + "%;background:" + barColor + '"></i></div>' +
      "</div>" +
      (st.auto_off ? ('<div class="bt-gpu-note">' + escHtml(st.auto_off.reason || "") + "</div>") : "");
    if (toggle) {
      toggle.textContent = (active || booting) ? "Turn off" : "Turn on";
      toggle.disabled = false;
    }
  }

  function runGpuToggle() {
    if (typeof _gpuBusy !== "undefined" && _gpuBusy) return;
    _gpuBusy = true;
    var st = _lastGpuStatus || {};
    var status = String(st.status || "").toUpperCase();
    var online = ["ONLINE","MODEL_READY_AND_WARMED","HEARTBEAT_ONLINE","MODEL_READY","MODEL_READY_COLD","TUNNEL_ONLINE"];
    var active = !!(st.active || online.indexOf(status) >= 0);
    var booting = !!st.booting || /BOOTING|STARTING|DOWNLOAD|WARMING|ALLOCAT/i.test(status);
    var isOn = active || booting;
    var endpoint = isOn ? "/api/kaggle-gpu/turn-off?reason=user" : "/api/kaggle-gpu/turn-on";
    var toggle = document.getElementById("bt-gpu-panel-toggle");
    if (toggle) { toggle.disabled = true; toggle.textContent = isOn ? "Stopping…" : "Starting…"; }
    var btn = findGpuButton();
    if (btn) {
      Array.from(btn.querySelectorAll("span")).forEach(function (s) {
        if (/GPU/i.test(s.textContent || "")) s.textContent = isOn ? "GPU …" : "GPU starting…";
      });
    }
    fetch(endpoint, { method: "POST", credentials: "include", headers: authHeaders() })
      .then(function () { return pollGpu(); })
      .catch(function () {})
      .then(function () {
        _gpuBusy = false;
        if (toggle) toggle.disabled = false;
      });
  }


  function wireGpuButton() {
    var btn = findGpuButton();
    if (!btn || btn.dataset.btGpuWired === "1") return;
    btn.dataset.btGpuWired = "1";
    btn.style.cursor = "pointer";
    btn.setAttribute("aria-haspopup", "dialog");
    btn.addEventListener("click", function (ev) {
      ev.preventDefault();
      ev.stopPropagation();
      openGpuPanel();
    });
  }

  /* ---- Clear blocking overlays ---- */
  function clearBlockingOverlays() {
    if (!document.body.classList.contains("bt-history-open")) {
      var bd = document.getElementById("bt-history-backdrop");
      if (bd) {
        bd.style.pointerEvents = "none";
        bd.style.display = "none";
      }
    }
    if (!document.body.classList.contains("bt-nav-open")) {
      // Remove any leftover pseudo-overlays from prior CSS
      document.querySelectorAll("[data-bt-nav-backdrop]").forEach(function (el) { el.remove(); });
    }
    // Ensure chat controls are clickable
    document.body.style.pointerEvents = "";
    var root = document.getElementById("root");
    if (root) root.style.pointerEvents = "";
  }

  /* ---- History panel ---- */
  function ensureHistoryPanel() {
    if (document.getElementById("bt-history-panel")) return;
    var bd = document.createElement("div");
    bd.id = "bt-history-backdrop";
    bd.style.cssText = "position:fixed;inset:0;z-index:80;background:rgba(0,0,0,0.45);display:none;pointer-events:none;";
    bd.addEventListener("click", function () {
      document.body.classList.remove("bt-history-open");
      clearBlockingOverlays();
    });
    document.body.appendChild(bd);

    var panel = document.createElement("aside");
    panel.id = "bt-history-panel";
    panel.setAttribute("aria-label", "Chat history");
    // Inline fixed styles so panel never drops into page flow under the composer
    panel.style.cssText = "position:fixed;top:0;right:0;bottom:0;width:min(280px,92vw);max-width:320px;z-index:90;" +
      "background:#121214;border-left:1px solid rgba(255,255,255,0.08);display:flex;flex-direction:column;" +
      "transform:translateX(105%);transition:transform 0.2s ease;box-shadow:-8px 0 32px rgba(0,0,0,0.4);";
    panel.innerHTML =
      '<div class="bt-hist-head" style="display:flex;align-items:center;gap:8px;padding:12px 14px;border-bottom:1px solid rgba(255,255,255,0.08);flex-shrink:0">' +
      '<h2 style="flex:1;margin:0;font-size:14px;font-weight:600;color:#ececef">Chats</h2>' +
      '<button type="button" class="bt-hdr-btn bt-hdr-newchat" id="bt-hist-new" title="New chat">' + ICONS.plus + " New</button>" +
      '<button type="button" class="bt-menu-btn" id="bt-hist-close" title="Close" aria-label="Close history">' + ICONS.close + "</button>" +
      "</div>" +
      '<div class="bt-hist-list" id="bt-hist-list" style="flex:1;overflow-y:auto;padding:8px"><p style="color:#a1a1aa;font-size:12px;padding:12px">Loading…</p></div>';
    document.body.appendChild(panel);

    document.getElementById("bt-hist-close").addEventListener("click", function () {
      document.body.classList.remove("bt-history-open");
      var p = document.getElementById("bt-history-panel");
      var b = document.getElementById("bt-history-backdrop");
      if (p) p.style.transform = "translateX(105%)";
      if (b) { b.style.display = "none"; b.style.pointerEvents = "none"; }
      clearBlockingOverlays();
    });
    document.getElementById("bt-hist-new").addEventListener("click", function () {
      document.body.classList.remove("bt-history-open");
      clearBlockingOverlays();
      if (typeof window.__btNewChat === "function") {
        try { window.__btNewChat(); return; } catch (e) {}
      }
      var btn = document.querySelector('button[aria-label*="Start a new chat" i]');
      if (btn) btn.click();
    });
  }


  /** Load a session instantly via HermesStudio's real loader — no full page reload. */
  function openSessionFast(sessionId, title) {
    if (!sessionId) return;
    // Close drawer immediately for snappy UX
    document.body.classList.remove("bt-history-open");
    var _p = document.getElementById("bt-history-panel");
    var _b = document.getElementById("bt-history-backdrop");
    if (_p) _p.style.transform = "translateX(105%)";
    if (_b) { _b.style.display = "none"; _b.style.pointerEvents = "none"; }
    clearBlockingOverlays();

    var onChat = /\/chat(\/|$|\?)/.test(location.pathname) || location.pathname === "/";
    // If not on the chat route, go there with session query (HermesStudio will pick it up)
    if (!onChat && !location.pathname.startsWith("/chat")) {
      window.location.href = "/chat?session=" + encodeURIComponent(sessionId);
      return;
    }

    // Update URL without reload
    try {
      var u = new URL(window.location.href);
      u.pathname = "/chat";
      u.searchParams.set("session", sessionId);
      history.pushState({ session: sessionId }, "", u.toString());
    } catch (e) {}

    function tryLoad() {
      if (typeof window.__btLoadSession === "function") {
        try {
          window.__btLoadSession(sessionId);
          return true;
        } catch (e) {
          console.warn("btLoadSession failed", e);
        }
      }
      var match =
        document.querySelector('button[data-session-id="' + CSS.escape(sessionId) + '"]') ||
        Array.from(document.querySelectorAll("button[data-session-id]")).find(function (b) {
          return b.getAttribute("data-session-id") === sessionId;
        });
      if (match) {
        match.click();
        return true;
      }
      return false;
    }

    if (tryLoad()) return;

    // Wait briefly for HermesStudio to mount and expose the loader
    var tries = 0;
    (function waitLoader() {
      tries += 1;
      if (tryLoad()) return;
      if (tries < 25) {
        setTimeout(waitLoader, 80);
        return;
      }
      // Hard navigation as last resort
      window.location.href = "/chat?session=" + encodeURIComponent(sessionId);
    })();
  }

  function loadHistory() {
    var list = document.getElementById("bt-hist-list");
    if (!list) return;
    fetch("/api/studio/sessions?limit=80", { credentials: "include", headers: authHeaders() })
      .then(function (r) { return r.ok ? r.json() : { sessions: [] }; })
      .then(function (data) {
        var sessions = (data && data.sessions) || [];
        if (!sessions.length) {
          list.innerHTML = '<p style="color:#a1a1aa;font-size:12px;padding:12px">No conversations yet.</p>';
          return;
        }
        list.innerHTML = "";
        var pinIds = [];
        try { pinIds = JSON.parse(localStorage.getItem("bt_pinned_sessions") || "[]"); } catch (e) {}
        var titleMap = {};
        try { titleMap = JSON.parse(localStorage.getItem("bt_session_titles") || "{}"); } catch (e) {}
        sessions.sort(function (a, b) {
          var ap = pinIds.indexOf(a.id); var bp = pinIds.indexOf(b.id);
          var aPinned = ap >= 0; var bPinned = bp >= 0;
          if (aPinned !== bPinned) return aPinned ? -1 : 1;
          if (aPinned && bPinned) return ap - bp;
          return (b.last_at || 0) - (a.last_at || 0);
        });
        var pinnedRendered = false; var restRendered = false;
        sessions.forEach(function (s) {
          var isPinned = pinIds.indexOf(s.id) >= 0;
          if (isPinned && !pinnedRendered) {
            var sec = document.createElement("div"); sec.className = "bt-hist-section"; sec.textContent = "Pinned";
            list.appendChild(sec); pinnedRendered = true;
          }
          if (!isPinned && !restRendered) {
            var sec2 = document.createElement("div"); sec2.className = "bt-hist-section"; sec2.textContent = "Recent";
            list.appendChild(sec2); restRendered = true;
          }
          var row = document.createElement("div");
          row.className = "bt-hist-item" + (isPinned ? " is-pinned" : "");
          row.dataset.sid = s.id;
          var titleBtn = document.createElement("button");
          titleBtn.type = "button";
          titleBtn.className = "bt-hist-title";
          var displayTitle = titleMap[s.id] || s.title || "New chat";
          titleBtn.textContent = (isPinned ? "📌 " : "") + displayTitle;
          titleBtn.addEventListener("click", function () {
            openSessionFast(s.id, s.title);
          });
          var meta = document.createElement("div");
          meta.className = "bt-hist-meta";
          meta.textContent = (s.message_count || 0) + " msg";
          var left = document.createElement("div");
          left.style.cssText = "flex:1;min-width:0;display:flex;flex-direction:column";
          left.appendChild(titleBtn);
          left.appendChild(meta);

          var menuBtn = document.createElement("button");
          menuBtn.type = "button";
          menuBtn.className = "bt-menu-btn";
          menuBtn.setAttribute("aria-label", "Chat options");
          menuBtn.innerHTML = ICONS.menu;
          menuBtn.addEventListener("click", function (ev) {
            ev.preventDefault();
            ev.stopPropagation();
            openItemMenu(row, s);
          });

          row.appendChild(left);
          row.appendChild(menuBtn);
          list.appendChild(row);
        });
      })
      .catch(function () {
        list.innerHTML = '<p style="color:#a1a1aa;font-size:12px;padding:12px">Could not load history.</p>';
      });
  }

  function closeAllMenus() {
    document.querySelectorAll(".bt-ctx-menu").forEach(function (m) { m.remove(); });
  }

  function openItemMenu(row, session) {
    closeAllMenus();
    var menu = document.createElement("div");
    menu.className = "bt-ctx-menu";
    menu.setAttribute("role", "menu");

    function pins() {
      try { return JSON.parse(localStorage.getItem("bt_pinned_sessions") || "[]"); } catch (e) { return []; }
    }
    function savePins(arr) {
      localStorage.setItem("bt_pinned_sessions", JSON.stringify(arr));
    }
    var pinned = pins().indexOf(session.id) >= 0;

    var pinBtn = document.createElement("button");
    pinBtn.type = "button";
    pinBtn.innerHTML = "<span>" + (pinned ? "Unpin" : "Pin") + "</span>";
    pinBtn.addEventListener("click", function (ev) {
      ev.stopPropagation();
      closeAllMenus();
      var arr = pins().filter(function (id) { return id !== session.id; });
      if (!pinned) arr.unshift(session.id);
      savePins(arr);
      loadHistory();
    });

    var ren = document.createElement("button");
    ren.type = "button";
    ren.innerHTML = "<span>Rename</span>";
    ren.addEventListener("click", function (ev) {
      ev.stopPropagation();
      closeAllMenus();
      var next = window.prompt("Rename chat", session.title || "New chat");
      if (next == null) return;
      next = String(next).trim();
      if (!next) return;
      fetch("/api/studio/sessions/" + encodeURIComponent(session.id), {
        method: "PATCH", credentials: "include", headers: authHeaders(),
        body: JSON.stringify({ title: next })
      }).then(function (r) {
        if (!r.ok) {
          // local fallback map
          try {
            var map = JSON.parse(localStorage.getItem("bt_session_titles") || "{}");
            map[session.id] = next;
            localStorage.setItem("bt_session_titles", JSON.stringify(map));
          } catch (e) {}
        }
        loadHistory();
      }).catch(function () {
        try {
          var map = JSON.parse(localStorage.getItem("bt_session_titles") || "{}");
          map[session.id] = next;
          localStorage.setItem("bt_session_titles", JSON.stringify(map));
        } catch (e) {}
        loadHistory();
      });
    });

    var arch = document.createElement("button");
    arch.type = "button";
    arch.innerHTML = ICONS.archive + "<span>Archive</span>";
    arch.addEventListener("click", function (ev) {
      ev.stopPropagation();
      closeAllMenus();
      fetch("/api/studio/sessions/" + encodeURIComponent(session.id) + "/archive", {
        method: "POST", credentials: "include", headers: authHeaders()
      }).then(function () { loadHistory(); }).catch(function () { loadHistory(); });
    });

    var del = document.createElement("button");
    del.type = "button";
    del.className = "bt-danger";
    del.innerHTML = ICONS.trash + "<span>Delete</span>";
    del.addEventListener("click", function (ev) {
      ev.stopPropagation();
      closeAllMenus();
      confirmDelete(session);
    });

    menu.appendChild(pinBtn);
    menu.appendChild(ren);
    menu.appendChild(arch);
    menu.appendChild(del);
    row.style.position = "relative";
    row.appendChild(menu);
    // close on outside click
    setTimeout(function () {
      function onDoc(ev) {
        if (!menu.contains(ev.target) && ev.target !== row.querySelector(".bt-menu-btn")) {
          closeAllMenus();
          document.removeEventListener("click", onDoc, true);
        }
      }
      document.addEventListener("click", onDoc, true);
    }, 0);
  }

  function confirmDelete(session) {
    var modal = document.getElementById("bt-confirm");
    if (!modal) {
      modal = document.createElement("div");
      modal.id = "bt-confirm";
      modal.innerHTML =
        '<div class="bt-sp-card" role="dialog" aria-modal="true">' +
        "<header><span>Delete conversation</span>" +
        '<button type="button" class="bt-menu-btn" id="bt-confirm-x" aria-label="Cancel">' + ICONS.close + "</button></header>" +
        "<p>Permanently delete <strong id=\"bt-confirm-title\"></strong> and all of its messages? This cannot be undone.</p>" +
        '<div class="bt-sp-foot"><span class="bt-sp-status"></span>' +
        '<button type="button" id="bt-confirm-cancel">Cancel</button>' +
        '<button type="button" class="bt-primary" id="bt-confirm-ok" style="background:rgba(239,68,68,0.2);border-color:rgba(239,68,68,0.5);color:#fca5a5">Delete</button>' +
        "</div></div>";
      document.body.appendChild(modal);
      document.getElementById("bt-confirm-x").onclick =
      document.getElementById("bt-confirm-cancel").onclick = function () {
        modal.classList.remove("bt-open");
      };
    }
    document.getElementById("bt-confirm-title").textContent = session.title || "this chat";
    modal.classList.add("bt-open");
    document.getElementById("bt-confirm-ok").onclick = function () {
      fetch("/api/studio/sessions/" + encodeURIComponent(session.id), {
        method: "DELETE", credentials: "include", headers: authHeaders()
      }).then(function () {
        modal.classList.remove("bt-open");
        loadHistory();
      }).catch(function () {
        modal.classList.remove("bt-open");
        loadHistory();
      });
    };
  }

  /* ---- System prompt ---- */
  function ensureSystemPromptUI() {
    if (document.getElementById("bt-sp-modal")) return;
    var modal = document.createElement("div");
    modal.id = "bt-sp-modal";
    modal.innerHTML =
      '<div class="bt-sp-card" role="dialog" aria-modal="true" aria-label="System Prompt">' +
      "<header><span>System Prompt</span>" +
      '<button type="button" class="bt-menu-btn" id="bt-sp-close" aria-label="Close">' + ICONS.close + "</button></header>" +
      '<textarea id="bt-sp-text" spellcheck="false" placeholder="Enter the system prompt used by the Blackthorn agent…"></textarea>' +
      '<div class="bt-sp-foot"><span class="bt-sp-status" id="bt-sp-status">Loading…</span>' +
      '<button type="button" id="bt-sp-cancel">Cancel</button>' +
      '<button type="button" class="bt-primary" id="bt-sp-save">Save</button></div></div>';
    document.body.appendChild(modal);

    function close() { modal.classList.remove("bt-open"); }
    document.getElementById("bt-sp-close").onclick = close;
    document.getElementById("bt-sp-cancel").onclick = close;
    modal.addEventListener("click", function (e) { if (e.target === modal) close(); });

    document.getElementById("bt-sp-save").onclick = function () {
      var text = document.getElementById("bt-sp-text").value;
      var st = document.getElementById("bt-sp-status");
      st.textContent = "Saving…";
      fetch("/api/system-prompt", {
        method: "PUT",
        credentials: "include",
        headers: authHeaders(),
        body: JSON.stringify({ prompt: text })
      }).then(function (r) {
        if (!r.ok) throw new Error("save failed");
        st.textContent = "Saved — applies to new agent turns";
        window._btSpSaved = text;
      }).catch(function () {
        st.textContent = "Save failed";
      });
    };
  }

  function openSystemPrompt() {
    ensureSystemPromptUI();
    var modal = document.getElementById("bt-sp-modal");
    var ta = document.getElementById("bt-sp-text");
    var st = document.getElementById("bt-sp-status");
    st.textContent = "Loading…";
    modal.classList.add("bt-open");
    fetch("/api/system-prompt", { credentials: "include", headers: authHeaders() })
      .then(function (r) { return r.ok ? r.json() : { prompt: "" }; })
      .then(function (data) {
        ta.value = (data && data.prompt) || "";
        window._btSpSaved = ta.value;
        st.textContent = ta.value ? "Loaded — edit and save to update the agent" : "Empty — default agent prompt is used until you save one";
      })
      .catch(function () {
        st.textContent = "Could not load system prompt";
      });
  }

  function mergeHeaderControls() {
    if (!document.body.classList.contains("bt-chat-mode")) return;
    var chatHeader = null;
    document.querySelectorAll("header").forEach(function (h) {
      var t = (h.innerText || "").replace(/\s+/g, " ").trim();
      var cls = String(h.className || "");
      if (h.querySelector('button[aria-label*="conversation" i], button[aria-label*="Start a new chat" i]') ||
          (/New chat/i.test(t) && /Export|Files|☰/.test(t))) {
        h.style.setProperty("display", "none", "important");
        h.setAttribute("data-bt-hidden-studio", "1");
        return;
      }
      if ((/^Chat\b/i.test(t) || /GPU\s*(ON|OFF|\d)/i.test(t)) && !cls.includes("lg:hidden")) {
        chatHeader = h;
        h.style.removeProperty("display");
      }
    });
    if (!chatHeader) return;

    var host = document.getElementById("bt-chat-actions");
    if (!host) {
      host = document.createElement("div");
      host.id = "bt-chat-actions";
      host.style.cssText = "display:inline-flex;align-items:center;gap:6px;margin-left:10px;flex-shrink:0;";
      var titleEl = null;
      chatHeader.querySelectorAll("h1,h2,span,div").forEach(function (el) {
        if (!titleEl && /^Chat$/i.test((el.textContent || "").trim()) && el.children.length === 0) titleEl = el;
      });
      if (titleEl && titleEl.parentElement) titleEl.parentElement.insertBefore(host, titleEl.nextSibling);
      else (chatHeader.firstElementChild || chatHeader).appendChild(host);
    }
    if (!document.getElementById("bt-hist-btn")) {
      var hBtn = document.createElement("button");
      hBtn.type = "button";
      hBtn.id = "bt-hist-btn";
      hBtn.className = "bt-hdr-btn";
      hBtn.style.cssText = "height:28px;min-height:28px;padding:0 8px;font-size:12px;border-radius:7px;";
      hBtn.title = "Chat history";
      hBtn.setAttribute("aria-label", "Open chat history");
      hBtn.innerHTML = ICONS.hist;
      hBtn.addEventListener("click", function (ev) {
        ev.preventDefault();
        ev.stopPropagation();
        ensureHistoryPanel();
        document.body.classList.toggle("bt-history-open");
        var panel = document.getElementById("bt-history-panel");
        var bd = document.getElementById("bt-history-backdrop");
        if (document.body.classList.contains("bt-history-open")) {
          if (panel) panel.style.transform = "translateX(0)";
          if (bd) { bd.style.display = "block"; bd.style.pointerEvents = "auto"; }
          loadHistory();
        } else {
          if (panel) panel.style.transform = "translateX(105%)";
          if (bd) { bd.style.display = "none"; bd.style.pointerEvents = "none"; }
          clearBlockingOverlays();
        }
      });
      host.appendChild(hBtn);
    }
    if (!document.getElementById("bt-newchat-btn")) {
      var nBtn = document.createElement("button");
      nBtn.type = "button";
      nBtn.id = "bt-newchat-btn";
      nBtn.className = "bt-hdr-btn bt-hdr-newchat";
      nBtn.style.cssText = "height:28px;min-height:28px;padding:0 8px;font-size:12px;border-radius:7px;";
      nBtn.title = "New chat";
      nBtn.innerHTML = ICONS.plus + " New";
      nBtn.addEventListener("click", function (ev) {
        ev.preventDefault();
        var orig = document.querySelector('button[aria-label*="Start a new chat" i]');
        if (orig) orig.click();
      });
      host.appendChild(nBtn);
    }
    if (!document.getElementById("bt-sys-prompt-btn")) {
      var sBtn = document.createElement("button");
      sBtn.type = "button";
      sBtn.id = "bt-sys-prompt-btn";
      sBtn.className = "bt-hdr-btn";
      sBtn.style.cssText = "height:28px;min-height:28px;padding:0 8px;font-size:12px;border-radius:7px;";
      sBtn.title = "System Prompt";
      sBtn.innerHTML = ICONS.prompt + " System Prompt";
      sBtn.addEventListener("click", function (ev) {
        ev.preventDefault();
        openSystemPrompt();
      });
      host.appendChild(sBtn);
    }
  }

  function forceLayout() {
    if (!document.body.classList.contains("bt-chat-mode")) return;
    // Hide React history asides (we use custom panel)
    document.querySelectorAll("aside").forEach(function (a) {
      if (a.id === "app-sidebar" || a.id === "bt-history-panel") return;
      a.style.setProperty("display", "none", "important");
      a.style.pointerEvents = "none";
    });
    var nav = document.getElementById("app-sidebar");
    if (nav && window.innerWidth >= 1024) {
      if (document.body.classList.contains("bt-nav-open")) {
        nav.style.setProperty("width", "260px", "important");
        nav.style.setProperty("min-width", "260px", "important");
      } else {
        nav.style.setProperty("width", "56px", "important");
        nav.style.setProperty("min-width", "56px", "important");
        nav.style.setProperty("transform", "none", "important");
      }
    }
    mergeHeaderControls();
    wireGpuButton();
    stripMockActivity(); enhanceCodeBlocks(document); collapseThinking();
    if (!document.body.classList.contains("bt-history-open") &&
        !document.body.classList.contains("bt-nav-open")) {
      clearBlockingOverlays();
    }
  }

  function wireSidebar() {
    document.addEventListener("click", function (ev) {
      var t = ev.target;
      if (!t || !t.closest) return;
      var navBtn = t.closest('button[aria-controls="app-sidebar"], button[aria-label*="navigation" i], button[aria-label*="Open nav" i], button[aria-label*="Close nav" i], button[aria-label*="Collapse" i]');
      if (navBtn) {
        document.body.classList.toggle("bt-nav-open");
        forceLayout();
        if (!document.body.classList.contains("bt-nav-open")) clearBlockingOverlays();
        return;
      }
      // Outside click closes history
      if (document.body.classList.contains("bt-history-open")) {
        var panel = document.getElementById("bt-history-panel");
        if (panel && !panel.contains(t) && !t.closest("#bt-hist-btn") && !t.closest("#bt-history-backdrop")) {
          // backdrop handles most cases
        }
      }
      if (!t.closest(".bt-ctx-menu") && !t.closest(".bt-menu-btn")) closeAllMenus();
    }, true);

    document.addEventListener("keydown", function (ev) {
      if (ev.key === "Escape") {
        document.body.classList.remove("bt-nav-open");
        document.body.classList.remove("bt-history-open");
        closeAllMenus();
        var sp = document.getElementById("bt-sp-modal");
        if (sp) sp.classList.remove("bt-open");
        var cf = document.getElementById("bt-confirm");
        if (cf) cf.classList.remove("bt-open");
        clearBlockingOverlays();
      }
    });
  }

  function wireMobileEnter() {
    if (window._btEnterWired) return;
    window._btEnterWired = true;
    document.addEventListener("keydown", function (ev) {
      if (ev.key !== "Enter" && ev.keyCode !== 13) return;
      var t = ev.target;
      if (!t || t.tagName !== "TEXTAREA") return;
      var isMobile = window.innerWidth < 768 ||
        window.matchMedia("(pointer: coarse)").matches ||
        /Mobi|Android|iPhone|iPad/i.test(navigator.userAgent || "");
      if (!isMobile) return;
      ev.preventDefault();
      ev.stopPropagation();
      ev.stopImmediatePropagation();
      var start = t.selectionStart || 0;
      var end = t.selectionEnd || 0;
      var val = t.value || "";
      var next = val.slice(0, start) + "\n" + val.slice(end);
      var setter = Object.getOwnPropertyDescriptor(window.HTMLTextAreaElement.prototype, "value");
      if (setter && setter.set) setter.set.call(t, next);
      else t.value = next;
      t.selectionStart = t.selectionEnd = start + 1;
      t.dispatchEvent(new Event("input", { bubbles: true }));
    }, true);
  }

  function watchRoute() {
    var last = location.pathname;
    setInterval(function () {
      if (location.pathname !== last) {
        last = location.pathname;
        syncChatMode();
        forceLayout();
      }
    }, 400);
    var _push = history.pushState;
    history.pushState = function () {
      _push.apply(this, arguments);
      setTimeout(function () { syncChatMode(); forceLayout(); }, 0);
    };
  }

  var n = 0;
  function tick() {
    syncChatMode();
    pollGpu().then(function () {
      forceLayout();
      stripMockActivity(); enhanceCodeBlocks(document); collapseThinking();
      n += 1;
      setTimeout(tick, n < 20 ? 1500 : 8000);
    });
  }

  
  function forceWireHeaderButtons() {
    // System Prompt — bind any visible matching control
    document.querySelectorAll("button, a").forEach(function (el) {
      var t = (el.textContent || "").replace(/\s+/g, " ").trim();
      if (/^System Prompt$/i.test(t) && !el.__btSpWired) {
        el.__btSpWired = true;
        el.addEventListener("click", function (ev) {
          ev.preventDefault();
          ev.stopPropagation();
          openSystemPrompt();
        }, true);
      }
      if ((el.id === "bt-hist-btn" || /Open chat history/i.test(el.getAttribute("aria-label") || "")) && !el.__btHistWired) {
        el.__btHistWired = true;
        el.addEventListener("click", function (ev) {
          ev.preventDefault();
          ev.stopPropagation();
          ensureHistoryPanel();
          document.body.classList.add("bt-history-open");
          var panel = document.getElementById("bt-history-panel");
          var bd = document.getElementById("bt-history-backdrop");
          if (panel) panel.style.transform = "translateX(0)";
          if (bd) { bd.style.display = "block"; bd.style.pointerEvents = "auto"; }
          loadHistory();
        }, true);
      }
    });
  }

  function boot() {
    ensureGpuPanel();
    wireGpuButton();
    syncChatMode();
    ensureHistoryPanel();
    ensureSystemPromptUI();
    forceLayout();
    wireSidebar();
    wireMobileEnter();
    watchRoute();
    tick();
    forceWireHeaderButtons();
    setInterval(forceWireHeaderButtons, 1500);
    var _btMoTimer = null;
    var mo = new MutationObserver(function () {
      if (_btMoTimer) return;
      _btMoTimer = setTimeout(function () {
        _btMoTimer = null;
        stripMockActivity(); enhanceCodeBlocks(document); collapseThinking();
        syncChatMode();
        forceLayout();
      }, 120);
    });
    mo.observe(document.documentElement, { childList: true, subtree: true });
  }

  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", boot);
  else boot();
})();
