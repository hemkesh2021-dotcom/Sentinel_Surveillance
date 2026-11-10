/* Sentinel status pages: in-place updates (D73) and the live player (D75).

   Served from this origin only (CSP: script-src 'self'; connect-src 'self'). No third-party code. Without it the
   pages still work, read-only: they show what the server rendered and turn NOT CURRENT after the watchdog window;
   reload to update.

   - Every fragment is rendered and escaped by the server; this script only places it, keyed, so that elements stay
     where they are (focus, expanded sections, text that did not change, the selected incident) and the whole page is
     never rebuilt or reloaded.
   - The D71 timers (the CSS watchdog) restart only after a document that was received with HTTP 200, parsed and
     validated, and is newer than the last one (server instance and sequence). Their window counts from when that
     document's request was sent, so a delayed answer gets less time, never more, and one that arrives after the
     window is not applied. Nothing else restarts them: not a failed or repeated answer, not a picture, not the chat.
   - One request at a time; a request is abandoned after the window; failures back off (2, 4, 8, at most 15 s). A
     hidden tab is not polled, and its claims expire when it is hidden and again when it is shown; on return the page
     updates at once.
   - The live picture (D75) is the player's, apart from the status updates: it asks GET /live/frame for one picture
     at a time, the newest newer than the one it has, as soon as it has the last; it decodes it off the page and draws
     it on a canvas, so the picture never blanks. Every picture restarts the picture's own timer (the CSS watchdog
     for the picture and the LIVE label): its window is the device's stale threshold minus the picture's age when
     sent, counted from when this browser asked for it, so a picture is never shown as live once its frame could be
     older than the threshold. A picture that cannot be had leaves the last one in place, dimmed and labelled as an
     old frame with its time. Nothing else restarts that timer: not a status update, not the chat, not an animation.
     The player stops asking while the tab is hidden or the picture is out of sight, and backs off on failures.
   - Chat (D74, only where the runtime offers it): one question at a time, about one identified frame; answers are
     shown as text, labelled with that frame's time and age, and never touch the timers above. The conversation and
     the draft stay in this tab; the token in the form belongs to the server that sent this page.
   - Navigation and appearance (D96): the section in view is marked in the sidebar; the sidebar collapses to a rail
     (wide screens) or opens as a drawer (narrow ones: Escape or the scrim closes it, focus stays inside while open and
     returns to the menu button); the camera's focus view shows the picture and recognition only; the loaded
     incidents can be filtered by status and severity (only what this page loaded, and it says so); theme, palette and
     density apply in place and are saved in this browser. None of these restarts a timer, reloads the page or touches
     the conversation, the draft, the selected incident or an expanded section.
   - The controls' states (session 114): each control gets its edge and pulse ring when first used (decorations that
     updates keep); a control is marked pending (aria-busy) only while its own request is in flight. */
(() => {
  "use strict";
  const root = document.documentElement;
  const SOURCE = root.dataset.source;
  const EVERY = Number(root.dataset.refreshMs);
  const WINDOW = Number(root.dataset.currentMs);
  const KIND = root.dataset.page;
  if (!SOURCE || !SOURCE.startsWith("/") || SOURCE.startsWith("//") || !(EVERY > 0) || !(WINDOW > 0)) return;
  let source = SOURCE;  // the appearance chosen in place is dropped from it (D96), nothing else changes it
  const MAX_TEXT = 600000;
  const MAX_WAIT = 15000;
  const INSTANCE = /^[0-9a-f]{16}$/;
  const TOKEN = /^[0-9a-f]{32}$/;
  const INCIDENT = /^[A-Za-z0-9][A-Za-z0-9_-]{0,127}$/;
  const state = {
    instance: root.dataset.instance || null, sequence: Number(root.dataset.sequence || "-1"),
    last: Object.create(null), timer: 0, inflight: null, failures: 0, token: null,
    selected: null, detail: null, detailTimer: 0,
    // D96: the appearance the address chose (kept in links until a choice is made in place), and whether one was
    appearanceQuery: root.hasAttribute("data-pref") ? (SOURCE.split("?")[1] || "").split("&")
      .filter((part) => /^(theme|palette|density)=[a-z-]{1,40}$/.test(part)).join("&") : "",
    chosenInPlace: false,
  };
  root.classList.add("js");
  let refreshAsk = () => {};  // the chat's Ask button follows chat's availability, which updates with the page
  let refreshControls = () => {};  // D99: the feature switches follow the page's currency and the runtime's state
  let player = null;  // the live player, on the status page with the live view on

  // ------------------------------------------------------------------ the watchdog timers

  // The delay is given to each timed element itself (animation-delay, not inherited), never as a custom property on
  // the root: a root property restyles the whole document on every update, which on a slow device delays the live
  // picture (D96, measured). The stylesheet's var(--fresh-delay, 0s) stays the delay without script.
  function delayTimers(ms) {
    const value = `${-Math.round(ms)}ms`;
    for (const node of document.querySelectorAll(".cur, .old, .report")) {
      if (node.style.animationDelay !== value) node.style.animationDelay = value;
    }
  }

  function restart(name, sentAt) {
    // A new animation name restarts every timer of this kind; the negative delay ends the window WINDOW ms after
    // the request was sent (a bound on the age of what it brought, on this browser's monotonic clock).
    delayTimers(Math.max(0, performance.now() - sentAt));
    root.dataset[name] = root.dataset[name] === "1" ? "0" : "1";
  }

  function expireNow() {  // after a freeze or a back-forward restore: nothing is current until an update arrives
    state.currentUntil = 0;
    delayTimers(10 * WINDOW);
    root.dataset.fresh = root.dataset.fresh === "1" ? "0" : "1";
    if (player) player.expire();
  }

  const navigation = performance.getEntriesByType ? performance.getEntriesByType("navigation")[0] : null;
  const requested = navigation && navigation.requestStart > 0 ? navigation.requestStart : 0;
  restart("fresh", requested);
  state.currentUntil = requested + WINDOW;  // the served document's own window (D71), for the feature switches (D99)

  // ------------------------------------------------------------------ keyed placement of server fragments

  function key(node) {
    return node.nodeType === 1 ? node.getAttribute("data-key") || node.id || null : null;
  }

  // A control's decorations (session 114: its edge and pulse ring, added here) are never the server's: an update
  // neither removes them nor makes one of them into a server node.
  function deco(node) {
    return node.nodeType === 1 && node.hasAttribute("data-deco");
  }

  function same(a, b) {
    return !deco(a) && a.nodeType === b.nodeType && a.nodeName === b.nodeName && key(a) === key(b);
  }

  function morph(target, source) {
    const keyed = new Map();
    for (let node = target.firstChild; node; node = node.nextSibling) {
      const k = key(node);
      if (k) keyed.set(k, node);
    }
    let cursor = target.firstChild;
    for (const node of [...source.childNodes]) {
      const k = key(node);
      let match = k ? keyed.get(k) : cursor && same(cursor, node) ? cursor : null;
      if (match && !same(match, node)) match = null;
      if (match) {
        if (match === cursor) cursor = cursor.nextSibling;
        else target.insertBefore(match, cursor);
        if (k) keyed.delete(k);
        patch(match, node);
      } else {
        target.insertBefore(document.importNode(node, true), cursor);
      }
    }
    while (cursor) {
      const next = cursor.nextSibling;
      if (!deco(cursor)) target.removeChild(cursor);
      cursor = next;
    }
  }

  function patch(current, wanted) {
    if (current.nodeType === 3 || current.nodeType === 8) {
      if (current.nodeValue !== wanted.nodeValue) current.nodeValue = wanted.nodeValue;
      return;
    }
    if (current.nodeType !== 1) return;
    const own = current.tagName === "DETAILS" ? "open" : null;  // the reader decides what stays expanded
    for (const {name} of [...current.attributes]) {
      if (name !== own && !wanted.hasAttribute(name)) current.removeAttribute(name);
    }
    for (const {name, value} of [...wanted.attributes]) {
      if (name !== own && current.getAttribute(name) !== value) current.setAttribute(name, value);
    }
    morph(current, wanted);
  }

  function place(name, html) {
    if (state.last[name] === html) return;
    const target = document.querySelector(`[data-region="${name}"]`);
    if (!target) return;
    const template = document.createElement("template");
    template.innerHTML = html;  // rendered and escaped by the server, from this origin only
    morph(target, template.content);
    state.last[name] = html;
  }

  // ------------------------------------------------------------------ validation

  function valid(doc) {
    if (!doc || typeof doc !== "object" || !doc.page || typeof doc.page !== "object") return false;
    const page = doc.page;
    if (typeof page.instance !== "string" || !INSTANCE.test(page.instance)) return false;
    if (!Number.isSafeInteger(page.sequence) || page.sequence < 0) return false;
    if (!page.regions || typeof page.regions !== "object") return false;
    for (const [name, html] of Object.entries(page.regions)) {
      if (!/^[a-z-]{1,40}$/.test(name) || typeof html !== "string") return false;
    }
    if (KIND === "status") {
      if (typeof page.frame_empty !== "string") return false;
      if (page.token !== undefined && (typeof page.token !== "string" || !TOKEN.test(page.token))) return false;
      const preview = doc.preview;
      if (preview !== undefined && (preview === null || typeof preview !== "object")) return false;
    }
    return true;
  }

  function newer(page) {
    return page.instance !== state.instance || page.sequence > state.sequence;
  }

  // ------------------------------------------------------------------ updating

  const sync = document.getElementById("sync");
  function say(text, mode) {
    if (!sync) return;
    if (sync.textContent !== text) sync.textContent = text;
    if (mode) sync.dataset.state = mode;
    else delete sync.dataset.state;
  }
  let lastApplied = null;  // when the last valid update arrived (this browser's clock), for the reconnect line

  function schedule(ms) {
    clearTimeout(state.timer);
    state.timer = setTimeout(update, ms);
  }

  function why(error) {
    if (error && error.name === "AbortError") return "no answer in time";
    if (error instanceof TypeError) return "connection failed";
    return error && error.message ? error.message : "update failed";
  }

  async function update() {
    clearTimeout(state.timer);
    if (state.inflight) return;
    if (document.visibilityState === "hidden") {
      say("Updates are paused while this tab is hidden.");
      return;
    }
    const sent = performance.now();
    const controller = new AbortController();
    state.inflight = controller;
    const timeout = setTimeout(() => controller.abort(), WINDOW);
    try {
      const response = await fetch(source, {
        cache: "no-store", credentials: "same-origin", redirect: "error", signal: controller.signal,
        headers: {Accept: "application/json"},
      });
      const type = response.headers.get("Content-Type") || "";
      if (response.status !== 200 || !type.startsWith("application/json")) throw new Error(`HTTP ${response.status}`);
      const text = await response.text();
      if (text.length > MAX_TEXT) throw new Error("answer too large");
      let doc;
      try {
        doc = JSON.parse(text);
      } catch (error) {
        throw new Error("answer not readable");
      }
      if (!valid(doc)) throw new Error("answer not valid");
      if (document.visibilityState === "hidden") return;  // arrived after the tab was hidden: asked again on return
      if (newer(doc.page)) {
        if (performance.now() - sent >= WINDOW) throw new Error("answer arrived too late");
        apply(doc, sent);
      }  // an older or repeated document is ignored: it restarts nothing
      state.failures = 0;
      say("");
      schedule(EVERY);
    } catch (error) {
      state.failures += 1;
      const wait = Math.min(EVERY * 2 ** state.failures, MAX_WAIT);
      // One steady line while reconnecting (no countdown, so assistive technology is not interrupted every retry).
      const since = lastApplied ? ` since ${lastApplied.toISOString().slice(11, 19)} UTC` : "";
      say(`Reconnecting: no current update${since} (${why(error)}). Retrying automatically; what is shown is the last `
        + "report, marked NOT CURRENT.", "reconnecting");
      schedule(wait);
    } finally {
      clearTimeout(timeout);
      if (state.inflight === controller) state.inflight = null;
    }
  }

  function apply(doc, sent) {
    if (doc.page.instance !== state.instance) state.last = Object.create(null);
    // Placed elements get their bounded delay from the restart below, in this same task (before any rendering).
    for (const [name, html] of Object.entries(doc.page.regions)) place(name, html);
    if (KIND === "status") {
      if (doc.page.token) state.token = doc.page.token;
      if (player) player.status(doc);
    }
    lastApplied = new Date();
    state.currentUntil = sent + WINDOW;  // D99: what this document says is current until then, at most
    if (state.chosenInPlace) stripLinks();  // a region asked for before an in-place choice keeps no old appearance
    filterIncidents();
    markSelected();
    state.instance = doc.page.instance;
    state.sequence = doc.page.sequence;
    root.dataset.instance = doc.page.instance;
    root.dataset.sequence = String(doc.page.sequence);
    refreshAsk();
    refreshControls();
    restart("fresh", sent);
  }

  // ------------------------------------------------------------------ small helpers

  function clock(iso) {
    const when = new Date(iso);
    return Number.isNaN(when.getTime()) ? "an unknown time" : `${when.toISOString().slice(11, 23)} UTC`;
  }

  function svgIcon(name) {
    const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
    svg.setAttribute("class", "ic");
    svg.setAttribute("aria-hidden", "true");
    svg.setAttribute("focusable", "false");
    const use = document.createElementNS("http://www.w3.org/2000/svg", "use");
    use.setAttribute("href", `#i-${name}`);
    svg.append(use);
    return svg;
  }

  function el(tag, attributes, children) {
    const node = document.createElement(tag);
    for (const [name, value] of Object.entries(attributes || {})) node.setAttribute(name, value);
    for (const child of children || []) node.append(child);
    return node;
  }

  function badge(kind, text, glyph, big) {
    return el("span", {class: `badge k-${kind}${big ? " big" : ""}`}, [svgIcon(glyph), text]);
  }

  // ------------------------------------------------------------------ the controls' edge and pulse (session 114)

  // A control gets, the first time it is hovered, focused or pending, one SVG rectangle on its own outline (its edge:
  // hover and focus send the highlight once around it, the stylesheet's trace; it circles while the control's request
  // is pending) and an empty ring for the press pulse. Both are decorations (aria-hidden, data-deco: updates keep
  // them). The rectangle's corners are the control's own, measured each time, so a density or rail change never
  // leaves them behind; a pill's corners are half its height, as the browser draws them.
  const CONTROLS = ".button, .ghost, .peek, .plain, .seg, .swatch, .icon-btn, a.inc, .back, .nav a, .switch, .field, "
    + ".pick";
  const SVG = "http://www.w3.org/2000/svg";

  function dress(control) {
    let edge = control.querySelector(":scope > .edge");
    if (!edge) {
      edge = document.createElementNS(SVG, "svg");
      for (const [name, value] of [["class", "edge"], ["aria-hidden", "true"], ["focusable", "false"],
                                   ["data-deco", ""]]) {
        edge.setAttribute(name, value);
      }
      const outline = document.createElementNS(SVG, "rect");
      for (const [name, value] of [["x", "0"], ["y", "0"], ["width", "100%"], ["height", "100%"], ["pathLength", "100"]]) {
        outline.setAttribute(name, value);
      }
      edge.append(outline);
      control.append(edge, el("span", {class: "ring", "aria-hidden": "true", "data-deco": ""}));
    }
    const box = edge.getBoundingClientRect();
    const corner = parseFloat(getComputedStyle(control).borderTopLeftRadius) || 0;
    const radius = String(Math.round(Math.max(0, Math.min(corner, box.width / 2, box.height / 2)) * 10) / 10);
    const outline = edge.firstElementChild;
    if (outline.getAttribute("rx") !== radius) {
      outline.setAttribute("rx", radius);
      outline.setAttribute("ry", radius);
    }
  }

  function controlOf(target) {
    return target instanceof Element ? target.closest(CONTROLS) : null;
  }

  function busy(control, on) {  // pending: only while the control's own request is in flight
    if (on) {
      dress(control);
      control.setAttribute("aria-busy", "true");
    } else {
      control.removeAttribute("aria-busy");
    }
  }

  function pulse(control) {
    if (control.disabled || control.getAttribute("aria-disabled") === "true"
        || control.getAttribute("aria-busy") === "true" || control.matches(".field, .pick")) return;
    dress(control);
    control.classList.remove("pulse");
    void control.offsetWidth;  // restarts the pulse
    control.classList.add("pulse");
  }

  document.addEventListener("pointerover", (event) => {
    const control = controlOf(event.target);
    if (control && !(event.relatedTarget instanceof Node && control.contains(event.relatedTarget))) dress(control);
  });
  document.addEventListener("focusin", (event) => {
    const control = controlOf(event.target);
    if (control) dress(control);
  });
  document.addEventListener("pointerdown", (event) => {
    const control = controlOf(event.target);
    if (control) pulse(control);
  });
  document.addEventListener("keydown", (event) => {  // Enter presses a link or a button; Space, a button
    if (event.repeat || (event.key !== "Enter" && event.key !== " ")) return;
    const control = controlOf(event.target);
    if (control && (event.key === "Enter" || control.tagName === "BUTTON")) pulse(control);
  });
  document.addEventListener("animationend", (event) => {
    if (event.animationName === "pulse" && event.target.parentElement) event.target.parentElement.classList.remove("pulse");
  });

  function median(values) {
    if (!values.length) return null;
    const sorted = [...values].sort((a, b) => a - b);
    const middle = sorted.length >> 1;
    return sorted.length % 2 ? sorted[middle] : (sorted[middle - 1] + sorted[middle]) / 2;
  }

  // ------------------------------------------------------------------ the live player (D75)

  function makePlayer() {
    const slot = document.getElementById("frame");
    const panel = document.getElementById("live");
    const label = document.getElementById("player-state");
    const facts = document.getElementById("player-facts");
    if (!slot || !panel || !label || slot.dataset.live !== "on") return null;
    const LIVE = "/live/frame";
    const FRAME = /^[0-9]{1,9}-[0-9]{1,12}$/;
    const SIZE = /^([0-9]{1,4})x([0-9]{1,4})$/;
    const STAMP = /^[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9:.]{8,15}Z$/;
    const STALE_AFTER = Math.min(Math.max(Number(slot.dataset.staleAfterMs) || 2000, 250), 30000);
    const ASK_TIMEOUT = 5000;  // the device waits at most 1.5 s for a newer picture; the rest is transfer
    const p = {
      token: TOKEN.test(slot.dataset.token || "") ? slot.dataset.token : "",
      after: null, shown: null, state: null, reason: "", failures: 0, running: false, wake: null,
      pending: null, decoding: false, canvas: null, context: null, seen: [], inSight: true, statsTimer: 0,
    };

    // ------------------------------------------------ its timer (the picture and the LIVE label share it)

    function restartPicture(sentAt, windowMs) {
      const elapsed = Math.max(0, performance.now() - sentAt);
      panel.style.setProperty("--pic-window", `${Math.max(0, Math.round(windowMs))}ms`);
      panel.style.setProperty("--pic-delay", `${-Math.round(elapsed)}ms`);
      panel.dataset.pic = panel.dataset.pic === "1" ? "0" : "1";
    }

    function expire() {  // nothing shown is live until a new picture arrives
      panel.style.setProperty("--pic-delay", `${-10 * STALE_AFTER}ms`);
      panel.dataset.pic = panel.dataset.pic === "1" ? "0" : "1";
    }

    // ------------------------------------------------ what it says

    const LABELS = {
      connecting: ["off", "clock", "CONNECTING", "asking the device for live pictures"],
      paused: ["off", "clock", "PAUSED", "not asked for while the picture is out of sight or this tab is hidden"],
      stale: ["warn", "clock", "STALE: not current", ""], offline: ["crit", "crit", "OFFLINE", ""],
      starting: ["off", "clock", "STARTING", ""], closed: ["off", "off", "STOPPED", "the runtime is stopping"],
      disconnected: ["crit", "off", "DISCONNECTED", ""], busy: ["warn", "warn", "BUSY", ""],
      reload: ["off", "off", "RELOAD TO RESUME", "the runtime restarted after this page was loaded; reload the page"],
      error: ["crit", "crit", "NO PICTURE", ""], off: ["off", "off", "OFF", ""],
      turned_off: ["off", "off", "OFF", "the live view is turned off in Features"],  // D99
    };

    function clearPicture() {  // D99: turned off: the picture is removed, not kept dimmed; the server's empty frame
      p.shown = null;
      p.after = null;
      p.pending = null;
      if (slot.querySelector("figure.frame")) slot.replaceChildren();
      delete state.last["frame-empty"];
      if (p.empty) place("frame-empty", p.empty);
    }

    function setState(name, reason) {
      const text = reason || "";
      if (p.state === name && p.reason === text) return;
      p.state = name;
      p.reason = text;
      panel.dataset.player = name;
      if (name === "live") {  // a claim that the picture is current: only while its own timer runs (D75)
        label.replaceChildren(el("span", {class: "fs inl"}, [
          el("span", {class: "pcur"}, [badge("ok", "LIVE", "video", true)]),
          el("span", {class: "pold"}, [badge("stale", "NOT CURRENT: no newer picture in time", "clock", true)]),
        ]));
      } else {
        const [kind, glyph, title, standing] = LABELS[name] || LABELS.error;
        label.replaceChildren(badge(kind, title, glyph, true), el("span", {class: "muted"}, [text || standing]));
      }
      if (name !== "live") keep(name);
    }

    function keep(name) {  // the last picture stays, dimmed, with why it is old and when it was received
      const figure = slot.querySelector("figure.frame");
      const shown = p.shown || (figure && figure.dataset.frame ? {id: figure.dataset.frame, ingest: figure.dataset.ingest}
                                                                : null);
      if (!figure || !shown || name === "connecting") return;
      figure.dataset.state = name === "disconnected" || name === "error" || name === "busy" ? "failed" : "retained";
      const cause = {
        stale: "the camera has no current frame", offline: "the camera has no current frame",
        disconnected: "the page cannot reach the device", busy: "the device is busy with other viewers",
        error: "the newest picture could not be shown", paused: "pictures are not asked for while out of sight",
        reload: "the runtime restarted", closed: "the runtime is stopping", starting: "the runtime is starting",
      }[name] || "no newer picture has arrived";
      const [epoch, seq] = (shown.id || "").split("-");
      const text = `OLD FRAME: ${cause}; this one was received at ${clock(shown.ingest)} (stream epoch ${epoch || "?"}, `
        + `frame ${seq || "?"}).`;
      const span = figure.querySelector(".kept span");
      if (span && span.textContent !== text) span.textContent = text;
    }

    // ------------------------------------------------ the picture

    function figure(width, height) {
      let shown = slot.querySelector("figure.frame");
      if (!shown || !shown.querySelector("canvas.pic")) {
        const canvas = el("canvas", {class: "pic", width: String(width), height: String(height), role: "img",
                                      "aria-label": "Live camera picture"});
        const built = el("figure", {class: "frame player"}, [
          canvas,
          el("figcaption", {class: "pic-old"}, [svgIcon("clock"), el("span")]),
          el("figcaption", {class: "kept"}, [svgIcon("clock"), el("span")]),
        ]);
        slot.replaceChildren(built);
        delete state.last["frame-empty"];
        shown = built;
        p.canvas = canvas;
        p.context = canvas.getContext("2d", {alpha: false});
      }
      return shown;
    }

    async function decode(blob) {
      if (typeof createImageBitmap === "function") return createImageBitmap(blob);
      const url = URL.createObjectURL(blob);
      try {
        const image = new Image();
        image.src = url;
        await image.decode();
        return image;
      } finally {
        URL.revokeObjectURL(url);
      }
    }

    function newerThanShown(meta) {
      return !p.shown || meta.epoch > p.shown.epoch || (meta.epoch === p.shown.epoch && meta.seq > p.shown.seq);
    }

    function draw(image, meta) {
      if (!wanted()) return;  // hidden or out of sight: a late picture restarts nothing (asked again on return)
      const shown = figure(meta.width, meta.height);
      if (p.canvas.width !== meta.width || p.canvas.height !== meta.height) {
        p.canvas.width = meta.width;
        p.canvas.height = meta.height;
      }
      p.context.drawImage(image, 0, 0, meta.width, meta.height);
      meta.drawn = performance.now();
      p.canvas.dataset.drawn = "1";
      shown.dataset.state = "live";
      shown.dataset.frame = meta.id;
      shown.dataset.frameShown = meta.id;
      shown.dataset.ingest = meta.ingest;
      p.shown = meta;
      const old = shown.querySelector(".pic-old span");
      const text = `NOT CURRENT: no newer picture has arrived in time; this one was received at ${clock(meta.ingest)}.`;
      if (old && old.textContent !== text) old.textContent = text;
      restartPicture(meta.sent, STALE_AFTER - meta.age);
      setState("live");
      p.seen.push(meta);
      if (p.seen.length > 120) p.seen.splice(0, p.seen.length - 120);
      requestAnimationFrame((at) => { meta.presented = at; });
    }

    async function decodeNewest() {
      if (p.decoding) return;
      p.decoding = true;
      try {
        while (p.pending) {
          const {meta, blob} = p.pending;
          p.pending = null;  // a newer one that arrives meanwhile replaces nothing older than it
          if (!newerThanShown(meta)) continue;
          let image = null;
          try {
            image = await decode(blob);
          } catch (error) {
            image = null;
          }
          meta.decoded = performance.now();
          if (!image) {
            setState("error", "a picture could not be decoded");
            continue;
          }
          if (newerThanShown(meta)) draw(image, meta);
          if (typeof image.close === "function") image.close();
        }
      } finally {
        p.decoding = false;
      }
    }

    function readMeta(headers, sent) {
      const id = headers.get("X-Frame") || "";
      const age = Number(headers.get("X-Frame-Age-Ms"));
      const size = SIZE.exec(headers.get("X-Frame-Size") || "");
      const ingest = headers.get("X-Frame-Ingest") || "";
      if (!FRAME.test(id) || !Number.isSafeInteger(age) || age < 0 || age > 3600000 || !size || !STAMP.test(ingest)) {
        return null;
      }
      const width = Number(size[1]);
      const height = Number(size[2]);
      if (width < 1 || height < 1 || width > 4096 || height > 4096) return null;
      const [epoch, seq] = id.split("-").map(Number);
      return {id, epoch, seq, age, width, height, ingest, sent, received: 0, decoded: 0, drawn: 0, presented: 0};
    }

    // ------------------------------------------------ asking

    function wanted() {
      return document.visibilityState !== "hidden" && p.inSight;
    }

    function sleep(ms) {
      return new Promise((resolve) => {
        const timer = setTimeout(() => { p.wake = null; resolve(); }, ms);
        p.wake = () => { clearTimeout(timer); p.wake = null; resolve(); };
      });
    }

    async function readJson(response) {
      const text = await response.text();
      if (text.length > 20000) return null;
      try {
        const doc = JSON.parse(text);
        return doc && typeof doc === "object" && !Array.isArray(doc) ? doc : null;
      } catch (error) {
        return null;
      }
    }

    async function askOnce() {  // the time to wait before asking again (0: at once)
      const sent = performance.now();
      const controller = new AbortController();
      const timeout = setTimeout(() => controller.abort(), ASK_TIMEOUT);
      const token = state.token && state.token !== p.token && p.failures ? state.token : p.token;
      try {
        const response = await fetch(p.after ? `${LIVE}?after=${p.after}` : LIVE, {
          cache: "no-store", credentials: "same-origin", redirect: "error", signal: controller.signal,
          headers: {"X-Sentinel-Token": token},
        });
        const type = response.headers.get("Content-Type") || "";
        if (response.status === 200 && type === "image/jpeg") {
          const meta = readMeta(response.headers, sent);
          const blob = await response.blob();
          if (!meta || blob.size < 4 || blob.size > 1048576) throw new Error("the picture was not valid");
          meta.received = performance.now();
          p.token = token;
          p.after = meta.id;
          p.failures = 0;
          p.pending = {meta, blob};  // the newest wins: an older one not yet decoded is dropped
          decodeNewest();
          return 0;
        }
        const doc = type.startsWith("application/json") ? await readJson(response) : null;
        const reason = doc && typeof doc.reason === "string" ? doc.reason.slice(0, 200) : "";
        if (response.status === 403) {
          if (state.token && state.token !== token) {  // the runtime restarted: its newer page brought a new token
            p.token = state.token;
            p.after = null;
            return 0;
          }
          p.failures += 1;
          setState("reload");
          return Math.min(2000 * p.failures, MAX_WAIT);
        }
        if (response.status === 503) {
          p.failures += 1;
          setState("busy", reason || "too many live picture requests at once");
          return 1000;
        }
        if (response.status === 200 && doc && typeof doc.state === "string") {
          p.token = token;
          p.failures = 0;
          if (doc.state === "live") return 0;  // live, but nothing newer within the device's wait: ask again
          if (doc.state === "paused") {  // D99: turned off in Features; asked again slowly, to notice it back on
            setState("turned_off", reason);
            clearPicture();
            return 2000;
          }
          setState(LABELS[doc.state] ? doc.state : "error", reason);
          return 500;
        }
        if (response.status === 404 && doc && doc.state === "off") {
          setState("off", reason);
          return MAX_WAIT;
        }
        throw new Error(`HTTP ${response.status}`);
      } catch (error) {
        p.failures += 1;
        const wait = Math.min(500 * 2 ** (p.failures - 1), 5000);
        setState("disconnected", `${why(error)}; trying again in ${Math.max(1, Math.round(wait / 1000))} s`);
        return wait;
      } finally {
        clearTimeout(timeout);
      }
    }

    async function run() {
      if (p.running) return;
      p.running = true;
      try {
        if (!p.shown && !slot.querySelector("figure.frame")) setState("connecting");
        while (wanted()) {
          const wait = await askOnce();
          if (wait > 0 && wanted()) await sleep(wait);
        }
        setState("paused");
      } finally {
        p.running = false;
      }
    }

    function resume() {
      if (p.wake) p.wake();
      if (wanted()) run();
    }

    // ------------------------------------------------ measuring (delivered rate apart from age)

    function stats() {
      const now = performance.now();
      const recent = p.seen.filter((m) => now - m.drawn <= 5000);
      if (!recent.length || p.state !== "live") {
        facts.hidden = !p.shown;
        slot.dataset.fps = "0";
        return;
      }
      const span = Math.max(1, Math.min(5000, now - recent[0].drawn));
      const fps = recent.length > 1 ? ((recent.length - 1) * 1000) / (recent[recent.length - 1].drawn - recent[0].drawn || 1) : 0;
      const sentAge = median(recent.map((m) => m.age));
      const bound = recent.map((m) => m.age + ((m.presented || m.drawn) - m.sent));
      const transfer = median(recent.map((m) => m.received - m.sent));
      const decoding = median(recent.map((m) => m.decoded - m.received));
      Object.assign(slot.dataset, {fps: fps.toFixed(1), ageSentMs: String(Math.round(sentAge)),
                                   ageBoundMs: String(Math.round(median(bound))),
                                   ageBoundMaxMs: String(Math.round(Math.max(...bound))), span: String(Math.round(span))});
      const shown = p.shown;
      const rows = [
        ["Delivered", `${fps.toFixed(1)} pictures/s (last ${Math.round(span / 1000)} s)`],
        ["Age when sent", `${Math.round(sentAge)} ms median, on the device's clock (ingest to sending)`],
        ["Age when shown", `at most ${Math.round(median(bound))} ms median, ${Math.round(Math.max(...bound))} ms `
          + "largest (age when sent plus the time since this browser asked)"],
        ["Asking to arrival", `${Math.round(transfer)} ms median (includes waiting for the next frame)`],
        ["Decoding", `${Math.round(decoding)} ms median`],
        ["Shown frame", `stream epoch ${shown.epoch}, frame ${shown.seq}; received by the device at ${clock(shown.ingest)}; `
          + "the camera's capture time is not known"],
      ];
      const nodes = rows.map(([term, value]) => el("div", {}, [el("dt", {}, [term]), el("dd", {}, [value])]));
      facts.replaceChildren(...nodes);
      facts.hidden = false;
    }

    // ------------------------------------------------ start

    const first = slot.querySelector("figure.frame");
    if (first && first.dataset.frame && FRAME.test(first.dataset.frame)) {  // the page's own first picture
      const [epoch, seq] = first.dataset.frame.split("-").map(Number);
      const age = Number(first.dataset.ageMs);
      p.after = first.dataset.frame;
      p.shown = {id: first.dataset.frame, epoch, seq, ingest: first.dataset.ingest || "", age: Number.isFinite(age) ? age : 0,
                 sent: requested, drawn: performance.now()};
      restartPicture(requested, STALE_AFTER - p.shown.age);
      setState("live");
    } else {
      expire();
    }
    panel.classList.add("playing");
    if (typeof IntersectionObserver === "function") {
      new IntersectionObserver((entries) => {
        for (const entry of entries) p.inSight = entry.isIntersecting;
        if (p.inSight) resume();
        else if (p.wake) p.wake();
      }).observe(slot);
    }
    p.statsTimer = setInterval(stats, 1000);
    return {
      run, resume, expire,
      status(doc) {  // a status update may say the live view is off or starting; it never restarts the picture's timer
        const preview = doc.preview;
        p.empty = doc.page.frame_empty;
        if (preview && preview.enabled === false) setState("off", preview.reason);
        if (preview && preview.state === "paused") {  // D99
          setState("turned_off", preview.reason);
          clearPicture();
        }
        if (!p.shown && !slot.querySelector("figure.frame")) place("frame-empty", doc.page.frame_empty);
      },
    };
  }

  // ------------------------------------------------------------------ the selected incident (status page)

  function markSelected() {
    for (const item of document.querySelectorAll(".inc-list > li[data-key]")) {
      const on = state.selected !== null && item.dataset.key === `i-${state.selected}`;
      item.classList.toggle("selected", on);
      const button = item.querySelector(".peek");
      if (button) {
        button.setAttribute("aria-pressed", on ? "true" : "false");
        const label = on ? "Shown below" : "Preview";
        const span = button.querySelector("span");
        if (span && span.textContent !== label) span.textContent = label;
      }
    }
  }

  const SEVERITY = {info: ["off", "info"], warning: ["warn", "warn"], critical: ["crit", "crit"]};
  const STATUS = {candidate: ["off", "clock"], open: ["warn", "bell"], acknowledged: ["done", "check"],
                  resolved: ["done", "check"], dismissed: ["off", "off"]};

  function capital(text) {
    return text ? text[0].toUpperCase() + text.slice(1) : "";
  }

  function renderPreview(detail) {
    const pane = document.getElementById("inc-preview");
    if (!pane || !detail || detail.state !== "available" || !detail.incident) return;
    const incident = detail.incident;
    if (incident.incident_id !== state.selected) return;
    const [sevKind, sevIcon] = SEVERITY[incident.severity] || ["off", "info"];
    const [stKind, stIcon] = STATUS[incident.status] || ["off", "info"];
    const opening = ((detail.evidence || {}).shown || []).find((item) => item.phase === "entered");
    const ack = detail.acknowledgment || {};
    const rows = ((detail.deliveries || {}).shown || []).map((d) => `${d.channel}: ${d.state}`);
    const snapshot = detail.snapshot || {};
    const theme = state.appearanceQuery ? `?${state.appearanceQuery}` : "";
    const children = [
      el("div", {class: "phead"}, [el("h3", {}, [incident.title || ""]),
        el("button", {type: "button", class: "plain", "data-close": "1"}, ["Close"])]),
      el("div", {class: "tags"}, [badge(sevKind, capital(incident.severity), sevIcon),
        badge(stKind, capital(incident.status), stIcon),
        ack.state === "acknowledged" ? badge("done", "Acknowledged", "check") : badge("off", "Not acknowledged", "bell")]),
    ];
    const view = `/incidents/${incident.incident_id}/snapshot.jpg`;
    if (snapshot.state === "saved" && snapshot.file === "verified" && snapshot.view_url === view) {
      // The same picture is kept, not fetched again, so a refresh never flashes it (D96).
      const kept = pane.querySelector(`.frame.thumb img[src="${view}"]`);
      children.push(kept ? kept.parentNode : el("div", {class: "frame thumb"}, [el("img", {src: view,
        alt: "Incident snapshot", width: String(snapshot.width || 640), height: String(snapshot.height || 480)})]));
    } else {
      children.push(el("p", {class: "note"}, [snapshot.state === "saved" ? "Snapshot not verified." : "No snapshot."]));
    }
    children.push(el("p", {class: "reason"}, [opening ? opening.reason : "Opening observation not shown."]));
    children.push(el("p", {class: "note"}, [
      `First seen ${clock(incident.first_observed_utc)}; latest ${clock(incident.last_observed_utc)}. `
      + (rows.length ? `Alerts: ${rows.join(", ")}.` : "No alerts.")]));
    children.push(el("a", {class: "button", href: `/incidents/${incident.incident_id}${theme}`},
      [svgIcon("chevron"), "Open the full incident"]));
    pane.replaceChildren(...children);
    pane.hidden = false;
  }

  async function loadSelected() {
    clearTimeout(state.detailTimer);
    if (!state.selected) return;
    if (state.detail) state.detail.abort();
    const controller = new AbortController();
    state.detail = controller;
    const timeout = setTimeout(() => controller.abort(), 8000);
    const id = state.selected;
    try {
      const response = await fetch(`/incidents/${id}.json`, {cache: "no-store", credentials: "same-origin",
        redirect: "error", signal: controller.signal, headers: {Accept: "application/json"}});
      if (response.status === 404) {
        select(null);
        return;
      }
      if (response.status !== 200) throw new Error(`HTTP ${response.status}`);
      const text = await response.text();
      if (text.length > MAX_TEXT) throw new Error("too large");
      if (state.selected === id) renderPreview(JSON.parse(text));
    } catch (error) {
      // keep what is shown; the next refresh tries again
    } finally {
      clearTimeout(timeout);
      if (state.detail === controller) state.detail = null;
      if (state.selected === id) state.detailTimer = setTimeout(loadSelected, 10000);
    }
  }

  function select(id) {
    const before = state.selected;
    state.selected = id && INCIDENT.test(id) ? id : null;
    const pane = document.getElementById("inc-preview");
    if (!state.selected && pane) {
      pane.hidden = true;
      pane.replaceChildren();
      if (state.detail) state.detail.abort();
      clearTimeout(state.detailTimer);
    } else if (pane && state.selected !== before) {  // a steady placeholder while the first answer is on its way
      pane.replaceChildren(el("p", {class: "note"}, ["Loading the incident…"]));
      pane.hidden = false;
    }
    const here = location.pathname + location.search;
    history.replaceState(null, "", state.selected ? `${here}#incident-${state.selected}` : here);
    markSelected();
    if (state.selected) loadSelected();
  }

  document.addEventListener("click", (event) => {
    const peek = event.target.closest(".peek");
    if (peek && peek.dataset.select) {
      select(peek.dataset.select === state.selected ? null : peek.dataset.select);
      return;
    }
    if (event.target.closest("[data-close]")) select(null);
  });

  // ------------------------------------------------------------------ chat (D74): advisory questions about one frame

  // The conversation and the draft live only in this tab: never in a region, so updates leave them alone, and never in
  // storage. Answers are shown as text. Nothing here touches the freshness timers: an answer is not camera data.
  const chatForm = document.getElementById("chat-form");
  if (chatForm) {
    const box = document.getElementById("chat-q");
    const ask = document.getElementById("chat-ask");
    const note = document.getElementById("chat-note");
    const log = document.getElementById("chat-log");
    const token = chatForm.dataset.token || "";
    const servedBy = state.instance;  // the token belongs to the server that sent this page
    const LIMIT = Number(box.getAttribute("maxlength")) || 300;
    const JOB = /^[0-9a-f]{16}$/;
    const chat = {pending: false, history: []};
    const STATES = {
      sending: "Sending your question…",
      waiting: "Waiting for the scene model; monitoring comes first…",
      answering: "The scene model is answering…",
    };
    const TITLES = {busy: "Busy", unavailable: "Unavailable", off: "Chat is off", no_fresh_frame: "No fresh frame",
                    timeout: "No answer in time", invalid: "Not sent", refused: "Refused", unknown: "Lost",
                    cancelled: "Cancelled", paused: "Paused", incomplete: "Incomplete",
                    picture_expired: "Picture no longer held"};
    const REASONS = {unknown: "the server no longer has this question; it may have restarted"};
    const GLYPHS = {sending: "clock", waiting: "clock", answering: "clock", answered: "check", busy: "warn",
                    no_fresh_frame: "warn", timeout: "warn", unavailable: "off", off: "off", unknown: "off",
                    cancelled: "warn", paused: "clock", incomplete: "warn", picture_expired: "warn"};

    const chatState = () => {
      const element = document.querySelector("[data-chat-state]");
      return element ? element.dataset.chatState : "off";
    };

    refreshAsk = () => {  // declared at the top, so that updates can call it
      const available = chatState() === "available";
      const restarted = state.instance !== servedBy;
      const text = box.value.trim();
      ask.disabled = chat.pending || !available || restarted || !text || text.length > LIMIT;
      for (const button of log.querySelectorAll("button[data-again], button[data-auto=start]")) {
        button.disabled = chat.pending || !available || restarted;  // part 28: one question at a time, as Ask
      }
      busy(ask, chat.pending);  // its edge circles while the question is on its way (session 114)
      const reason = restarted ? "The runtime restarted after this page was loaded: reload the page to ask."
        : !available ? "Chat is not available now: see the reason above."
          : chat.pending ? "One question at a time." : "";
      if (note.textContent !== reason) note.textContent = reason;
    };

    const line = (exchange, kind, text) => {
      let node = exchange.querySelector(`.${kind}`);
      if (!node) {
        node = el("p", {class: kind});
        exchange.append(node);
      }
      node.textContent = text;
      return node;
    };

    // The exchange's state, as a label with its icon (the text says it; the icon and the edge colour repeat it).
    const status = (exchange, state_, text) => {
      let node = exchange.querySelector(".status");
      if (node && exchange.dataset.state === state_ && node.textContent === text) return;  // unchanged: left alone
      exchange.dataset.state = state_;
      if (!node) {
        node = el("p", {class: "status"});
        exchange.append(node);
      }
      node.replaceChildren(svgIcon(GLYPHS[state_] || "crit"), text);
    };

    const number = (value) => (Number.isSafeInteger(value) && value >= 0 ? String(value) : "?");
    const seconds = (ms) => (Number.isSafeInteger(ms) && ms >= 0 ? `${(ms / 1000).toFixed(1)} s` : "an unknown time");

    // Which frame an answer describes, and how old it was. Historical unless the server says it was still live when
    // the answer was ready: an answer is never about the scene now. When the frame asked about was no longer live by
    // the time the question could start, the server sent the newest live one instead, and the label says so.
    const label = (document_, exchange_) => {
      const frame = document_.frame;
      if (!frame || typeof frame !== "object") return "";
      const replaced = document_.frame_replaced === true;
      const which = `the frame received at ${clock(frame.ingest_utc)} (stream epoch ${number(frame.stream_epoch)}, `
        + `frame ${number(frame.frame_seq)})`
        + (replaced ? ", sent in place of the frame asked about, which was no longer live:" : ",");
      const age = replaced ? `${seconds(frame.age_ms_when_sent)} old when sent`
        : `${seconds(frame.age_ms_when_asked)} old when asked`;
      if (document_.state !== "answered") return `About ${which} ${age}.`;
      // Part 28: asked again, the card says which picture this result is about: the same (held) one, or a new one.
      const again_ = document_.same_picture === true ? "The same picture again (the held one, not a new picture). "
        : exchange_ && exchange_.dataset.again === "latest" ? "A new picture (the latest), not the one before. " : "";
      const lead = document_.historical === false ? "About" : "Historical: about";
      return `${again_}${lead} ${which} ${age} and `
        + `${seconds(document_.age_ms_when_answered)} old when answered. It describes that frame only, not the scene now.`;
    };

    const numbers = (list, count) => Array.isArray(list) && list.length === count
      && list.every((v) => Number.isFinite(v)) ? list : null;
    const regionText = (region) => {
      const px = numbers(region.pixels, 4);
      const size = numbers(region.source_size, 2);
      const sent = numbers(region.sent_size, 2);
      const from = region.source === "main_stream" ? "the main stream's picture" : "the working picture (no larger one is recorded on this stream)";
      return `Region: recorded pixels ${px ? `${px[0]},${px[1]} to ${px[2]},${px[3]}` : "?"} of ${from}`
        + `${size ? ` (${size[0]} × ${size[1]})` : ""}, cut from that picture and sent as ${sent ? `${sent[0]} × ${sent[1]}` : "?"}. `
        + "It shows no more than was recorded.";
    };

    // Shows a server document on one exchange; true when the exchange is finished.
    const show = (exchange, document_, question) => {
      const state_ = typeof document_.state === "string" ? document_.state : "error";
      if (STATES[state_]) {
        status(exchange, state_, STATES[state_]);
        return false;
      }
      if (state_ === "answered" && document_.kind === "recorded" && typeof document_.answer === "string") {
        // Part 28: answered from the system's own records, not by the scene model and not from the picture. Shown as
        // text only; it never joins the context sent with a later question about the picture.
        status(exchange, "answered", "Answered from recorded facts, not from the picture.");
        exchange.dataset.kind = "recorded";
        line(exchange, "answer", document_.answer.slice(0, 600));
        for (const [name, limit] of [["lines", 16], ["limits", 6]]) {
          const items = Array.isArray(document_[name]) ? document_[name].filter((t) => typeof t === "string") : [];
          let list = exchange.querySelector(`ul.${name}`);
          if (!list) {
            list = el("ul", {class: name});
            exchange.append(list);
          }
          list.replaceChildren(...items.slice(0, limit).map((text) => el("li", {}, [text.slice(0, 500)])));
          list.hidden = !items.length;
        }
        return true;
      }
      if (state_ === "answered" && typeof document_.answer === "string") {
        // Shortened: its time before the next scene check ran out, so it shows its complete sentences and may have
        // said more; what was left unfinished is shown apart, never dropped.
        status(exchange, "answered", document_.shortened === true
          ? "Answered, shortened: it had to stop before the next scene check and may have said more."
          : document_.cut_short === true ? "Answered (cut short at the answer limit)." : "Answered.");
        line(exchange, "answer", document_.answer.slice(0, 1200));
        if (typeof document_.unfinished === "string" && document_.unfinished) {
          line(exchange, "unfinished", `Left unfinished when it stopped: “${document_.unfinished.slice(0, 600)}…”`);
        }
        if (typeof document_.read_as === "string" && document_.read_as) {
          line(exchange, "readas", `Read as a follow-up to your earlier question: “${document_.read_as.slice(0, 400)}”`);
        }
        line(exchange, "frame", label(document_, exchange));
        if (document_.region && typeof document_.region === "object") {
          // Part 28: a region of a held picture. Never part of the context sent with a later question.
          line(exchange, "region", regionText(document_.region));
          return true;
        }
        // the context sent with the next question: the operator's questions and the model's own earlier statements
        // (never a recorded fact); the same question asked again replaces its own earlier entry
        chat.history = chat.history.filter((turn) => turn.question !== question);
        chat.history.push({question, answer: document_.answer.slice(0, 400)});
        chat.history = chat.history.slice(-2);
        return true;
      }
      const reason = typeof document_.reason === "string" ? document_.reason.slice(0, 300)
        : REASONS[state_] || "the question could not be answered";
      status(exchange, TITLES[state_] ? state_ : "error", `${TITLES[state_] || "Error"}: ${reason}.`);
      if (document_.frame) line(exchange, "frame", label(document_, exchange));
      return true;
    };

    // One bounded request; always an object. ``lost`` marks a request that did not get a readable answer.
    // ``given`` (optional): an AbortSignal of the caller's; aborted, the request ends as one that ran out of time.
    const request = async (url, options, ms, given) => {
      const controller = new AbortController();
      const timer = setTimeout(() => controller.abort(), ms);
      if (given) {
        if (given.aborted) controller.abort();
        else given.addEventListener("abort", () => controller.abort(), {once: true});
      }
      try {
        const response = await fetch(url, {cache: "no-store", credentials: "same-origin", redirect: "error",
          signal: controller.signal, ...options});
        const type = response.headers.get("Content-Type") || "";
        if (!type.startsWith("application/json")) return {state: "error", reason: `HTTP ${response.status}`};
        const text = await response.text();
        if (text.length > 20000) return {state: "error", reason: "the answer was too large"};
        const parsed = JSON.parse(text);
        return parsed && typeof parsed === "object" && !Array.isArray(parsed) ? parsed
          : {state: "error", reason: "the answer was not readable"};
      } catch (error) {
        const reason = error && error.name === "AbortError" ? "no answer in time"
          : error && error.name === "SyntaxError" ? "the answer was not readable" : "the connection failed";
        return {state: "error", reason, lost: true};
      } finally {
        clearTimeout(timer);
      }
    };

    // Part 28. One question to its end on one exchange: a new one, or the same card again ("Analyse again", "Try
    // again", an automatic refresh). ``extra`` is {again: {use, picture}} and/or {auto: true}; ``typed``: the question
    // came from the box, which is cleared once it is taken and gets it back if it ends without an answer. One at a
    // time: chat.pending covers manual and automatic questions alike, so they never overlap. Returns the last
    // document shown.
    const run = async (exchange, question, extra, typed) => {
      chat.pending = true;
      refreshAsk();
      status(exchange, "sending", STATES.sending);
      let shown = {state: "error", reason: "the question could not be sent"};
      try {
        let document_ = await request("/chat", {method: "POST", headers: {"Content-Type": "application/json",
          "X-Sentinel-Token": token}, body: JSON.stringify({question, history: chat.history, ...extra})}, 8000);
        const job = typeof document_.job === "string" && JOB.test(document_.job) ? document_.job : null;
        // Handed over, the box is cleared for the next question; one that was not taken stays in it, and one that
        // ends without an answer (cancelled, busy, ...) comes back into an empty box, to send again without retyping.
        if (typed && (job || document_.kind === "recorded")) box.value = "";
        const started = performance.now();
        for (;;) {
          shown = document_;
          if (document_.lost && job) {  // a poll that got no answer: the question may still be answered
            status(exchange, "waiting", `No reply from the page server (${document_.reason}); still waiting…`);
          } else if (show(exchange, document_, question)) {
            break;
          } else if (!job) {
            shown = {state: "error", reason: "the answer was not readable"};
            show(exchange, shown, question);
            break;
          }
          if (performance.now() - started > 40000) {
            shown = {state: "timeout", reason: "no answer in time"};
            show(exchange, shown, question);
            break;
          }
          await new Promise((resolve) => setTimeout(resolve, 1000));
          document_ = await request(`/chat/${job}`, {headers: {"X-Sentinel-Token": token}}, 5000);
        }
      } finally {
        if (typed && exchange.dataset.state !== "answered" && !box.value.trim()) box.value = question;
        chat.pending = false;
        actions(exchange, shown);
        refreshAsk();
      }
      return shown;
    };

    // A card run again keeps its earlier results, the newest first, at most EARLIER of them: a bounded card, never an
    // endless list of repeated replies.
    const EARLIER = 4;
    const archive = (exchange) => {
      const answer = exchange.querySelector("p.answer");
      const frame = exchange.querySelector("p.frame");
      if (exchange.dataset.state !== "answered" || !answer) return;
      let earlier = exchange.querySelector("details.earlier");
      if (!earlier) {
        earlier = el("details", {class: "earlier"}, [el("summary", {}, ["Earlier results"]), el("ul", {})]);
        exchange.append(earlier);
      }
      const list = earlier.querySelector("ul");
      list.prepend(el("li", {}, [`${answer.textContent.slice(0, 300)} — ${frame ? frame.textContent.slice(0, 300) : ""}`]));
      while (list.children.length > EARLIER) list.lastElementChild.remove();
      earlier.querySelector("summary").textContent = `Earlier results (${list.children.length}, newest first)`;
      for (const name of ["unfinished", "readas"]) {
        const stale = exchange.querySelector(`p.${name}`);
        if (stale) stale.remove();
      }
    };

    // The card's own controls once a question about the picture has ended: answered, "Analyse again" on the same
    // (held) picture or on the latest one, and automatic refresh; not answered, "Try again" without retyping.
    const AUTO_MIN = Number(chatForm.dataset.autoMin) || 15;
    const AUTO_MAX = Number(chatForm.dataset.autoMax) || 120;
    const AUTO_DEFAULT = Number(chatForm.dataset.autoDefault) || 30;
    const auto = {exchange: null, every: 0, wait: 0, timer: null};
    const actions = (exchange, document_) => {
      if (exchange.dataset.kind === "recorded" || exchange.dataset.kind === "scan" || !exchange.isConnected) return;
      if (exchange.dataset.region) {  // a region of a held picture: only "Try again" on that same region
        let row = exchange.querySelector("div.again");
        if (exchange.dataset.state === "answered" || document_.state === "picture_expired") {
          if (row) row.remove();
          return;
        }
        if (!row) {
          row = el("div", {class: "again"});
          exchange.append(row);
        }
        row.replaceChildren(el("button", {type: "button", class: "ghost", "data-again": "region"}, ["Try again"]));
        return;
      }
      const answered = exchange.dataset.state === "answered";
      if (typeof document_.picture === "string" && JOB.test(document_.picture)) exchange.dataset.picture = document_.picture;
      else if (document_.state === "picture_expired") delete exchange.dataset.picture;
      let row = exchange.querySelector("div.again");
      if (!row) {
        row = el("div", {class: "again"});
        exchange.append(row);
      }
      const button = (use, text) => el("button", {type: "button", class: "ghost", "data-again": use}, [text]);
      const items = [];
      if (answered) {
        items.push(el("span", {class: "muted"}, ["Analyse again:"]));
        if (exchange.dataset.picture) items.push(button("same", "Same picture"));
        items.push(button("latest", "Latest picture"));
      } else {
        items.push(button("latest", document_.state === "picture_expired" ? "Use the latest picture" : "Try again"));
      }
      row.replaceChildren(...items);
      // the automatic refresh's line: running or paused with Stop; off with its interval and Start (answered cards
      // only); and, off, why it last stopped, which stays on the card whatever state it is in
      let line_ = exchange.querySelector("div.autorow");
      const stopped = exchange.dataset.autoNote || "";
      if (auto.exchange !== exchange && !answered && !stopped) {
        if (line_) line_.remove();
        return;
      }
      if (!line_) {
        line_ = el("div", {class: "autorow"});
        exchange.append(line_);
      }
      if (auto.exchange === exchange) {
        if (!line_.querySelector("button[data-auto=stop]")) {
          line_.replaceChildren(el("span", {class: "auto-state", role: "status"}, [""]),
            el("button", {type: "button", class: "ghost", "data-auto": "stop"}, ["Stop"]));
        }
      } else if (!answered) {
        line_.replaceChildren(el("span", {class: "auto-state", role: "status"}, [stopped]));
      } else {
        if (!line_.querySelector("select")) {
          const select = el("select", {class: "auto-every", "aria-label": "Seconds between automatic refreshes"});
          for (const seconds of [...new Set([AUTO_MIN, AUTO_DEFAULT, 60, AUTO_MAX])].filter(
            (n) => n >= AUTO_MIN && n <= AUTO_MAX).sort((x, y) => x - y)) {
            const option = el("option", {value: String(seconds)}, [`${seconds} s`]);
            if (seconds === AUTO_DEFAULT) option.selected = true;
            select.append(option);
          }
          line_.replaceChildren(el("span", {class: "muted"}, ["Auto-refresh (off): every"]),
            el("span", {class: "pick"}, [select]),
            el("button", {type: "button", class: "ghost", "data-auto": "start"}, ["Start"]),
            el("span", {class: "auto-state", role: "status"}, [""]));
        }
        autoSay(exchange, stopped);
      }
    };

    const again = (exchange, use, automatic) => {
      if (use === "region") {
        return run(exchange, exchange.dataset.question || "", {region: JSON.parse(exchange.dataset.region), history: []},
          false);
      }
      const extra = {again: use === "same" ? {use: "same", picture: exchange.dataset.picture} : {use: "latest"}};
      if (automatic) extra.auto = true;
      exchange.dataset.again = use;
      archive(exchange);
      return run(exchange, exchange.dataset.question || "", extra, false);
    };

    // Automatic refresh (part 28): off until started on one answered card; one cycle at a time, the next scheduled
    // only when the last has ended (no backlog); each cycle asks about the latest picture and updates that card in
    // place. It sends nothing while this tab is hidden, the page is not current, the camera is not fresh, chat is
    // paused or a question is being answered, and says why; busy answers back off up to AUTO_MAX; it stops, saying
    // so, on Stop, a restarted runtime, a closed card, or chat off or refused.
    const autoSay = (exchange, text) => {
      const node = exchange.querySelector(".auto-state");
      if (node && node.textContent !== text) node.textContent = text;
    };
    const stopAuto = (why) => {
      const exchange = auto.exchange;
      if (!exchange) return;
      clearTimeout(auto.timer);
      auto.exchange = null;
      auto.timer = null;
      exchange.dataset.auto = "off";
      exchange.dataset.autoNote = why;
      if (exchange.isConnected) {
        const line_ = exchange.querySelector("div.autorow");
        if (line_) line_.replaceChildren();
        actions(exchange, {state: exchange.dataset.state});
        refreshAsk();
      }
    };
    const pauseReason = () => {
      if (document.hidden) return "this tab is hidden";
      if (!(Number.isFinite(state.currentUntil) && performance.now() < state.currentUntil)) return "this page is not current";
      const row = document.querySelector("[data-chat-state]");
      if (row && row.dataset.chatState === "paused") return "chat is paused by the runtime (monitoring comes first)";
      if (row && row.dataset.video && row.dataset.video !== "fresh") {
        return `the camera is ${row.dataset.video === "unknown" ? "not reporting" : row.dataset.video}`;
      }
      if (chat.pending) return "a question is being answered";
      return "";
    };
    const tick = async () => {
      const exchange = auto.exchange;
      if (!exchange) return;
      if (!exchange.isConnected) {
        stopAuto("");
        return;
      }
      if (state.instance !== servedBy) {
        stopAuto("Auto-refresh stopped: the runtime restarted.");
        return;
      }
      const now = chatState();
      if (now === "off" || now === "unavailable") {
        stopAuto(`Auto-refresh stopped: chat is ${now}.`);
        return;
      }
      const reason = pauseReason();
      if (reason) {
        exchange.dataset.auto = "paused";
        autoSay(exchange, `Auto-refresh paused: ${reason}. Nothing is sent meanwhile.`);
        auto.timer = setTimeout(tick, 1000);
        return;
      }
      exchange.dataset.auto = "running";
      autoSay(exchange, "Auto-refresh running: refreshing now…");
      const document_ = await again(exchange, "latest", true);
      if (auto.exchange !== exchange) return;  // stopped meanwhile
      const ended = typeof document_.state === "string" ? document_.state : "error";
      if (["off", "unavailable", "refused", "invalid", "unknown"].includes(ended)) {
        stopAuto(`Auto-refresh stopped: the last refresh was not accepted (${TITLES[ended] || ended}).`);
        return;
      }
      const backoff = ended !== "answered";
      auto.wait = backoff ? Math.min(AUTO_MAX, Math.max(auto.every, auto.wait * 2)) : auto.every;
      exchange.dataset.auto = "running";
      autoSay(exchange, `Auto-refresh running: every ${auto.every} s`
        + (backoff ? `; the last one was not answered, so the next try is in ${auto.wait} s.` : "."));
      auto.timer = setTimeout(tick, auto.wait * 1000);
    };
    const startAuto = (exchange) => {
      stopAuto("Auto-refresh stopped here: it was started on another question.");
      const select = exchange.querySelector("select.auto-every");
      const every = Math.min(AUTO_MAX, Math.max(AUTO_MIN, Number(select && select.value) || AUTO_DEFAULT));
      Object.assign(auto, {exchange, every, wait: every});
      delete exchange.dataset.autoNote;
      exchange.dataset.auto = "running";
      actions(exchange, {state: exchange.dataset.state});
      autoSay(exchange, `Auto-refresh running: every ${every} s.`);
      auto.timer = setTimeout(tick, every * 1000);
    };

    log.addEventListener("click", (event) => {
      const button = event.target.closest("button[data-again], button[data-auto]");
      const exchange = button && button.closest("li.exchange");
      if (!button || button.disabled || !exchange) return;
      if (button.dataset.auto === "stop") stopAuto("Auto-refresh stopped.");
      else if (button.dataset.auto === "start") startAuto(exchange);
      else if (!chat.pending) again(exchange, button.dataset.again, false);  // one at a time: a second click is nothing
    });

    chatForm.addEventListener("submit", (event) => {
      event.preventDefault();
      refreshAsk();
      if (ask.disabled) return;
      const question = box.value.trim();
      const exchange = el("li", {class: "exchange", "data-question": question},
        [el("p", {class: "question"}, [question])]);
      log.append(exchange);
      while (log.children.length > 12) log.firstElementChild.remove();
      run(exchange, question, {}, true);
    });
    // ---------------------------------------------------------------- part 28: zoom into a region, scan the room
    //
    // "Zoom" holds the newest live frame on the device (POST /chat/hold) and shows that held picture here; the viewer
    // drags a rectangle on it (or uses the arrow keys, the numbers, or a tracked person's own box) and asks about it.
    // The server cuts the rectangle from the held frame's recorded pixels: nothing is magnified in the browser, and
    // no coordinates are ever guessed. "Scan" asks about the held picture's four overlapping parts in turn, each as
    // its own question through every usual limit, on one card that says what was and was not analysed.
    const zoomOpen = document.getElementById("zoom-open");
    const scanOpen = document.getElementById("scan-open");
    const zoomPanel = document.getElementById("zoom-panel");
    if (zoomOpen && scanOpen && zoomPanel) {
      const surface = document.getElementById("zoom-surface");
      const canvas = document.getElementById("zoom-canvas");
      const marker = document.getElementById("zoom-box");
      const zoomNote = document.getElementById("zoom-note");
      const zoomAsk = document.getElementById("zoom-ask");
      const sizeNote = document.getElementById("zoom-size");
      const inputs = ["x1", "y1", "x2", "y2"].map((name) => document.getElementById(`zoom-${name}`));
      // ``unset``: why the four numbers are not the region (while they are being typed), or "" when they are.
      // ``loading``: the one held picture on its way (being held on the device, fetched and decoded), or null. While
      // it is set, Zoom and Scan stay off, through the page's own updates too (refreshZoom), so a second picture is
      // never asked for beside it. Closing the panel gives it up there and then (dropLoad), which also frees the
      // controls; whatever arrives for it afterwards is no longer the picture being waited for and changes nothing.
      const zoom = {pin: null, box: null, drag: null, unset: "", loading: null};
      const PICTURE_MS = 8000;  // the held picture is fetched and decoded within this, or it is given up
      const RESTARTED = "The runtime restarted while the held picture was on its way, so it is not shown.";
      const headers = {"Content-Type": "application/json", "X-Sentinel-Token": token};
      const sayZoom = (text) => {
        if (zoomNote.textContent !== text) zoomNote.textContent = text;
      };
      const clamp = (v) => Math.min(1, Math.max(0, v));
      const pixels = () => {  // the selection in the source's whole pixels, as the server maps it
        const [w, h] = zoom.pin.source_size;
        const [x1, y1, x2, y2] = zoom.box;
        return [Math.min(w - 1, Math.floor(x1 * w)), Math.min(h - 1, Math.floor(y1 * h)),
          Math.min(w, Math.ceil(x2 * w)), Math.min(h, Math.ceil(y2 * h))];
      };
      const saySize = (text) => {
        if (sizeNote.textContent !== text) sizeNote.textContent = text;
      };
      // Draws the region. ``write``: the region was set on the picture (a drag, the arrow keys, a tracked person's
      // box), so the four numbers follow it, all of them. Otherwise the numbers are left alone: they are the viewer's
      // own text while typed, and a field is never rewritten under them, by this or by the page's updates.
      const drawBox = (write) => {
        const usable = !!zoom.box && zoom.box[2] > zoom.box[0] && zoom.box[3] > zoom.box[1];
        marker.hidden = !usable;
        if (write) {
          zoom.unset = "";
          inputs.forEach((input, i) => {
            input.value = usable ? String(Math.round(zoom.box[i] * 100)) : "";
            input.removeAttribute("aria-invalid");
          });
        }
        let enough = false;
        if (usable) {
          const [x1, y1, x2, y2] = zoom.box;
          Object.assign(marker.style, {left: `${x1 * 100}%`, top: `${y1 * 100}%`, width: `${(x2 - x1) * 100}%`,
            height: `${(y2 - y1) * 100}%`});
          const [px1, py1, px2, py2] = pixels();
          const w = px2 - px1;
          const h = py2 - py1;
          enough = w >= zoom.pin.min_px && h >= zoom.pin.min_px;
          const from = zoom.pin.source === "main_stream" ? "main stream" : "working picture";
          if (!zoom.unset) {
            saySize(enough ? `${w} × ${h} recorded pixels (${from}).`
              : `${w} × ${h} recorded pixels: too small to analyse (at least ${zoom.pin.min_px} on a side).`);
          }
        } else if (!zoom.unset) {
          saySize("Drag on the picture to select a region.");
        }
        if (zoom.unset) saySize(zoom.unset);
        zoomAsk.disabled = !enough || !!zoom.unset || chat.pending;  // never asked while the numbers are not the region
      };
      const setBox = (box) => {
        zoom.box = [clamp(Math.min(box[0], box[2])), clamp(Math.min(box[1], box[3])),
          clamp(Math.max(box[0], box[2])), clamp(Math.max(box[1], box[3]))];
        drawBox(true);
      };
      // The four numbers as typed: each edge's fraction (NaN for a field that is empty or not from 0 to 100), and
      // which fields keep them from being a region (not a number, or left not below right, top not below bottom).
      const typedBox = () => {
        const values = inputs.map((input) => {
          const text = input.value.trim();
          const value = text === "" ? NaN : Number(text);
          return Number.isFinite(value) && value >= 0 && value <= 100 ? value / 100 : NaN;
        });
        const bad = values.map((value) => Number.isNaN(value));
        if (!bad[0] && !bad[2] && values[0] >= values[2]) bad[0] = bad[2] = true;
        if (!bad[1] && !bad[3] && values[1] >= values[3]) bad[1] = bad[3] = true;
        return {values, bad};
      };
      const dropLoad = () => {  // the picture on its way, if there is one, is no longer wanted
        const load = zoom.loading;
        if (!load) return;
        zoom.loading = null;
        load.controller.abort();  // its request (the hold, or the picture) ends here
        refreshAsk();
      };
      const closeZoom = (text) => {
        dropLoad();
        zoomPanel.hidden = true;
        zoom.pin = null;
        zoom.box = null;
        zoom.unset = "";
        canvas.width = canvas.height = 0;  // the held picture is not kept in this page either
        sayZoom(text || "");
      };
      // Holds the newest live frame: the held document, or (as text) why no picture was held. It says nothing on
      // the page itself: its caller does, if the answer is still the one it waits for. ``given``: a signal that
      // ends the request when the picture is given up, so that the next hold is not sent beside it.
      const hold = async (given) => {
        const held = await request("/chat/hold", {method: "POST", headers, body: "{}"}, 8000, given);
        if (held.state !== "held" || typeof held.picture !== "string" || !JOB.test(held.picture)
            || !numbers(held.source_size, 2) || !numbers(held.shown_size, 2)) {
          return `No picture was held: ${typeof held.reason === "string" ? held.reason.slice(0, 200) : "the request failed"}.`;
        }
        return held;
      };
      const discard = (bitmap) => {
        if (bitmap && typeof bitmap.close === "function") bitmap.close();
      };
      // The held picture, decoded, or null. It ends within PICTURE_MS, fetching and decoding together, and at once
      // when ``controller`` is aborted (the picture was given up); one decoded after that is closed, not kept.
      const heldBitmap = async (picture, controller) => {
        let timer = null;
        const over = new Promise((resolve) => {
          controller.signal.addEventListener("abort", () => resolve(null), {once: true});
          timer = setTimeout(() => controller.abort(), PICTURE_MS);
        });
        const made = (async () => {
          try {
            const response = await fetch(`/chat/picture/${picture}`, {cache: "no-store", credentials: "same-origin",
              redirect: "error", signal: controller.signal, headers: {"X-Sentinel-Token": token}});
            return response.ok ? await createImageBitmap(await response.blob()) : null;
          } catch (error) {
            return null;
          }
        })();
        const bitmap = await Promise.race([made, over]);
        clearTimeout(timer);
        if (!bitmap) made.then(discard);
        return bitmap;
      };
      const facts = (held) => `the frame received at ${clock(held.frame && held.frame.ingest_utc)} `
        + `(stream epoch ${number(held.frame && held.frame.stream_epoch)}, frame ${number(held.frame && held.frame.frame_seq)})`;

      zoomOpen.addEventListener("click", async () => {
        if (zoomOpen.disabled || zoom.loading || chat.pending) return;
        const load = {controller: new AbortController()};
        zoom.loading = load;
        refreshAsk();  // Zoom and Scan are off from here until this picture is shown or given up
        try {
          sayZoom("Holding the newest picture…");
          const held = await hold(load.controller.signal);
          if (zoom.loading !== load) return;  // given up meanwhile (the panel was closed): nothing of it is shown
          if (typeof held === "string" || state.instance !== servedBy) {
            sayZoom(typeof held === "string" ? held : RESTARTED);
            return;
          }
          const bitmap = await heldBitmap(held.picture, load.controller);
          if (zoom.loading !== load || state.instance !== servedBy) {
            discard(bitmap);  // given up, or the runtime restarted: what is shown stays as it is
            if (zoom.loading === load) sayZoom(RESTARTED);
            return;
          }
          if (!bitmap) {
            sayZoom("The held picture could not be shown, so no region can be selected.");
            return;
          }
          zoom.pin = held;
          canvas.width = bitmap.width;
          canvas.height = bitmap.height;
          canvas.getContext("2d").drawImage(bitmap, 0, 0);
          discard(bitmap);  // drawn: the canvas is the page's only copy
          document.getElementById("zoom-facts").textContent = `Held picture: ${facts(held)}, kept for at most `
            + `${number(held.keep_s)} s. A region is cut from ${held.source === "main_stream" ? "the main stream's picture"
              : "the working picture"} (${held.source_size[0]} × ${held.source_size[1]} recorded pixels).`;
          const people = document.getElementById("zoom-people");
          people.replaceChildren(...(Array.isArray(held.people) ? held.people : []).filter((p) => p && numbers(p.box, 4))
            .slice(0, 8).map((p) => el("button", {type: "button", class: "ghost", "data-track-box": JSON.stringify(p.box)},
              [`Track ${number(p.track_id)} (the tracker's box)`])));
          zoom.box = null;
          zoomPanel.hidden = false;
          sayZoom("");
          drawBox(true);  // nothing selected yet: the four numbers are empty
          surface.focus();
        } finally {
          if (zoom.loading === load) {  // still this picture's turn: it ends here (given up, dropLoad ended it)
            zoom.loading = null;
            refreshAsk();
          }
        }
      });
      const at = (event) => {
        const rect = surface.getBoundingClientRect();
        return [clamp((event.clientX - rect.left) / rect.width), clamp((event.clientY - rect.top) / rect.height)];
      };
      surface.addEventListener("pointerdown", (event) => {
        if (!zoom.pin) return;
        zoom.drag = at(event);
        surface.setPointerCapture(event.pointerId);
        setBox([...zoom.drag, ...zoom.drag]);
        event.preventDefault();
      });
      surface.addEventListener("pointermove", (event) => {
        if (zoom.drag) setBox([...zoom.drag, ...at(event)]);
      });
      for (const name of ["pointerup", "pointercancel"]) {
        surface.addEventListener(name, () => { zoom.drag = null; });
      }
      surface.addEventListener("keydown", (event) => {  // the keyboard's way: arrows move, Shift+arrows resize
        const step = {ArrowLeft: [-0.02, 0], ArrowRight: [0.02, 0], ArrowUp: [0, -0.02], ArrowDown: [0, 0.02]}[event.key];
        if (!step || !zoom.pin) return;
        event.preventDefault();
        const [x1, y1, x2, y2] = zoom.box || [0.35, 0.35, 0.65, 0.65];
        if (event.shiftKey) setBox([x1, y1, clamp(x2 + step[0]), clamp(y2 + step[1])]);
        else {
          const dx = Math.min(1 - x2, Math.max(-x1, step[0]));
          const dy = Math.min(1 - y2, Math.max(-y1, step[1]));
          setBox([x1 + dx, y1 + dy, x2 + dx, y2 + dy]);
        }
      });
      // Typing a number. The four fields become the region only when they are one together, exactly as shown: no
      // edge is swapped with its opposite, no field is rewritten, and the edges that were not edited keep the numbers
      // they show. Until then (a first digit, a cleared field) the marked region stays as it was and cannot be asked
      // about, so what is asked is always what the numbers say.
      inputs.forEach((input) => input.addEventListener("input", () => {
        if (!zoom.pin) return;
        const {values, bad} = typedBox();
        inputs.forEach((item, i) => {
          if (bad[i]) item.setAttribute("aria-invalid", "true");
          else item.removeAttribute("aria-invalid");
        });
        if (bad.some(Boolean)) {
          zoom.unset = "These numbers are not a region yet (each from 0 to 100, left below right, top below bottom): "
            + "nothing changes and nothing can be asked until they are.";
        } else {
          zoom.unset = "";
          zoom.box = values;
        }
        drawBox(false);
      }));
      zoomPanel.addEventListener("click", (event) => {
        const person = event.target.closest("button[data-track-box]");
        if (person && zoom.pin) setBox(JSON.parse(person.dataset.trackBox));
      });
      document.getElementById("zoom-cancel").addEventListener("click", () => closeZoom("Cancelled: nothing was asked."));
      zoomAsk.addEventListener("click", () => {
        if (zoomAsk.disabled || !zoom.pin || !zoom.box || zoom.unset || chat.pending) return;
        const typed = box.value.trim();
        const question = typed && typed.length <= LIMIT ? typed : "Describe this area.";
        const region = {picture: zoom.pin.picture, box: zoom.box.map((v) => Math.round(v * 1e6) / 1e6)};
        const exchange = el("li", {class: "exchange", "data-question": question, "data-region": JSON.stringify(region)},
          [el("p", {class: "question"}, [`${question} (a selected region of the held picture)`])]);
        log.append(exchange);
        while (log.children.length > 12) log.firstElementChild.remove();
        closeZoom("");
        run(exchange, question, {region, history: []}, !!typed && question === typed);
      });

      // One question to its end without a card of its own (the scan's tiles); the last document.
      const askOnce = async (payload, limitMs) => {
        let document_ = await request("/chat", {method: "POST", headers, body: JSON.stringify(payload)}, 8000);
        const job = typeof document_.job === "string" && JOB.test(document_.job) ? document_.job : null;
        const started = performance.now();
        while (job && STATES[document_.state] || (document_.lost && job)) {
          if (performance.now() - started > limitMs) return {state: "timeout", reason: "no answer in time"};
          await new Promise((resolve) => setTimeout(resolve, 1000));
          document_ = await request(`/chat/${job}`, {headers: {"X-Sentinel-Token": token}}, 5000);
        }
        return document_;
      };
      const SCAN_QUESTION = "Describe what is visible in this part of the picture.";
      const SCAN_LIMIT_MS = (Number(chatForm.dataset.scanDeadline) || 120) * 1000;
      scanOpen.addEventListener("click", async () => {
        if (scanOpen.disabled || chat.pending || zoom.loading) return;
        chat.pending = true;  // the whole scan is one piece of work: no other question, manual or automatic, meanwhile
        refreshAsk();
        sayZoom("Holding the newest picture…");
        const answer = await hold();
        const held = typeof answer === "string" ? null : answer;
        if (!held) sayZoom(answer);
        const tiles = held && Array.isArray(held.tiles) ? held.tiles.filter((t) => t && typeof t.name === "string"
          && numbers(t.box, 4)).slice(0, 4) : [];
        if (!held || !tiles.length) {
          chat.pending = false;
          refreshAsk();
          return;
        }
        sayZoom("");
        const list = el("ul", {class: "lines tiles"});
        const stop = el("button", {type: "button", class: "ghost", "data-scan": "stop"}, ["Stop the scan"]);
        const exchange = el("li", {class: "exchange", "data-kind": "scan"},
          [el("p", {class: "question"}, ["Scan of the visible room (one held picture, four overlapping parts)"]), list,
            el("div", {class: "again"}, [stop])]);
        log.append(exchange);
        while (log.children.length > 12) log.firstElementChild.remove();
        const rows = tiles.map((tile) => {
          const row = el("li", {"data-tile": tile.name, "data-state": "waiting"}, [`${tile.name}: waiting…`]);
          list.append(row);
          return row;
        });
        let stopped = "";
        stop.addEventListener("click", () => { stopped = "stopped by you"; });
        status(exchange, "answering", `Scanning ${facts(held)}: one part at a time; monitoring comes first…`);
        const began = performance.now();
        let analysed = 0;
        for (let i = 0; i < tiles.length; i += 1) {
          const row = rows[i];
          let result = null;
          while (!stopped && !result) {
            if (!exchange.isConnected) stopped = "the card was closed";
            else if (performance.now() - began > SCAN_LIMIT_MS) stopped = "the scan's time ran out";
            else if (state.instance !== servedBy) stopped = "the runtime restarted";
            else if (["off", "unavailable"].includes(chatState())) stopped = `chat is ${chatState()}`;
            if (stopped) break;
            if (chatState() !== "available") {  // paused by the runtime: wait, sending nothing
              row.textContent = `${tiles[i].name}: waiting (chat is paused by the runtime)…`;
              await new Promise((resolve) => setTimeout(resolve, 1000));
              continue;
            }
            row.textContent = `${tiles[i].name}: asking…`;
            const document_ = await askOnce({question: SCAN_QUESTION, region: {picture: held.picture,
              box: tiles[i].box, tile: tiles[i].name}}, 40000);
            if (document_.state === "busy" || document_.state === "paused") {
              row.textContent = `${tiles[i].name}: waiting for its turn (${String(document_.reason || "busy").slice(0, 120)})…`;
              await new Promise((resolve) => setTimeout(resolve, 1500));
            } else {
              result = document_;
            }
          }
          if (!result) break;  // stopped before this part was asked; a part already answered is still shown
          if (result.state === "answered" && typeof result.answer === "string") {
            analysed += 1;
            row.dataset.state = "answered";
            const px = result.region && numbers(result.region.pixels, 4);
            row.textContent = `${tiles[i].name}${px ? ` (recorded pixels ${px[0]},${px[1]} to ${px[2]},${px[3]})` : ""}: `
              + result.answer.slice(0, 600);
          } else {
            row.dataset.state = "failed";
            row.textContent = `${tiles[i].name}: not analysed (${String(result.reason || result.state || "no answer").slice(0, 200)}).`;
            if (result.state === "picture_expired") stopped = "the held picture is gone";
          }
          if (stopped) break;
        }
        for (const row of rows) {
          if (row.dataset.state === "waiting") {
            row.dataset.state = "skipped";
            row.textContent = `${row.dataset.tile}: not analysed (${stopped || "the scan ended"}).`;
          }
        }
        stop.remove();
        status(exchange, analysed ? "answered" : "cancelled", `Scan ended: ${analysed} of ${tiles.length} parts analysed`
          + `${stopped ? ` (${stopped})` : ""}.`);
        line(exchange, "frame", `Historical: about ${facts(held)}, one held picture. Each part was asked about by itself; `
          + "nothing here combines them. It covers what the camera saw in that picture: not areas out of view, and not "
          + "a camera movement.");
        chat.pending = false;
        refreshAsk();
      });
      const refreshZoom = () => {
        const off = chat.pending || !!zoom.loading || chatState() !== "available" || state.instance !== servedBy;
        zoomOpen.disabled = off;
        scanOpen.disabled = off;
        if (zoom.pin) drawBox(false);  // the Ask button only: a number being typed is left as it is
      };
      const refreshAskBefore = refreshAsk;
      refreshAsk = () => {
        refreshAskBefore();
        refreshZoom();
      };
    }

    box.addEventListener("input", () => refreshAsk());
    refreshAsk();
  }

  // ------------------------------------------------------------------ feature controls (D99)

  // The operator's key is kept in this closure only: never in storage, a region, the address or a log; reloading the
  // page forgets it, and so does a runtime restart (its key is new). A switch can be used only while the key is
  // unlocked, this page is current (a stale page never acts on what it last saw), the runtime accepts changes and the
  // feature is switchable; one request at a time. The switches show the REQUESTED state; the actual state is the
  // server's, in the region, and updates with the page.
  const unlockForm = document.getElementById("controls-unlock");
  if (unlockForm) {
    const keyBox = document.getElementById("control-key");
    const unlocked = document.getElementById("controls-unlocked");
    const lock = document.getElementById("controls-lock");
    const note = document.getElementById("controls-note");
    const panel = document.querySelector('[data-region="features"]');
    const KEY = /^[A-Za-z2-7\s-]{24,40}$/;
    const ctl = {key: null, servedBy: null, sending: false, feature: null};
    const tell = (text) => {
      if (note.textContent !== text) note.textContent = text;
    };
    const meta = () => panel && panel.querySelector(".fmeta");
    const current = () => Number.isFinite(state.currentUntil) && performance.now() < state.currentUntil;
    const forget = (why) => {
      ctl.key = null;
      ctl.servedBy = null;
      if (why) tell(why);
    };
    const pageToken = () => state.token || unlockForm.dataset.token || "";

    refreshControls = () => {  // declared at the top, so that updates can call it
      const m = meta();
      const enabled = !!m && m.dataset.controls === "on";
      const accepting = !!m && m.dataset.accepting === "1";
      if (ctl.key && ctl.servedBy !== state.instance) {
        forget("The runtime restarted after these controls were unlocked: unlock them again with its new key.");
      }
      unlockForm.hidden = !enabled || !!ctl.key;
      unlocked.hidden = !enabled || !ctl.key;
      const live = current();
      for (const button of panel ? panel.querySelectorAll("button.switch") : []) {
        button.disabled = !(ctl.key && enabled && accepting && live && button.dataset.allowed === "1" && !ctl.sending);
        busy(button, ctl.sending && button.dataset.feature === ctl.feature);  // again after an update re-renders it
      }
      if (ctl.key && enabled && !live) tell("This page is not current: the switches wait for a current update.");
      else if (ctl.key && enabled && !accepting) tell("The runtime accepts changes only while it is running.");
      else if (ctl.key && note.dataset.stale === "1") tell("");
      note.dataset.stale = ctl.key && enabled && (!live || !accepting) ? "1" : "0";
    };

    const post = async (url, body, key) => {
      const controller = new AbortController();
      const timer = setTimeout(() => controller.abort(), 8000);
      try {
        const response = await fetch(url, {method: "POST", cache: "no-store", credentials: "same-origin",
          redirect: "error", signal: controller.signal, body: JSON.stringify(body),
          headers: {"Content-Type": "application/json", "X-Sentinel-Token": pageToken(), "X-Sentinel-Control": key}});
        const type = response.headers.get("Content-Type") || "";
        const text = type.startsWith("application/json") ? await response.text() : "";
        const parsed = text && text.length < 4000 ? JSON.parse(text) : null;
        return {status: response.status, doc: parsed && typeof parsed === "object" ? parsed : {}};
      } catch (error) {
        return {status: 0, doc: {reason: error && error.name === "AbortError" ? "no answer in time"
          : "the connection failed"}};
      } finally {
        clearTimeout(timer);
      }
    };
    const reason = (doc) => (typeof doc.reason === "string" ? doc.reason.slice(0, 200) : "");

    unlockForm.addEventListener("submit", async (event) => {
      event.preventDefault();
      const key = keyBox.value.trim();
      if (!KEY.test(key)) {
        tell("That is not a control key: six groups of four letters and digits.");
        return;
      }
      tell("Checking the key…");
      const submit = unlockForm.querySelector("button[type=submit]");
      busy(submit, true);
      const {status, doc} = await post("/features/unlock", {}, key);
      busy(submit, false);
      if (status === 200 && doc.state === "unlocked") {
        ctl.key = key;
        ctl.servedBy = state.instance;
        keyBox.value = "";
        tell("Unlocked. A switch asks the runtime for a change; the actual state below follows from the runtime.");
      } else {
        tell(status === 0 ? `Not checked: ${reason(doc)}.` : `Not unlocked: ${reason(doc) || `HTTP ${status}`}.`);
      }
      refreshControls();
    });
    lock.addEventListener("click", () => {
      forget("Locked: the key is forgotten in this tab.");
      refreshControls();
      keyBox.focus();
    });
    if (panel) {
      panel.addEventListener("click", async (event) => {
        const button = event.target.closest("button.switch");
        if (!button || button.disabled || !ctl.key) return;
        refreshControls();
        if (button.disabled) return;
        const feature = button.dataset.feature;
        const want = button.dataset.want;
        if (!/^[a-z_]{1,20}$/.test(feature) || (want !== "on" && want !== "off")) return;
        ctl.sending = true;
        ctl.feature = feature;
        refreshControls();
        tell(`Asking the runtime to turn ${want} ${feature.replace("_", " ")}…`);
        const {status, doc} = await post("/features", {feature, state: want}, ctl.key);
        ctl.sending = false;
        ctl.feature = null;
        if (status === 202) tell(`Requested: ${feature.replace("_", " ")} ${want} (change ${doc.change}). The actual `
          + "state below follows when the runtime applies it.");
        else if (status === 200) tell(`Already requested: ${feature.replace("_", " ")} ${want}.`);
        else if (status === 403 && doc.state === "refused" && /key/.test(reason(doc))) {
          forget("The key was refused (the runtime may have restarted): unlock again with its key.");
        } else tell(`Not changed: ${reason(doc) || (status ? `HTTP ${status}` : "no answer")}.`);
        refreshControls();
        if (status === 202) schedule(250);  // show the actual state soon
      });
    }
    setInterval(refreshControls, 1000);  // a page that stops being current disables the switches by itself
    refreshControls();
  }

  // ------------------------------------------------------------------ appearance (D96): applied in place, saved here

  const STORE = "sentinel.appearance";
  const PALETTES = (root.dataset.palettes || "").split(" ").filter(Boolean);

  function readSaved() {
    try {
      const saved = JSON.parse(window.localStorage.getItem(STORE) || "null");
      return saved && typeof saved === "object" && !Array.isArray(saved) ? saved : {};
    } catch (error) {
      return {};  // storage blocked: choices still apply to this view
    }
  }

  function save(changes) {
    try {
      window.localStorage.setItem(STORE, JSON.stringify({...readSaved(), ...changes}));
    } catch (error) {
      // not saved (private mode, blocked storage); the choice still applies to this page
    }
  }

  function current() {
    return {theme: root.dataset.theme || "auto", palette: root.dataset.palette || PALETTES[0] || "",
            density: root.dataset.density || "comfortable"};
  }

  function stripped(href) {  // a same-origin link without the address's appearance (the saved one applies instead)
    const url = new URL(href, location.href);
    if (url.origin !== location.origin) return null;
    for (const name of ["theme", "palette", "density"]) url.searchParams.delete(name);
    return url.pathname + url.search + url.hash;
  }

  function stripLinks() {
    for (const link of document.querySelectorAll("a[href^='/']")) {
      if (link.dataset.set) continue;
      const next = stripped(link.getAttribute("href"));
      if (next !== null && next !== link.getAttribute("href")) link.setAttribute("href", next);
    }
  }

  function markChoices() {
    const now = current();
    for (const control of document.querySelectorAll("[data-set]")) {
      if (control.dataset.value === now[control.dataset.set]) control.setAttribute("aria-current", "true");
      else control.removeAttribute("aria-current");
    }
  }

  function choose(kind, value) {
    if (kind === "theme" && ["auto", "light", "dark"].includes(value)) {
      if (value === "auto") delete root.dataset.theme;
      else root.dataset.theme = value;
    } else if (kind === "palette" && PALETTES.includes(value)) {
      if (value === PALETTES[0]) delete root.dataset.palette;
      else root.dataset.palette = value;
    } else if (kind === "density" && ["comfortable", "compact"].includes(value)) {
      if (value === "comfortable") delete root.dataset.density;
      else root.dataset.density = value;
    } else {
      return;
    }
    const now = current();
    save({theme: now.theme, palette: now.palette, density: now.density});
    // From now on the saved choice applies on every page of this server (appearance.js), so the address, the polled
    // source and the links stop carrying the address's appearance; nothing is reloaded.
    state.chosenInPlace = true;
    state.appearanceQuery = "";
    root.removeAttribute("data-pref");
    source = stripped(source) || source;
    const here = stripped(location.href);
    if (here !== null) history.replaceState(history.state, "", here);
    stripLinks();
    markChoices();
  }

  document.addEventListener("click", (event) => {
    const control = event.target.closest("[data-set]");
    if (!control || event.metaKey || event.ctrlKey || event.shiftKey || event.altKey) return;
    event.preventDefault();
    choose(control.dataset.set, control.dataset.value);
  });
  markChoices();

  // ------------------------------------------------------------------ viewing preferences (D100): this browser only

  // The live picture's size (fit, or its own pixels) and which optional panels show. Saved with the appearance; never
  // sent anywhere and never a feature change: hiding Ask does not turn chat off, and the picture is the same one.
  const OPTIONAL = ["chat", "health", "delivery"];
  const hidden = () => (root.dataset.hide || "").split(" ").filter((name) => OPTIONAL.includes(name));

  function markViewing() {
    const view = root.dataset.view === "actual" ? "actual" : "fit";
    for (const button of document.querySelectorAll("[data-view]")) {
      button.setAttribute("aria-pressed", button.dataset.view === view ? "true" : "false");
    }
    const hide = hidden();
    for (const button of document.querySelectorAll("button[data-panel]")) {
      const gone = hide.includes(button.dataset.panel);
      button.setAttribute("aria-pressed", gone ? "false" : "true");
      const word = button.querySelector(".pstate");
      if (word) word.textContent = gone ? ": hidden" : ": shown";
    }
  }

  function setHidden(list) {
    const unique = OPTIONAL.filter((name) => list.includes(name));
    if (unique.length) root.dataset.hide = unique.join(" ");
    else delete root.dataset.hide;
    save({hide: unique});
    markViewing();
  }

  document.addEventListener("click", (event) => {
    const view = event.target.closest("button[data-view]");
    if (view && (view.dataset.view === "fit" || view.dataset.view === "actual")) {
      if (view.dataset.view === "actual") root.dataset.view = "actual";
      else delete root.dataset.view;
      save({view: view.dataset.view});
      markViewing();
      return;
    }
    const panel = event.target.closest("button[data-panel]");
    if (panel && OPTIONAL.includes(panel.dataset.panel)) {
      const hide = hidden();
      setHidden(hide.includes(panel.dataset.panel) ? hide.filter((n) => n !== panel.dataset.panel)
        : [...hide, panel.dataset.panel]);
      return;
    }
    const show = event.target.closest("button[data-show]");
    if (show && OPTIONAL.includes(show.dataset.show)) {
      setHidden(hidden().filter((n) => n !== show.dataset.show));
      const target = document.getElementById(show.dataset.show);
      if (target) target.scrollIntoView({block: "start"});
    }
  });
  markViewing();

  // ------------------------------------------------------------------ the sidebar: rail, drawer, the section in view

  const sidebar = document.getElementById("sidebar");
  const menu = document.getElementById("menu");
  const scrim = document.getElementById("scrim");
  const railToggle = document.getElementById("rail-toggle");
  const narrow = window.matchMedia ? window.matchMedia("(max-width: 899px)") : {matches: false};

  function setRail(compact) {
    if (compact) root.dataset.rail = "compact";
    else delete root.dataset.rail;
    if (railToggle) {
      railToggle.setAttribute("aria-expanded", compact ? "false" : "true");
      railToggle.querySelector(".vh").textContent = compact ? "Expand the sidebar" : "Collapse the sidebar";
    }
  }
  if (railToggle) {
    setRail(root.dataset.rail === "compact");
    railToggle.addEventListener("click", () => {
      const compact = root.dataset.rail !== "compact";
      setRail(compact);
      save({rail: compact ? "compact" : "full"});
    });
  }

  function focusable() {
    return [...sidebar.querySelectorAll("a[href], button:not([disabled])")].filter((node) => node.offsetParent !== null);
  }

  function openDrawer() {
    if (!sidebar || !menu) return;
    root.classList.add("drawer-open");
    menu.setAttribute("aria-expanded", "true");
    if (scrim) scrim.hidden = false;
    const first = sidebar.querySelector(".nav a[aria-current]") || sidebar.querySelector(".nav a");
    if (first) first.focus();
  }

  function closeDrawer(returnFocus) {
    if (!root.classList.contains("drawer-open")) return;
    root.classList.remove("drawer-open");
    if (menu) menu.setAttribute("aria-expanded", "false");
    if (scrim) scrim.hidden = true;
    if (returnFocus && menu) menu.focus();
  }

  if (menu) menu.addEventListener("click", () => openDrawer());
  const drawerClose = document.getElementById("drawer-close");
  if (drawerClose) drawerClose.addEventListener("click", () => closeDrawer(true));
  if (scrim) scrim.addEventListener("click", () => closeDrawer(true));
  if (narrow.addEventListener) narrow.addEventListener("change", () => { if (!narrow.matches) closeDrawer(false); });
  document.addEventListener("keydown", (event) => {
    if (event.key === "Escape" && root.classList.contains("drawer-open")) {
      event.preventDefault();
      event.stopImmediatePropagation();  // Escape closes the drawer only, not the focus view behind it
      closeDrawer(true);
      return;
    }
    if (event.key === "Tab" && root.classList.contains("drawer-open") && sidebar) {  // focus stays in the open drawer
      const nodes = focusable();
      if (!nodes.length) return;
      const first = nodes[0];
      const last = nodes[nodes.length - 1];
      if (event.shiftKey && document.activeElement === first) {
        event.preventDefault();
        last.focus();
      } else if (!event.shiftKey && (document.activeElement === last || !sidebar.contains(document.activeElement))) {
        event.preventDefault();
        first.focus();
      }
    }
  });

  const sectionLinks = sidebar ? [...sidebar.querySelectorAll(".nav a[data-nav]")] : [];
  let heldUntil = 0;  // after a click, the chosen section stays marked while the page scrolls to it
  if (KIND === "status" && sectionLinks.length) {
    sidebar.addEventListener("click", (event) => {
      const link = event.target.closest(".nav a[data-nav]");
      if (!link) return;
      if (root.dataset.focus) leaveFocus(false);  // a section outside the focus view is shown again first
      markSection(link.dataset.nav);
      heldUntil = performance.now() + 1200;
      if (narrow.matches) closeDrawer(false);
    });
    const sections = sectionLinks.map((link) => document.getElementById(link.dataset.nav)).filter(Boolean);
    let pending = false;
    const spy = () => {
      pending = false;
      if (performance.now() < heldUntil) return;
      const line = window.innerHeight * 0.3;
      let shown = sections[0];
      let best = -Infinity;
      for (const section of sections) {  // the lowest top above the line; side by side, the first of them
        const top = section.getBoundingClientRect().top;
        if (section.offsetParent !== null && top <= line && top > best + 1) {
          shown = section;
          best = top;
        }
      }
      if (window.innerHeight + window.scrollY >= document.documentElement.scrollHeight - 2) {
        shown = sections.filter((section) => section.offsetParent !== null).pop() || shown;
      }
      if (shown) markSection(shown.id);
    };
    window.addEventListener("scroll", () => {
      if (!pending) {
        pending = true;
        requestAnimationFrame(spy);
      }
    }, {passive: true});
    spy();
  }

  function markSection(id) {
    for (const link of sectionLinks) {
      if (link.dataset.nav === id) link.setAttribute("aria-current", "true");
      else link.removeAttribute("aria-current");
    }
  }

  // ------------------------------------------------------------------ Settings: Appearance and Features in a dialog
  //
  // Part 28. The page's own Appearance and Features sections (the same elements: one set of controls, the same region
  // for updates, the same unlock form) are moved into the Settings dialog, one shown at a time. The sidebar's
  // Appearance and Features open it instead of scrolling; so does an address ending in #settings or #features, and
  // the "Open Features" button of a warning kept on the page. A native modal dialog: the page behind is inert, Escape
  // and the close button close it, focus stays inside and returns to what opened it. Opening, switching and closing
  // change nothing else: no request is made, the typed key and the unlocked state stay as they are, and the page's
  // updates go on behind and inside it. Without script none of this runs and both sections stay on the page.
  const settingsDialog = document.getElementById("settings-dialog");
  const SETTINGS = ["settings", "features"];
  let openSettings = () => false;
  if (KIND === "status" && settingsDialog && typeof settingsDialog.showModal === "function") {
    const panel = document.getElementById("sd-panel");
    const tabs = [...settingsDialog.querySelectorAll("a[data-sd]")];
    const parts = SETTINGS.map((id) => document.getElementById(id)).filter(Boolean);
    let opener = null;
    for (const part of parts) {
      part.hidden = true;
      panel.append(part);
    }
    root.classList.add("sd");
    const select = (id) => {
      for (const part of parts) part.hidden = part.id !== id;
      for (const tab of tabs) {
        if (tab.dataset.sd === id) tab.setAttribute("aria-current", "true");
        else tab.removeAttribute("aria-current");
      }
      if (settingsDialog.dataset.section !== id) panel.scrollTop = 0;
      settingsDialog.dataset.section = id;
      panel.setAttribute("aria-labelledby", `h-${id}`);
    };
    const inside = () => [...settingsDialog.querySelectorAll(
      "a[href], button:not([disabled]), input:not([disabled]), select, textarea, summary, [tabindex]:not([tabindex='-1'])",
    )].filter((node) => node.offsetParent !== null);
    openSettings = (id, from) => {
      if (!parts.some((part) => part.id === id)) return false;
      select(id);
      if (!settingsDialog.open) {
        opener = from || null;
        settingsDialog.showModal();
        markSection(id);
        heldUntil = Infinity;  // the opened section stays marked in the sidebar while the dialog is open
      } else {
        markSection(id);
      }
      const tab = tabs.find((item) => item.dataset.sd === id);
      if (tab) tab.focus();
      return true;
    };
    settingsDialog.addEventListener("close", () => {
      heldUntil = 0;
      window.dispatchEvent(new Event("scroll"));  // the sidebar marks the section in view again
      if (SETTINGS.includes(location.hash.slice(1))) {
        history.replaceState(null, "", location.pathname + location.search);
      }
      // Focus returns to what opened the dialog. When that is gone or hidden (a warning's button that an update
      // removed meanwhile, the drawer's link, an address that opened it), it goes to a control that is in view and
      // opens the same thing: the menu button where the sidebar is a drawer, otherwise the sidebar's own link for
      // the section that was shown. Never left on the page body.
      const shown = (node) => !!node && node.isConnected && node.offsetParent !== null;
      const link = sectionLinks.find((item) => item.dataset.nav === settingsDialog.dataset.section);
      const back = [opener, menu, link].find(shown);
      opener = null;
      if (back) back.focus();
    });
    document.getElementById("sd-close").addEventListener("click", () => settingsDialog.close());
    settingsDialog.addEventListener("click", (event) => {
      const tab = event.target.closest("a[data-sd]");
      if (tab) {
        event.preventDefault();
        select(tab.dataset.sd);
        markSection(tab.dataset.sd);
      } else if (event.target === settingsDialog) {
        settingsDialog.close();  // a click on the dimmed page behind it
      }
    });
    settingsDialog.addEventListener("keydown", (event) => {
      if (event.key === "Escape") {
        event.stopPropagation();  // Escape closes the dialog only (the browser does), not the focus view behind it
        return;
      }
      const at = tabs.indexOf(document.activeElement);
      if (at >= 0 && (event.key === "ArrowDown" || event.key === "ArrowUp")) {
        event.preventDefault();
        const next = tabs[(at + (event.key === "ArrowDown" ? 1 : tabs.length - 1)) % tabs.length];
        select(next.dataset.sd);
        markSection(next.dataset.sd);
        next.focus();
        return;
      }
      if (event.key === "Tab") {  // focus stays in the dialog, wrapping at both ends
        const items = inside();
        if (!items.length) return;
        const first = items[0];
        const last = items[items.length - 1];
        if (event.shiftKey && (document.activeElement === first || !settingsDialog.contains(document.activeElement))) {
          event.preventDefault();
          last.focus();
        } else if (!event.shiftKey && document.activeElement === last) {
          event.preventDefault();
          first.focus();
        }
      }
    });
    document.addEventListener("click", (event) => {  // before the sidebar's own handler: these never scroll the page
      const link = event.target.closest('.nav a[data-nav="settings"], .nav a[data-nav="features"]');
      const button = link ? null : event.target.closest("button[data-open-settings]");
      const id = link ? link.dataset.nav : button ? button.dataset.openSettings : null;
      if (!id || !SETTINGS.includes(id)) return;
      event.preventDefault();
      event.stopPropagation();
      const drawer = root.classList.contains("drawer-open");
      if (drawer) closeDrawer(false);
      openSettings(id, drawer ? menu : link || button);
    }, true);
    const fromAddress = () => {
      const id = location.hash.slice(1);
      if (SETTINGS.includes(id)) openSettings(id, null);
    };
    window.addEventListener("hashchange", fromAddress);
    fromAddress();
  }

  // ------------------------------------------------------------------ the camera's focus view (D96)

  const focusToggle = document.getElementById("focus-toggle");
  let focusReturn = 0;

  function enterFocus() {
    focusReturn = window.scrollY;
    root.dataset.focus = "live";
    focusToggle.setAttribute("aria-pressed", "true");
    focusToggle.querySelector("span").textContent = "Exit focus";
    focusToggle.querySelector("use").setAttribute("href", "#i-shrink");
    window.scrollTo(0, 0);  // the freshness bar stays in view above the picture
    if (player) player.resume();
  }

  function leaveFocus(restore) {
    delete root.dataset.focus;
    focusToggle.setAttribute("aria-pressed", "false");
    focusToggle.querySelector("span").textContent = "Focus";
    focusToggle.querySelector("use").setAttribute("href", "#i-expand");
    if (restore) window.scrollTo(0, focusReturn);
  }

  if (focusToggle) {
    focusToggle.addEventListener("click", () => (root.dataset.focus ? leaveFocus(true) : enterFocus()));
    document.addEventListener("keydown", (event) => {
      if (event.key === "Escape" && root.dataset.focus) {
        leaveFocus(true);
        focusToggle.focus();
      }
    });
  }

  // ------------------------------------------------------------------ filtering the loaded incidents (D96)

  const filterStatus = document.getElementById("f-status");
  const filterSeverity = document.getElementById("f-severity");
  const filterCount = document.getElementById("f-count");
  const filterEmpty = document.getElementById("f-empty");

  function filterIncidents() {
    if (!filterStatus || !filterSeverity) return;
    const wantStatus = filterStatus.value;
    const wantSeverity = filterSeverity.value;
    const items = [...document.querySelectorAll(".inc-list > li")];
    let shown = 0;
    for (const item of items) {
      const match = (!wantStatus || item.dataset.status === wantStatus)
        && (!wantSeverity || item.dataset.severity === wantSeverity);
      item.hidden = !match;
      if (match) shown += 1;
    }
    const filtering = Boolean(wantStatus || wantSeverity);
    const text = !items.length ? ""
      : filtering ? `Showing ${shown} of the ${items.length} loaded incidents (the latest only; older ones are not searched).`
        : `${items.length} loaded incidents (the latest only).`;
    if (filterCount && filterCount.textContent !== text) filterCount.textContent = text;
    if (filterEmpty) filterEmpty.hidden = !(filtering && items.length && !shown);
  }

  for (const control of [filterStatus, filterSeverity]) {
    if (control) control.addEventListener("change", filterIncidents);
  }
  filterIncidents();

  // ------------------------------------------------------------------ lifecycle

  document.addEventListener("visibilitychange", () => {
    // A browser may pause animations in a hidden tab and resume them on return: expire every claim at both edges, so
    // that only an update received after the return can make anything current again.
    expireNow();
    if (document.visibilityState === "visible") {
      update();
      if (player) player.resume();
    }
  });
  window.addEventListener("pageshow", (event) => {
    if (event.persisted) {
      expireNow();
      update();
      if (player) player.resume();
    }
  });
  document.addEventListener("resume", () => {
    expireNow();
    update();
    if (player) player.resume();
  });
  window.addEventListener("online", () => {
    update();
    if (player) player.resume();
  });

  setTimeout(() => root.classList.remove("enter"), 700);  // the entrance plays once, never on an update (D96)
  if (KIND === "status") {
    player = makePlayer();
    if (player) player.run();
    const wanted = /^#incident-([A-Za-z0-9][A-Za-z0-9_-]{0,127})$/.exec(location.hash);
    if (wanted) select(wanted[1]);
  }
  schedule(EVERY);
})();
