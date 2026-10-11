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
     the draft stay in this tab; the token in the form belongs to the server that sent this page. */
(() => {
  "use strict";
  const root = document.documentElement;
  const SOURCE = root.dataset.source;
  const EVERY = Number(root.dataset.refreshMs);
  const WINDOW = Number(root.dataset.currentMs);
  const KIND = root.dataset.page;
  if (!SOURCE || !SOURCE.startsWith("/") || SOURCE.startsWith("//") || !(EVERY > 0) || !(WINDOW > 0)) return;
  const MAX_TEXT = 600000;
  const MAX_WAIT = 15000;
  const INSTANCE = /^[0-9a-f]{16}$/;
  const TOKEN = /^[0-9a-f]{32}$/;
  const INCIDENT = /^[A-Za-z0-9][A-Za-z0-9_-]{0,127}$/;
  const state = {
    instance: root.dataset.instance || null, sequence: Number(root.dataset.sequence || "-1"),
    last: Object.create(null), timer: 0, inflight: null, failures: 0, token: null,
    selected: null, detail: null, detailTimer: 0,
  };
  root.classList.add("js");
  let refreshAsk = () => {};  // the chat's Ask button follows chat's availability, which updates with the page
  let player = null;  // the live player, on the status page with the live view on

  // ------------------------------------------------------------------ the watchdog timers

  function restart(name, sentAt) {
    // A new animation name restarts every timer of this kind; the negative delay ends the window WINDOW ms after
    // the request was sent (a bound on the age of what it brought, on this browser's monotonic clock).
    const elapsed = Math.max(0, performance.now() - sentAt);
    root.style.setProperty(`--${name}-delay`, `${-Math.round(elapsed)}ms`);
    root.dataset[name] = root.dataset[name] === "1" ? "0" : "1";
  }

  function expireNow() {  // after a freeze or a back-forward restore: nothing is current until an update arrives
    root.style.setProperty("--fresh-delay", `${-10 * WINDOW}ms`);
    root.dataset.fresh = root.dataset.fresh === "1" ? "0" : "1";
    if (player) player.expire();
  }

  const navigation = performance.getEntriesByType ? performance.getEntriesByType("navigation")[0] : null;
  const requested = navigation && navigation.requestStart > 0 ? navigation.requestStart : 0;
  restart("fresh", requested);

  // ------------------------------------------------------------------ keyed placement of server fragments

  function key(node) {
    return node.nodeType === 1 ? node.getAttribute("data-key") || node.id || null : null;
  }

  function same(a, b) {
    return a.nodeType === b.nodeType && a.nodeName === b.nodeName && key(a) === key(b);
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
      target.removeChild(cursor);
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
  function say(text) {
    if (sync && sync.textContent !== text) sync.textContent = text;
  }

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
      const response = await fetch(SOURCE, {
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
      say(`No current update (${why(error)}); trying again in ${Math.round(wait / 1000)} s.`);
      schedule(wait);
    } finally {
      clearTimeout(timeout);
      if (state.inflight === controller) state.inflight = null;
    }
  }

  function apply(doc, sent) {
    if (doc.page.instance !== state.instance) state.last = Object.create(null);
    // New elements pick up the bounded delay as they appear; the restart below then aligns every timer.
    root.style.setProperty("--fresh-delay", `${-Math.round(performance.now() - sent)}ms`);
    for (const [name, html] of Object.entries(doc.page.regions)) place(name, html);
    if (KIND === "status") {
      if (doc.page.token) state.token = doc.page.token;
      if (player) player.status(doc);
    }
    markSelected();
    state.instance = doc.page.instance;
    state.sequence = doc.page.sequence;
    root.dataset.instance = doc.page.instance;
    root.dataset.sequence = String(doc.page.sequence);
    refreshAsk();
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
    };

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
        if (preview && preview.enabled === false) setState("off", preview.reason);
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
    const theme = root.dataset.theme ? `?theme=${root.dataset.theme}` : "";
    const children = [
      el("div", {class: "phead"}, [el("h3", {}, [incident.title || ""]),
        el("button", {type: "button", class: "plain", "data-close": "1"}, ["Close"])]),
      el("div", {class: "tags"}, [badge(sevKind, capital(incident.severity), sevIcon),
        badge(stKind, capital(incident.status), stIcon),
        ack.state === "acknowledged" ? badge("done", "Acknowledged", "check") : badge("off", "Not acknowledged", "bell")]),
    ];
    const view = `/incidents/${incident.incident_id}/snapshot.jpg`;
    if (snapshot.state === "saved" && snapshot.file === "verified" && snapshot.view_url === view) {
      children.push(el("div", {class: "frame thumb"}, [el("img", {src: view, alt: "Incident snapshot",
        width: String(snapshot.width || 640), height: String(snapshot.height || 480)})]));
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
    state.selected = id && INCIDENT.test(id) ? id : null;
    const pane = document.getElementById("inc-preview");
    if (!state.selected && pane) {
      pane.hidden = true;
      pane.replaceChildren();
      if (state.detail) state.detail.abort();
      clearTimeout(state.detailTimer);
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
                    cancelled: "Cancelled", paused: "Paused"};
    const REASONS = {unknown: "the server no longer has this question; it may have restarted"};
    const GLYPHS = {sending: "clock", waiting: "clock", answering: "clock", answered: "check", busy: "warn",
                    no_fresh_frame: "warn", timeout: "warn", unavailable: "off", off: "off", unknown: "off",
                    cancelled: "warn", paused: "clock"};

    const chatState = () => {
      const element = document.querySelector("[data-chat-state]");
      return element ? element.dataset.chatState : "off";
    };

    refreshAsk = () => {  // declared at the top, so that updates can call it
      const available = chatState() === "available";
      const restarted = state.instance !== servedBy;
      const text = box.value.trim();
      ask.disabled = chat.pending || !available || restarted || !text || text.length > LIMIT;
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
    // the answer was ready: an answer is never about the scene now.
    const label = (document_) => {
      const frame = document_.frame;
      if (!frame || typeof frame !== "object") return "";
      const which = `the frame received at ${clock(frame.ingest_utc)} (stream epoch ${number(frame.stream_epoch)}, `
        + `frame ${number(frame.frame_seq)})`;
      if (document_.state !== "answered") return `About ${which}, ${seconds(frame.age_ms_when_asked)} old when asked.`;
      const lead = document_.historical === false ? "About" : "Historical: about";
      return `${lead} ${which}, ${seconds(frame.age_ms_when_asked)} old when asked and `
        + `${seconds(document_.age_ms_when_answered)} old when answered. It describes that frame only, not the scene now.`;
    };

    // Shows a server document on one exchange; true when the exchange is finished.
    const show = (exchange, document_, question) => {
      const state_ = typeof document_.state === "string" ? document_.state : "error";
      if (STATES[state_]) {
        status(exchange, state_, STATES[state_]);
        return false;
      }
      if (state_ === "answered" && typeof document_.answer === "string") {
        status(exchange, "answered",
          document_.cut_short === true ? "Answered (cut short at the answer limit)." : "Answered.");
        line(exchange, "answer", document_.answer.slice(0, 1200));
        line(exchange, "frame", label(document_));
        chat.history.push({question, answer: document_.answer.slice(0, 400)});
        chat.history = chat.history.slice(-2);
        return true;
      }
      const reason = typeof document_.reason === "string" ? document_.reason.slice(0, 300)
        : REASONS[state_] || "the question could not be answered";
      status(exchange, TITLES[state_] ? state_ : "error", `${TITLES[state_] || "Error"}: ${reason}.`);
      if (document_.frame) line(exchange, "frame", label(document_));
      return true;
    };

    // One bounded request; always an object. ``lost`` marks a request that did not get a readable answer.
    const request = async (url, options, ms) => {
      const controller = new AbortController();
      const timer = setTimeout(() => controller.abort(), ms);
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

    chatForm.addEventListener("submit", async (event) => {
      event.preventDefault();
      refreshAsk();
      if (ask.disabled) return;
      const question = box.value.trim();
      chat.pending = true;
      refreshAsk();
      const exchange = el("li", {class: "exchange"}, [el("p", {class: "question"}, [question])]);
      log.append(exchange);
      while (log.children.length > 12) log.firstElementChild.remove();
      status(exchange, "sending", STATES.sending);
      try {
        let document_ = await request("/chat", {method: "POST", headers: {"Content-Type": "application/json",
          "X-Sentinel-Token": token}, body: JSON.stringify({question, history: chat.history})}, 8000);
        const job = typeof document_.job === "string" && JOB.test(document_.job) ? document_.job : null;
        if (job) box.value = "";  // handed over; a question that was not taken stays in the box, to send again
        const started = performance.now();
        for (;;) {
          if (document_.lost && job) {  // a poll that got no answer: the question may still be answered
            status(exchange, "waiting", `No reply from the page server (${document_.reason}); still waiting…`);
          } else if (show(exchange, document_, question)) {
            break;
          } else if (!job) {
            show(exchange, {state: "error", reason: "the answer was not readable"}, question);
            break;
          }
          if (performance.now() - started > 40000) {
            show(exchange, {state: "timeout", reason: "no answer in time"}, question);
            break;
          }
          await new Promise((resolve) => setTimeout(resolve, 1000));
          document_ = await request(`/chat/${job}`, {headers: {"X-Sentinel-Token": token}}, 5000);
        }
      } finally {
        chat.pending = false;
        refreshAsk();
      }
    });
    box.addEventListener("input", () => refreshAsk());
    refreshAsk();
  }

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

  if (KIND === "status") {
    player = makePlayer();
    if (player) player.run();
    const wanted = /^#incident-([A-Za-z0-9][A-Za-z0-9_-]{0,127})$/.exec(location.hash);
    if (wanted) select(wanted[1]);
  }
  schedule(EVERY);
})();
