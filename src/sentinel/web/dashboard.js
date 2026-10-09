/* Sentinel status pages: in-place updates (D73).

   Served from this origin only (CSP: script-src 'self'; connect-src 'self'). No storage, no cookies, no third-party
   code. Without it the pages still work, read-only: they show what the server rendered and turn NOT CURRENT after the
   watchdog window; reload to update.

   - Every fragment is rendered and escaped by the server; this script only places it, keyed, so that elements stay
     where they are (focus, expanded sections, text that did not change, the selected incident) and the whole page is
     never rebuilt or reloaded.
   - The D71 timers (the CSS watchdog) restart only after a document that was received with HTTP 200, parsed and
     validated, and is newer than the last one (server instance and sequence). Their window counts from when that
     document's request was sent, so a delayed answer gets less time, never more, and one that arrives after the
     window is not applied. Nothing else restarts them: not a failed or repeated answer, not an image, not the chat.
   - One request at a time; a request is abandoned after the window; failures back off (2, 4, 8, at most 15 s). A
     hidden tab is not polled, and its claims expire when it is hidden and again when it is shown; on return the page
     updates at once.
   - The live picture is loaded and decoded before it replaces the one shown, so it never blanks. It has its own
     timer, from the request that brought it. A picture that could not be loaded, or a frame that is no longer live,
     leaves the last picture in place, dimmed and labelled as an old frame with its time. */
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
  const PICTURE = /^\/preview\/[0-9]{1,9}-[0-9]{1,12}\.jpg$/;
  const INSTANCE = /^[0-9a-f]{16}$/;
  const INCIDENT = /^[A-Za-z0-9][A-Za-z0-9_-]{0,127}$/;
  const state = {
    instance: root.dataset.instance || null, sequence: Number(root.dataset.sequence || "-1"),
    last: Object.create(null), timer: 0, inflight: null, failures: 0, picture: null,
    selected: null, detail: null, detailTimer: 0,
  };
  root.classList.add("js");

  // ------------------------------------------------------------------ the watchdog timers

  function restart(name, sentAt) {
    // A new animation name restarts every timer of this kind; the negative delay ends the window WINDOW ms after
    // the request was sent (a bound on the age of what it brought, on this browser's monotonic clock).
    const elapsed = Math.max(0, performance.now() - sentAt);
    root.style.setProperty(`--${name}-delay`, `${-Math.round(elapsed)}ms`);
    root.dataset[name] = root.dataset[name] === "1" ? "0" : "1";
  }

  function expireNow() {  // after a freeze or a back-forward restore: nothing is current until an update arrives
    for (const name of ["fresh", "pic"]) {
      root.style.setProperty(`--${name}-delay`, `${-10 * WINDOW}ms`);
      root.dataset[name] = root.dataset[name] === "1" ? "0" : "1";
    }
  }

  const navigation = performance.getEntriesByType ? performance.getEntriesByType("navigation")[0] : null;
  const requested = navigation && navigation.requestStart > 0 ? navigation.requestStart : 0;
  restart("fresh", requested);
  restart("pic", requested);

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
      const preview = doc.preview;
      if (preview !== undefined && (preview === null || typeof preview !== "object")) return false;
      const picture = preview && preview.picture;
      if (picture != null) {
        if (typeof picture !== "object" || typeof picture.url !== "string" || !PICTURE.test(picture.url)) return false;
        for (const field of ["width", "height", "stream_epoch", "frame_seq"]) {
          if (!Number.isSafeInteger(picture[field]) || picture[field] < 0) return false;
        }
        if (picture.width < 1 || picture.height < 1 || picture.width > 4096 || picture.height > 4096) return false;
      }
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
      if (newer(doc.page)) {
        if (performance.now() - sent >= WINDOW) throw new Error("answer arrived too late");
        await apply(doc, sent);
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

  async function apply(doc, sent) {
    const next = KIND === "status" ? await nextPicture(doc.preview) : null;
    if (doc.page.instance !== state.instance) state.last = Object.create(null);
    // New elements pick up the bounded delay as they appear; the restart below then aligns every timer.
    root.style.setProperty("--fresh-delay", `${-Math.round(performance.now() - sent)}ms`);
    for (const [name, html] of Object.entries(doc.page.regions)) place(name, html);
    if (next) showFrame(doc, next, sent);
    markSelected();
    state.instance = doc.page.instance;
    state.sequence = doc.page.sequence;
    root.dataset.instance = doc.page.instance;
    root.dataset.sequence = String(doc.page.sequence);
    restart("fresh", sent);
  }

  // ------------------------------------------------------------------ the live picture

  function preload(url) {
    return new Promise((resolve) => {
      const img = new Image();
      let done = false;
      const finish = (value) => {
        if (!done) {
          done = true;
          clearTimeout(timer);
          resolve(value);
        }
      };
      const timer = setTimeout(() => finish(null), Math.min(WINDOW, 4000));
      img.onload = () => (img.decode ? img.decode() : Promise.resolve()).then(() => finish(img), () => finish(null));
      img.onerror = () => finish(null);
      img.src = url;
    });
  }

  async function nextPicture(preview) {
    const picture = preview && preview.picture;
    if (!picture) return {state: "none", preview};
    if (state.picture && state.picture.url === picture.url) return {state: "same", picture};
    const img = await preload(picture.url);
    return img ? {state: "loaded", picture, img} : {state: "failed", picture};
  }

  function clock(iso) {
    const when = new Date(iso);
    return Number.isNaN(when.getTime()) ? "an unknown time" : `${when.toISOString().slice(11, 23)} UTC`;
  }

  function showFrame(doc, next, sent) {
    const slot = document.getElementById("frame");
    if (!slot) return;
    let figure = slot.querySelector("figure.frame");
    if (next.state === "loaded") {
      if (!figure) {
        slot.replaceChildren(buildFigure());
        figure = slot.querySelector("figure.frame");
        delete state.last["frame-empty"];  // the placeholder is gone; place it again when it is needed
      }
      const picture = next.picture;
      const img = next.img;
      img.width = picture.width;
      img.height = picture.height;
      img.alt = `Camera picture, stream epoch ${picture.stream_epoch}, frame ${picture.frame_seq}`;
      const shown = figure.querySelector("img");
      if (shown) shown.replaceWith(img);
      else figure.prepend(img);
      figure.dataset.state = "live";
      figure.dataset.frame = `${picture.stream_epoch}-${picture.frame_seq}`;
      figure.dataset.ingest = picture.ingest_utc || "";
      state.picture = picture;
      restart("pic", sent);
    } else if (next.state === "same" && figure && figure.dataset.state !== "failed") {
      figure.dataset.state = "live";  // a newer, valid update says this very frame is still live
      restart("pic", sent);
    } else if (next.state === "failed" || next.state === "none") {
      if (figure && state.picture) {
        figure.dataset.state = next.state === "failed" ? "failed" : "retained";
        const kept = figure.querySelector(".kept span");
        const when = clock(state.picture.ingest_utc);
        const which = `stream epoch ${state.picture.stream_epoch}, frame ${state.picture.frame_seq}`;
        const label = next.state === "failed"
          ? `OLD FRAME: the newest picture could not be loaded; this one was received at ${when} (${which}).`
          : `OLD FRAME: the camera has no current frame; this one was received at ${when} (${which}).`;
        if (kept && kept.textContent !== label) kept.textContent = label;
      } else {
        place("frame-empty", doc.page.frame_empty);
      }
    }
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

  function buildFigure() {  // the same structure the server renders around a live picture
    return el("figure", {class: "frame"}, [
      el("figcaption", {class: "pic-old"}, [svgIcon("clock"),
        `NOT CURRENT: this picture is old: no newer one has arrived for ${Math.round(WINDOW / 1000)} s.`]),
      el("figcaption", {class: "kept"}, [svgIcon("clock"), el("span")]),
    ]);
  }

  function initialPicture() {
    const figure = document.querySelector("#frame figure.frame");
    const img = figure && figure.querySelector("img");
    if (!img) return;
    const [epoch, seq] = (figure.dataset.frame || "").split("-").map(Number);
    state.picture = {url: img.getAttribute("src"), ingest_utc: figure.dataset.ingest || "", stream_epoch: epoch,
                     frame_seq: seq};
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

  function badge(kind, text, glyph) {
    return el("span", {class: `badge k-${kind}`}, [svgIcon(glyph), text]);
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

  // ------------------------------------------------------------------ lifecycle

  document.addEventListener("visibilitychange", () => {
    // A browser may pause animations in a hidden tab and resume them on return: expire every claim at both edges, so
    // that only an update received after the return can make anything current again.
    expireNow();
    if (document.visibilityState === "visible") update();
  });
  window.addEventListener("pageshow", (event) => {
    if (event.persisted) {
      expireNow();
      update();
    }
  });
  document.addEventListener("resume", () => {
    expireNow();
    update();
  });
  window.addEventListener("online", () => update());

  if (KIND === "status") {
    initialPicture();
    const wanted = /^#incident-([A-Za-z0-9][A-Za-z0-9_-]{0,127})$/.exec(location.hash);
    if (wanted) select(wanted[1]);
  }
  schedule(EVERY);
})();
