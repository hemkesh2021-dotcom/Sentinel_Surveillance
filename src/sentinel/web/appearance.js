/* Sentinel status pages: the viewer's saved appearance, applied before the first paint (D96).

   Loaded synchronously in the head (CSP: script-src 'self'), so a saved theme, palette, density or compact sidebar never
   flashes the defaults first; so does the live picture's size and which optional panels are hidden (D100). It reads
   only this browser's storage for this origin and writes nothing; a page whose
   address chose its appearance (data-pref, set by the server for ?theme=, ?palette= or ?density=) keeps that choice for
   this view. Only exact values are used: a palette must be one the server listed in data-palettes. Without script, or
   when storage is unavailable, the server's choice (or the browser's preference) applies. */
(() => {
  "use strict";
  const root = document.documentElement;
  root.classList.add("js");
  let saved = null;
  try {
    saved = JSON.parse(window.localStorage.getItem("sentinel.appearance") || "null");
  } catch (error) {
    saved = null;  // storage blocked or unreadable: the defaults stand
  }
  if (saved && typeof saved === "object" && !Array.isArray(saved)) {
    const palettes = (root.dataset.palettes || "").split(" ").filter(Boolean);
    if (!root.hasAttribute("data-pref")) {
      if (saved.theme === "light" || saved.theme === "dark") root.dataset.theme = saved.theme;
      if (typeof saved.palette === "string" && palettes.indexOf(saved.palette) > 0) root.dataset.palette = saved.palette;
      if (saved.density === "compact") root.dataset.density = "compact";
    }
    if (saved.rail === "compact") root.dataset.rail = "compact";
    if (saved.view === "actual") root.dataset.view = "actual";  // D100: the live picture at its own pixel size
    if (Array.isArray(saved.hide)) {  // D100: only these optional panels can be hidden, never a warning
      const hide = saved.hide.filter((name) => ["chat", "health", "delivery"].indexOf(name) >= 0);
      if (hide.length) root.dataset.hide = hide.join(" ");
    }
  }
  root.classList.add("enter");  // the panels' one entrance; dashboard.js removes it, so no update ever replays it
})();
