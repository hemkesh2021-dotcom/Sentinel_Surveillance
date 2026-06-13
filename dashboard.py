"""
dashboard.py — SENTINEL Surveillance Dashboard v4
Changes vs v3:
  - Redesigned AI chat section with distinct cyan/neural identity
  - Real chat bubbles with avatars, names, timestamps
  - Animated typing dots
  - Categorized suggestion chips (ID / Threat / Scene)
  - Context tags on AI replies (Jessica, Stranger, Scene, etc.)
  - Modern pill input with circular send button
"""

import os, json, time, threading, base64, requests
from flask import Flask, Response, jsonify, render_template_string, request as freq

HOME_DIR     = os.path.expanduser("~")
STATE_FILE   = "/tmp/surv_state.json"
FRAME_FILE   = "/tmp/surv_frame.jpg"
INTRUDER_LOG = os.path.join(HOME_DIR, "intruder_log.json")
LFM2_SERVER  = "http://localhost:8080"

app = Flask(__name__)

# ══════════════════════════════════════════════════════════════
# DOUBLE-BUFFER FRAME STORE
# ══════════════════════════════════════════════════════════════
_bufs   = [None, None]
_front  = 0
_flip   = threading.Lock()

def _is_valid_jpeg(data: bytes) -> bool:
    return (len(data) > 4
            and data[:2] == b'\xff\xd8'
            and data[-2:] == b'\xff\xd9')

def _push_frame(data: bytes):
    global _front
    back = 1 - _front
    _bufs[back] = data
    with _flip:
        _front = back

def _get_frame() -> bytes | None:
    with _flip:
        idx = _front
    return _bufs[idx]

def frame_feeder():
    last_mtime = 0.0
    while True:
        try:
            mtime = os.path.getmtime(FRAME_FILE)
            if mtime != last_mtime:
                last_mtime = mtime
                with open(FRAME_FILE, 'rb') as fh:
                    raw = fh.read()
                if _is_valid_jpeg(raw):
                    _push_frame(raw)
        except Exception:
            pass
        time.sleep(0.030)

threading.Thread(target=frame_feeder, daemon=True).start()

# ══════════════════════════════════════════════════════════════
# MJPEG GENERATOR
# ══════════════════════════════════════════════════════════════
def generate_stream():
    last_id = None
    while True:
        frame = _get_frame()
        if frame is None or id(frame) == last_id:
            time.sleep(0.033)
            continue
        last_id = id(frame)
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n'
               + frame + b'\r\n')
        time.sleep(0.050)

# ══════════════════════════════════════════════════════════════
# STATE + LOG READERS
# ══════════════════════════════════════════════════════════════
def read_state() -> dict:
    try:
        with open(STATE_FILE, 'r') as f:
            return json.load(f)
    except Exception:
        return {"status": "OFFLINE", "threat": "none", "description": "",
                "persons": [], "fire": False, "fps": 0.0, "room_count": 0}

def read_log() -> list:
    try:
        with open(INTRUDER_LOG, 'r') as f:
            return json.load(f)
    except Exception:
        return []

# ══════════════════════════════════════════════════════════════
# AI CHAT
# ══════════════════════════════════════════════════════════════
def ask_ai(question: str) -> str:
    try:
        frame = _get_frame()
        content: list = []
        if frame:
            b64 = base64.b64encode(frame).decode()
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{b64}"}
            })
        content.append({"type": "text", "text": question})

        payload = {
            "model": "lfm2-vl",
            "max_tokens": 250,
            "temperature": 0.2,
            "messages": [
                {"role": "system",
                 "content": ("You are a security AI with live camera access. "
                              "Answer concisely about what you see. "
                              "If unsure, say so honestly.")},
                {"role": "user", "content": content}
            ]
        }
        r = requests.post(f"{LFM2_SERVER}/v1/chat/completions",
                          json=payload, timeout=30)
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"].strip()
    except Exception as e:
        return f"⚠ Error: {e}"

# ══════════════════════════════════════════════════════════════
# HTML
# ══════════════════════════════════════════════════════════════
HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1,maximum-scale=1,user-scalable=no">
<meta name="apple-mobile-web-app-capable" content="yes">
<meta name="apple-mobile-web-app-status-bar-style" content="black">
<title>SENTINEL</title>
<link href="https://fonts.googleapis.com/css2?family=Courier+Prime:wght@400;700&family=Barlow+Condensed:wght@300;400;600;700;900&display=swap" rel="stylesheet">
<style>
:root{
  --ink:#0a0a08; --ink2:#0f0f0c; --ink3:#141410; --ink4:#1a1a15;
  --amber:#e8a020; --amber2:#f5b830; --amber3:#ffd060;
  --green:#48d984; --red:#f03550; --orange:#f07020;
  --dim:#4a4035; --muted:#8a7a60; --body:#c8b890;
  --mono:'Courier Prime','Courier New',monospace;
  --disp:'Barlow Condensed',sans-serif;
  --border:rgba(232,160,32,.12); --border2:rgba(232,160,32,.06);
  /* AI chat palette */
  --cy:#3ed8ff; --cy2:#7ee9ff; --cy-d:#1a8eb0;
  --vio:#a78bfa;
}
*{margin:0;padding:0;box-sizing:border-box;-webkit-tap-highlight-color:transparent}
html,body{height:100%;background:var(--ink);color:var(--body);font-family:var(--disp);overflow-x:hidden}

body::after{
  content:'';position:fixed;inset:0;pointer-events:none;z-index:9998;
  background-image:url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='300' height='300'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='.75' numOctaves='4' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='300' height='300' filter='url(%23n)' opacity='.04'/%3E%3C/svg%3E");
  opacity:.5
}

/* ─── HEADER ─── */
.hdr{display:flex;align-items:center;justify-content:space-between;padding:9px 14px;background:var(--ink2);border-bottom:1px solid var(--border);position:sticky;top:0;z-index:100}
.logo{font-family:var(--disp);font-weight:900;font-size:18px;letter-spacing:6px;color:var(--amber);text-transform:uppercase}
.rec{width:6px;height:6px;border-radius:50%;background:var(--red);display:inline-block;margin-right:7px;animation:blink 1.4s ease-in-out infinite;box-shadow:0 0 8px var(--red)}
@keyframes blink{0%,100%{opacity:1}50%{opacity:.1}}
.hdr-meta{display:flex;align-items:center;gap:14px}
.hdr-item{font-family:var(--mono);font-size:9px;color:var(--dim);text-align:right;line-height:1.6}
.hdr-item strong{color:var(--amber);font-weight:700;display:block;font-size:11px}
#clock{font-family:var(--mono);font-size:12px;color:var(--muted)}

.prog{height:1px;background:var(--ink4);overflow:hidden}
.prog-fill{height:100%;background:var(--amber);animation:prog 2s linear infinite}
@keyframes prog{0%{width:0;opacity:1}85%{width:100%;opacity:1}100%{width:100%;opacity:0}}

/* ─── VIDEO ─── */
.feed-wrap{position:relative;background:#000;border-bottom:1px solid var(--border)}
#feed-a,#feed-b{width:100%;height:auto;display:block;image-rendering:auto;position:relative}
#feed-b{position:absolute;top:0;left:0;opacity:0}
.feed-hud{position:absolute;inset:0;pointer-events:none}
.brk{position:absolute;width:18px;height:18px}
.brk::before,.brk::after{content:'';position:absolute;background:var(--amber);opacity:.7}
.brk-tl{top:7px;left:7px}.brk-tl::before{top:0;left:0;width:2px;height:100%}.brk-tl::after{top:0;left:0;width:100%;height:2px}
.brk-tr{top:7px;right:7px}.brk-tr::before{top:0;right:0;width:2px;height:100%}.brk-tr::after{top:0;right:0;width:100%;height:2px}
.brk-bl{bottom:7px;left:7px}.brk-bl::before{bottom:0;left:0;width:2px;height:100%}.brk-bl::after{bottom:0;left:0;width:100%;height:2px}
.brk-br{bottom:7px;right:7px}.brk-br::before{bottom:0;right:0;width:2px;height:100%}.brk-br::after{bottom:0;right:0;width:100%;height:2px}
.feed-tag{position:absolute;top:10px;left:30px;font-family:var(--mono);font-size:9px;color:var(--amber);letter-spacing:2px;text-shadow:0 0 10px var(--amber)}
.feed-fps{position:absolute;bottom:9px;right:10px;font-family:var(--mono);font-size:9px;color:var(--dim)}
.scanline{position:absolute;left:0;right:0;height:1px;background:linear-gradient(transparent,rgba(232,160,32,.18),transparent);animation:scan 5s linear infinite;pointer-events:none}
@keyframes scan{from{top:-1px}to{top:100%}}

/* ─── STATUS / CARDS / SECTIONS / PERSONS / AI DESC ─── */
.sbar{display:flex;align-items:center;gap:9px;padding:7px 13px;background:var(--ink2);border-bottom:1px solid var(--border)}
.badge{font-family:var(--mono);font-size:9px;font-weight:700;padding:4px 9px;border-radius:1px;letter-spacing:2px;transition:all .25s;text-transform:uppercase}
.b-idle{background:rgba(74,64,53,.2);color:var(--dim);border:1px solid var(--dim)}
.b-ok{background:rgba(72,217,132,.07);color:var(--green);border:1px solid rgba(72,217,132,.3)}
.b-warn{background:rgba(240,112,32,.07);color:var(--orange);border:1px solid rgba(240,112,32,.3)}
.b-alert{background:rgba(240,53,80,.07);color:var(--red);border:1px solid rgba(240,53,80,.4);animation:palert .9s infinite}
@keyframes palert{0%,100%{box-shadow:0 0 0 0 rgba(240,53,80,.3)}50%{box-shadow:0 0 0 5px rgba(240,53,80,0)}}
.sbar-fps{font-family:var(--mono);font-size:9px;color:var(--dim);margin-left:auto}
#threat-pill{font-family:var(--mono);font-size:9px;padding:2px 7px;border-radius:1px;display:none;letter-spacing:1px}

.cards{display:grid;grid-template-columns:repeat(3,1fr);gap:1px;background:var(--border2)}
.card{background:var(--ink2);padding:9px 11px}
.card-lbl{font-family:var(--mono);font-size:8px;color:var(--dim);letter-spacing:2px;margin-bottom:3px}
.card-val{font-family:var(--disp);font-size:20px;font-weight:700;color:var(--body);line-height:1}
.c-none{color:var(--green)}.c-low{color:var(--amber3)}.c-medium{color:var(--orange)}.c-high,.c-fire{color:var(--red)}

.sec{padding:9px 13px;border-bottom:1px solid var(--border2)}
.sec-head{font-family:var(--mono);font-size:8px;color:var(--dim);letter-spacing:2px;text-transform:uppercase;margin-bottom:7px;display:flex;align-items:center;gap:6px}
.sec-head::after{content:'';flex:1;height:1px;background:var(--border)}

.chip{display:inline-flex;align-items:center;gap:5px;padding:4px 8px;margin:2px;border-radius:1px;font-family:var(--mono);font-size:10px;font-weight:700}
.ch-k{background:rgba(72,217,132,.07);color:var(--green);border:1px solid rgba(72,217,132,.2)}
.ch-s{background:rgba(240,53,80,.07);color:var(--red);border:1px solid rgba(240,53,80,.2)}
.ch-c{background:rgba(74,64,53,.15);color:var(--dim);border:1px solid var(--border)}
.pip{width:5px;height:5px;border-radius:50%}
.pk{background:var(--green)}.ps{background:var(--red)}.pc{background:var(--dim)}
.no-p{font-family:var(--mono);font-size:10px;color:var(--dim)}

.ai-sec{padding:9px 13px;background:var(--ink3);border-bottom:1px solid var(--border2)}
.ai-text{font-family:var(--mono);font-size:11px;color:var(--amber2);line-height:1.7;min-height:18px;transition:opacity .3s}

/* ═══════════════════════════════════════════════════════════
   ─── AI CHAT v2 — distinct cyan/neural identity ───
   ═══════════════════════════════════════════════════════════ */
.chat-panel{
  position:relative;margin:4px 8px;
  background:linear-gradient(180deg,rgba(62,216,255,.04) 0%,rgba(10,10,8,.95) 60%);
  border:1px solid rgba(62,216,255,.18);border-radius:4px;overflow:hidden
}
.chat-panel::before{
  content:'';position:absolute;inset:0;pointer-events:none;
  background:radial-gradient(circle at 12% 0%,rgba(62,216,255,.10),transparent 50%),
             radial-gradient(circle at 100% 0%,rgba(167,139,250,.08),transparent 50%)
}

/* Glow header */
.chat-head{display:flex;align-items:center;justify-content:space-between;padding:11px 13px 10px;border-bottom:1px solid rgba(62,216,255,.12);position:relative}
.chat-head-left{display:flex;align-items:center;gap:10px}
.ai-orb{position:relative;width:30px;height:30px;display:flex;align-items:center;justify-content:center}
.ai-orb-core{width:14px;height:14px;border-radius:50%;
  background:radial-gradient(circle at 35% 30%,var(--cy2),var(--cy) 50%,var(--cy-d));
  box-shadow:0 0 12px var(--cy),inset 0 0 4px rgba(255,255,255,.4);
  animation:orbPulse 2.4s ease-in-out infinite}
.ai-orb-ring{position:absolute;inset:0;border-radius:50%;border:1px solid rgba(62,216,255,.4);animation:orbRing 2.4s ease-out infinite}
@keyframes orbPulse{
  0%,100%{box-shadow:0 0 8px var(--cy),inset 0 0 3px rgba(255,255,255,.4)}
  50%{box-shadow:0 0 16px var(--cy),0 0 24px rgba(62,216,255,.4),inset 0 0 4px rgba(255,255,255,.5)}
}
@keyframes orbRing{0%{transform:scale(.7);opacity:.9}100%{transform:scale(1.4);opacity:0}}
.chat-title{font-family:var(--mono);font-size:11px;font-weight:700;color:var(--cy2);letter-spacing:2.5px;text-shadow:0 0 8px rgba(62,216,255,.4)}
.chat-sub{font-family:var(--mono);font-size:8px;color:var(--cy-d);letter-spacing:1.5px;margin-top:2px;display:flex;align-items:center;gap:5px}
.chat-dot{width:4px;height:4px;border-radius:50%;background:var(--cy);box-shadow:0 0 6px var(--cy);animation:blink 1.6s infinite}
.chat-meta{display:flex;flex-direction:column;align-items:flex-end;gap:1px;font-family:var(--mono)}
.chat-meta-k{font-size:7px;color:var(--cy-d);letter-spacing:1.5px}
.chat-meta-v{font-size:9px;color:var(--cy);letter-spacing:1px;border:1px solid rgba(62,216,255,.25);padding:1px 5px;border-radius:1px}

/* Chip rail */
.chip-rail{display:flex;align-items:center;gap:8px;padding:8px 13px;border-bottom:1px solid rgba(62,216,255,.08)}
.chip-rail-lbl{font-family:var(--mono);font-size:7px;color:var(--cy-d);letter-spacing:1.5px;flex-shrink:0}
.chip-rail-scroll{display:flex;gap:5px;overflow-x:auto;flex:1;scrollbar-width:none}
.chip-rail-scroll::-webkit-scrollbar{display:none}
.sugg2{display:inline-flex;align-items:center;gap:4px;font-family:var(--mono);font-size:9px;font-weight:600;
  padding:4px 8px;border-radius:99px;background:rgba(62,216,255,.06);
  border:1px solid rgba(62,216,255,.2);color:var(--cy2);
  cursor:pointer;white-space:nowrap;flex-shrink:0;letter-spacing:.5px;transition:all .15s}
.sugg2:hover,.sugg2:active{background:rgba(62,216,255,.14);border-color:var(--cy);box-shadow:0 0 10px rgba(62,216,255,.2)}
.sugg-icon{font-size:7px;opacity:.85}
.sugg-id{color:var(--cy2);border-color:rgba(62,216,255,.25);background:rgba(62,216,255,.06)}
.sugg-threat{color:#ffb37a;border-color:rgba(240,112,32,.25);background:rgba(240,112,32,.06)}
.sugg-scene{color:#c9b8ff;border-color:rgba(167,139,250,.25);background:rgba(167,139,250,.06)}

/* Messages */
.msgs2{max-height:260px;overflow-y:auto;padding:10px 11px;display:flex;flex-direction:column;gap:9px;scrollbar-width:thin;scrollbar-color:var(--cy-d) transparent}
.msgs2::-webkit-scrollbar{width:2px}
.msgs2::-webkit-scrollbar-thumb{background:var(--cy-d)}
.bubble{display:flex;gap:7px;align-items:flex-start;animation:bIn .25s ease-out}
@keyframes bIn{from{opacity:0;transform:translateY(4px)}to{opacity:1;transform:none}}
.bubble-u{flex-direction:row-reverse}
.bubble-avatar{width:22px;height:22px;flex-shrink:0;border-radius:50%;display:flex;align-items:center;justify-content:center;font-family:var(--mono);font-size:7px;font-weight:700}
.av-u{width:22px;height:22px;border-radius:50%;background:rgba(232,160,32,.12);color:var(--amber);border:1px solid rgba(232,160,32,.3);display:flex;align-items:center;justify-content:center;font-size:6px}
.av-a{width:22px;height:22px;border-radius:50%;
  background:radial-gradient(circle at 35% 30%,rgba(126,233,255,.3),rgba(62,216,255,.1));
  color:var(--cy2);border:1px solid rgba(62,216,255,.4);
  display:flex;align-items:center;justify-content:center;box-shadow:0 0 8px rgba(62,216,255,.15)}
.bubble-body{flex:1;min-width:0;max-width:calc(100% - 30px)}
.bubble-u .bubble-body{text-align:right}
.bubble-meta{display:flex;gap:6px;align-items:baseline;font-family:var(--mono);font-size:7px;letter-spacing:1px;margin-bottom:3px}
.bubble-u .bubble-meta{justify-content:flex-end}
.bubble-name{color:var(--cy);font-weight:700}
.bubble-u .bubble-name{color:var(--amber)}
.bubble-time{color:var(--dim)}
.bubble-text{font-family:var(--mono);font-size:10.5px;line-height:1.55;padding:7px 10px;border-radius:8px;display:inline-block;text-align:left}
.bubble-a .bubble-text{
  background:linear-gradient(180deg,rgba(62,216,255,.08),rgba(62,216,255,.04));
  border:1px solid rgba(62,216,255,.18);color:#d8f4ff;border-top-left-radius:2px}
.bubble-u .bubble-text{
  background:linear-gradient(180deg,rgba(232,160,32,.12),rgba(232,160,32,.05));
  border:1px solid rgba(232,160,32,.25);color:var(--amber3);border-top-right-radius:2px}
.bubble-refs{display:flex;gap:4px;flex-wrap:wrap;margin-top:5px}
.bubble-u .bubble-refs{justify-content:flex-end}
.ref-tag{font-family:var(--mono);font-size:7.5px;font-weight:600;padding:2px 6px;border-radius:99px;letter-spacing:.5px}
.ref-known{background:rgba(72,217,132,.1);color:var(--green);border:1px solid rgba(72,217,132,.3)}
.ref-stranger{background:rgba(240,53,80,.1);color:var(--red);border:1px solid rgba(240,53,80,.3)}
.ref-scene{background:rgba(167,139,250,.1);color:#c9b8ff;border:1px solid rgba(167,139,250,.3)}
.ref-low{background:rgba(255,208,96,.1);color:var(--amber3);border:1px solid rgba(255,208,96,.3)}
.ref-count{background:rgba(62,216,255,.1);color:var(--cy2);border:1px solid rgba(62,216,255,.3)}

/* Typing dots */
.typing-dots{display:inline-flex;gap:3px;padding:9px 12px;
  background:rgba(62,216,255,.06);border:1px solid rgba(62,216,255,.18);
  border-radius:8px;border-top-left-radius:2px;align-items:center}
.typing-dots i{width:5px;height:5px;border-radius:50%;background:var(--cy);display:inline-block;animation:tDot 1.2s infinite}
.typing-dots i:nth-child(2){animation-delay:.15s}
.typing-dots i:nth-child(3){animation-delay:.3s}
@keyframes tDot{0%,60%,100%{opacity:.25;transform:translateY(0)}30%{opacity:1;transform:translateY(-2px)}}

/* Input */
.inp2-wrap{padding:8px 10px 10px;border-top:1px solid rgba(62,216,255,.12);background:rgba(62,216,255,.025)}
.inp2{display:flex;align-items:center;gap:6px;
  background:rgba(10,15,18,.7);border:1px solid rgba(62,216,255,.25);
  border-radius:99px;padding:4px 4px 4px 12px;transition:all .15s}
.inp2:focus-within{border-color:var(--cy);box-shadow:0 0 0 3px rgba(62,216,255,.1),0 0 14px rgba(62,216,255,.15)}
.inp2-prefix{font-family:var(--mono);color:var(--cy);font-size:13px;font-weight:700;line-height:1}
#ci2{flex:1;background:transparent;border:none;outline:none;font-family:var(--mono);font-size:11px;color:#d8f4ff;padding:6px 0}
#ci2::placeholder{color:var(--cy-d);opacity:.8}
.send-btn{width:28px;height:28px;flex-shrink:0;border-radius:50%;border:none;cursor:pointer;
  background:linear-gradient(135deg,var(--cy2),var(--cy));color:#042033;
  display:flex;align-items:center;justify-content:center;
  box-shadow:0 0 10px rgba(62,216,255,.35);transition:all .15s}
.send-btn:hover:not(:disabled){transform:scale(1.05);box-shadow:0 0 16px rgba(62,216,255,.55)}
.send-btn:disabled{background:rgba(62,216,255,.15);color:var(--cy-d);cursor:not-allowed;box-shadow:none}
.send-btn svg{transform:translateX(-1px)}
.inp2-foot{display:flex;justify-content:space-between;font-family:var(--mono);font-size:7px;color:var(--cy-d);letter-spacing:1px;padding:5px 10px 0;opacity:.8}

/* ─── LOG ─── */
.log-wrap{padding:9px 13px}
.filters{display:flex;gap:4px;flex-wrap:wrap;margin-bottom:7px}
.flt{font-family:var(--mono);font-size:8px;padding:3px 7px;border:1px solid var(--border);color:var(--dim);background:transparent;border-radius:1px;cursor:pointer;transition:all .15s;letter-spacing:1px}
.flt.on,.flt:hover{border-color:var(--amber);color:var(--amber)}
.lentry{display:flex;align-items:flex-start;gap:7px;padding:6px 0;border-bottom:1px solid var(--border2)}
.lentry:last-child{border-bottom:none}
.ltime{font-family:var(--mono);font-size:8px;color:var(--dim);white-space:nowrap;min-width:72px;padding-top:1px}
.ltag{font-family:var(--mono);font-size:7px;padding:2px 5px;border-radius:1px;text-transform:uppercase;letter-spacing:1px;white-space:nowrap}
.ldesc{flex:1;font-family:var(--mono);font-size:9px;color:var(--muted);line-height:1.5}
.log-empty{font-family:var(--mono);font-size:9px;color:var(--dim)}

/* ─── HEALTH BAR ─── */
.health{display:flex;gap:1px;padding:6px 13px;background:var(--ink2);border-top:1px solid var(--border);position:sticky;bottom:0}
.hi{display:flex;align-items:center;gap:4px;flex:1;justify-content:center}
.hd{width:5px;height:5px;border-radius:50%;transition:background .5s}
.h-ok{background:var(--green)}.h-warn{background:var(--orange)}.h-err{background:var(--red)}
.hl{font-family:var(--mono);font-size:7px;color:var(--dim);letter-spacing:1px}
</style>
</head>
<body>

<!-- ─── HEADER ─── -->
<div class="hdr">
  <div class="logo"><span class="rec"></span>SENTINEL</div>
  <div class="hdr-meta">
    <div class="hdr-item">CAMERA<strong id="h-cam-s">ONLINE</strong></div>
    <div class="hdr-item">AI ENGINE<strong id="h-ai-s">IDLE</strong></div>
    <div id="clock">--:--:--</div>
  </div>
</div>
<div class="prog"><div class="prog-fill"></div></div>

<!-- ─── VIDEO FEED ─── -->
<div class="feed-wrap" id="feed-wrap">
  <img id="feed-a" src="/stream" alt="live">
  <img id="feed-b" src="" alt="live-b">
  <div class="feed-hud">
    <div class="scanline"></div>
    <div class="brk brk-tl"></div><div class="brk brk-tr"></div>
    <div class="brk brk-bl"></div><div class="brk brk-br"></div>
    <div class="feed-tag">◉ CAM-01 &nbsp;LIVE</div>
    <div class="feed-fps" id="feed-fps">-- FPS</div>
  </div>
</div>

<!-- ─── STATUS ─── -->
<div class="sbar">
  <div class="badge b-idle" id="badge">LOADING</div>
  <div id="threat-pill"></div>
  <div class="sbar-fps" id="fps-lbl">0.0 FPS</div>
</div>

<!-- ─── CARDS ─── -->
<div class="cards">
  <div class="card"><div class="card-lbl">Threat</div><div class="card-val c-none" id="threat-val">NONE</div></div>
  <div class="card"><div class="card-lbl">Room</div><div class="card-val" id="room-val">0</div></div>
  <div class="card"><div class="card-lbl">Persons</div><div class="card-val" id="p-count">0</div></div>
</div>

<!-- ─── PERSONS ─── -->
<div class="sec">
  <div class="sec-head">Detected Persons</div>
  <div id="persons"></div>
</div>

<!-- ─── AI DESCRIPTION ─── -->
<div class="ai-sec">
  <div class="sec-head" style="margin-bottom:5px">Live AI Analysis</div>
  <div class="ai-text" id="ai-desc">Waiting for analysis…</div>
</div>

<!-- ═══ AI CHAT (redesigned) ═══ -->
<div class="chat-panel">
  <!-- Glow header -->
  <div class="chat-head">
    <div class="chat-head-left">
      <div class="ai-orb">
        <div class="ai-orb-core"></div>
        <div class="ai-orb-ring"></div>
      </div>
      <div>
        <div class="chat-title">NEURAL ASSISTANT</div>
        <div class="chat-sub">
          <span class="chat-dot"></span>
          LFM2-VL · vision-online
        </div>
      </div>
    </div>
    <div class="chat-meta">
      <span class="chat-meta-k">CTX</span>
      <span class="chat-meta-v">CAM-01</span>
    </div>
  </div>

  <!-- Chip rail -->
  <div class="chip-rail">
    <div class="chip-rail-lbl">PROMPTS</div>
    <div class="chip-rail-scroll" id="sugg-row"></div>
  </div>

  <!-- Messages -->
  <div class="msgs2" id="msgs"></div>

  <!-- Input -->
  <div class="inp2-wrap">
    <div class="inp2">
      <span class="inp2-prefix">›</span>
      <input id="ci2" type="text" placeholder="Ask Sentinel AI about the live feed…" maxlength="200"
             oninput="updateFoot()"
             onkeydown="if(event.key==='Enter'&&!event.shiftKey){event.preventDefault();send()}">
      <button class="send-btn" id="cs" onclick="send()" disabled>
        <svg width="14" height="14" viewBox="0 0 14 14" fill="none">
          <path d="M1 7L13 1L7 13L5.5 8.5L1 7Z" fill="currentColor"/>
        </svg>
      </button>
    </div>
    <div class="inp2-foot">
      <span>↵ to send</span>
      <span class="inp2-foot-r">vision frame attached · <span id="char-count">0</span>/200</span>
    </div>
  </div>
</div>

<!-- ─── EVENT LOG ─── -->
<div class="log-wrap">
  <div class="sec-head">Event Log</div>
  <div class="filters">
    <button class="flt on" onclick="filt('all',this)">ALL</button>
    <button class="flt" onclick="filt('high',this)">HIGH</button>
    <button class="flt" onclick="filt('medium',this)">MED</button>
    <button class="flt" onclick="filt('low',this)">LOW</button>
    <button class="flt" onclick="filt('none',this)">INFO</button>
  </div>
  <div id="log"></div>
</div>

<!-- ─── HEALTH BAR ─── -->
<div class="health">
  <div class="hi"><div class="hd h-ok" id="hd-cam"></div><div class="hl">CAMERA</div></div>
  <div class="hi"><div class="hd h-warn" id="hd-ai"></div><div class="hl">AI</div></div>
  <div class="hi"><div class="hd h-warn" id="hd-face"></div><div class="hl">FACE-ID</div></div>
  <div class="hi"><div class="hd h-ok" id="hd-net"></div><div class="hl">NETWORK</div></div>
</div>

<script>
// ── Clock ──
setInterval(()=>{document.getElementById('clock').textContent=new Date().toTimeString().slice(0,8)},1000)

// ── Feed reconnect ──
let _frontImg='a',_errCount=0,_errTimer=null
function activateFeed(id){
  const a=document.getElementById('feed-a'),b=document.getElementById('feed-b')
  if(id==='a'){a.style.position='relative';a.style.opacity='1';b.style.opacity='0'}
  else{b.style.position='relative';b.style.opacity='1';a.style.position='absolute';a.style.opacity='0'}
  _frontImg=id
}
function reconnectFeed(){
  const back=_frontImg==='a'?'feed-b':'feed-a',backId=_frontImg==='a'?'b':'a'
  const el=document.getElementById(back)
  el.onload=()=>{activateFeed(backId);_errCount=0}
  el.onerror=()=>{scheduleReconnect()}
  el.src='/stream?t='+Date.now()
}
function scheduleReconnect(){
  if(_errTimer)return
  const delay=Math.min(500*Math.pow(2,_errCount),8000)
  _errCount++
  _errTimer=setTimeout(()=>{_errTimer=null;reconnectFeed()},delay)
}
document.getElementById('feed-a').onerror=scheduleReconnect
document.getElementById('feed-b').onerror=scheduleReconnect

// ── Badge ──
function badge(status){
  const el=document.getElementById('badge'),s=(status||'').toUpperCase()
  el.className='badge'
  if(s.includes('FIRE')||s.includes('INTRUDER'))el.classList.add('b-alert')
  else if(s.includes('STRANGER'))el.classList.add('b-warn')
  else if(s.includes('KNOWN')||s.includes('MONITORING'))el.classList.add('b-ok')
  else el.classList.add('b-idle')
  el.textContent=s
}

const TC={none:'c-none',low:'c-low',medium:'c-medium',high:'c-high',fire:'c-fire'}
const TK={none:'#48d984',low:'#ffd060',medium:'#f07020',high:'#f03550'}

// ── Status poll ──
let _prevDesc=''
async function pollStatus(){
  try{
    const d=await fetch('/status').then(r=>r.json())
    badge(d.status)
    const fps=d.fps||0
    document.getElementById('fps-lbl').textContent=fps+' FPS'
    document.getElementById('feed-fps').textContent=fps+' FPS'
    document.getElementById('room-val').textContent=d.room_count||0
    document.getElementById('p-count').textContent=(d.persons||[]).length

    const t=(d.threat||'none').toLowerCase()
    const tv=document.getElementById('threat-val')
    tv.textContent=t.toUpperCase()
    tv.className='card-val '+(TC[t]||'c-none')

    const tp=document.getElementById('threat-pill')
    if(t!=='none'){
      tp.style.cssText=`display:inline-block;background:${TK[t]}18;color:${TK[t]};border:1px solid ${TK[t]}44;padding:2px 7px;font-family:var(--mono);font-size:9px;letter-spacing:1px;border-radius:1px`
      tp.textContent='▲ '+t.toUpperCase()
    }else{tp.style.display='none'}

    const desc=d.description||''
    if(desc!==_prevDesc){
      const el=document.getElementById('ai-desc')
      el.style.opacity='0'
      setTimeout(()=>{el.textContent=desc||'No analysis yet';el.style.opacity='1'},200)
      _prevDesc=desc
    }

    const pl=document.getElementById('persons')
    const ps=d.persons||[]
    if(ps.length){
      pl.innerHTML=ps.map(p=>{
        if(p.name==='Checking...')
          return `<span class="chip ch-c"><span class="pip pc"></span>#${p.tid} Checking…</span>`
        const k=!p.stranger
        return `<span class="chip ${k?'ch-k':'ch-s'}"><span class="pip ${k?'pk':'ps'}"></span>#${p.tid} ${p.name}${p.score>0?' ('+p.score+')':''}${k?' ✓':''}</span>`
      }).join('')
    }else{pl.innerHTML='<span class="no-p">No persons detected</span>'}

    document.getElementById('hd-cam').className='hd '+(fps>0?'h-ok':'h-err')
    document.getElementById('hd-ai').className='hd '+(desc?'h-ok':'h-warn')
    document.getElementById('h-cam-s').textContent=fps>0?'ONLINE':'OFFLINE'
    document.getElementById('h-ai-s').textContent=desc?'ACTIVE':'IDLE'
    const hasFace=ps.some(p=>!p.stranger&&p.name!=='Checking...')
    document.getElementById('hd-face').className='hd '+(hasFace?'h-ok':'h-warn')
    document.getElementById('hd-net').className='hd h-ok'
  }catch(e){document.getElementById('hd-net').className='hd h-err'}
}

// ── Log ──
let _logData=[],_logFilter='all'
function filt(f,btn){
  _logFilter=f
  document.querySelectorAll('.flt').forEach(b=>b.classList.remove('on'))
  btn.classList.add('on')
  renderLog()
}
function renderLog(){
  const el=document.getElementById('log')
  let data=_logData.slice()
  if(_logFilter!=='all')data=data.filter(e=>(e.threat||'none')===_logFilter)
  if(!data.length){el.innerHTML='<span class="log-empty">No events</span>';return}
  el.innerHTML=data.slice(-12).reverse().map(e=>{
    const t=e.threat||'none',c=TK[t]||'#4a4035'
    const ts=(e.timestamp||'').split(' ')
    return `<div class="lentry">
      <div class="ltime">${ts[1]||e.timestamp||''}<br><span style="font-size:7px">${ts[0]||''}</span></div>
      <div class="ltag" style="color:${c};border:1px solid ${c}33;background:${c}11">${t.toUpperCase()}</div>
      <div class="ldesc">${e.description||'Stranger detected'}</div>
    </div>`
  }).join('')
}
async function pollLog(){
  try{_logData=await fetch('/log').then(r=>r.json());renderLog()}catch(e){}
}

// ═══ AI CHAT v2 ═══
const SUGGESTIONS=[
  {text:'Who is in the room?',cat:'id',icon:'◉'},
  {text:'Is anyone suspicious?',cat:'threat',icon:'▲'},
  {text:'Describe the scene',cat:'scene',icon:'▦'},
  {text:'Any unusual activity?',cat:'threat',icon:'▲'},
  {text:'Count the people',cat:'id',icon:'◉'},
  {text:'Is the room empty?',cat:'scene',icon:'▦'}
]

function buildSuggestions(){
  document.getElementById('sugg-row').innerHTML=
    SUGGESTIONS.map(s=>`<button class="sugg2 sugg-${s.cat}" onclick="quickAsk('${s.text.replace(/'/g,"\\'")}')">
      <span class="sugg-icon">${s.icon}</span><span>${s.text}</span>
    </button>`).join('')
}

function quickAsk(q){
  document.getElementById('ci2').value=q
  updateFoot()
  send()
}

function updateFoot(){
  const v=document.getElementById('ci2').value
  document.getElementById('char-count').textContent=v.length
  document.getElementById('cs').disabled=!v.trim()
}

// Auto-tag AI replies with referenced entities
function inferRefs(text){
  const refs=[]
  const lc=text.toLowerCase()
  if(lc.match(/jessica|known|recogn/))refs.push({cls:'ref-known',label:'◉ Known'})
  if(lc.match(/stranger|unidentified|unknown/))refs.push({cls:'ref-stranger',label:'◉ Stranger'})
  if(lc.match(/scene|room|living|kitchen|lighting/))refs.push({cls:'ref-scene',label:'▦ Scene'})
  if(lc.match(/\b(\d+) people|\bcount\b|persons present/))refs.push({cls:'ref-count',label:'∑ Count'})
  if(lc.match(/no threat|nothing unusual|relaxed|normal/))refs.push({cls:'ref-low',label:'▲ Low'})
  return refs
}

function nowStr(){return new Date().toTimeString().slice(0,8)}

function addBubble(text,type){
  const el=document.getElementById('msgs')
  const isUser=type==='u',isErr=type==='e'
  const time=nowStr()
  const refsHtml=(!isUser&&!isErr)
    ? (()=>{const r=inferRefs(text);return r.length
        ?`<div class="bubble-refs">${r.map(x=>`<span class="ref-tag ${x.cls}">${x.label}</span>`).join('')}</div>`
        :''})()
    : ''
  const avatarSvg = isUser ? 'YOU' :
    `<svg width="11" height="11" viewBox="0 0 11 11" fill="none">
       <circle cx="5.5" cy="5.5" r="4.5" stroke="currentColor" stroke-width="1"/>
       <circle cx="5.5" cy="5.5" r="1.5" fill="currentColor"/>
       <path d="M5.5 1V2.5M5.5 8.5V10M1 5.5H2.5M8.5 5.5H10" stroke="currentColor" stroke-width="1" stroke-linecap="round"/>
     </svg>`
  const d=document.createElement('div')
  d.className=`bubble bubble-${type}`
  d.innerHTML=`
    <div class="bubble-avatar"><span class="${isUser?'av-u':'av-a'}">${avatarSvg}</span></div>
    <div class="bubble-body">
      <div class="bubble-meta">
        <span class="bubble-name">${isUser?'OPERATOR':isErr?'SYSTEM':'SENTINEL AI'}</span>
        <span class="bubble-time">${time}</span>
      </div>
      <div class="bubble-text">${text}</div>
      ${refsHtml}
    </div>`
  el.appendChild(d)
  el.scrollTop=el.scrollHeight
}

function showTyping(){
  const el=document.getElementById('msgs')
  const d=document.createElement('div')
  d.className='bubble bubble-a bubble-typing'
  d.id='typing-bubble'
  d.innerHTML=`
    <div class="bubble-avatar"><span class="av-a">
      <svg width="11" height="11" viewBox="0 0 11 11" fill="none">
        <circle cx="5.5" cy="5.5" r="4.5" stroke="currentColor" stroke-width="1"/>
        <circle cx="5.5" cy="5.5" r="1.5" fill="currentColor"/>
      </svg></span></div>
    <div class="bubble-body">
      <div class="bubble-meta"><span class="bubble-name">SENTINEL AI</span><span class="bubble-time">analysing frame</span></div>
      <div class="typing-dots"><i></i><i></i><i></i></div>
    </div>`
  el.appendChild(d)
  el.scrollTop=el.scrollHeight
}
function hideTyping(){
  const t=document.getElementById('typing-bubble')
  if(t)t.remove()
}

async function send(){
  const inp=document.getElementById('ci2')
  const btn=document.getElementById('cs')
  const q=inp.value.trim()
  if(!q||btn.disabled)return
  inp.value=''
  updateFoot()
  addBubble(q,'u')
  btn.disabled=true
  showTyping()
  try{
    const r=await fetch('/chat',{
      method:'POST',
      headers:{'Content-Type':'application/json'},
      body:JSON.stringify({message:q})
    })
    const d=await r.json()
    hideTyping()
    addBubble(d.response||'No response','a')
  }catch(e){
    hideTyping()
    addBubble('Connection error — is llama-server running?','e')
  }finally{
    updateFoot()
  }
}

// ── Boot ──
buildSuggestions()
pollStatus();pollLog()
setInterval(pollStatus,2000)
setInterval(pollLog,8000)
</script>
</body>
</html>"""


# ── Routes ──
@app.route('/')
def index():
    return render_template_string(HTML)

@app.route('/stream')
def stream():
    return Response(generate_stream(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/status')
def status():
    return jsonify(read_state())

@app.route('/log')
def log():
    return jsonify(read_log())

@app.route('/chat', methods=['POST'])
def chat():
    data = freq.get_json(force=True) or {}
    question = data.get('message', '').strip()
    if not question:
        return jsonify({"response": "Please ask a question."})
    answer = ask_ai(question)
    return jsonify({"response": answer})


if __name__ == '__main__':
    print("🌐 SENTINEL Dashboard  →  http://0.0.0.0:5000")
    print("   iPhone / iPad       →  http://192.168.55.1:5000")
    app.run(host='0.0.0.0', port=5000, threaded=True)
