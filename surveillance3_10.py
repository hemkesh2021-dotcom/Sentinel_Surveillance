import os
os.environ["CUDA_VISIBLE_DEVICES"]      = ""
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
os.environ["TF_GPU_ALLOCATOR"]          = "cuda_malloc_async"

import cv2, time, threading, numpy as np, pickle, json, base64, requests
from collections import defaultdict
from ultralytics import YOLO
from datetime import datetime
from deepface import DeepFace
from queue import Queue, Empty
from dotenv import load_dotenv

load_dotenv()

HOME_DIR = os.path.expanduser("~")

# ─── CONFIG ───────────────────────────────────────────
RTSP_URL          = os.getenv("RTSP_URL", "rtsp://<user>:<pass>@<ip>:554/cam/realmonitor?channel=1&subtype=1")
YOLO_MODEL        = os.getenv("YOLO_MODEL", os.path.join(HOME_DIR, "yolov8n.engine"))
FACE_DB_PATH      = os.getenv("FACE_DB_PATH", os.path.join(HOME_DIR, "face_db.pkl"))
BOT_TOKEN         = os.getenv("TELEGRAM_BOT_TOKEN", "")
CHAT_ID           = os.getenv("TELEGRAM_CHAT_ID", "")
DISPLAY_WIDTH     = 640
DISPLAY_HEIGHT    = 480
CONF_THRESHOLD    = 0.4
CONFIRM_FRAMES    = 2
PERSIST_FRAMES    = 8
FACE_MATCH_THRESH = 0.40
FACE_CHECK_EVERY  = 3
YOLO_FRAME_SKIP   = 2
RESTRICTED_HOURS  = (22, 6)
MODEL_NAME        = "Facenet512"
LFM2_SERVER       = "http://localhost:8080"
LFM2_INTERVAL     = 4.0
LFM2_IMG_W        = 480
LFM2_IMG_H        = 360
LFM2_IMG_QUALITY  = 60
INTRUDER_LOG      = os.getenv("INTRUDER_LOG", os.path.join(HOME_DIR, "intruder_log.json"))
STATE_FILE        = "/tmp/surv_state.json"
FRAME_FILE        = "/tmp/surv_frame.jpg"

# ── Face Re-ID config ──
FACE_MAX_RETRIES  = 15      # more chances before giving up
REID_THRESHOLD    = 0.48    # slightly lower than match threshold
MAX_SESSION_EMBS  = 5       # embeddings stored per known person

# ── Load face DB ──
print("Loading face database...")
with open(FACE_DB_PATH, 'rb') as f:
    face_db = pickle.load(f)
print(f"✅ {len(face_db)} persons: {list(face_db.keys())}")

# ── Dashboard state ──
dashboard_state  = {"status":"STARTING","threat":"none","description":"",
                    "persons":[],"fire":False,"fps":0.0,"room_count":0}
dashboard_lock   = threading.Lock()
_last_state_write = 0

def update_dashboard(status, threat, description, persons, fire, fps, room_count):
    global _last_state_write
    with dashboard_lock:
        dashboard_state.update({"status":status,"threat":threat,
            "description":description,"persons":persons,"fire":fire,
            "fps":round(fps,1),"room_count":room_count})
    now = time.time()
    if now - _last_state_write > 1.0:
        _last_state_write = now
        try:
            with open(STATE_FILE,'w') as f: json.dump(dashboard_state,f)
        except: pass

# ── Intruder log ──
log_lock = threading.Lock()

def load_intruder_log():
    try:
        with open(INTRUDER_LOG,'r') as f: return json.load(f)
    except: return []

def save_intruder_event(photo_path, threat_level="none", description=""):
    entry = {"timestamp":datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
             "photo":photo_path,"threat":threat_level,"description":description}
    with log_lock:
        log = load_intruder_log()
        log.append(entry)
        with open(INTRUDER_LOG,'w') as f: json.dump(log,f,indent=2)
    print(f"[LOG] Intruder saved → {photo_path}")

# ── Alert Queue ──
alert_queue     = Queue()
alert_cooldowns = defaultdict(float)

def alert_worker():
    while True:
        try:
            item = alert_queue.get(timeout=1)
            if item is None: break
            priority, atype, path, caption = item
            try:
                if path:
                    with open(path,"rb") as img:
                        requests.post(f"https://api.telegram.org/bot{BOT_TOKEN}/sendPhoto",
                            data={"chat_id":CHAT_ID,"caption":caption,"parse_mode":"HTML"},
                            files={"photo":img}, timeout=10)
                else:
                    requests.post(f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage",
                        data={"chat_id":CHAT_ID,"text":caption,"parse_mode":"HTML"},timeout=10)
                print(f"[TG P{priority}] {atype}")
            except Exception as e: print(f"[TG ERR] {e}")
            alert_queue.task_done()
        except Empty: continue

threading.Thread(target=alert_worker, daemon=True).start()

def send_alert(atype, priority=2, frame=None, caption=""):
    cooldown = {1:15,2:10,3:20,4:60}.get(priority,10)
    now = time.time()
    if now - alert_cooldowns[atype] < cooldown: return
    alert_cooldowns[atype] = now
    path = None
    if frame is not None:
        path = os.path.join(HOME_DIR, f"alert_{atype}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg")
        cv2.imwrite(path, frame)
    alert_queue.put_nowait((priority, atype, path, caption))
    print(f"[ALERT P{priority}] {atype}")

def send_message(text):
    alert_queue.put_nowait((3,"MSG",None,text))

# ── LFM2 JSON parser ──
def parse_lfm2_response(raw):
    default = {"persons":0,"activities":[],"postures":["normal"],
               "harmful":False,"threat":"none","fire_smoke":False,"description":""}
    raw = raw.replace('```json','').replace('```','').strip()
    s = raw.find('{'); e = raw.rfind('}')+1
    if s==-1 or e<=s: default["description"]=raw[:50]; return default
    try: data = json.loads(raw[s:e])
    except: default["description"]=raw[s:s+50]; return default
    p = data.get("persons",0)
    default["persons"] = len(p) if isinstance(p,list) else (1 if isinstance(p,dict) else int(p or 0))
    a = data.get("activities",[])
    default["activities"] = [str(x) for x in a] if isinstance(a,list) else [str(a)]
    pos = data.get("postures",["normal"])
    default["postures"] = [str(x) for x in pos] if isinstance(pos,list) else [str(pos)]
    t = str(data.get("threat") or "none").lower().strip()
    default["threat"] = t if t in ("none","low","medium","high") else "none"
    for key in ("harmful","fire_smoke"):
        v = data.get(key,False)
        default[key] = v.lower()=="true" if isinstance(v,str) else bool(v)
    d = data.get("description","")
    default["description"] = str(d)[:60] if d else ""
    return default

# ── LFM2-VL worker ──
last_ai_result = {}
ai_lock        = threading.Lock()
ai_in_progress = False
last_ai_time   = 0.0
ai_frame_queue = Queue(maxsize=1)

LFM2_SYSTEM = ("You are a security camera AI. Respond with valid JSON only. "
               "No markdown, no explanation, only a JSON object.")
LFM2_PROMPT = """\
Analyze this security camera image. Return ONLY this JSON, nothing else:
{
  "persons": <integer count>,
  "activities": [<what each person is doing>],
  "postures": [<normal/suspicious/aggressive/fallen/running>],
  "harmful": <true or false>,
  "threat": "<none/low/medium/high>",
  "fire_smoke": <true or false>,
  "description": "<max 10 words>"
}"""

def lfm2_worker():
    global last_ai_result, ai_in_progress, last_ai_time
    while True:
        try: frame = ai_frame_queue.get(timeout=1)
        except Empty: continue
        ai_in_progress = True
        t0 = time.time()
        try:
            sf = cv2.resize(frame,(LFM2_IMG_W,LFM2_IMG_H))
            _,buf = cv2.imencode('.jpg',sf,[cv2.IMWRITE_JPEG_QUALITY,LFM2_IMG_QUALITY])
            b64 = base64.b64encode(buf).decode('utf-8')
            payload = {"model":"lfm2-vl","max_tokens":200,"temperature":0.05,
                "messages":[{"role":"system","content":LFM2_SYSTEM},
                    {"role":"user","content":[
                        {"type":"image_url","image_url":{"url":f"data:image/jpeg;base64,{b64}"}},
                        {"type":"text","text":LFM2_PROMPT}]}]}
            resp = requests.post(f"{LFM2_SERVER}/v1/chat/completions",json=payload,timeout=30)
            resp.raise_for_status()
            result = parse_lfm2_response(resp.json()["choices"][0]["message"]["content"])
            with ai_lock: last_ai_result = result
            last_ai_time = time.time()
            print(f"[LFM2 {time.time()-t0:.1f}s] {result['description']} | threat:{result['threat']}")
            if result["harmful"]:    print(f"[LFM2] ⚠️ {result['threat'].upper()}")
            if result["fire_smoke"]: print("[LFM2] 🔥 FIRE!")
        except Exception as e:
            print(f"[LFM2 ERR] {e}")
            with ai_lock:
                if not last_ai_result:
                    last_ai_result = {"persons":0,"activities":[],"postures":["normal"],
                        "harmful":False,"threat":"none","fire_smoke":False,"description":"error"}
        finally:
            ai_in_progress = False
            ai_frame_queue.task_done()

threading.Thread(target=lfm2_worker, daemon=True).start()

def request_ai_analysis(frame):
    if ai_in_progress or time.time()-last_ai_time < LFM2_INTERVAL: return
    try:
        if not ai_frame_queue.empty():
            try: ai_frame_queue.get_nowait()
            except: pass
        ai_frame_queue.put_nowait(frame.copy())
    except: pass

# ── Load Models ──
print("Loading YOLO...")
yolo = YOLO(YOLO_MODEL, task='detect')
print("Loading DeepFace...")
DeepFace.build_model(MODEL_NAME)

print("Warming up LFM2-VL...")
try:
    r = requests.post(f"{LFM2_SERVER}/v1/chat/completions",
        json={"model":"lfm2-vl","max_tokens":5,"messages":[{"role":"user","content":"hi"}]},
        timeout=15)
    r.raise_for_status()
    print("✅ LFM2-VL ready!")
except Exception as e: print(f"⚠ LFM2-VL warmup: {e}")

# ══════════════════════════════════════════════════════
# FACE RECOGNITION — with Re-ID for returning persons
# ══════════════════════════════════════════════════════
verified_faces    = {}              # {tid: (name, score)}
verified_lock     = threading.Lock()
face_retry_count  = defaultdict(int)
session_embeddings = defaultdict(list)  # {name: [emb, emb, ...]}
session_emb_lock  = threading.Lock()

def identify_face(crop):
    """Returns (name, score, face_found, embedding)."""
    try:
        result = DeepFace.represent(img_path=crop, model_name=MODEL_NAME,
                                    detector_backend="yunet", enforce_detection=False)
        if not result: return "Unknown", 0.0, False, None
        fa = result[0].get('facial_area',{})
        if fa.get('w',0) < 20 or fa.get('h',0) < 20:  # lowered from 25
            return "Unknown", 0.0, False, None
        emb  = np.array(result[0]['embedding'])
        norm = np.linalg.norm(emb)
        if norm == 0: return "Unknown", 0.0, False, None
        emb = emb / norm
        best_name, best_score = "Stranger", 0.0
        for name, db_emb in face_db.items():
            score = float(np.dot(emb, db_emb))
            if score > best_score: best_score = score; best_name = name
        if best_score < FACE_MATCH_THRESH:
            return "Stranger", best_score, True, emb
        return best_name, best_score, True, emb
    except Exception as e:
        print(f"[FACE DBG] {e}")
        return "Unknown", 0.0, False, None

def reid_from_session(emb):
    """Match embedding against session-stored embeddings of known persons."""
    if emb is None: return None, 0.0
    best_name, best_score = None, 0.0
    with session_emb_lock:
        for name, emb_list in session_embeddings.items():
            for stored in emb_list:
                score = float(np.dot(emb, stored))
                if score > best_score: best_score = score; best_name = name
    return (best_name, best_score) if best_score >= REID_THRESHOLD else (None, 0.0)

def store_session_embedding(name, emb):
    """Save embedding of known person for future re-identification."""
    if name in ("Stranger","Unknown") or emb is None: return
    with session_emb_lock:
        session_embeddings[name].append(emb)
        if len(session_embeddings[name]) > MAX_SESSION_EMBS:
            session_embeddings[name] = session_embeddings[name][-MAX_SESSION_EMBS:]

def _resolve_identity(name, score, emb):
    """If Stranger, try Re-ID against session embeddings."""
    if name == "Stranger" and emb is not None:
        reid_name, reid_score = reid_from_session(emb)
        if reid_name: return reid_name, reid_score
    return name, score

def face_recognition_worker(frame, tid_box_map):
    """
    Identify unverified tracks.
    Strategy (in order):
      1. Crop-based recognition (fast, precise)
      2. Full-frame recognition (fallback when crop fails — proven to work in debug)
      3. Re-ID against session embeddings
    Writes ONLY to verified_faces.
    """
    # Run full-frame recognition once per call — shared across all tids
    # Debug proved full frame gives good scores (Hemkesh 0.58, Yogesh 0.51)
    full_name, full_score, full_found, full_emb = identify_face(frame)
    if full_found:
        full_name, full_score = _resolve_identity(full_name, full_score, full_emb)

    for tid, box in tid_box_map.items():
        with verified_lock:
            if tid in verified_faces: continue

        x1,y1,x2,y2 = box
        pad  = 20
        crop = frame[max(0,y1-pad):min(frame.shape[0],y2+pad),
                     max(0,x1-pad):min(frame.shape[1],x2+pad)]

        name, score, face_found, emb = "Unknown", 0.0, False, None
        if crop.size > 0:
            name, score, face_found, emb = identify_face(crop)

        # Strategy 2: fall back to full-frame result if crop failed
        if not face_found and full_found:
            name, score, face_found, emb = full_name, full_score, True, full_emb
            print(f"[FACE] #{tid} crop failed — using full-frame result")

        if face_found:
            name, score = _resolve_identity(name, score, emb)
            store_session_embedding(name, emb)
            with verified_lock: verified_faces[tid] = (name, score)
            face_retry_count.pop(tid, None)
            status = 'KNOWN' if name not in ('Stranger','Unknown') else 'Stranger'
            print(f"[FACE] #{tid} → {name} ({score:.3f}) [{status}]")
        else:
            face_retry_count[tid] += 1
            attempts = face_retry_count[tid]
            print(f"[FACE] #{tid} no face in crop or full frame (try {attempts}/{FACE_MAX_RETRIES})")
            if attempts >= FACE_MAX_RETRIES:
                with verified_lock: verified_faces[tid] = ("Stranger", 0.0)
                face_retry_count.pop(tid, None)
                print(f"[FACE] #{tid} → Stranger (max retries)")

# ── Camera ──
def open_capture():
    cap = cv2.VideoCapture(RTSP_URL)
    cap.set(cv2.CAP_PROP_BUFFERSIZE,1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,DISPLAY_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT,DISPLAY_HEIGHT)
    return cap

class FrameReader:
    def __init__(self):
        self.cap=open_capture(); self.frame=None
        self.lock=threading.Lock(); self.running=True
        threading.Thread(target=self._read,daemon=True).start()
    def _read(self):
        while self.running:
            ret,frame = self.cap.read()
            if ret:
                with self.lock: self.frame=frame
            else:
                print("Stream lost. Reconnecting...")
                self.cap.release(); time.sleep(2); self.cap=open_capture()
    def get(self):
        with self.lock: return self.frame.copy() if self.frame is not None else None
    def stop(self): self.running=False; self.cap.release()

# ── Track State ──
class TrackState:
    def __init__(self):
        self.confirm_count=0; self.last_seen=0; self.last_box=None
        self.visible=False; self.name="Checking..."; self.face_score=0.0
        self.is_stranger=True; self.face_checked=False

track_states = defaultdict(TrackState)

def update_tracks(boxes, frame_num):
    seen = set()
    if boxes is not None and len(boxes)>0 and boxes.id is not None:
        for box,tid in zip(boxes.xyxy,boxes.id):
            tid=int(tid); seen.add(tid)
            s=track_states[tid]
            s.confirm_count=min(s.confirm_count+1,CONFIRM_FRAMES+2)
            s.last_seen=frame_num; s.last_box=box.cpu().numpy().astype(int)
            if s.confirm_count>=CONFIRM_FRAMES: s.visible=True
    for tid in list(track_states.keys()):
        s=track_states[tid]
        if tid not in seen:
            s.confirm_count=max(0,s.confirm_count-1)
            if frame_num-s.last_seen>PERSIST_FRAMES:
                s.visible=False
                if s.confirm_count==0: del track_states[tid]
    return [(tid,s) for tid,s in track_states.items() if s.visible and s.last_box is not None]

# ── Entry/Exit Counter ──
class EntryExitCounter:
    def __init__(self,line_y=240): self.line_y=line_y; self.entries=0; self.exits=0; self.hist={}
    def update(self,track_list):
        for tid,state in track_list:
            if state.last_box is None: continue
            _,y1,_,y2=state.last_box; cy=(y1+y2)/2
            if tid not in self.hist: self.hist[tid]=[]
            h=self.hist[tid]; h.append(cy)
            if len(h)>=2:
                if h[-2]<self.line_y<=h[-1]:
                    self.entries+=1; print(f"[ENTRY] #{tid} Room:{max(0,self.entries-self.exits)}")
                elif h[-2]>self.line_y>=h[-1]:
                    self.exits+=1; print(f"[EXIT]  #{tid} Room:{max(0,self.entries-self.exits)}")
            if len(h)>10: self.hist[tid]=h[-10:]
    def draw(self,frame):
        w=frame.shape[1]
        cv2.line(frame,(0,self.line_y),(w,self.line_y),(255,165,0),2)
        cv2.putText(frame,f"IN:{self.entries} OUT:{self.exits} ROOM:{max(0,self.entries-self.exits)}",
                    (10,self.line_y-8),cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,165,0),2)

def is_restricted_time():
    h=datetime.now().hour; return h>=RESTRICTED_HOURS[0] or h<RESTRICTED_HOURS[1]

# ── Init ──
counter = EntryExitCounter(line_y=DISPLAY_HEIGHT//2)
send_message(f"✅ <b>Surveillance Online</b>\n👤 Known: {list(face_db.keys())}\n"
             f"🤖 LFM2-VL 1.6B Q4 (llama.cpp)\n🕐 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

print("🟢 Surveillance + LFM2-VL started!")
reader      = FrameReader()
bg_sub      = cv2.createBackgroundSubtractorMOG2(history=500,varThreshold=50,detectShadows=False)
frame_count = 0; fps_time=time.time(); fps=0
track_list  = []; last_alert=0; last_fire=0; last_intruder_log=0
face_thread_busy = False

while True:
    frame = reader.get()
    if frame is None: time.sleep(0.005); continue

    frame_count += 1
    small   = cv2.resize(frame,(DISPLAY_WIDTH,DISPLAY_HEIGHT))
    display = small.copy()

    if frame_count%3==0:
        try: cv2.imwrite(FRAME_FILE,small,[cv2.IMWRITE_JPEG_QUALITY,70])
        except: pass

    if frame_count%30==0:
        fps=30/(time.time()-fps_time); fps_time=time.time()

    if frame_count%YOLO_FRAME_SKIP==0:
        results   = yolo.track(small,imgsz=640,device=0,half=True,verbose=False,
                               classes=[0],conf=CONF_THRESHOLD,persist=True,tracker="bytetrack.yaml")
        track_list = update_tracks(results[0].boxes,frame_count)

    if not track_list:
        cv2.putText(display,f"FPS:{fps:.1f} | IDLE — no persons",
                    (10,30),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,0),2)
        counter.draw(display); cv2.imshow("Surveillance",display)
        if cv2.waitKey(1)&0xFF==ord('q'): break
        continue

    any_stranger = False

    # ── Sync verified_faces → TrackState ──
    with verified_lock: vf_snap = dict(verified_faces)

    for tid,state in track_list:
        if tid in vf_snap:
            name,score = vf_snap[tid]
            state.name=name; state.face_score=score
            state.is_stranger=name in ("Stranger","Unknown"); state.face_checked=True

    # ── Reset retry + clear wrong Stranger label when person is close ──
    for tid,s in track_list:
        if tid not in vf_snap and s.last_box is not None:
            x1,y1,x2,y2 = s.last_box
            if (y2-y1) > DISPLAY_HEIGHT*0.4:   # large box = close to camera
                if face_retry_count.get(tid,0) > 0:
                    face_retry_count[tid]=0
                    print(f"[FACE] #{tid} close — retry reset")
                with verified_lock:
                    cur = verified_faces.get(tid)
                    if cur and cur[0]=="Stranger" and cur[1]==0.0:
                        del verified_faces[tid]
                        print(f"[FACE] #{tid} removed Stranger label — retrying")

    # ── Spawn face recognition ──
    # Re-read verified_faces fresh here — not the stale vf_snap
    with verified_lock: current_verified = set(verified_faces.keys())
    unverified = {tid:s.last_box for tid,s in track_list
                  if tid not in current_verified and s.last_box is not None}
    if unverified and not face_thread_busy:
        face_thread_busy = True
        def _face_thread(f,u):
            global face_thread_busy
            face_recognition_worker(f,u); face_thread_busy=False
        threading.Thread(target=_face_thread,args=(small.copy(),dict(unverified)),daemon=True).start()

    if track_list: request_ai_analysis(small)
    counter.update(track_list)

    # ── Re-sync verified_faces right before drawing (catches results from this frame's thread) ──
    with verified_lock:
        vf_snap2 = dict(verified_faces)
    for tid,state in track_list:
        if tid in vf_snap2:
            name,score = vf_snap2[tid]
            state.name=name; state.face_score=score
            state.is_stranger=name in ("Stranger","Unknown"); state.face_checked=True

    # ── Draw boxes ──
    for tid,state in track_list:
        if state.last_box is None: continue
        x1,y1,x2,y2 = state.last_box
        color = (0,0,255) if state.is_stranger else (0,255,0)
        cv2.rectangle(display,(x1,y1),(x2,y2),color,2)
        label = f"#{tid} {state.name}"
        if state.face_checked and not state.is_stranger: label+=f"({state.face_score:.2f}) ✓"
        elif state.face_checked: label+=f"({state.face_score:.2f})"
        cv2.putText(display,label,(x1,y1-8),cv2.FONT_HERSHEY_SIMPLEX,0.45,color,1)
        if state.is_stranger: any_stranger=True

    counter.draw(display)

    # ── AI overlay ──
    with ai_lock: ar=last_ai_result.copy()

    if ar:
        threat=str(ar.get('threat') or 'none').lower().strip()
        desc  =str(ar.get('description') or '')[:55]
        fire  =bool(ar.get('fire_smoke',False))
        cq    =(0,0,255) if threat=='high' else (0,165,255) if threat=='medium' else \
               (0,255,255) if threat=='low' else (180,180,180)
        cv2.putText(display,f"AI:{desc}",(10,DISPLAY_HEIGHT-38),cv2.FONT_HERSHEY_SIMPLEX,0.42,cq,1)
        postures=ar.get('postures',['?'])
        if not isinstance(postures,list): postures=['?']
        cv2.putText(display,f"THREAT:{threat.upper()} | {','.join(str(p) for p in postures[:2])}",
                    (10,DISPLAY_HEIGHT-15),cv2.FONT_HERSHEY_SIMPLEX,0.46,cq,1)

        now=time.time()
        if fire and now-last_fire>30:
            last_fire=now
            send_alert("FIRE",1,small,f"🔥 <b>FIRE/SMOKE</b>\n🕐{datetime.now().strftime('%H:%M:%S')}\n🚒 Immediate action!")
        if bool(ar.get('harmful',False)) and threat in ['high','medium']:
            if now-last_alert>10:
                last_alert=now
                send_alert(f"THREAT_{threat.upper()}",2,small,
                           f"⚠️ <b>THREAT:{threat.upper()}</b>\n🕐{datetime.now().strftime('%H:%M:%S')}\n🤖{desc}")

    # ── Intruder handling ──
    now=time.time()
    if any_stranger:
        with ai_lock:
            cur_threat=str(last_ai_result.get('threat') or 'none').lower()
            cur_desc  =str(last_ai_result.get('description') or '')
        if now-last_intruder_log>60:
            last_intruder_log=now
            path=os.path.join(HOME_DIR, f"intruder_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg")
            cv2.imwrite(path,small)
            threading.Thread(target=save_intruder_event,args=(path,cur_threat,cur_desc),daemon=True).start()
        if cur_threat in ['medium','high'] and is_restricted_time():
            if now-last_alert>10:
                last_alert=now
                send_alert("INTRUDER",2,small,
                           f"🚨 <b>INTRUDER — {cur_threat.upper()} THREAT</b>\n"
                           f"🕐{datetime.now().strftime('%H:%M:%S')}\n🤖{cur_desc}\n"
                           f"👥 Room:{max(0,counter.entries-counter.exits)}")

    # ── Recompute any_stranger after re-sync (verified_faces may have updated) ──
    any_stranger = any(s.is_stranger for _,s in track_list if s.last_box is not None)

    # ── Status bar ──
    with ai_lock: fire_active=bool(last_ai_result.get('fire_smoke',False))
    ai_icon="🤖⏳" if ai_in_progress else "🤖✅"

    if fire_active:                             status,sc="FIRE!",(0,0,255)
    elif any_stranger and is_restricted_time(): status,sc="INTRUDER",(0,0,255)
    elif any_stranger:                          status,sc="STRANGER",(0,165,255)
    elif track_list:                            status,sc="KNOWN",(0,255,0)
    else:                                       status,sc="MONITORING",(255,255,255)

    cv2.putText(display,f"FPS:{fps:.1f}|{status}|{ai_icon}",(10,30),cv2.FONT_HERSHEY_SIMPLEX,0.65,sc,2)

    person_list=[{"tid":tid,"name":s.name,"score":round(s.face_score,2),"stranger":s.is_stranger}
                 for tid,s in track_list if s.last_box is not None]
    with ai_lock:
        _t=str(last_ai_result.get('threat') or 'none').lower()
        _d=str(last_ai_result.get('description') or '')
        _f=bool(last_ai_result.get('fire_smoke',False))
    update_dashboard(status,_t,_d,person_list,_f,fps,max(0,counter.entries-counter.exits))

    cv2.imshow("Surveillance",display)
    if cv2.waitKey(1)&0xFF==ord('q'): break

reader.stop()
cv2.destroyAllWindows()
