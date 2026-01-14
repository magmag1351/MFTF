import cv2
import os
import mediapipe as mp
import numpy as np
import time
import threading
import base64
import asyncio
import logging
import json
from collections import deque
from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from sklearn.linear_model import LinearRegression
from pynput import keyboard, mouse
from datetime import datetime
from sqlalchemy import Column, Integer, Float, String, DateTime, func, select, desc
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker, declarative_base

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("MFTF_Backend")

app = FastAPI()

# Allow CORS for dev
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Database Setup
db_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../db")
os.makedirs(db_dir, exist_ok=True)
DATABASE_URL = f"sqlite+aiosqlite:///{os.path.join(db_dir, 'fatigue_history.db')}"
Base = declarative_base()

class SessionRecord(Base):
    __tablename__ = "sessions"
    id = Column(Integer, primary_key=True)
    start_time = Column(DateTime, default=datetime.now)
    end_time = Column(DateTime)
    avg_fatigue = Column(Float)
    peak_fatigue = Column(Float)
    alert_count = Column(Integer)
    duration_seconds = Column(Integer)

engine = create_async_engine(DATABASE_URL)
AsyncSessionLocal = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

async def init_db():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

# ==========================================
# 1. INPUT MONITORING (Background Thread)
# ==========================================
class InputMonitor:
    def __init__(self):
        self.key_count = 0
        self.click_count = 0
        self.lock = threading.Lock()
        self.running = True
        
        # Start listeners
        try:
            self.kb_listener = keyboard.Listener(on_press=self._on_press)
            self.mouse_listener = mouse.Listener(on_click=self._on_click)
            self.kb_listener.start()
            self.mouse_listener.start()
        except Exception as e:
            logger.error(f"Failed to start input listeners: {e}")

    def _on_press(self, key):
        with self.lock:
            self.key_count += 1

    def _on_click(self, x, y, button, pressed):
        if pressed:
            with self.lock:
                self.click_count += 1

    def get_and_reset_counts(self):
        with self.lock:
            k, c = self.key_count, self.click_count
            self.key_count = 0
            self.click_count = 0
        return k, c

    def stop(self):
        self.running = False
        if hasattr(self, 'kb_listener'): self.kb_listener.stop()
        if hasattr(self, 'mouse_listener'): self.mouse_listener.stop()

# ==========================================
# 2. FACE & POSE ESTIMATION
# ==========================================
class FaceMonitor:
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            min_detection_confidence=0.5, 
            min_tracking_confidence=0.5,
            refine_landmarks=True
        )
        # self.cap = cv2.VideoCapture(camera_id) # Removed for frontend stream
        
        self.cam_matrix = None
        self.dist_matrix = np.zeros((4, 1), dtype=np.float64)
        self.pitch_history = deque(maxlen=5)

    def process_image(self, image):
        # Image is already BGR (opencv format)
        img_h, img_w, _ = image.shape
        results = self.face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        pitch = None
        
        if self.cam_matrix is None:
            focal_length = 1 * img_w
            self.cam_matrix = np.array([ [focal_length, 0, img_h / 2],
                                         [0, focal_length, img_w / 2],
                                         [0, 0, 1]])

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                face_3d = []
                face_2d = []
                idx_list = [1, 152, 33, 263, 61, 291]

                for idx in idx_list:
                    lm = face_landmarks.landmark[idx]
                    x, y = int(lm.x * img_w), int(lm.y * img_h)
                    face_2d.append([x, y])
                    face_3d.append([x, y, lm.z])       
                
                face_2d = np.array(face_2d, dtype=np.float64)
                face_3d = np.array(face_3d, dtype=np.float64)

                success, rot_vec, trans_vec = cv2.solvePnP(face_3d, face_2d, self.cam_matrix, self.dist_matrix)
                
                if success:
                    rmat, jac = cv2.Rodrigues(rot_vec)
                    angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)
                    
                    raw_pitch = angles[0]
                    self.pitch_history.append(raw_pitch)
                    pitch = sum(self.pitch_history) / len(self.pitch_history)
                    
                    # Visual Guide
                    nose_2d = (int(face_landmarks.landmark[1].x * img_w), int(face_landmarks.landmark[1].y * img_h))
                    p1 = (int(nose_2d[0] + angles[1] * 10), int(nose_2d[1] - angles[0] * 10))
                    cv2.line(image, nose_2d, p1, (255, 0, 0), 3)

        if pitch is None:
            self.pitch_history.clear()

        return pitch, image

    def release(self):
        pass # self.cap.release()

# ==========================================
# 3. BACKEND SERVICE Logic
# ==========================================
class FatigueService:
    def __init__(self):
        self.input_mon = InputMonitor()
        self.face_mon = FaceMonitor()
        
        self.history_len = int(os.getenv('HISTORY_LEN', 200))
        self.timestamps = deque(maxlen=self.history_len)
        self.fatigue_history = deque(maxlen=self.history_len)
        self.prediction_history = deque(maxlen=self.history_len)
        
        self.model = LinearRegression()
        self.smooth_fatigue = 0.0  
        self.smoothing_factor = float(os.getenv('SMOOTHING_FACTOR', 0.2))

        self.last_alert_time = 0
        self.alert_cooldown = float(os.getenv('ALERT_COOLDOWN', 3.0))

        self.pause_end_time = 0
        self.start_time = time.time()
        
        # Session Metrics tracking
        self.current_session_start = None
        self.session_peak = 0.0
        self.session_sum = 0.0
        self.session_count = 0
        self.session_alerts = 0
        
        # State shared with frontend
        self.current_state = {
            "fatigue_current": 0,
            "fatigue_pred": 0,
            "status": "active", # active, away
            "away_remaining": 0,
            "alert": False, 
            "chart_data": [],
            "image_base64": ""
        }
        
        self.is_running = False
        self.thread = None
        self.latest_frame = None
        self.frame_lock = threading.Lock()

    def update_frame(self, base64_data):
        try:
             # Decode base64 to numpy
             img_bytes = base64.b64decode(base64_data)
             nparr = np.frombuffer(img_bytes, np.uint8)
             img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
             if img is not None:
                 with self.frame_lock:
                     self.latest_frame = img
        except Exception:
             pass

    def calculate_current_fatigue(self, keys, clicks, pitch):
        activity_score = min((keys * 5) + (clicks * 10), 50) 
        posture_fatigue = 0
        if pitch < -0.1: posture_fatigue = 50 
        elif pitch < -0.02: posture_fatigue = 10 
        base_fatigue = 50 - activity_score 
        return max(0, min(100, base_fatigue + posture_fatigue))

    def get_prediction(self):
        if len(self.fatigue_history) < 5:
            return self.fatigue_history[-1] if self.fatigue_history else 0

        y = np.array(self.fatigue_history)
        X = np.array(self.timestamps).reshape(-1, 1)

        weights = np.exp(np.linspace(-2, 0, len(y)))
        self.model.fit(X, y, sample_weight=weights)
        
        if not self.timestamps:
            return 0
        current_elapsed = self.timestamps[-1]
        future_time = np.array([[current_elapsed + 60]])
        prediction = self.model.predict(future_time)
        return max(0, min(100, prediction[0]))

    def start(self):
        if self.is_running: return
        self.is_running = True
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()
        logger.info("Service Started")

    def stop(self):
        self.is_running = False
        self.input_mon.stop()
        self.face_mon.release()

    async def _save_session(self):
        if self.current_session_start is None or self.session_count == 0:
            return
            
        end_time = datetime.now()
        duration = int((time.time() - self.session_unix_start))
        avg = self.session_sum / self.session_count
        
        async with AsyncSessionLocal() as db:
            record = SessionRecord(
                start_time=self.current_session_start,
                end_time=end_time,
                avg_fatigue=round(avg, 1),
                peak_fatigue=round(self.session_peak, 1),
                alert_count=self.session_alerts,
                duration_seconds=duration
            )
            db.add(record)
            await db.commit()
            logger.info(f"Session saved: {duration}s, peak {self.session_peak}")
            
        # Reset
        self.current_session_start = None
        self.session_peak = 0.0
        self.session_sum = 0.0
        self.session_count = 0
        self.session_alerts = 0

    def set_away(self, duration=300):
        if self.current_state["status"] == "active":
             # End current session
             asyncio.run_coroutine_threadsafe(self._save_session(), asyncio.get_event_loop())
        self.pause_end_time = time.time() + duration
    
    def resume(self):
        self.pause_end_time = 0

    def _loop(self):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        while self.is_running:
            try:
                loop_start = time.time()
                elapsed_time = loop_start - self.start_time
                
                # Check Away Mode
                remaining_pause = self.pause_end_time - loop_start
                if remaining_pause > 0:
                    if self.current_state["status"] == "active":
                        # This shouldn't happen usually if set_away is called via method
                        pass
                    self.current_state["status"] = "away"
                    self.current_state["away_remaining"] = int(remaining_pause)
                    self.current_state["alert"] = False
                    
                    time.sleep(0.1)
                    continue

                if self.current_state["status"] == "away" or self.current_session_start is None:
                    # Starting new session
                    self.current_session_start = datetime.now()
                    self.session_unix_start = time.time()
                    self.session_peak = 0.0
                    self.session_sum = 0.0
                    self.session_count = 0
                    self.session_alerts = 0
                    logger.info("New session started")

                self.current_state["status"] = "active"
                self.current_state["away_remaining"] = 0
                
                # Input & Face
                keys, clicks = self.input_mon.get_and_reset_counts()
                
                # Process latest frame if available
                img_to_process = None
                with self.frame_lock:
                    if self.latest_frame is not None:
                        img_to_process = self.latest_frame
                        self.latest_frame = None # Consume it
                
                pitch = None
                if img_to_process is not None:
                    pitch, _ = self.face_mon.process_image(img_to_process)
                
                if pitch is None:
                    pitch = -20 # Penalty for no face / no frame
                
                target = self.calculate_current_fatigue(keys, clicks, pitch)
                self.smooth_fatigue = (self.smooth_fatigue * (1.0 - self.smoothing_factor)) + (target * self.smoothing_factor)
                
                # Update Session Metrics
                self.session_count += 1
                self.session_sum += self.smooth_fatigue
                self.session_peak = max(self.session_peak, self.smooth_fatigue)
                
                self.timestamps.append(elapsed_time)
                self.fatigue_history.append(self.smooth_fatigue)
                
                # Predict
                pred = self.get_prediction()
                self.prediction_history.append(pred)
                
                # Check Alert
                self.current_state["alert"] = False
                fatigue_threshold = float(os.getenv('FATIGUE_THRESHOLD', 90))
                if self.smooth_fatigue >= fatigue_threshold:
                    if loop_start - self.last_alert_time > self.alert_cooldown:
                        self.current_state["alert"] = True
                        self.last_alert_time = loop_start
                        self.session_alerts += 1

                # Update State
                self.current_state["fatigue_current"] = round(self.smooth_fatigue, 1)
                self.current_state["fatigue_pred"] = round(pred, 1)
                
                # We do NOT send image back to frontend (saves bandwidth)
                self.current_state["image_base64"] = "" 
                    
                # Chart Data 
                data_points = []
                ts_list = list(self.timestamps)
                val_list = list(self.fatigue_history)
                pred_list = list(self.prediction_history)
                for t, v, p in zip(ts_list, val_list, pred_list):
                     data_points.append({
                         "time": round(t, 1), 
                         "value": round(v, 1),
                         "pred": round(p, 1)
                     })
                self.current_state["chart_data"] = data_points

            except Exception as e:
                logger.error(f"FatigueService Loop Error: {e}")
            
            time.sleep(0.1) # 10 FPS

service = FatigueService()

@app.on_event("startup")
async def startup_event():
    await init_db()
    service.start()

@app.on_event("shutdown")
async def shutdown_event():
    await service._save_session()
    service.stop()

@app.get("/history")
async def get_history():
    async with AsyncSessionLocal() as db:
        # Fetch sessions
        stmt = select(SessionRecord).order_by(desc(SessionRecord.start_time)).limit(50)
        result = await db.execute(stmt)
        sessions = result.scalars().all()
        
        history_list = []
        for s in sessions:
            history_list.append({
                "id": s.id,
                "timestamp": s.start_time.strftime("%Y-%m-%d %H:%M") if s.start_time else "",
                "duration": f"{s.duration_seconds//60}m {s.duration_seconds%60}s" if s.duration_seconds else "0s",
                "peak": s.peak_fatigue,
                "avg": s.avg_fatigue,
                "alerts": s.alert_count
            })
        
        # Calculate overall stats
        avg_res = await db.execute(select(func.avg(SessionRecord.avg_fatigue)))
        alert_res = await db.execute(select(func.sum(SessionRecord.alert_count)))
        time_res = await db.execute(select(func.sum(SessionRecord.duration_seconds)))
        
        stats = {
            "avg_fatigue": round(avg_res.scalar() or 0, 1),
            "total_alerts": int(alert_res.scalar() or 0),
            "total_time": round((time_res.scalar() or 0) / 3600, 1)
        }
        
        return {"history": history_list, "stats": stats}

# Mount Static Files
base_dir = os.path.dirname(os.path.abspath(__file__))
# Frontend
frontend_dir = os.path.join(base_dir, "../frontend")
if not os.path.exists(frontend_dir):
    os.makedirs(frontend_dir)
app.mount("/static", StaticFiles(directory=frontend_dir, html=True), name="static")

@app.get("/")
async def root():
    return RedirectResponse(url="/static/index.html")

# Resources (for audio)
resources_dir = os.path.join(base_dir, "../resources")
if not os.path.exists(resources_dir):
    os.makedirs(resources_dir, exist_ok=True)
app.mount("/resources", StaticFiles(directory=resources_dir), name="resources")

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            try:
                # Non-blocking receive for commands
                try:
                    data = await asyncio.wait_for(websocket.receive_text(), 0.01)
                    cmd = json.loads(data)
                    if cmd.get("type") == "set_away":
                        service.set_away(cmd.get("value", 300))
                    elif cmd.get("type") == "resume":
                        service.resume()
                    elif cmd.get("type") == "frame":
                        service.update_frame(cmd.get("data"))
                except asyncio.TimeoutError:
                    pass

                # Send State
                await websocket.send_json(service.current_state)
                await asyncio.sleep(0.1) 
            except Exception as e:
                logger.error(f"WS Loop Error: {e}")
                await asyncio.sleep(1)
                break
    except WebSocketDisconnect:
        pass