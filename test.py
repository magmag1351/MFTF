import cv2
import mediapipe as mp
import numpy as np
import time
import threading
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from pynput import keyboard, mouse
from sklearn.linear_model import LinearRegression
from collections import deque

# ==========================================
# 1. INPUT MONITORING (Background Thread)
# ==========================================
class InputMonitor:
    def __init__(self):
        self.key_count = 0
        self.click_count = 0
        self.lock = threading.Lock()
        self.running = True
        
        self.kb_listener = keyboard.Listener(on_press=self._on_press)
        self.mouse_listener = mouse.Listener(on_click=self._on_click)
        
        self.kb_listener.start()
        self.mouse_listener.start()

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
        self.kb_listener.stop()
        self.mouse_listener.stop()

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
        self.cap = cv2.VideoCapture(0)

    def get_head_pose(self):
        success, image = self.cap.read()
        if not success:
            return None, None

        image = cv2.flip(image, 1)
        img_h, img_w, _ = image.shape
        results = self.face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        # 【修正箇所】初期値を 0 ではなく None に設定
        # これにより、顔が見つからない場合に明確に「不在」として扱える
        pitch = None
        
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

                focal_length = 1 * img_w
                cam_matrix = np.array([ [focal_length, 0, img_h / 2],
                                        [0, focal_length, img_w / 2],
                                        [0, 0, 1]])
                dist_matrix = np.zeros((4, 1), dtype=np.float64)

                success, rot_vec, trans_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix)
                
                if success:
                    rmat, jac = cv2.Rodrigues(rot_vec)
                    angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)
                    
                    pitch = angles[0]
                    
                    # Visual Guide
                    nose_2d = (int(face_landmarks.landmark[1].x * img_w), int(face_landmarks.landmark[1].y * img_h))
                    p1 = (int(nose_2d[0] + angles[1] * 10), int(nose_2d[1] - angles[0] * 10))
                    cv2.line(image, nose_2d, p1, (255, 0, 0), 3)

        return pitch, image

    def release(self):
        self.cap.release()

# ==========================================
# 3. MACHINE LEARNING & APP LOGIC
# ==========================================
class FatigueApp:
    def __init__(self):
        self.input_mon = InputMonitor()
        self.face_mon = FaceMonitor()
        
        self.history_len = 200 
        self.timestamps = deque(maxlen=self.history_len)
        self.fatigue_history = deque(maxlen=self.history_len)
        
        self.time_counter = 0
        
        self.model = LinearRegression()
        
        self.start_time = time.time()
        self.update_interval = 1.0
        self.last_update = time.time()

        plt.style.use('dark_background')
        self.fig, self.ax = plt.subplots(figsize=(12, 6))
        
        self.line_current, = self.ax.plot([], [], 'c-', label='History (Past 60s)', linewidth=2)
        self.line_pred, = self.ax.plot([], [], 'm--', label='Forecast (+60s)', linewidth=2)
        
        self.ax.set_xlim(-60, 60)
        self.ax.set_ylim(0, 100)
        
        self.ax.set_title("Fatigue Level: History & Prediction")
        self.ax.set_xlabel("Time (Seconds)")
        self.ax.set_ylabel("Fatigue Level (%)")
        self.ax.legend(loc='upper left')
        self.ax.grid(True, alpha=0.3)
        
        self.vline = self.ax.axvline(x=0, color='white', alpha=0.5, linestyle=':')
        
        self.txt_curr = self.ax.text(0, 95, "Current: --", color='cyan', fontsize=12)
        self.txt_pred = self.ax.text(0, 88, "Pred (+60s): --", color='magenta', fontsize=12)
        
        self.fig.canvas.mpl_connect('close_event', self.on_close)
        self.is_running = True

    def calculate_current_fatigue(self, keys, clicks, pitch):
        activity_score = min((keys * 5) + (clicks * 10), 50) 
        
        posture_fatigue = 0
        # 感度が鋭敏すぎるため、少し緩和した閾値に戻すことを推奨しますが、
        # 現在の設定(-0.1)のままにしてあります。
        if pitch < -0.1: posture_fatigue = 40
        elif pitch < -0.02: posture_fatigue = 10
            
        base_fatigue = 50 - activity_score 
        return max(0, min(100, base_fatigue + posture_fatigue))

    def get_prediction_trajectory(self):
        if len(self.fatigue_history) < 5:
            return self.fatigue_history[-1] if self.fatigue_history else 0

        y = np.array(self.fatigue_history)
        X = np.array(self.timestamps).reshape(-1, 1)
        
        self.model.fit(X, y)
        
        future_time = np.array([[self.time_counter + 60]])
        prediction = self.model.predict(future_time)
        
        return max(0, min(100, prediction[0]))

    def update(self, frame):
        if not self.is_running: return

        pitch, cam_img = self.face_mon.get_head_pose()
        if cam_img is not None:
            cv2.imshow('Face Monitor', cam_img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                plt.close(self.fig)
                self.on_close(None)
                return

        current_ts = time.time()
        if current_ts - self.last_update > self.update_interval:
            self.last_update = current_ts
            
            keys, clicks = self.input_mon.get_and_reset_counts()
            
            # 【ここが機能するようになります】
            # FaceMonitorがNoneを返すようになったため、ここに入り、-20（激しい疲労/不在）がセットされる
            if pitch is None:
                pitch = -20
                print("Face Lost: Setting Pitch to -20") # 確認用ログ
            
            # debug
            print(f"現在のPitch角度: {pitch:.2f}")

            f_curr = self.calculate_current_fatigue(keys, clicks, pitch)
            
            self.time_counter += 1
            self.timestamps.append(self.time_counter)
            self.fatigue_history.append(f_curr)
            
            f_pred_target = self.get_prediction_trajectory()

            self.line_current.set_data(self.timestamps, self.fatigue_history)
            
            pred_x = [self.time_counter, self.time_counter + 60]
            pred_y = [f_curr, f_pred_target]
            self.line_pred.set_data(pred_x, pred_y)
            
            left_limit = self.time_counter - 60
            right_limit = self.time_counter + 60
            self.ax.set_xlim(left_limit, right_limit)
            
            self.vline.set_xdata([self.time_counter])
            
            self.txt_curr.set_text(f"Current: {f_curr:.1f}")
            self.txt_pred.set_text(f"Pred (+60s): {f_pred_target:.1f}")
            
            self.txt_curr.set_position((left_limit + 2, 95))
            self.txt_pred.set_position((left_limit + 2, 88))

    def on_close(self, event):
        self.is_running = False
        self.input_mon.stop()
        self.face_mon.release()
        cv2.destroyAllWindows()

    def run(self):
        ani = FuncAnimation(self.fig, self.update, interval=100, cache_frame_data=False)
        try:
            plt.show()
        except KeyboardInterrupt:
            self.on_close(None)

if __name__ == "__main__":
    app = FatigueApp()
    app.run()