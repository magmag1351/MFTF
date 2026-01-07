import cv2
import os
from ffpyplayer.player import MediaPlayer
import mediapipe as mp
import numpy as np
import time
import threading
import winsound  # Windows標準サウンドライブラリ
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Button
from pynput import keyboard, mouse
from sklearn.linear_model import LinearRegression
from collections import deque
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# ==========================================
# 1. INPUT MONITORING (Background Thread)
# ==========================================
class InputMonitor:
    def __init__(self):
        self.key_count = 0
        self.click_count = 0
        self.lock = threading.Lock()
        self.running = True
        
        # 入力監視リスナーの起動
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
# 2. FACE & POSE ESTIMATION (With Smoothing)
# ==========================================
class FaceMonitor:
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            min_detection_confidence=0.5, 
            min_tracking_confidence=0.5,
            refine_landmarks=True
        )
        camera_id = int(os.getenv('CAMERA_ID', 0))
        self.cap = cv2.VideoCapture(camera_id)
        
        # 【改善】カメラ行列計算のキャッシュ用
        self.cam_matrix = None
        self.dist_matrix = np.zeros((4, 1), dtype=np.float64)
        
        # 【改善】入力値の平滑化用（直近5フレームの平均をとる）
        self.pitch_history = deque(maxlen=5)

    def get_head_pose(self):
        success, image = self.cap.read()
        if not success:
            return None, None

        image = cv2.flip(image, 1)
        img_h, img_w, _ = image.shape
        results = self.face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        pitch = None
        
        # 初回のみカメラ行列を計算
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
                    
                    # 【改善】Pitchの移動平均を計算
                    self.pitch_history.append(raw_pitch)
                    pitch = sum(self.pitch_history) / len(self.pitch_history)
                    
                    # Visual Guide (鼻先の線を描画)
                    nose_2d = (int(face_landmarks.landmark[1].x * img_w), int(face_landmarks.landmark[1].y * img_h))
                    p1 = (int(nose_2d[0] + angles[1] * 10), int(nose_2d[1] - angles[0] * 10))
                    cv2.line(image, nose_2d, p1, (255, 0, 0), 3)

        # 顔を見失った場合は履歴をクリア
        if pitch is None:
            self.pitch_history.clear()

        return pitch, image

    def release(self):
        self.cap.release()

# ==========================================
# 2.5. BACKGROUND AUDIO PLAYER
# ==========================================
class BackgroundAudioPlayer:
    def __init__(self, video_path):
        self.video_path = video_path
        # ff_opts={'t': 'audio'} ensures we strictly prioritize audio logic if needed, 
        # but MediaPlayer usually handles audio automatically.
        self.player = MediaPlayer(video_path)
        
    def update(self):
        # ffpyplayer handles audio in its own thread. 
        # We just need to check for EOF to loop.
        frame, val = self.player.get_frame()
        if val == 'eof':
            # Restart audio
            self.player.seek(0)
            
    def set_pause(self, pause):
        self.player.set_pause(pause)

    def release(self):
        self.player.close_player()

# ==========================================
# 3. MACHINE LEARNING & APP LOGIC (Complete)
# ==========================================
class FatigueApp:
    def __init__(self):
        self.input_mon = InputMonitor()
        self.face_mon = FaceMonitor()
        
        self.history_len = int(os.getenv('HISTORY_LEN', 200))
        self.timestamps = deque(maxlen=self.history_len)
        self.fatigue_history = deque(maxlen=self.history_len)
        
        # self.time_counter = 0  # Frame-based counter removed
        self.model = LinearRegression()
        
        # 【改善】滑らかな変化（慣性）のための変数
        self.smooth_fatigue = 0.0  
        self.smoothing_factor = float(os.getenv('SMOOTHING_FACTOR', 0.2)) # 小さいほど変化がゆっくりになる（0.05〜0.2推奨）

        # 【追加】警告音制御用
        self.last_alert_time = 0
        self.alert_cooldown = float(os.getenv('ALERT_COOLDOWN', 3.0)) # 警告音のインターバル（秒）

        self.pause_end_time = 0

        # 【追加】背景音声プレーヤー (MP4の音声のみ再生)
        video_path = os.getenv('VIDEO_PATH', os.path.join('resources', 'videoplayback.mp4'))
        self.bg_player = BackgroundAudioPlayer(video_path)

        # グラフ初期設定
        # plt.style.use('dark_background') # Remove default dark mode to customize
        
        # iOS Dark Mode Colors
        self.ios_bg = '#000000'
        self.ios_card = '#1C1C1E'
        self.ios_text = '#FFFFFF'
        self.ios_subtext = '#8E8E93'
        self.ios_cyan = '#64D2FF'
        self.ios_purple = '#BF5AF2'
        self.ios_blue = '#0A84FF'
        self.ios_grid = '#3A3A3C'
        
        # Custom rcParams for iOS look
        plt.rcParams['font.family'] = 'sans-serif'
        plt.rcParams['text.color'] = self.ios_text
        plt.rcParams['axes.labelcolor'] = self.ios_subtext
        plt.rcParams['xtick.color'] = self.ios_subtext
        plt.rcParams['ytick.color'] = self.ios_subtext
        
        # 画面レイアウト変更: 横に2つ並べる (左:カメラ, 右:グラフ)
        self.fig = plt.figure(figsize=(16, 7), facecolor=self.ios_bg)
        gs = self.fig.add_gridspec(1, 2, width_ratios=[1, 1])
        
        self.ax_cam = self.fig.add_subplot(gs[0])
        self.ax_graph = self.fig.add_subplot(gs[1])
        
        plt.subplots_adjust(right=0.85, wspace=0.2) # ボタン用のスペースを確保
        
        # --- カメラ表示用設定 ---
        self.ax_cam.set_title("Camera Feed", color=self.ios_text)
        self.ax_cam.axis('off')
        # 初期画像プレースホルダー (真っ黒な画像)
        self.im_cam = self.ax_cam.imshow(np.zeros((480, 640, 3), dtype='uint8'))

        # --- グラフ表示用設定 ---
        self.ax_graph.set_facecolor(self.ios_card) # Card-like background
        # Remove top and right spines for cleaner look
        self.ax_graph.spines['top'].set_visible(False)
        self.ax_graph.spines['right'].set_visible(False)
        self.ax_graph.spines['bottom'].set_color(self.ios_grid)
        self.ax_graph.spines['left'].set_color(self.ios_grid)
        
        self.line_current, = self.ax_graph.plot([], [], color=self.ios_cyan, linewidth=2)
        self.line_pred, = self.ax_graph.plot([], [], color=self.ios_purple, linestyle='--', linewidth=2)
        
        self.ax_graph.set_xlim(-60, 60)
        self.ax_graph.set_ylim(0, 100) # 0〜100%
        
        self.ax_graph.set_title("Fatigue Level Monitor", color=self.ios_text, fontweight='bold')
        self.ax_graph.set_xlabel("Time (Seconds)")
        self.ax_graph.set_ylabel("Fatigue Level (%)")
        
        # Legend styling
        legend = self.ax_graph.legend(loc='upper left', frameon=False)
        for text in legend.get_texts():
            text.set_color(self.ios_subtext)
            
        self.ax_graph.grid(True, color=self.ios_grid, linestyle=':', alpha=0.5)
        
        self.vline = self.ax_graph.axvline(x=0, color=self.ios_text, alpha=0.3, linestyle=':')
        self.txt_curr = self.ax_graph.text(0, 95, "Current: --", color=self.ios_cyan, fontsize=12, fontweight='bold')
        self.txt_pred = self.ax_graph.text(0, 88, "Pred (+60s): --", color=self.ios_purple, fontsize=12)
        self.txt_status = self.ax_graph.text(0, 81, "", color=self.ios_subtext, fontsize=12)
        
        # 離籍ボタンの追加 (iOS Style)
        self.ax_button = plt.axes([0.87, 0.45, 0.1, 0.05]) # [left, bottom, width, height]
        # Matplotlib buttons are hard to style perfectly flat without borders, trying best effort
        self.btn_away = Button(self.ax_button, 'Away\n(5min)', color=self.ios_card, hovercolor=self.ios_grid)
        self.btn_away.label.set_color(self.ios_text)
        self.btn_away.on_clicked(self.on_away_button_click)

        # 再開ボタンの追加 (iOS Style)
        self.ax_resume_button = plt.axes([0.87, 0.35, 0.1, 0.05]) # [left, bottom, width, height]
        self.btn_resume = Button(self.ax_resume_button, 'Resume', color=self.ios_blue, hovercolor='#409CFF')
        self.btn_resume.label.set_color('white')
        self.btn_resume.on_clicked(self.on_resume_button_click)

        # 音楽停止・再生ボタンの追加 (横並び)
        self.ax_music_stop = plt.axes([0.87, 0.25, 0.045, 0.05])
        self.ax_music_play = plt.axes([0.925, 0.25, 0.045, 0.05])
        
        self.btn_music_stop = Button(self.ax_music_stop, 'Stop\nMusic', color=self.ios_card, hovercolor=self.ios_grid)
        self.btn_music_play = Button(self.ax_music_play, 'Play\nMusic', color=self.ios_blue, hovercolor='#409CFF')
        
        self.btn_music_stop.label.set_color(self.ios_text)
        self.btn_music_stop.label.set_fontsize(8)
        self.btn_music_play.label.set_color('white')
        self.btn_music_play.label.set_fontsize(8)
        
        self.btn_music_stop.on_clicked(self.on_music_stop_click)
        self.btn_music_play.on_clicked(self.on_music_play_click)

        self.fig.canvas.mpl_connect('close_event', self.on_close)
        self.is_running = True
        
        self.start_time = time.time()
        self.last_update = time.time()
        self.update_interval = 1.0

    def on_away_button_click(self, event):
        """離籍ボタンが押された時の処理"""
        away_time = int(os.getenv('AWAY_TIME_SECONDS', 300))
        self.pause_end_time = time.time() + away_time # 5分間停止
        print(f"Away mode started ({away_time // 60} minutes)")

    def on_resume_button_click(self, event):
        """再開ボタンが押された時の処理"""
        self.pause_end_time = 0
        print("Resumed monitoring")

    def on_music_stop_click(self, event):
        """音楽停止ボタンが押された時の処理"""
        self.bg_player.set_pause(True)
        print("Music stopped")

    def on_music_play_click(self, event):
        """音楽再生ボタンが押された時の処理"""
        self.bg_player.set_pause(False)
        print("Music resumed")

    def play_warning_sound(self):
        """別スレッドでビープ音を再生"""
        try:
            # 1000Hz, 500ms
            winsound.Beep(1000, 500)
        except Exception as e:
            print(f"Sound Error: {e}")

    def calculate_current_fatigue(self, keys, clicks, pitch):
        activity_score = min((keys * 5) + (clicks * 10), 50) 
        
        posture_fatigue = 0
        # 【修正】最大疲労度が100になるようにペナルティを調整
        if pitch < -0.1: posture_fatigue = 50 # 強い下向き
        elif pitch < -0.02: posture_fatigue = 10 # 軽い下向き
            
        base_fatigue = 50 - activity_score 
        return max(0, min(100, base_fatigue + posture_fatigue))

    def get_prediction_trajectory(self):
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

    def update(self, frame):
        if not self.is_running: return

        # 音声のループ確認
        self.bg_player.update()

        pitch, cam_img = self.face_mon.get_head_pose()
        if cam_img is not None:
            # BGR -> RGB変換してMatplotlibで表示
            rgb_img = cv2.cvtColor(cam_img, cv2.COLOR_BGR2RGB)
            self.im_cam.set_data(rgb_img)
            
        if cv2.waitKey(1) & 0xFF == ord('q'):
            plt.close(self.fig)
            self.on_close(None)
            return

        current_ts = time.time()
        elapsed_time = current_ts - self.start_time # Elapsed time for axis calculations

        if current_ts - self.last_update > self.update_interval:
            self.last_update = current_ts
            
            # 離籍モードのチェック
            remaining_pause = self.pause_end_time - current_ts
            if remaining_pause > 0:
                mins, secs = divmod(int(remaining_pause), 60)
                status_text = f"Away Mode: {mins}:{secs:02d}"
                self.txt_status.set_text(status_text)
                
                # グラフのX軸範囲を更新して、テキスト位置を維持する
                left_limit = elapsed_time - 60
                
                self.txt_status.set_position((left_limit + 2, 81))
                self.txt_curr.set_position((left_limit + 2, 95))
                self.txt_pred.set_position((left_limit + 2, 88))
                
                # Ensure axis is updated even in away mode to scroll
                self.ax_graph.set_xlim(left_limit, elapsed_time + 60)
                
                return # 計測・更新をスキップ

            # 通常モード
            self.txt_status.set_text("")
            
            keys, clicks = self.input_mon.get_and_reset_counts()
            
            # 顔が見つからない場合（不在/顔隠し）
            if pitch is None:
                pitch = -20
                print("Face Lost: Applying max penalty")
            
            # 1. 瞬間的な目標疲労度を計算
            target_fatigue = self.calculate_current_fatigue(keys, clicks, pitch)
            
            # 2. 【改善】慣性をつけて滑らかに変化させる（ローパスフィルタ）
            # 新しい値 = (今の値 * 0.9) + (目標値 * 0.1)
            self.smooth_fatigue = (self.smooth_fatigue * (1.0 - self.smoothing_factor)) + (target_fatigue * self.smoothing_factor)

            # 3. リスト更新
            self.timestamps.append(elapsed_time)
            self.fatigue_history.append(self.smooth_fatigue)
            
            # 4. 警告判定（滑らかな値で判定するため誤検知が減る）
            fatigue_threshold = float(os.getenv('FATIGUE_THRESHOLD', 90))
            if self.smooth_fatigue >= fatigue_threshold:
                if current_ts - self.last_alert_time > self.alert_cooldown:
                    threading.Thread(target=self.play_warning_sound, daemon=True).start()
                    self.last_alert_time = current_ts
                    print(f"⚠️ WARNING: Fatigue Critical ({self.smooth_fatigue:.1f}%)")

            # 予測計算
            f_pred_target = self.get_prediction_trajectory()

            # グラフ描画更新
            self.line_current.set_data(self.timestamps, self.fatigue_history)
            
            # 予測線: 現在時刻〜現在時刻+60秒
            pred_x = [elapsed_time, elapsed_time + 60]
            pred_y = [self.smooth_fatigue, f_pred_target]
            self.line_pred.set_data(pred_x, pred_y)
            
            # X軸の範囲: 現在時刻の前後60秒
            left_limit = elapsed_time - 60
            right_limit = elapsed_time + 60
            self.ax_graph.set_xlim(left_limit, right_limit)
            
            self.vline.set_xdata([elapsed_time])
            
            self.txt_curr.set_text(f"Current: {self.smooth_fatigue:.1f}")
            self.txt_pred.set_text(f"Pred (+60s): {f_pred_target:.1f}")
            
            self.txt_curr.set_position((left_limit + 2, 95))
            self.txt_pred.set_position((left_limit + 2, 88))
            self.txt_status.set_position((left_limit + 2, 81))

    def on_close(self, event):
        self.is_running = False
        self.is_running = False
        self.input_mon.stop()
        self.face_mon.release()
        self.bg_player.release()
        cv2.destroyAllWindows()

    def run(self):
        ani = FuncAnimation(self.fig, self.update, interval=100, cache_frame_data=False)
        try:
            plt.show()
        except KeyboardInterrupt:
            self.on_close(None)

if __name__ == "__main__":
    print("Starting Fatigue Monitor...")
    print("Press 'q' on the camera window or close the graph to exit.")
    app = FatigueApp()
    app.run()