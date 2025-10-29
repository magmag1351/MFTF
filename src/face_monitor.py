import cv2
import mediapipe as mp
import numpy as np
from datetime import datetime

class FaceMonitor:
    def __init__(self):
        self.mp_face = mp.solutions.face_mesh
        self.face_mesh = self.mp_face.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True)
        self.cap = cv2.VideoCapture(0)
        self.fatigue_score = 0  # 疲労度スコア（簡易）
    
    def calculate_eye_aspect_ratio(self, landmarks, eye_indices):
        """目の開き具合を簡易的に算出"""
        p1, p2, p3, p4, p5, p6 = [np.array([landmarks[i].x, landmarks[i].y]) for i in eye_indices]
        vertical1 = np.linalg.norm(p2 - p6)
        vertical2 = np.linalg.norm(p3 - p5)
        horizontal = np.linalg.norm(p1 - p4)
        ear = (vertical1 + vertical2) / (2.0 * horizontal)
        return ear
    
    def get_fatigue_score(self):
        success, frame = self.cap.read()
        if not success:
            return None
        
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(frame_rgb)
        
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                # 左目と右目のランドマークインデックス（MediaPipe FaceMesh仕様）
                LEFT_EYE = [33, 160, 158, 133, 153, 144]
                RIGHT_EYE = [362, 385, 387, 263, 373, 380]

                left_ear = self.calculate_eye_aspect_ratio(face_landmarks.landmark, LEFT_EYE)
                right_ear = self.calculate_eye_aspect_ratio(face_landmarks.landmark, RIGHT_EYE)
                ear = (left_ear + right_ear) / 2.0

                # 閾値より小さいほど目が閉じている（疲労度上昇）
                fatigue = max(0, (0.3 - ear) * 100)
                self.fatigue_score = round(fatigue, 2)
                return self.fatigue_score

        return None

    def release(self):
        self.cap.release()
        cv2.destroyAllWindows()
