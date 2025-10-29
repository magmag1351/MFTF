import cv2
import mediapipe as mp

class WebcamMonitor:
    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(max_num_faces=1)

    def get_fatigue_indicator(self):
        ret, frame = self.cap.read()
        if not ret:
            return 0

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self.face_mesh.process(rgb)

        if not result.multi_face_landmarks:
            return 0

        # 簡易的に「検出された顔あり」→覚醒度1、「なし」→0とする
        return 1

    def __del__(self):
        if self.cap.isOpened():
            self.cap.release()
