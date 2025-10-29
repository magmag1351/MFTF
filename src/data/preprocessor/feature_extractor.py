import numpy as np

class FeatureExtractor:
    def extract(self, kb_count, mouse_count, face_score):
        return np.array([
            kb_count / 100.0,
            mouse_count / 50.0,
            face_score
        ])
