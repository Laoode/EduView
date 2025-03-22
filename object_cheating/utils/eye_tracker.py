# object_cheating/utils/eye_tracker.py
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from typing import Tuple, List, Optional

class EyeTracker:
    def __init__(self):
        # Model parameters
        self.IMG_SIZE = (64, 56)
        self.CLASS_LABELS = ['center', 'left', 'right']
        self.FRAMES_TO_ALERT = 6
        self.last_timestamp = 0
        
        # Initialize models
        self.face_landmarker = self._init_mediapipe()
        self.eye_model = load_model('object_cheating/models/eye_model2.h5')

    def _init_mediapipe(self) -> vision.FaceLandmarker:
        """Initialize MediaPipe Face Landmarker"""
        base_options = python.BaseOptions(
            model_asset_path='object_cheating/models/face_landmarker.task'
        )
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        return vision.FaceLandmarker.create_from_options(options)

    def process_frame(
        self, 
        frame: np.ndarray,
        alert_counter: int,
        frame_counter: int
    ) -> Tuple[np.ndarray, List[str], int, int]:
        """Process frame and return updated state"""
        alerts = []
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Update timestamp logic
        current_timestamp = int(frame_counter * 33.33)  # Convert to milliseconds
        if current_timestamp <= self.last_timestamp:
            current_timestamp = self.last_timestamp + 1
        self.last_timestamp = current_timestamp
        
        # Process with MediaPipe
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        result = self.face_landmarker.detect_for_video(mp_image, current_timestamp)
        frame_counter += 1

        if result.face_landmarks:
            for landmarks in result.face_landmarks:
                # Gambar bounding box dan teks
                frame = self._draw_eye_annotations(frame, landmarks)
                
                # Logika deteksi
                suspicious = self._process_eyes(frame, landmarks)
                
                # Update alert counter
                if suspicious:
                    alert_counter += 1
                    if alert_counter >= self.FRAMES_TO_ALERT:
                        alerts.append("Suspicious eye movement detected!")
                else:
                    alert_counter = 0

        return frame, alerts, alert_counter, frame_counter

    def _draw_eye_annotations(self, frame: np.ndarray, landmarks) -> np.ndarray:
        """Tambahkan anotasi seperti di model3.py"""
        try:
            # Left eye
            left_gaze, left_rect = self._get_eye_data(frame, landmarks, [33, 133, 159, 145])
            if left_rect:
                x_min, y_min, x_max, y_max = left_rect
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                cv2.putText(frame, f"Left: {left_gaze}", (x_min, y_min-10), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Right eye
            right_gaze, right_rect = self._get_eye_data(frame, landmarks, [362, 263, 386, 374])
            if right_rect:
                x_min, y_min, x_max, y_max = right_rect
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                cv2.putText(frame, f"Right: {right_gaze}", (x_min, y_min-10),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
        except Exception as e:
            print(f"Annotation error: {str(e)}")
        
        return frame

    def _get_eye_data(self, frame, landmarks, indices):
        """Helper untuk mendapatkan data mata"""
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            h, w = gray.shape
            
            # Perbaikan sintaksis di sini
            eye_points = np.array([
                [int(landmarks[i].x * w), int(landmarks[i].y * h)] 
                for i in indices
            ], dtype=np.int64)
            
            x1, y1 = np.min(eye_points, axis=0)
            x2, y2 = np.max(eye_points, axis=0)
            eye_img = gray[y1:y2, x1:x2]
            
            processed = cv2.resize(eye_img, self.IMG_SIZE)
            processed = processed.reshape((1, *self.IMG_SIZE, 1)).astype(np.float32) / 255.0
            prediction = self.eye_model.predict(processed, verbose=0)
            
            return (
                self.CLASS_LABELS[np.argmax(prediction)],
                [x1, y1, x2, y2]
            )
        except Exception as e:
            print(f"Error processing eye data: {str(e)}")
            return None, None

    def _process_eyes(self, frame: np.ndarray, landmarks) -> bool:
        """Process both eyes and return suspicious status"""
        # Left eye indices: [33, 133, 159, 145]
        # Right eye indices: [362, 263, 386, 374]
        suspicious = False
        
        # Process left eye
        left_gaze = self._process_single_eye(frame, landmarks, [33, 133, 159, 145])
        # Process right eye
        right_gaze = self._process_single_eye(frame, landmarks, [362, 263, 386, 374])
        
        if left_gaze in ['left', 'right'] or right_gaze in ['left', 'right']:
            suspicious = True
            
        return suspicious

    def _process_single_eye(self, frame: np.ndarray, landmarks, indices) -> str:
        """Process single eye and return gaze direction"""
        try:
            _, gaze = self._get_eye_data(frame, landmarks, indices)
            return gaze
        except:
            return 'center'