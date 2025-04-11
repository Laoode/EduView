import cv2
import numpy as np
from tensorflow.keras.models import load_model
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from typing import Tuple, List, Optional
import time

class EyeTracker:
    def __init__(self):
        # Model parameters
        self.IMG_SIZE = (56, 64)
        self.CLASS_LABELS = ['center', 'left', 'right']
        self.last_timestamp = time.time()
        self.last_direction = "center"
        self.direction_start_time = time.time()
        self.alert_level = 0

        # Initialize single landmarker for all modes
        self.face_landmarker = self._init_mediapipe()
        self.eye_model = load_model('object_cheating/models/eye_modelv3.h5')

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

    def process_frame(self, frame, alert_counter, frame_counter, 
                     cnn_threshold=0.6, movement_threshold=0.3, duration_threshold=5.0,
                     is_video=False):
        alerts = []
        current_time = time.time()
        processed_frame = frame.copy()
        
        # Inisialisasi nilai default untuk arah dan kepercayaan
        left_direction, left_conf = "center", 0.0
        right_direction, right_conf = "center", 0.0
        
        try:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            timestamp_ms = int(current_time * 1000)
            
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            detection_result = self.face_landmarker.detect_for_video(mp_image, timestamp_ms)
            
            if detection_result.face_landmarks:
                landmarks = detection_result.face_landmarks[0]
                
                # Process both eyes
                left_result = self._process_single_eye(frame, landmarks, [33, 133, 159, 145])
                right_result = self._process_single_eye(frame, landmarks, [362, 263, 386, 374])
                
                left_direction, left_conf = left_result
                right_direction, right_conf = right_result
                
                # Log untuk debugging
                print(f"Left eye: direction={left_direction}, confidence={left_conf}")
                print(f"Right eye: direction={right_direction}, confidence={right_conf}")
                
                # Check if either eye has sufficient confidence
                if left_conf > cnn_threshold or right_conf > cnn_threshold:
                    current_direction = "center"
                    if left_direction in ["left", "right"] or right_direction in ["left", "right"]:
                        current_direction = "side"
                    
                    # Update timing for alerts
                    if current_direction != self.last_direction:
                        self.direction_start_time = current_time
                        self.last_direction = current_direction
                    
                    direction_duration = current_time - self.direction_start_time
                    
                    # Process alerts based on mode
                    if is_video:
                        if current_direction == "side":
                            if direction_duration > duration_threshold * 2:
                                self.alert_level = 2
                                alerts.append("CHEATING: Prolonged side viewing")
                            elif direction_duration > duration_threshold:
                                self.alert_level = 1
                                alerts.append("WARNING: Suspicious movement")
                        else:
                            if direction_duration > 1.0:
                                self.alert_level = 0
                    else:
                        if current_direction == "side":
                            self.alert_level = 2
                            alerts.append("Side-looking detected")
                        else:
                            self.alert_level = 0
                
                # Draw annotations
                color = [(0, 252, 124),  # Lawn green for normal
                        (0, 165, 255),  # Orange for suspicious
                        (71, 99, 255)][self.alert_level]  # Tomato for cheating
                
                # Draw eye boxes and labels
                h, w = frame.shape[:2]
                
                # Draw left eye only if detected with sufficient confidence
                if left_conf > cnn_threshold:
                    left_points = np.array([[int(landmarks[i].x * w), int(landmarks[i].y * h)] 
                                        for i in [33, 133, 159, 145]], dtype=np.int32)
                    left_rect = cv2.boundingRect(left_points)
                    cv2.rectangle(processed_frame, 
                                (left_rect[0], left_rect[1]), 
                                (left_rect[0] + left_rect[2], left_rect[1] + left_rect[3]), 
                                color, 2)
                    cv2.putText(processed_frame, left_direction, 
                               (left_rect[0], left_rect[1] - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                
                # Draw right eye only if detected with sufficient confidence
                if right_conf > cnn_threshold:
                    right_points = np.array([[int(landmarks[i].x * w), int(landmarks[i].y * h)] 
                                         for i in [362, 263, 386, 374]], dtype=np.int32)
                    right_rect = cv2.boundingRect(right_points)
                    cv2.rectangle(processed_frame, 
                                (right_rect[0], right_rect[1]), 
                                (right_rect[0] + right_rect[2], right_rect[1] + right_rect[3]), 
                                color, 2)
                    cv2.putText(processed_frame, right_direction, 
                               (right_rect[0], right_rect[1] - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                
                # Add alert status to top-right
                status_text = (
                    "Normal" if self.alert_level == 0 else
                    "SUSPICIOUS" if self.alert_level == 1 else
                    "CHEATING DETECTED"
                )
                cv2.putText(processed_frame, status_text,
                           (w - 300, 30), cv2.FONT_HERSHEY_SIMPLEX,
                           0.7, color, 2)
                
        except Exception as e:
            print(f"Eye tracking error: {str(e)}")
            
        return processed_frame, alerts, alert_counter, frame_counter, left_direction, left_conf, right_direction, right_conf

    def _process_single_eye(self, frame: np.ndarray, landmarks, indices) -> Tuple[str, float]:
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            h, w = gray.shape
            
            eye_points = np.array([
                [int(landmarks[i].x * w), int(landmarks[i].y * h)] 
                for i in indices
            ], dtype=np.int64)
            
            x1, y1 = np.min(eye_points, axis=0)
            x2, y2 = np.max(eye_points, axis=0)
            eye_img = gray[y1:y2, x1:x2]
            
            if eye_img.size == 0:
                return "center", 0.0
                
            processed = cv2.resize(eye_img, (self.IMG_SIZE[1], self.IMG_SIZE[0]))
            processed = processed.reshape((1, *self.IMG_SIZE, 1)).astype(np.float32) / 255.0
            
            prediction = self.eye_model.predict(processed, verbose=0)[0]
            direction = self.CLASS_LABELS[np.argmax(prediction)]
            confidence = float(np.max(prediction))
            
            return direction, confidence
            
        except Exception as e:
            print(f"Error in eye processing: {str(e)}")
            return "center", 0.0

    def process_eye_detections(self, frame, alert_counter, frame_counter, 
                              cnn_threshold=0.6, movement_threshold=0.3, 
                              duration_threshold=5.0, is_video=False):
        """
        Process eye tracking detections and return processed frame, alerts, total detections, and process time.
        
        Args:
            frame: Input frame to process
            alert_counter: Counter for alerts
            frame_counter: Counter for frames
            cnn_threshold: Confidence threshold for eye direction detection
            movement_threshold: Movement threshold for eye tracking
            duration_threshold: Duration threshold for alerts
            is_video: Boolean indicating if the input is a video
            
        Returns:
            tuple: (processed_frame, alerts, total_detections, process_time)
        """
        start_time = time.time()
        total_detections = 0
        
        try:
            # Proses frame menggunakan fungsi yang sudah ada
            processed_frame, alerts, alert_counter, frame_counter, left_direction, left_conf, right_direction, right_conf = self.process_frame(
                frame,
                alert_counter,
                frame_counter,
                cnn_threshold=cnn_threshold,
                movement_threshold=movement_threshold,
                duration_threshold=duration_threshold,
                is_video=is_video
            )

            # Tambah deteksi jika mata terdeteksi (confidence cukup, termasuk arah "center")
            if left_conf > cnn_threshold:
                total_detections += 1
            if right_conf > cnn_threshold:
                total_detections += 1

            # Hitung runtime
            end_time = time.time()
            process_time = round((end_time - start_time), 1)

            # Debugging output
            print(f"Total detections: {total_detections}")
            print(f"Alerts: {alerts}")

            return processed_frame, alerts, total_detections, process_time

        except Exception as e:
            print(f"Eye tracking error: {str(e)}")
            return frame, [], 0, 0.0