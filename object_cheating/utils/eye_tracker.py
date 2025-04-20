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
        self.EYE_CLOSURE_THRESHOLD = 0.009 

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
                
                # Log debugging
                print(f"Left eye: direction={left_direction}, confidence={left_conf}")
                print(f"Right eye: direction={right_direction}, confidence={right_conf}")
                
                # Check if either eye is open and has sufficient confidence for gaze detection
                if (left_conf > cnn_threshold and left_direction != "closed") or \
                   (right_conf > cnn_threshold and right_direction != "closed"):
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
                closed_color = (128, 128, 128)  # Gray for closed eyes
                
                # Draw eye boxes and labels
                h, w = frame.shape[:2]
                
                # Draw left eye
                left_points = np.array([[int(landmarks[i].x * w), int(landmarks[i].y * h)] 
                                    for i in [33, 133, 159, 145]], dtype=np.int32)
                left_rect = cv2.boundingRect(left_points)
                if left_direction == "closed":
                    # Draw bounding box for closed eye with gray color
                    cv2.rectangle(processed_frame, 
                                (left_rect[0], left_rect[1]), 
                                (left_rect[0] + left_rect[2], left_rect[1] + left_rect[3]), 
                                closed_color, 2)
                    cv2.putText(processed_frame, "closed", 
                               (left_rect[0], left_rect[1] - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, closed_color, 2)
                elif left_conf > cnn_threshold:
                    # Draw bounding box for open eye with gaze direction
                    cv2.rectangle(processed_frame, 
                                (left_rect[0], left_rect[1]), 
                                (left_rect[0] + left_rect[2], left_rect[1] + left_rect[3]), 
                                color, 2)
                    cv2.putText(processed_frame, left_direction, 
                               (left_rect[0], left_rect[1] - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                
                # Draw right eye
                right_points = np.array([[int(landmarks[i].x * w), int(landmarks[i].y * h)] 
                                     for i in [362, 263, 386, 374]], dtype=np.int32)
                right_rect = cv2.boundingRect(right_points)
                if right_direction == "closed":
                    # Draw bounding box for closed eye with gray color
                    cv2.rectangle(processed_frame, 
                                (right_rect[0], right_rect[1]), 
                                (right_rect[0] + right_rect[2], right_rect[1] + right_rect[3]), 
                                closed_color, 2)
                    cv2.putText(processed_frame, "closed", 
                               (right_rect[0], right_rect[1] - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, closed_color, 2)
                elif right_conf > cnn_threshold:
                    # Draw bounding box for open eye with gaze direction
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
            
            # Get eye points from landmarks
            eye_points = np.array([
                [landmarks[i].x, landmarks[i].y]  # Use normalized coordinates
                for i in indices
            ], dtype=np.float32)
            
            # Check if the eye is closed using the vertical distance between upper and lower eyelid
            # For left eye: indices 159 (upper) and 145 (lower)
            # For right eye: indices 386 (upper) and 374 (lower)
            if indices == [33, 133, 159, 145]:  # Left eye
                upper_idx, lower_idx = 159, 145
            else:  # Right eye
                upper_idx, lower_idx = 386, 374
            
            # Calculate vertical distance between upper and lower eyelid (normalized)
            eye_height = abs(landmarks[upper_idx].y - landmarks[lower_idx].y)
            
            # If the eye height is below the threshold, consider the eye closed
            if eye_height < self.EYE_CLOSURE_THRESHOLD:
                return "closed", 1.0  # Confidence set to 1.0 for rule-based detection
            
            # If the eye is open, proceed with gaze direction detection
            # Convert normalized coordinates to pixel coordinates for cropping
            eye_points_pixel = np.array([
                [int(landmarks[i].x * w), int(landmarks[i].y * h)]
                for i in indices
            ], dtype=np.int64)
            
            x1, y1 = np.min(eye_points_pixel, axis=0)
            x2, y2 = np.max(eye_points_pixel, axis=0)
            
            eye_img = gray[y1:y2, x1:x2]
            
            if eye_img.size == 0:
                return "closed", 0.0  # If eye region is invalid, treat as closed
                
            # Resize eye image for gaze detection
            processed = cv2.resize(eye_img, (self.IMG_SIZE[1], self.IMG_SIZE[0]))
            processed = processed.reshape((1, *self.IMG_SIZE, 1)).astype(np.float32) / 255.0
            
            # Predict gaze direction
            gaze_pred = self.eye_model.predict(processed, verbose=0)[0]
            direction = self.CLASS_LABELS[np.argmax(gaze_pred)]
            confidence = float(np.max(gaze_pred))
            
            return direction, confidence
            
        except Exception as e:
            print(f"Error in eye processing: {str(e)}")
            return "closed", 0.0

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
            tuple: (processed_frame, alerts, total_detections, process_time, highest_class, highest_conf, coords)
        """
        start_time = time.time()
        total_detections = 0
        coords = {"xmin": 0, "ymin": 0, "xmax": 0, "ymax": 0}
        
        try:
            processed_frame, alerts, alert_counter, frame_counter, left_direction, left_conf, right_direction, right_conf = self.process_frame(
                frame,
                alert_counter,
                frame_counter,
                cnn_threshold=cnn_threshold,
                movement_threshold=movement_threshold,
                duration_threshold=duration_threshold,
                is_video=is_video
            )

            if left_conf > cnn_threshold and left_direction != "closed":
                total_detections += 1
            if right_conf > cnn_threshold and right_direction != "closed":
                total_detections += 1
                
            h, w = frame.shape[:2]
            landmarks = self.face_landmarker.detect_for_video(
                mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)),
                int(time.time() * 1000)
            ).face_landmarks[0]

            # Get coordinates for both eyes
            left_points = np.array([[int(landmarks[i].x * w), int(landmarks[i].y * h)] 
                                for i in [33, 133, 159, 145]], dtype=np.int32)
            right_points = np.array([[int(landmarks[i].x * w), int(landmarks[i].y * h)] 
                                for i in [362, 263, 386, 374]], dtype=np.int32)
            
            left_rect = cv2.boundingRect(left_points)
            right_rect = cv2.boundingRect(right_points)
                
            highest_class = "N/A"
            highest_conf = 0.0
            
            # Determine which eye's coordinates to use based on confidence
            if left_direction == "closed" and right_direction == "closed":
                highest_class = "closed"
                highest_conf = 1.0 
                total_detections=2
                # Use combined eye area for closed eyes
                coords["xmin"] = min(left_rect[0], right_rect[0])
                coords["ymin"] = min(left_rect[1], right_rect[1])
                coords["xmax"] = max(left_rect[0] + left_rect[2], right_rect[0] + right_rect[2])
                coords["ymax"] = max(left_rect[1] + left_rect[3], right_rect[1] + right_rect[3])
            else:
                if left_direction != "closed" and right_direction != "closed":
                    # Both eyes open - use coordinates of eye with higher confidence
                    if left_conf >= right_conf:
                        highest_class = left_direction
                        highest_conf = left_conf
                        coords["xmin"] = left_rect[0]
                        coords["ymin"] = left_rect[1]
                        coords["xmax"] = left_rect[0] + left_rect[2]
                        coords["ymax"] = left_rect[1] + left_rect[3]
                    else:
                        highest_class = right_direction
                        highest_conf = right_conf
                        coords["xmin"] = right_rect[0]
                        coords["ymin"] = right_rect[1]
                        coords["xmax"] = right_rect[0] + right_rect[2]
                        coords["ymax"] = right_rect[1] + right_rect[3]
                elif left_direction != "closed":
                    # Only left eye open
                    highest_class = left_direction
                    highest_conf = left_conf
                    coords["xmin"] = left_rect[0]
                    coords["ymin"] = left_rect[1]
                    coords["xmax"] = left_rect[0] + left_rect[2]
                    coords["ymax"] = left_rect[1] + left_rect[3]
                    total_detections += 1
                elif right_direction != "closed":
                    # Only right eye open
                    highest_class = right_direction
                    highest_conf = right_conf
                    coords["xmin"] = right_rect[0]
                    coords["ymin"] = right_rect[1]
                    coords["xmax"] = right_rect[0] + right_rect[2]
                    coords["ymax"] = right_rect[1] + right_rect[3]
                    total_detections += 1

            # Runtime
            end_time = time.time()
            process_time = round((end_time - start_time), 1)

            # Debugging output
            print(f"Total detections: {total_detections}")
            print(f"Alerts: {alerts}")
            print(f"Highest class: {highest_class}, Highest confidence: {highest_conf}")

            highest_conf = round(highest_conf * 100)

            return processed_frame, alerts, total_detections, process_time, highest_class, highest_conf, coords

        except Exception as e:
            print(f"Eye tracking error: {str(e)}")
            return frame, [], 0, 0.0, "N/A", 0, coords