import reflex as rx
from typing import TypedDict, List
import cv2
import base64
import numpy as np
import asyncio
import os
from object_cheating.utils.eye_tracker import EyeTracker
from ultralytics import YOLO

class DetectionResult(TypedDict):
    id: int
    x: int
    y: int
    width: int
    height: int

class CameraState(rx.State):
    # Model state
    active_model: int = 1  # Contoh definisi state variable

    @rx.event
    def prev_model(self):
        if self.active_model > 1:
            self.active_model -= 1

    @rx.event
    def next_model(self):
        if self.active_model < 3:  # Ganti 3 dengan jumlah maksimum model Anda
            self.active_model += 1
            
    detection_enabled: bool = False
    eye_alerts: list[str] = []
    
    # Eye tracking state
    eye_alert_counter: int = 0
    eye_frame_counter: int = 0
    
    _original_frame_bytes: bytes = b""
    
    # Stream state
    camera_active: bool = False
    processing_active: bool = False
    current_frame: str = ""  # Base64 encoded image
    error_message: str = ""
    
    # Tambahkan state untuk upload gambar
    uploaded_image: str = ""  # Untuk menyimpan gambar yang diupload
    
    video_playing: bool = False
    video_path: str = ""
    # Face detection state
    face_detection_active: bool = False
    detection_results: List[DetectionResult] = []
    min_neighbors: int = 5
    scale_factor: float = 1.3
    
    # Performance metrics
    fps: float = 0.0
    frame_count: int = 0
    last_frame_time: float = 0.0
    face_count: int = 0
    
    # Model YOLO
    _yolo_model = None
    
    # Add new YOLO model for Model 2
    _yolo_model_2 = None
    
    @classmethod
    def get_yolo_model(cls):
        """Get or initialize YOLO model"""
        if cls._yolo_model is None:
            cls._yolo_model = YOLO("object_cheating/models/modelv8.pt")
        return cls._yolo_model
    
    @classmethod
    def get_yolo_model_2(cls):
        """Get or initialize YOLO model 2 for cheating detection"""
        if cls._yolo_model_2 is None:
            cls._yolo_model_2 = YOLO("object_cheating/models/modelv8-2.pt")
        return cls._yolo_model_2
    
    @classmethod
    def get_class_color(cls, class_name: str) -> tuple:
        """Get color for each class in Model 1"""
        color_map = {
            "Normal": (0, 255, 128),        # Green
            "Bend Over The Desk": (255, 255, 0),    # Aqua
            "Hand Under Table": (255, 105, 65),      # Royal Blue
            "Look Around": (238, 130, 238),         # Violet
            "Stand Up": (250, 230, 230),           # Lavender
            "Wave": (193, 182, 255)                # Light Pink
        }
        return color_map.get(class_name, (0, 255, 128))  # Default to green if class not found
    
    def __init__(self, *args, **kwargs):
        """Initialize state with parent initialization."""
        super().__init__(*args, **kwargs)
        
    @rx.event
    def set_active_model(self, model_num: int):
        if 1 <= model_num <= 3:
            self.active_model = model_num
        else:
            print(f"Nomor model tidak valid: {model_num}. Harus antara 1 dan 3.")
        
    
    def get_face_cascade(self) -> cv2.CascadeClassifier:
        return cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    @rx.event
    def toggle_camera(self):
        self.camera_active = not self.camera_active
        if self.camera_active:
            return CameraState.process_camera_feed
        else:
            self.current_frame = ""
            
    @property
    def original_frame(self) -> np.ndarray:
        """Convert bytes back to numpy array when needed"""
        if not self._original_frame_bytes:
            return None
        nparr = np.frombuffer(self._original_frame_bytes, np.uint8)
        return cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    def set_original_frame(self, frame: np.ndarray):
        """Convert numpy array to bytes for storage"""
        if frame is None:
            self._original_frame_bytes = b""
        else:
            _, buffer = cv2.imencode('.jpg', frame)
            self._original_frame_bytes = buffer.tobytes()

    @rx.event
    async def handle_image_upload(self, files: list[rx.UploadFile]):
        """Handle image upload from local computer."""
        try:
            if not files or len(files) == 0:
                return

            file = files[0]
            upload_data = await file.read()
            
            # Convert image bytes to numpy array
            nparr = np.frombuffer(upload_data, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            # Store original frame using the new method
            self.set_original_frame(frame)
            
            # Convert to base64 for display
            img_base64 = base64.b64encode(upload_data).decode('utf-8')
            content_type = file.content_type or "image/jpeg"
            
            # Update state
            self.uploaded_image = f"data:{content_type};base64,{img_base64}"
            self.current_frame = self.uploaded_image
            self.camera_active = False
            
            # Proses gambar jika deteksi aktif
            if self.detection_enabled:
                return CameraState.process_uploaded_image
            
        except Exception as e:
            self.error_message = f"Upload error: {str(e)}"
                
    @rx.event
    async def toggle_detection(self, enabled: bool):
        """Toggle detection and process uploaded image if exists"""
        # Set state without async with
        self.detection_enabled = enabled
        self.eye_alerts = []
        self.eye_alert_counter = 0
        self.eye_frame_counter = 0
        
        if enabled and self._original_frame_bytes:
            # If enabled, process image with detection
            return CameraState.process_uploaded_image
        elif not enabled and self.uploaded_image:
            # If disabled, restore original uploaded image
            self.current_frame = self.uploaded_image

    @rx.event(background=True)
    async def process_uploaded_image(self):
        """Process uploaded image with selected model detection"""
        try:
            frame = self.original_frame
            if frame is None:
                return
            
            processed_frame = frame.copy()

            if self.detection_enabled:
                if self.active_model == 1:
                    # Model 1: YOLOv8 for classroom behavior
                    yolo_model = self.get_yolo_model()
                    results = yolo_model(processed_frame)
                    for result in results:
                        boxes = result.boxes
                        for box in boxes:
                            x1, y1, x2, y2 = box.xyxy[0]
                            conf = box.conf[0]
                            cls = box.cls[0]
                            class_name = yolo_model.names[int(cls)]
                            label = f"{class_name} {conf:.2f}"
                            color = self.get_class_color(class_name)
                            cv2.rectangle(processed_frame, (int(x1), int(y1)), 
                                    (int(x2), int(y2)), color, 2)
                            cv2.putText(processed_frame, label, (int(x1), int(y1)-10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                
                elif self.active_model == 2:
                    # Model 2: YOLOv8 for cheating detection
                    yolo_model = self.get_yolo_model_2()
                    results = yolo_model(processed_frame)
                    for result in results:
                        boxes = result.boxes
                        for box in boxes:
                            x1, y1, x2, y2 = box.xyxy[0]
                            conf = box.conf[0]
                            cls = box.cls[0]
                            label = f"{yolo_model.names[int(cls)]} {conf:.2f}"
                            # Use red color for cheating detection
                            color = (0, 0, 255) if yolo_model.names[int(cls)] == "cheating" else (0, 255, 0)
                            cv2.rectangle(processed_frame, (int(x1), int(y1)), 
                                        (int(x2), int(y2)), color, 2)
                            cv2.putText(processed_frame, label, (int(x1), int(y1)-10), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                
                elif self.active_model == 3:
                    # Model 3: Eye tracking (existing code)
                    eye_tracker = EyeTracker()
                    try:
                        processed_frame, alerts, _, _ = eye_tracker.process_frame(
                            processed_frame,
                            0,
                            0
                        )
                        if alerts:
                            async with self:
                                self.eye_alerts = alerts
                    except Exception as e:
                        print(f"Eye tracking error: {str(e)}")
            
            # Convert processed frame to base64
            _, buffer = cv2.imencode('.jpg', processed_frame)
            img_base64 = base64.b64encode(buffer).decode('utf-8')
            
            # Update display
            async with self:
                self.current_frame = f"data:image/jpeg;base64,{img_base64}"
        
        except Exception as e:
            print(f"Image processing error: {str(e)}")
            async with self:
                self.error_message = f"Image processing error: {str(e)}"

    @rx.event
    async def handle_video_upload(self, files: list[rx.UploadFile]):
        """Handle video upload."""
        try:
            if not files or len(files) == 0:
                return

            file = files[0]
            upload_data = await file.read()
            
            # Save video to temporary file
            self.video_path = os.path.join(rx.get_upload_dir(), file.name)  # Use .name instead of .filename
            with open(self.video_path, "wb") as f:
                f.write(upload_data)
            
            # Stop other media sources and start video processing
            self.camera_active = False
            self.uploaded_image = ""
            self.current_frame = ""
            self.video_playing = True
            self.detection_enabled = False  # Reset detection state
            self.eye_alerts = []  # Clear any existing alerts
            
            return CameraState.process_video_frames
            
        except Exception as e:
            self.error_message = f"Video upload error: {str(e)}"

    @rx.event(background=True)
    async def process_video_frames(self):
        """Process and display video frames."""
        try:
            cap = cv2.VideoCapture(self.video_path)
            if not cap.isOpened():
                async with self:
                    self.error_message = "Failed to open video file"
                    self.video_playing = False
                return

            # Initialize trackers and models
            eye_tracker = None
            yolo_model = None
            yolo_model_2 = None
            frame_counter = 0
            local_eye_alert_counter = 0
            local_eye_frame_counter = 0

            async with self:
                self.processing_active = True
                self.error_message = ""

            while self.video_playing and cap.isOpened():
                ret, frame = cap.read()
                if not ret:  # Reset video when it ends
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue

                processed_frame = frame.copy()

                # Process detections if enabled
                if self.detection_enabled:
                    if self.active_model == 1:
                        # Model 1: YOLOv8 for classroom behavior
                        if yolo_model is None:
                            yolo_model = self.get_yolo_model()
                        
                        try:
                            results = yolo_model(processed_frame)
                            for result in results:
                                boxes = result.boxes
                                for box in boxes:
                                    x1, y1, x2, y2 = box.xyxy[0]
                                    conf = box.conf[0]
                                    cls = box.cls[0]
                                    class_name = yolo_model.names[int(cls)]
                                    label = f"{class_name} {conf:.2f}"
                                    color = self.get_class_color(class_name)
                                    cv2.rectangle(processed_frame, (int(x1), int(y1)), 
                                            (int(x2), int(y2)), color, 2)
                                    cv2.putText(processed_frame, label, (int(x1), int(y1)-10),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                        except Exception as e:
                            print(f"YOLO detection error: {str(e)}")

                    elif self.active_model == 2:
                        # Model 2: YOLOv8 for cheating detection
                        if yolo_model_2 is None:
                            yolo_model_2 = self.get_yolo_model_2()
                        
                        try:
                            results = yolo_model_2(processed_frame)
                            for result in results:
                                boxes = result.boxes
                                for box in boxes:
                                    x1, y1, x2, y2 = box.xyxy[0]
                                    conf = box.conf[0]
                                    cls = box.cls[0]
                                    label = f"{yolo_model_2.names[int(cls)]} {conf:.2f}"
                                    # Use red color for cheating detection
                                    color = (0, 0, 255) if yolo_model_2.names[int(cls)] == "cheating" else (0, 255, 0)
                                    cv2.rectangle(processed_frame, (int(x1), int(y1)), 
                                            (int(x2), int(y2)), color, 2)
                                    cv2.putText(processed_frame, label, (int(x1), int(y1)-10),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                        except Exception as e:
                            print(f"YOLO Model 2 detection error: {str(e)}")

                    elif self.active_model == 3:
                        # Model 3: Eye tracking
                        if eye_tracker is None:
                            eye_tracker = EyeTracker()
                        
                        try:
                            processed_frame, alerts, local_eye_alert_counter, local_eye_frame_counter = eye_tracker.process_frame(
                                processed_frame,
                                local_eye_alert_counter,
                                local_eye_frame_counter
                            )
                            
                            if alerts:
                                async with self:
                                    self.eye_alerts = alerts
                                    self.eye_alert_counter = local_eye_alert_counter
                                    self.eye_frame_counter = local_eye_frame_counter
                        except Exception as e:
                            print(f"Eye tracking error in video: {str(e)}")

                # Convert frame to base64
                _, buffer = cv2.imencode('.jpg', processed_frame)
                img_base64 = base64.b64encode(buffer).decode('utf-8')
                
                async with self:
                    self.current_frame = f"data:image/jpeg;base64,{img_base64}"
                
                await asyncio.sleep(1/30)  # ~30 fps

            cap.release()
            
        except Exception as e:
            async with self:
                self.error_message = f"Video processing error: {str(e)}"
            
        finally:
            async with self:
                self.processing_active = False
                self.video_playing = False
                self.current_frame = ""

    @rx.event
    async def clear_camera(self):
        """Clear the camera state and stop the camera if it's running."""
        self.camera_active = False
        self.video_playing = False 
        self.current_frame = ""
        self.uploaded_image = ""
        self.detection_results = []
        self.face_count = 0
        self.error_message = ""

    @rx.event
    def toggle_face_detection(self):
        self.face_detection_active = not self.face_detection_active
        
    @rx.event
    def update_min_neighbors(self, value: str):
        self.min_neighbors = int(value)
        
    @rx.event
    def update_scale_factor(self, value: str):
        self.scale_factor = float(value) / 10

    @rx.event(background=True)
    async def process_camera_feed(self):
        try:
            # Initialize camera
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                async with self:
                    self.error_message = "Failed to open camera"
                    self.camera_active = False
                return

            # Initialize variables outside the loop
            eye_tracker = None
            yolo_model = None
            yolo_model_2 = None
            frame_counter = 0
            local_eye_alert_counter = 0
            local_eye_frame_counter = 0

            async with self:
                self.processing_active = True
                self.error_message = ""

            while self.camera_active:
                ret, frame = cap.read()
                if not ret:
                    break
                
                processed_frame = frame.copy()

                # Process detections if enabled
                if self.detection_enabled:
                    if self.active_model == 1:
                        # Model 1: YOLOv8 for classroom behavior
                        if yolo_model is None:
                            yolo_model = self.get_yolo_model()
                        
                        try:
                            results = yolo_model(processed_frame)
                            for result in results:
                                boxes = result.boxes
                                for box in boxes:
                                    x1, y1, x2, y2 = box.xyxy[0]
                                    conf = box.conf[0]
                                    cls = box.cls[0]
                                    class_name = yolo_model.names[int(cls)]
                                    label = f"{class_name} {conf:.2f}"
                                    color = self.get_class_color(class_name)
                                    cv2.rectangle(processed_frame, (int(x1), int(y1)), 
                                            (int(x2), int(y2)), color, 2)
                                    cv2.putText(processed_frame, label, (int(x1), int(y1)-10),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                        except Exception as e:
                            print(f"YOLO detection error: {str(e)}")

                    elif self.active_model == 2:
                        # Model 2: YOLOv8 for cheating detection
                        if yolo_model_2 is None:
                            yolo_model_2 = self.get_yolo_model_2()
                        
                        try:
                            results = yolo_model_2(processed_frame)
                            for result in results:
                                boxes = result.boxes
                                for box in boxes:
                                    x1, y1, x2, y2 = box.xyxy[0]
                                    conf = box.conf[0]
                                    cls = box.cls[0]
                                    label = f"{yolo_model_2.names[int(cls)]} {conf:.2f}"
                                    # Use red color for cheating detection
                                    color = (0, 0, 255) if yolo_model_2.names[int(cls)] == "cheating" else (0, 255, 0)
                                    cv2.rectangle(processed_frame, (int(x1), int(y1)), 
                                            (int(x2), int(y2)), color, 2)
                                    cv2.putText(processed_frame, label, (int(x1), int(y1)-10),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                        except Exception as e:
                            print(f"YOLO Model 2 detection error: {str(e)}")

                    elif self.active_model == 3:
                        # Model 3: Eye tracking
                        if eye_tracker is None:
                            eye_tracker = EyeTracker()
                        
                        try:
                            processed_frame, alerts, local_eye_alert_counter, local_eye_frame_counter = eye_tracker.process_frame(
                                processed_frame,
                                local_eye_alert_counter,
                                local_eye_frame_counter
                            )
                            
                            if alerts:
                                async with self:
                                    self.eye_alerts = alerts
                                    self.eye_alert_counter = local_eye_alert_counter
                                    self.eye_frame_counter = local_eye_frame_counter
                        except Exception as e:
                            print(f"Eye tracking error: {str(e)}")

                # Convert and display frame
                _, buffer = cv2.imencode('.jpg', processed_frame)
                img_base64 = base64.b64encode(buffer).decode('utf-8')
                
                async with self:
                    self.current_frame = f"data:image/jpeg;base64,{img_base64}"
                    self.frame_count += 1
                
                await asyncio.sleep(1/30)
                
        except Exception as e:
            async with self:
                self.error_message = f"Camera error: {str(e)}"
                self.camera_active = False
                self.processing_active = False
        
        finally:
            if 'cap' in locals():
                cap.release()
            async with self:
                self.processing_active = False
                self.detection_results = []