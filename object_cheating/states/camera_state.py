
import reflex as rx
from typing import TypedDict, List
import cv2
import base64
import numpy as np
import asyncio
import os
from object_cheating.utils.eye_tracker import EyeTracker

class DetectionResult(TypedDict):
    id: int
    x: int
    y: int
    width: int
    height: int

class CameraState(rx.State):
    # Model state
    active_model: int = 3
    detection_enabled: bool = False
    eye_alerts: list[str] = []
    
    # Eye tracking state
    eye_alert_counter: int = 0
    eye_frame_counter: int = 0
    
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

    @rx.event
    def set_active_model(self, model_num: int):
        self.active_model = model_num
        
    @rx.event
    def toggle_detection(self, enabled: bool):
        self.detection_enabled = enabled
        self.eye_alerts = []
        self.eye_alert_counter = 0
        self.eye_frame_counter = 0
    
    def get_face_cascade(self) -> cv2.CascadeClassifier:
        return cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    @rx.event
    def toggle_camera(self):
        self.camera_active = not self.camera_active
        if self.camera_active:
            return CameraState.process_camera_feed
        else:
            self.current_frame = ""

    @rx.event
    async def handle_image_upload(self, files: list[rx.UploadFile]):
        """Handle image upload from local computer."""
        try:
            if not files or len(files) == 0:
                return

            file = files[0]
            upload_data = await file.read()
            
            # Convert to base64 for display
            img_base64 = base64.b64encode(upload_data).decode('utf-8')
            
            # Get content type from file or default to jpeg
            content_type = file.content_type or "image/jpeg"
            
            # Update state
            self.uploaded_image = f"data:{content_type};base64,{img_base64}"
            self.current_frame = self.uploaded_image
            # Ensure camera is not active when displaying uploaded image
            self.camera_active = False
            
        except Exception as e:
            self.error_message = f"Upload error: {str(e)}"

    @rx.event
    async def handle_video_upload(self, files: list[rx.UploadFile]):
        """Handle video upload."""
        try:
            if not files or len(files) == 0:
                return

            file = files[0]
            upload_data = await file.read()
            
            # Save video to temporary file
            self.video_path = os.path.join(rx.get_upload_dir(), file.filename)
            with open(self.video_path, "wb") as f:
                f.write(upload_data)
            
            # Stop other media sources
            self.camera_active = False
            self.uploaded_image = ""
            self.current_frame = ""
            self.video_playing = True
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

            async with self:
                self.processing_active = True
                self.error_message = ""

            while self.video_playing and cap.isOpened():
                ret, frame = cap.read()
                if not ret:  # Jika video selesai
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Reset ke awal
                    continue

                # Convert frame to base64
                _, buffer = cv2.imencode('.jpg', frame)
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
        self.video_playing = False  # Tambahkan ini
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

                # Initialize eye tracker only when detection is enabled
                if self.detection_enabled and self.active_model == 3:
                    if eye_tracker is None:
                        eye_tracker = EyeTracker()
                    
                    try:
                        # Process frame with local variables
                        processed_frame, alerts, local_eye_alert_counter, local_eye_frame_counter = eye_tracker.process_frame(
                            processed_frame,
                            local_eye_alert_counter,
                            local_eye_frame_counter
                        )
                        
                        # Update state within context manager
                        if alerts:
                            async with self:
                                self.eye_alerts = alerts
                                self.eye_alert_counter = local_eye_alert_counter
                                self.eye_frame_counter = local_eye_frame_counter
                    except Exception as e:
                        print(f"Eye tracking error: {str(e)}")

                # Face detection processing
                if self.face_detection_active:
                    try:
                        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                        faces = self.get_face_cascade().detectMultiScale(
                            gray,
                            scaleFactor=self.scale_factor,
                            minNeighbors=self.min_neighbors,
                            minSize=(30, 30)
                        )
                        
                        detection_results = []
                        for i, (x, y, w, h) in enumerate(faces):
                            cv2.rectangle(processed_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                            detection_results.append({
                                "id": i,
                                "x": int(x),
                                "y": int(y),
                                "width": int(w),
                                "height": int(h)
                            })
                        
                        async with self:
                            self.detection_results = detection_results
                            self.face_count = len(faces)
                    except Exception as e:
                        print(f"Face detection error: {str(e)}")

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