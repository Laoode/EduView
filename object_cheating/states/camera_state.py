
import reflex as rx
from typing import TypedDict, List
import cv2
import base64
import numpy as np
import asyncio
import os

class DetectionResult(TypedDict):
    id: int
    x: int
    y: int
    width: int
    height: int

class CameraState(rx.State):
    # Stream state
    camera_active: bool = False
    processing_active: bool = False
    current_frame: str = ""  # Base64 encoded image
    error_message: str = ""
    
    # Tambahkan state untuk upload gambar
    uploaded_image: str = ""  # Untuk menyimpan gambar yang diupload
    
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
    async def clear_camera(self):
        """Clear the camera state and stop the camera if it's running."""
        self.camera_active = False
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
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                async with self:
                    self.error_message = "Failed to open camera"
                    self.camera_active = False
                return

            async with self:
                self.processing_active = True
                self.error_message = ""

            while self.camera_active:
                ret, frame = cap.read()
                if not ret:
                    break

                # Face detection processing
                if self.face_detection_active:
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    faces = self.get_face_cascade().detectMultiScale(
                        gray,
                        scaleFactor=self.scale_factor,
                        minNeighbors=self.min_neighbors,
                        minSize=(30, 30)
                    )
                    
                    # Draw rectangles around faces
                    detection_results = []
                    for i, (x, y, w, h) in enumerate(faces):
                        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
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

                # Convert frame to base64 for display
                _, buffer = cv2.imencode('.jpg', frame)
                img_base64 = base64.b64encode(buffer).decode('utf-8')
                
                async with self:
                    self.current_frame = f"data:image/jpeg;base64,{img_base64}"
                    self.frame_count += 1
                
                await asyncio.sleep(1/30)  # Limit to ~30 fps
                
            cap.release()
            
        except Exception as e:
            async with self:
                self.error_message = f"Camera error: {str(e)}"
                self.camera_active = False
                self.processing_active = False
        
        finally:
            async with self:
                self.processing_active = False
                self.detection_results = []