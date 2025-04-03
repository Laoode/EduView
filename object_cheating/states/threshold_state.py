import reflex as rx

class ThresholdState(rx.State):
    confidence_threshold: float = 0.25
    iou_threshold: float = 0.70
    
    def increment_confidence(self):
        if self.confidence_threshold < 1.0:
            self.confidence_threshold += 0.01
            self.confidence_threshold = round(self.confidence_threshold, 2)

    def decrement_confidence(self):
        if self.confidence_threshold > 0.0:
            self.confidence_threshold -= 0.01
            self.confidence_threshold = round(self.confidence_threshold, 2)

    def increment_iou(self):
        if self.iou_threshold < 1.0:
            self.iou_threshold += 0.01
            self.iou_threshold = round(self.iou_threshold, 2)

    def decrement_iou(self):
        if self.iou_threshold > 0.0:
            self.iou_threshold -= 0.01
            self.iou_threshold = round(self.iou_threshold, 2)

    def set_confidence_from_str(self, value: str):
        try:
            self.confidence_threshold = float(value)
        except ValueError:
            print("Invalid input for confidence threshold")

    def set_iou_from_str(self, value: str):
        try:
            self.iou_threshold = float(value)
        except ValueError:
            print("Invalid input for IoU threshold")
            
    def set_model_defaults(self, model_number: int):
        """Set default threshold values based on model number"""
        if model_number == 3:  # Eye tracking model
            self.confidence_threshold = 0.6
            self.iou_threshold = 0.3
        else:  # YOLO models
            self.confidence_threshold = 0.25
            self.iou_threshold = 0.70