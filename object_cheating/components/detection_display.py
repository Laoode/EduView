
import reflex as rx
from object_cheating.states.camera_state import CameraState

def detection_display() -> rx.Component:
    return rx.el.div(
        rx.el.div(
            rx.el.h2(
                "Detection Results",
                class_name="text-lg font-semibold text-gray-800 mb-4"
            ),
            rx.el.div(
                rx.cond(
                    CameraState.detection_results.length() > 0,
                    rx.foreach(
                        CameraState.detection_results,
                        lambda result: rx.el.div(
                            rx.el.div(
                                rx.el.span(
                                    result["label"],
                                    class_name="font-medium text-purple-600"
                                ),
                                rx.el.span(
                                    f" ({result['confidence']:.2f})",
                                    class_name="text-gray-500"
                                ),
                                class_name="flex items-center justify-between"
                            ),
                            class_name="p-3 bg-white rounded-lg shadow-sm border border-gray-100"
                        )
                    ),
                    rx.el.p(
                        "No detections",
                        class_name="text-gray-500 text-sm"
                    )
                ),
                class_name="space-y-2"
            ),
            class_name="bg-gray-50 p-4 rounded-lg"
        ),
        class_name="w-full"
    )