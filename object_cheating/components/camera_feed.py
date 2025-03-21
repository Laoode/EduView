
import reflex as rx
from object_cheating.states.camera_state import CameraState

def camera_feed() -> rx.Component:
    return rx.el.div(
        rx.el.div(
            rx.cond(
                CameraState.current_frame,
                rx.el.img(
                    src=CameraState.current_frame,
                    class_name="w-full h-full object-contain rounded-lg shadow-lg",
                ),
                rx.el.div(
                    "Camera Off",
                    class_name="w-full h-[480px] bg-gray-900 flex items-center justify-center text-gray-500 text-xl font-medium rounded-lg"
                )
            ),
            class_name="w-full h-[480px] bg-gray-900 rounded-lg overflow-hidden"
        ),
        rx.cond(
            CameraState.error_message,
            rx.el.p(
                CameraState.error_message,
                class_name="text-red-500 mt-2 text-sm"
            )
        ),
        class_name="w-full"
    )

