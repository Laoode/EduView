
import reflex as rx
from object_cheating.states.camera_state import CameraState

def controls() -> rx.Component:
    return rx.el.div(
        rx.el.div(
            # Camera toggle button
            rx.el.button(
                rx.cond(
                    CameraState.camera_active,
                    "Stop Camera",
                    "Start Camera"
                ),
                on_click=CameraState.toggle_camera,
                class_name=rx.cond(
                    CameraState.camera_active,
                    "bg-red-500 hover:bg-red-600",
                    "bg-green-500 hover:bg-green-600"
                ) + " text-white px-6 py-2 rounded-lg transition-colors duration-200 font-medium mr-4"
            ),
            # Face detection toggle
            rx.el.button(
                rx.cond(
                    CameraState.face_detection_active,
                    "Disable Face Detection",
                    "Enable Face Detection"
                ),
                on_click=CameraState.toggle_face_detection,
                class_name=rx.cond(
                    CameraState.face_detection_active,
                    "bg-purple-500 hover:bg-purple-600",
                    "bg-blue-500 hover:bg-blue-600"
                ) + " text-white px-6 py-2 rounded-lg transition-colors duration-200 font-medium"
            ),
            class_name="flex space-x-4"
        ),
        rx.cond(
            CameraState.face_detection_active,
            rx.el.div(
                rx.el.div(
                    rx.el.label(
                        "Min Neighbors:",
                        class_name="text-sm text-gray-600 mr-2"
                    ),
                    rx.el.input(
                        type="range",
                        min=1,
                        max=10,
                        value=CameraState.min_neighbors,
                        on_change=CameraState.update_min_neighbors,
                        class_name="w-48"
                    ),
                    rx.el.span(
                        CameraState.min_neighbors,
                        class_name="text-sm text-gray-600 ml-2"
                    ),
                    class_name="flex items-center mb-2"
                ),
                rx.el.div(
                    rx.el.label(
                        "Scale Factor:",
                        class_name="text-sm text-gray-600 mr-2"
                    ),
                    rx.el.input(
                        type="range",
                        min=11,
                        max=20,
                        value=CameraState.scale_factor * 10,
                        on_change=CameraState.update_scale_factor,
                        class_name="w-48"
                    ),
                    rx.el.span(
                        CameraState.scale_factor,
                        class_name="text-sm text-gray-600 ml-2"
                    ),
                    class_name="flex items-center"
                ),
                class_name="mt-4"
            )
        ),
        rx.el.div(
            rx.el.p(
                "FPS: ",
                CameraState.fps,
                class_name="text-sm text-gray-600"
            ),
            rx.cond(
                CameraState.face_detection_active,
                rx.el.p(
                    "Faces Detected: ",
                    CameraState.face_count,
                    class_name="text-sm text-gray-600"
                )
            ),
            class_name="mt-4"
        ),
        class_name="p-4 bg-white rounded-lg shadow-md space-y-4"
    )

