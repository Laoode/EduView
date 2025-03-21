import reflex as rx
from object_cheating.states.camera_state import CameraState

def input_panel() -> rx.Component:
    button_style = "bg-[#ffa94d] text-black px-6 py-2 w-full text-center rounded-lg shadow-md"
    disabled_style = "bg-gray-400 text-gray-600 px-6 py-2 w-full text-center rounded-lg cursor-not-allowed"

    return rx.el.div(
        rx.el.div(
            rx.el.h3("Media Controls", class_name="text-lg font-semibold mb-2 text-gray-900"),
            # First row (Image - Video)
            rx.el.div(
                rx.button("📷 Image", class_name=button_style),
                rx.button("🎬 Video", class_name=button_style),
                class_name="grid grid-cols-2 gap-4 w-full"
            ),

            # Second row (Webcam - Save)
            rx.el.div(
                rx.button(
                    "🎥 Webcam",
                    on_click=CameraState.toggle_camera,
                    class_name=rx.cond(
                        CameraState.camera_active,
                        disabled_style,
                        f"{button_style} hover:bg-[#ff922b]"
                    ),
                    is_disabled=CameraState.camera_active
                ),
                rx.button("💾 Save", class_name=button_style),
                class_name="grid grid-cols-2 gap-4 w-full mt-4"
            ),

            # Third row (Clear, centered)
            rx.el.div(
                rx.button(
                    "🗑️ Clear",
                    on_click=CameraState.clear_camera,
                    class_name=rx.cond(
                        CameraState.camera_active,
                        f"{button_style} hover:bg-[#ff922b]",
                        disabled_style
                    ),
                    is_disabled=~CameraState.camera_active
                ),
                class_name="flex justify-center w-full mt-4"
            ),

            class_name="w-full p-6 bg-[#ffec99] rounded-lg shadow-md max-w-md mx-auto"
        )
    )