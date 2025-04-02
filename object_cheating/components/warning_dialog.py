import reflex as rx
from object_cheating.states.camera_state import CameraState

def warning_dialog() -> rx.Component:
    return rx.dialog.root(
        rx.dialog.content(
            rx.dialog.title(
                "Detection Active",
                class_name="text-red-600 font-bold",
            ),
            rx.dialog.description(
                rx.text(
                    "Please disable detection before switching models. "
                    "This ensures proper cleanup and initialization of the new model.",
                    class_name="text-gray-700 my-4",
                ),
            ),
            rx.hstack(
                rx.dialog.close(
                    rx.button(
                        "Cancel",
                        class_name="bg-gray-200 text-gray-700 px-4 py-2 rounded-lg hover:bg-gray-300",
                    ),
                ),
                spacing="4",
                justify="end",
                padding_top="4",
            ),
        ),
        open=CameraState.show_warning_dialog,
        on_open_change=CameraState.close_warning_dialog,
    )