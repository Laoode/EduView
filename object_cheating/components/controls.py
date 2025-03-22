import reflex as rx
from object_cheating.states.camera_state import CameraState

def model_navigation() -> rx.Component:
    """Komponen navigasi model."""
    return rx.hstack(
        rx.icon_button(
            rx.icon("chevron-left"),
            on_click=CameraState.set_active_model((CameraState.active_model - 2) % 3 + 1),
            variant="surface",
            height="30px",
            width="30px",
        ),
        rx.badge(
            rx.center(
                rx.text(f"Model {CameraState.active_model}"),
                width="100%",
                height="28px",
            ),
            variant="surface",  # Light gray background; adjust if needed
            min_width="100px",
            text_align="center",
        ),
        rx.icon_button(
            rx.icon("chevron-right"),
            on_click=CameraState.set_active_model(CameraState.active_model % 3 + 1),
            variant="surface",
            height="30px",
            width="30px",
        ),
        spacing="2",
        align="center",
    )

def controls() -> rx.Component:
    return rx.hstack(
        rx.hstack(
            rx.text("Enable Detection", class_name="text-gray-700"),
            rx.switch(
                is_checked=CameraState.detection_enabled,
                on_change=CameraState.toggle_detection,
                color_scheme="grass",
                variant="surface",
            ),
            spacing="2",
            align="center",# Adds spacing between text and switch
        ),
        model_navigation(),
        spacing="2",
        class_name="flex justify-between"
    )