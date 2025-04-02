import reflex as rx
from object_cheating.states.camera_state import CameraState

def model_navigation() -> rx.Component:
    """Komponen navigasi model."""
    return rx.hstack(
        rx.icon_button(
            rx.icon("chevron-left"),
            on_click=lambda: CameraState.try_change_model(CameraState.active_model - 1),
            variant="surface",
            height="30px",
            width="30px",
            disabled=CameraState.active_model == 1,  # Dinonaktifkan saat di Model 1
        ),
        rx.badge(
            rx.center(
                rx.text(f"Model {CameraState.active_model}"),
                width="100%",
                height="28px",
            ),
            variant="surface",
            min_width="100px",
            text_align="center",
        ),
        rx.icon_button(
            rx.icon("chevron-right"),
            on_click=lambda: CameraState.try_change_model(CameraState.active_model + 1),
            variant="surface",
            height="30px",
            width="30px",
            disabled=CameraState.active_model == 3,  # Dinonaktifkan saat di Model 3
        ),
        spacing="2",
        align="center",
    )

def controls() -> rx.Component:
    """Controls component for detection and model selection."""
    # Check if any media input is active
    media_active = rx.cond(
        (CameraState.camera_active | 
         (CameraState.current_frame != "") | 
         CameraState.video_playing),
        True,
        False
    )
    return rx.hstack(
        rx.hstack(
            rx.text("Enable Detection", class_name="text-gray-700"),
            rx.switch(
                checked=CameraState.detection_enabled,
                on_change=CameraState.toggle_detection,
                color_scheme="grass",
                variant="surface",
                disabled=~media_active,
                transition="all 0.2s ease-in-out",
            ),
            spacing="2",
            align="center",# Adds spacing between text and switch
        ),
        model_navigation(),
        spacing="2",
        class_name="flex justify-between"
    )