import reflex as rx
from object_cheating.states.camera_state import CameraState

def stats_panel() -> rx.Component:
    # Define model specific classes
    model1_classes = [
        "All",
        "Bend Over The Desk",
        "Hand Under Table",
        "Look Around",
        "Normal",
        "Stand Up",
        "Wave"
    ]
    
    model2_classes = [
        "All",
        "cheating",
        "normal"
    ]
    
    model3_classes = [
        "All",
        "center",
        "left", 
        "right"
    ]
    
    # Select classes based on active model
    target_classes = rx.cond(
        CameraState.active_model == 1,
        model1_classes,
        rx.cond(
            CameraState.active_model == 2,
            model2_classes,
            model3_classes
        )
    )
    
    return rx.vstack(
        rx.el.h3("Detection Summary", class_name="text-lg font-semibold mb-2 text-gray-900"),
        rx.hstack(
            rx.text(
                f"Total Target: {CameraState.detection_count}", 
                class_name="text-gray-700"
            ),
            rx.text(
                rx.cond(
                    (CameraState.camera_active | CameraState.video_playing),
                    f"FPS: {CameraState.fps}",
                    "FPS: N/A"
                ),
                class_name="text-gray-700"
            ),
            justify="between",
            width="100%",
        ),
        rx.hstack(
            rx.text(
                f"Runtime: {CameraState.processing_time}s",
                class_name="text-gray-700"
            ),
            rx.hstack(
                rx.text("Target Selection: ", class_name="text-gray-700"),
                rx.select(
                    target_classes, 
                    value=CameraState.selected_target,
                    on_change=CameraState.set_selected_target,
                ),
                spacing="2",
            ),
            justify="between",
            width="100%",
        ),
        spacing="4",
        class_name="bg-[#ffec99] p-4 rounded-lg shadow-md w-full"
    )