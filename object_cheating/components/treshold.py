import reflex as rx
from object_cheating.states.threshold_state import ThresholdState
from object_cheating.states.camera_state import CameraState

def threshold() -> rx.Component:
    return rx.box(
        rx.vstack(
            rx.el.h3("Threshold Settings", class_name="text-lg font-semibold mb-2 text-gray-900"),
            # Confidence Threshold Section
            rx.hstack(
                rx.text("Confidence Threshold:", class_name="font-medium text-gray-700"),
                rx.spacer(),
                rx.hstack(
                    rx.input(
                        value=ThresholdState.confidence_threshold,
                        type="number",
                        min=0,
                        max=1,
                        step=0.01,
                        width="70px",
                        height="36px",
                        text_align="center",
                        border="1px solid #e2e8f0",
                        border_radius="md",
                        on_change=ThresholdState.set_confidence_from_str,
                    ),
                    rx.vstack( 
                        rx.icon_button(
                            rx.icon("chevron-up", size=15),
                            on_click=ThresholdState.increment_confidence,
                            border="1px solid #e2e8f0",
                            border_radius="md",
                            height="18px",
                            width="30px",
                            px="1",
                            ml="1",
                        ),
                        rx.icon_button(
                            rx.icon("chevron-down", size=15),
                            on_click=ThresholdState.decrement_confidence,
                            border="1px solid #e2e8f0",
                            border_radius="md",
                            height="18px",
                            width="30px",
                            px="1",
                            ml="1",
                        ),
                        spacing="0",
                    ),
                    align="center",
                ),
                width="100%",
                justify="between",
                align="center",
            ),
            # Second Threshold Section (IoU/Eye Movement)
            rx.hstack(
                rx.text(
                    rx.cond(
                        CameraState.active_model == 3,
                        "Duration Threshold (s):",
                        "IoU Threshold:"
                    ),
                    class_name="font-medium text-gray-700"
                ),
                rx.spacer(),
                rx.hstack(
                    rx.input(
                        value=ThresholdState.duration_threshold,
                        type="number",
                        min=0,
                        max=1,
                        step=0.01,
                        width="70px",
                        height="36px",
                        text_align="center",
                        border="1px solid #e2e8f0",
                        border_radius="md",
                        on_change=lambda value: ThresholdState.set_duration_from_str(value)
                    ),
                    rx.vstack( 
                        rx.icon_button(
                            rx.icon("chevron-up", size=15),
                            on_click=ThresholdState.increment_duration,
                            border="1px solid #e2e8f0",
                            border_radius="md",
                            height="18px",
                            width="30px",
                            px="1",
                            ml="1",
                        ),
                        rx.icon_button(
                            rx.icon("chevron-down", size=15),
                            on_click=ThresholdState.decrement_duration,
                            border="1px solid #e2e8f0",
                            border_radius="md",
                            height="18px",
                            width="30px",
                            px="1",
                            ml="1",
                        ),
                        spacing="0",
                    ),
                    align="center",
                ),
                width="100%",
                justify="between",
                align="center",
                mt="4",
            ),
            class_name="bg-[#ffec99] p-4 rounded-lg shadow-md w-full"
        ),
        width="100%",
        # display=rx.cond(CameraState.detection_enabled, "block", "none"),  # Fixed this line
    )