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
            # Second Threshold Section (IoU for Model 1 & 2, Duration for Model 3)
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
                        value=rx.cond(
                            CameraState.active_model == 3,
                            ThresholdState.duration_threshold,
                            ThresholdState.iou_threshold
                        ),
                        type="number",
                        min=rx.cond(CameraState.active_model == 3, 1, 0),  # Min: 1 for duration, 0 for IoU
                        max=rx.cond(CameraState.active_model == 3, 10, 1),  # Max: 10 for duration, 1 for IoU
                        step=rx.cond(CameraState.active_model == 3, 0.1, 0.01),  # Step: 0.1 for duration, 0.01 for IoU
                        width="70px",
                        height="36px",
                        text_align="center",
                        border="1px solid #e2e8f0",
                        border_radius="md",
                        on_change=lambda value: ThresholdState.set_second_threshold_from_str(value, CameraState.active_model),
                    ),
                    rx.vstack( 
                        rx.icon_button(
                            rx.icon("chevron-up", size=15),
                            on_click=lambda: ThresholdState.increment_second_threshold(CameraState.active_model),
                            border="1px solid #e2e8f0",
                            border_radius="md",
                            height="18px",
                            width="30px",
                            px="1",
                            ml="1",
                        ),
                        rx.icon_button(
                            rx.icon("chevron-down", size=15),
                            on_click=lambda: ThresholdState.decrement_second_threshold(CameraState.active_model),
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
    )