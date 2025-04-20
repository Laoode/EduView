import reflex as rx
from object_cheating.states.camera_state import CameraState

def coordinate_panel() -> rx.Component:
    return rx.el.div(
        rx.el.h3("Bounding Box Coordinates", class_name="text-lg font-semibold mb-2 text-gray-900"),
        rx.el.div(
            rx.el.div(
                rx.el.span(
                    rx.cond(
                        CameraState.detection_enabled & (CameraState.detection_count > 0),
                        f"xmin: {CameraState.highest_conf_xmin}",
                        "xmin: N/A"
                    ), 
                    class_name="text-gray-700"
                ),
                rx.el.span(
                    rx.cond(
                        CameraState.detection_enabled & (CameraState.detection_count > 0),
                        f"ymin: {CameraState.highest_conf_ymin}",
                        "ymin: N/A"
                    ), 
                    class_name="text-gray-700"
                ),
                class_name="flex justify-between"
            ),
            rx.el.div(
                rx.el.span(
                    rx.cond(
                        CameraState.detection_enabled & (CameraState.detection_count > 0),
                        f"xmax: {CameraState.highest_conf_xmax}",
                        "xmax: N/A"
                    ), 
                    class_name="text-gray-700"
                ),
                rx.el.span(
                    rx.cond(
                        CameraState.detection_enabled & (CameraState.detection_count > 0),
                        f"ymax: {CameraState.highest_conf_ymax}",
                        "ymax: N/A"
                    ), 
                    class_name="text-gray-700"
                ),
                class_name="flex justify-between mt-2"
            ),
            class_name="bg-[#fef3e2] p-2 rounded"
        ),
        class_name="bg-[#ffec99] p-4 rounded-lg shadow-md w-full"
    )