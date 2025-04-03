import reflex as rx
from object_cheating.components.camera_feed import camera_feed
from object_cheating.components.controls import controls
from object_cheating.components.treshold import threshold
from object_cheating.components.stats_panel import stats_panel
from object_cheating.components.behavior_panel import behavior_panel
from object_cheating.components.coordinate_panel import coordinate_panel
from object_cheating.components.table import _tables_v2
from object_cheating.components.input_panel import input_panel
from object_cheating.components.warning_dialog import warning_dialog
from object_cheating.components.delete_dialog import delete_dialog

def index() -> rx.Component:
    return rx.box(
        # Warning dialog at root level for proper overlay
        warning_dialog(),
        delete_dialog(),
        rx.el.div(
            rx.el.div(
                rx.el.h1(
                    "Edu View — Smart Online Proctoring Assist for Cheating",
                    class_name="text-3xl font-bold text-gray-800 mb-8 text-center"
                ),
                rx.el.div(
                    # Left Section: Camera Feed, Controls, and Table in separate sections
                    rx.el.div(
                        rx.el.div(
                            camera_feed(),
                            controls(),
                            class_name="bg-[#ffec99] p-4 rounded-lg shadow-md space-y-4"
                        ),
                        rx.el.div(
                            _tables_v2(),
                            class_name="bg-[#ffec99] p-4 rounded-lg shadow-md space-y-4"
                        ),
                        class_name="w-2/3 pr-4 space-y-4"  # Added space-y-4 for spacing between sections
                    ),
                    # Right Section: Threshold, Show Label Name, Behavior, Coordinate Panels
                    rx.el.div(
                        threshold(),
                        stats_panel(),
                        behavior_panel(),
                        coordinate_panel(),
                        input_panel(),
                        class_name="w-1/3 space-y-4"
                    ),
                    class_name="flex"
                ),
                class_name="max-w-6xl mx-auto px-4 py-8"
            ),
            class_name="min-h-screen bg-[#fff9db]"
        ),
    )

app = rx.App(
    theme=rx.theme(
        accent_color="grass",
    )
)
app.add_page(index)