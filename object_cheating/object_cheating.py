
import reflex as rx
from object_cheating.components.camera_feed import camera_feed
from object_cheating.components.controls import controls

def index() -> rx.Component:
    return rx.el.div(
        rx.el.div(
            rx.el.h1(
                "Camera Feed with Face Detection",
                class_name="text-3xl font-bold text-gray-800 mb-8"
            ),
            rx.el.div(
                camera_feed(),
                controls(),
                class_name="space-y-4"
            ),
            class_name="max-w-3xl mx-auto px-4 py-8"
        ),
        class_name="min-h-screen bg-gray-100"
    )

app = rx.App()
app.add_page(index)