import reflex as rx
from object_cheating.states.camera_state import CameraState

def stats_panel() -> rx.Component:
    return rx.vstack(
        rx.el.h3("Detection Summary", class_name="text-lg font-semibold mb-2 text-gray-900"),
        # Baris pertama: Total Target dan FPS
        rx.hstack(
            rx.text("Total Target: 5", class_name="text-gray-700"),
            rx.text("FPS: 30", class_name="text-gray-700"),
            justify="between",  # Memastikan elemen tersebar ke kiri dan kanan
            width="100%",  # Mengisi lebar penuh container
        ),
        # Baris kedua: Runtime dan Target Selection
        rx.hstack(
            rx.text("Runtime: 0.072s", class_name="text-gray-700"),
            rx.hstack(
                rx.text("Target Selection: ", class_name="text-gray-700"),
                rx.select(
                    ["All", "Looking Around", "Leaning to Copy"],
                    default_value="All",
                ),
                spacing="2",  # Jarak antara label dan dropdown
            ),
            justify="between",  # Memastikan elemen tersebar ke kiri dan kanan
            width="100%",  # Mengisi lebar penuh container
        ),
        spacing="4",  # Jarak vertikal antar elemen
        class_name="bg-[#ffec99] p-4 rounded-lg shadow-md w-full"  # Styling sesuai desain
    )