import reflex as rx

def input_panel() -> rx.Component:
    button_style = "bg-[#ffa94d] text-black px-6 py-2 w-full text-center rounded-lg shadow-md hover:bg-[#ff922b]"

    return rx.el.div(
        rx.el.div(
            rx.el.h3("Media Controls", class_name="text-lg font-semibold mb-2 text-gray-900"),
            # First row (Image - Video)
            rx.el.div(
                rx.button("📷 Image", class_name=button_style),
                rx.button("🎬 Video", class_name=button_style),
                class_name="grid grid-cols-2 gap-4 w-full"
            ),

            # Second row (Webcam - Save)
            rx.el.div(
                rx.button("🎥 Webcam", class_name=button_style),
                rx.button("💾 Save", class_name=button_style),
                class_name="grid grid-cols-2 gap-4 w-full mt-4"
            ),

            # Third row (Clear, centered)
            rx.el.div(
                rx.button("🗑️ Clear", class_name=button_style),
                class_name="flex justify-center w-full mt-4"
            ),

            class_name="w-full p-6 bg-[#ffec99] rounded-lg shadow-md max-w-md mx-auto"
        )
    )
