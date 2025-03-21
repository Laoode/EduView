import reflex as rx

def behavior_panel() -> rx.Component:
    return rx.el.div(
        rx.el.h3("Behavior Analysis", class_name="text-lg font-semibold mb-2 text-gray-900"),
        rx.el.div(
            rx.el.div(
                rx.el.span("Behavior: ", class_name="text-gray-700"),
                rx.el.span("Looking Around", class_name="text-purple-500 font-semibold"),
                class_name="flex justify-between"
            ),
            rx.el.div(
                rx.el.span("Confidence Level: ", class_name="text-gray-700"),
                rx.el.span("95%", class_name="text-red-500 font-semibold"),
                class_name="flex justify-between mt-2"
            ),
            class_name="bg-[#fef3e2] p-2 rounded"
        ),
        class_name="bg-[#ffec99] p-4 rounded-lg shadow-md w-full"
    )