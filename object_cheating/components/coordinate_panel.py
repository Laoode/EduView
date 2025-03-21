import reflex as rx

def coordinate_panel() -> rx.Component:
    return rx.el.div(
        rx.el.h3("Bounding Box Coordinates", class_name="text-lg font-semibold mb-2 text-gray-900"),
        rx.el.div(
            rx.el.div(
                rx.el.span("xmin: 1236", class_name="text-gray-700"),
                rx.el.span("ymin: 341", class_name="text-gray-700"),
                class_name="flex justify-between"
            ),
            rx.el.div(
                rx.el.span("xmax: 1512", class_name="text-gray-700"),
                rx.el.span("ymax: 765", class_name="text-gray-700"),
                class_name="flex justify-between mt-2"
            ),
            class_name="bg-[#fef3e2] p-2 rounded"
        ),
        class_name="bg-[#ffec99] p-4 rounded-lg shadow-md w-full"
    )