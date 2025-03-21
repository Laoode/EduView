import reflex as rx

def threshold() -> rx.Component:
    return rx.box(
        rx.vstack(
            rx.el.h3("Threshold Settings", class_name="text-lg font-semibold mb-2 text-gray-900"),
            rx.hstack(
                rx.text("Confidence Threshold:", class_name="font-medium text-gray-700"),
                rx.spacer(),
                rx.hstack(
                    rx.input(
                        value="0.25",
                        type="number",
                        min=0,
                        max=1,
                        step=0.01,
                        width="70px",
                        text_align="center",
                        border="1px solid #e2e8f0",
                        border_radius="md",
                    ),
                    rx.button(
                        rx.icon("chevron-up", size=12),
                        rx.icon("chevron-down", size=12),
                        display="flex",
                        flex_direction="column",
                        border="1px solid #e2e8f0",
                        border_radius="md",
                        height="100%",
                        px="1",
                        ml="1",
                    ),
                ),
                width="100%",
                justify="between",
                align="center",
            ),
            rx.hstack(
                rx.text("IoU Threshold:", class_name="font-medium text-gray-700"),
                rx.spacer(),
                rx.hstack(
                    rx.input(
                        value="0.70",
                        type="number",
                        min=0,
                        max=1,
                        step=0.01,
                        width="70px",
                        text_align="center",
                        border="1px solid #e2e8f0",
                        border_radius="md",
                    ),
                    rx.button(
                        rx.icon("chevron-up", size=12),
                        rx.icon("chevron-down", size=12),
                        display="flex",
                        flex_direction="column",
                        border="1px solid #e2e8f0",
                        border_radius="md",
                        height="100%",
                        px="1",
                        ml="1",
                    ),
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