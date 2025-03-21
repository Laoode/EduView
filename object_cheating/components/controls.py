import reflex as rx
def model_navigation() -> rx.Component:
    """Komponen navigasi model."""
    return rx.hstack(
        rx.icon_button(
            rx.icon("chevron-left"),
            variant="surface",
            height="30px",
            width="30px",
        ),
        rx.badge(
            rx.center(
                rx.text("Model 1"),
                width="100%",
                height="28px",
            ),
            variant="surface",  # Light gray background; adjust if needed
            min_width="100px",
            text_align="center",
        ),
        rx.icon_button(
            rx.icon("chevron-right"),
            variant="surface",
            height="30px",
            width="30px",
        ),
        spacing="2",
        align="center",
    )

def controls() -> rx.Component:
    return rx.hstack(
        rx.hstack(
            rx.text("Enable Detection", class_name="text-gray-700"),
            rx.switch(
                name="switch",
                color_scheme="grass",
                variant="surface",
            ),
            spacing="2",
            align="center",# Adds spacing between text and switch
        ),
        model_navigation(),
        spacing="2",
        class_name="flex justify-between"
    )