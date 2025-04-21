import reflex as rx
from typing import List, Dict

def create_data_row(data: Dict[str, str]):
    from object_cheating.states.camera_state import CameraState
    return rx.table.row(
        rx.table.cell(
            rx.text(
                data["no"],
                color="black",
                font_size="11px",
                weight="medium",
            ),
        ),
        rx.table.cell(
            rx.text(
                data["location_file"],
                color="black",
                font_size="11px",
                weight="medium",
            ),
        ),
        rx.table.cell(
            rx.badge(
                data["behaviour"],
                color_scheme=rx.match(
                    data["behaviour"],
                    ("cheating", "tomato"),
                    ("Look Around", "violet"),
                    ("Normal", "grass"),
                    ("normal", "grass"),
                    ("Bend Over The Desk", "cyan"),
                    ("Hand Under Table", "indigo"),
                    ("Stand Up", "sky"),
                    ("Wave", "pink"),
                    "gray"
                ),
                size="1"
            ),
        ),
        rx.table.cell(
            rx.text(
                data["coordinate"],
                color="black",
                font_size="11px",
                weight="regular",
            ),
        ),
        align="center",
        white_space="nowrap",
    )

def tables_v2():
    from object_cheating.states.camera_state import CameraState
    return rx.vstack(
        rx.scroll_area(
            rx.table.root(
                rx.table.header(
                    rx.table.row(
                        rx.foreach(
                            ["No", "Location File", "Behaviour", "Coordinate"],
                            lambda title: rx.table.column_header_cell(
                                rx.text(title, font_size="12px", weight="bold", color="black"),
                            ),
                        ),
                    ),
                    position="sticky",
                    top="0",
                    background_color="#ffec99",
                    z_index="1",
                ),
                rx.table.body(
                    rx.foreach(
                        CameraState.table_data,
                        create_data_row
                    ),
                ),
                width="100%",
                variant="surface",
                size="2",
            ),
            type="always",
            scrollbars="vertical",
            style={
                "height": "267px",  # Tinggi untuk 5 baris
                "border": "1px solid #e2e8f0",
                "border_radius": "8px",
                "background": "#fef3e2",
            },
        ),
        background="#ffec99",
        padding="4",
        border_radius="lg",
        width="100%",
        max_width="800px",
    )
def _tables_v2():
    from object_cheating.states.camera_state import CameraState
    return rx.table.root(
        rx.table.header(
            rx.table.row(
                rx.foreach(
                    ["No", "Location File", "Behaviour", "Coordinate"],
                    lambda title: rx.table.column_header_cell(
                        rx.text(title, font_size="12px", weight="bold", color="black"),
                    ),
                ),
            ),
        ),
        rx.table.body(
            rx.foreach(CameraState.table_data, create_data_row),
            style={"max_height": "200px", "overflow_y": "auto"},
        ),
        width="100%",
        variant="surface",
        size="2",
    )