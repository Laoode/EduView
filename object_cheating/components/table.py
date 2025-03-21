import reflex as rx


class Table(rx.State):
    data: list[dict[str, str]] = [
        {
            "no": "1",
            "location_file": "D:/Downloads/Rob Wolf",
            "behaviour": "Cheating",
            "coordinate": "[1236,1512,341,765]",
        },
        {
            "no": "2",
            "location_file": "D:/Downloads/Rob Wolf",
            "behaviour": "Cheating",
            "coordinate": "[146,112,341,765]",
        },
        {
            "no": "3",
            "location_file": "D:/Downloads/Rob Wolf",
            "behaviour": "Normal",
            "coordinate": "[236,412,361,876]",
        },
        {
            "no": "4",
            "location_file": "D:/Downloads/Rob Wolf",
            "behaviour": "Normal",
            "coordinate": "[479,617,381,576]",
        },
        {
            "no": "5",
            "location_file": "D:/Downloads/Rob Wolf",
            "behaviour": "Looking arround",
            "coordinate": "[579,457,241,719]",
        },
    ]

    color_map: dict[str, str] = {
        "Cheating": "blue",
        "Looking arround": "cyan",
        "Normal": "pink",
    }


def create_data_row(data: dict[str, str]):
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
                color_scheme=Table.color_map[data["behaviour"]],
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
    return rx.table.root(
        rx.table.header(
            rx.table.row(
                rx.foreach(
                    ["No", "Location File", "Behaviour", "Coordinate"],
                    lambda title: rx.table.column_header_cell(
                        rx.text(title, font_size="12px", weight="bold", color="black"),  # Added color="black"
                    ),
                ),
            ),
        ),
        rx.table.body(
            rx.foreach(Table.data, create_data_row)
        ),
        width="100%",
        variant="surface",
        max_width="800px",
        size="2",
    )


def _tables_v2():
    return rx.table.root(
        rx.table.header(
            rx.table.row(
                rx.foreach(
                    ["No", "Location File", "Behaviour", "Coordinate"],
                    lambda title: rx.table.column_header_cell(
                        rx.text(title, font_size="12px", weight="bold", color="black"),  # Added color="black"
                    ),
                ),
            ),
        ),
        rx.table.body(
            rx.foreach(Table.data, create_data_row)
        ),
        width="100%",
        variant="surface",
        size="2",
    )