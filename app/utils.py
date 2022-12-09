from dash import Dash, html, dcc
import plotly.express as px
import pandas as pd

from assets import colors


def generate_table(dataframe, max_rows=10):
    return html.Table(
        [
            html.Thead(html.Tr([html.Th(col) for col in dataframe.columns])),
            html.Tbody(
                [
                    html.Tr(
                        [html.Td(dataframe.iloc[i][col]) for col in dataframe.columns]
                    )
                    for i in range(min(len(dataframe), max_rows))
                ]
            ),
        ],
        style={"color": colors["text"]},
    )
