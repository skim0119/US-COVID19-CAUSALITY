# Run this app with `python app.py` and
# visit http://127.0.0.1:8050/ in your web browser.

from dash import Dash, html, dcc
from dash import Input, Output
import plotly.express as px
import pandas as pd

from assets import colors
from utils import generate_table

import state_prediction as sp_panels
import connectivity as conn_panels

app = Dash(__name__)

app.layout = html.Div(
    style={"backgroundColor": colors["background"]},
    children=[
        html.H1(
            children="US COVID-19 Spread Analysis/Visualization using Information Theory",
            style={"textAlign": "center", "color": colors["text"]},
        ),
        html.H2(
            children="Statewise COVID19 cases and death and prediction",
            style={"textAlign": "center", "color": colors["text"]},
        ),
        sp_panels.control_panel,
        sp_panels.display,
        html.H2(
            children="Causality network visualization and connectivity partitioning",
            style={"textAlign": "center", "color": colors["text"]},
        ),
        conn_panels.control_panel,
        html.Br(),
        conn_panels.display
    ],
)


@app.callback(
    Output("state-prediction-figure", "figure"),
    Input("state-prediction-selected-states", "value"),
    Input("state-prediction-predicting-month", "value"),
)
def update_state_prediction_figure(states, predicting_months):
    return sp_panels.get_figure(states, predicting_months)

@app.callback(
    Output("connectivity-figure", "figure"),
    Input("connectivity-year", "value"),
    Input("connectivity-month", "value"),
)
def update_connectivity_figure(year, month):
    return conn_panels.get_figure(year, month)

if __name__ == "__main__":
    app.run_server(debug=True)
