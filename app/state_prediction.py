__all__ = ["control_panel", "display", "get_figure"]

import os, sys

sys.path.append(
    os.path.join(sys.path[0], "..")
)  # TODO: probably need better file structure

from dash import Dash, html, dcc
import plotly.express as px
import pandas as pd

from assets import colors
from utils import generate_table

from CovidRawDataManager import CovidRawDataManager
from CovidTemporalPrediction import CovidTemporalPredictor

data = CovidRawDataManager()
states = data.generate_statewise_history_data("cases").columns.tolist()

_initial_states = ("Illinois",)

# create an encapsulated layout component
control_panel = html.Div(
    [
        html.Div(
            children=[
                html.Label("Select States of interest"),
                dcc.Dropdown(
                    states,
                    _initial_states,
                    multi=True,
                    id="state-prediction-selected-states",
                ),
                # html.Br(),
            ],
            style={"padding": 10, "flex": 1},
        ),
        html.Div(
            children=[
                html.Label("Prediction Month"),
                dcc.Slider(
                    min=1,
                    max=24,
                    marks={i: f"{i}" if i == 1 else str(i) for i in range(1, 25)},
                    value=12,
                    id="state-prediction-predicting-month",
                ),
            ],
            style={"padding": 10, "flex": 1},
        ),
    ],
    style={"display": "flex", "flex-direction": "row"},
)


def get_figure(states=_initial_states, future_months=12):
    y_labels = []
    dfs = []
    for state in states:
        temporal_predictor = CovidTemporalPredictor(state_name=state)
        temporal_predictor.setup_model(train_fraction=0.9)
        model = temporal_predictor.train_model(error_metric="MAE", verbose=False)
        # temporal_predictor.plot_model_fit()

        dfs.append(
            temporal_predictor.predict_and_plot_future(
                future_months=future_months, return_data_only=True
            )
        )
        y_labels.extend([state, state + " future prediction"])
    df = pd.concat(dfs)

    fig = px.line(df, x="date", y=y_labels)  # , color="City", barmode="group")
    fig.update_layout(
        # plot_bgcolor=colors['background'],
        # paper_bgcolor=colors['background'],
        # font_color=colors['text'],
        title="COVID19 daily cases per state and the prediction",
        yaxis_title="daily cases",
    )

    return fig


display = html.Div(
    [
        dcc.Graph(figure=get_figure(), id="state-prediction-figure"),
    ]
)
