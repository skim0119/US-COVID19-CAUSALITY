__all__ = ["control_panel", "display", "get_figure"]

import os, sys
from datetime import date
import networkx as nx
import yaml

sys.path.append(
    os.path.join(sys.path[0], "..")
)  # TODO: probably need better file structure

from dash import Dash, html, dcc
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import rgb2hex

from assets import colors
from utils import generate_table

from CovidRawDataManager import CovidRawDataManager
from CovidTemporalPrediction import CovidTemporalPredictor
from GeographyDataManager import GeographyDataManager
from EventDetection import EventDetection
from TransferEntropy import TransferEntropy
from GraphUtility import Network, pagerank

data = CovidRawDataManager()
statewise_cases = data.generate_statewise_history_data("cases")
df = (statewise_cases.diff().rolling("7D").sum().diff() > 0).dropna()

# Call Geography Data
geography_data_manager = GeographyDataManager()
state_xy = geography_data_manager.generate_us_state_coordinates()

xmin, xmax, ymin, ymax = 0, 0, 0, 0
for x, y in state_xy.values():
    xmin = min(x, xmin)
    xmax = max(x, xmax)
    ymin = min(y, ymin)
    ymax = max(y, ymax)

# Compute transfer entropy
te = TransferEntropy(3, 3, 16)
threshold = 0.2
node_size_scale = 0.30

# create an encapsulated layout component
control_panel = html.Div(
    [
        html.Label("Select year of interest"),
        dcc.Dropdown([{'label':x, 'value': x} for x in range(2020,2023)], 2021, id='connectivity-year'),
        html.Label("Select month of interest"),
        dcc.Dropdown([{'label':x, 'value': x} for x in range(1,13)], 1, id='connectivity-month'),
    ],
    style={"width": "15%"},
)

def create_traces_from_networkx(G):
    edge_x = []
    edge_y = []
    for edge in G.edges():
        x0, y0 = G.nodes[edge[0]]['pos']
        x1, y1 = G.nodes[edge[1]]['pos']
        edge_x.append(x0)
        edge_x.append(x1)
        edge_x.append(None)
        edge_y.append(y0)
        edge_y.append(y1)
        edge_y.append(None)

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.5, color='#888'),
        hoverinfo='none',
        mode='lines')

    node_x = []
    node_y = []
    for node in G.nodes():
        x, y = G.nodes[node]['pos']
        node_x.append(x)
        node_y.append(y)

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers',
        hoverinfo='text',
        marker=dict(
            showscale=True,
            # colorscale options
            #'Greys' | 'YlGnBu' | 'Greens' | 'YlOrRd' | 'Bluered' | 'RdBu' |
            #'Reds' | 'Blues' | 'Picnic' | 'Rainbow' | 'Portland' | 'Jet' |
            #'Hot' | 'Blackbody' | 'Earth' | 'Electric' | 'Viridis' |
            colorscale='YlGnBu',
            reversescale=True,
            color=[],
            size=10,
            colorbar=dict(
                thickness=15,
                title='Node Connections',
                xanchor='left',
                titleside='right'
            ),
            line_width=2))
    return edge_trace, node_trace

def get_figure(year=2021, month=1):
    title = f"Covid19 Entropy Transfer {year}/{month}"
    ydf = df[df.index.year.isin([year])]
    mdf = ydf[ydf.index.month.isin([month])]

    directional_entropy = te(mdf, disable_progbar=True)
    states = mdf.columns.tolist()

    # Get Population for node size
    new_case_df = statewise_cases.diff().dropna()
    y_new_case_df = new_case_df[new_case_df.index.year.isin([year])]
    m_new_case_df = y_new_case_df[y_new_case_df.index.month.isin([month])]

    with open("assets/StatesAbbreviation.yaml", "r") as file:
        states_abbreviation = yaml.load(file, Loader=yaml.Loader)

    G_1 = nx.DiGraph()
    n_state = len(states)

    edge_scale = lambda x: (x) ** 2

    # Nodes
    pos = {}
    new_cases = []
    for idx, state in enumerate(states):
        pos[idx] = state_xy[state]
        new_cases.append(np.sqrt(m_new_case_df[state].sum() + 1) * node_size_scale)

    # Edges
    edgelist = []
    for i in range(n_state):
        for j in range(i + 1, n_state):
            diff = directional_entropy[i, j] - directional_entropy[j, i]
            weight = np.abs(diff)
            if weight < threshold:
                continue
            if diff > 0:  # influence i->j
                edgelist.append([i, j, weight])
            else:  # influence j-> i
                edgelist.append([j, i, weight])
    G_1.add_weighted_edges_from(edgelist)

    # node tag
    tags = []
    for node in G_1.nodes:
        state = states[node]
        G_1.nodes[node]['pos'] = state_xy[state]
        tags.append(states_abbreviation[state])

    node_size = [new_cases[k]*0.25 for k in dict(G_1.degree).keys()]
    for u, v in G_1.edges():
        G_1[u][v]["weight"] = edge_scale(G_1[u][v]["weight"])

    edge_trace, node_trace = create_traces_from_networkx(G_1)
    node_trace.text = tags
    node_trace.marker.color = node_size
    node_trace.marker.size = node_size

    # Community Detection
    RUN_COMMUNITY_DETECTION = False
    if RUN_COMMUNITY_DETECTION:
        import igraph as ig
        import leidenalg as la

        try:
            pageranks = pagerank(G_1)
        except NameError:
            pass

        h = ig.Graph.from_networkx(G_1)
        partitions = la.find_partition(
            h, la.ModularityVertexPartition, max_comm_size=4
        )
        # partitions = la.find_partition(h, la.SignificanceVertexPartition, max_comm_size=4)
        # partitions = la.find_partition(h, la.SurpriseVertexPartition, max_comm_size=4)
        print(len(partitions))
        #colors = plt.cm.rainbow(np.linspace(0.1, 0.9, len(partitions)))

        node_color = []
        for node in G_1:
            color = [0.0, 0.0, 0.0, 1.0]
            for idx, p in enumerate(partitions):
                if node in p:
                    color = colors[idx]
                    break
            node_color.append(color)

        node_size = [pageranks[k] * 5000 for k in dict(G_1.degree).keys()]
        nx.draw(G_1, pos, node_color=node_color, width=weights, node_size=node_size)
        for node, (x, y) in pos.items():
            pass
            #plt.text(x, y, states_abbreviation[df.columns[node]])
        #plt.title(f"Pagerank {year}/{month}")

    def assign_colour(correlation):
        # Function to assign correlation colors (neg becomes red, pos becomes green)
        return rgb2hex(plt.cm.Greys(correlation))
        if correlation <= 0:
            return "#ffa09b"  # red
        else:
            return "#9eccb7"  # green

    def assign_thickness(correlation, benchmark_thickness=2, scaling_factor=1):
        # Function to assign correlation thickness based on absolute magnitide
        return benchmark_thickness * abs(correlation)**scaling_factor

    def edge_traces(attr="weight"):
        ### Function which creates a list of traces for the edges.
        ### In order to assign weights to edges they need to be added separately via a function like this.    
        
        # assign colours to edges depending on positive or negative correlation
        # assign edge thickness depending on magnitude of correlation
        edge_colours = []
        edge_width = []
        vals = np.array([value for value in nx.get_edge_attributes(G_1, 'weight').values()])
        for key, value in nx.get_edge_attributes(G_1, 'weight').items():
            value = (value - vals.min()) / (vals.max() - vals.min())
            edge_colours.append(assign_colour(value))
            edge_width.append(assign_thickness(value))
                
        edge_trace_ = []
        for i, e in enumerate(G_1.edges):
            x0, y0 = G_1.nodes[e[0]]['pos']
            x1, y1 = G_1.nodes[e[1]]['pos']
            trace_ = go.Scatter(
                x=[x0,x1],
                y=[y0,y1],
                mode="lines",
                line=dict(width=edge_width[i], color=edge_colours[i])
            )
            
            edge_trace_.append(trace_)
            
        return edge_trace_

    #fig = go.Figure(data=[edge_trace, node_trace],
    fig = go.Figure(data=[node_trace] + edge_traces()[0:],
            layout=go.Layout(
                title='Causality network using transfer entropy',
                titlefont_size=16,
                showlegend=False,
                hovermode='closest',
                margin=dict(b=20,l=5,r=5,t=40),
                annotations=[ dict(
                    text="",
                    showarrow=True,
                    xref="paper", yref="paper",
                    x=0.005, y=-0.002 ) ],
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                )
    #fig.add_annotation(
    #    x=30,  # arrows' head
    #    y=30,  # arrows' head
    #    ax=40,  # arrows' tail
    #    ay=40,  # arrows' tail
    #    xref='x',
    #    yref='y',
    #    axref='x',
    #    ayref='y',
    #    text='',  # if you want only the arrow
    #    showarrow=True,
    #    arrowhead=3,
    #    arrowsize=1,
    #    arrowwidth=1,
    #    arrowcolor='black'
    #)
    return fig

display = html.Div(
    [
        dcc.Graph(figure=get_figure(), id="connectivity-figure"),
    ]
)
