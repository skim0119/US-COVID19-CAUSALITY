import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd

import mpi4py
from mpi4py import MPI

# CONTRIBUTION: Necessity modules
from CovidRawDataManager import CovidRawDataManager
from GeographyDataManager import GeographyDataManager
from EventDetection import EventDetection
from TransferEntropy import TransferEntropy
from GraphUtility import Network, pagerank

# MPI setup
ROOT = 0
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
nprocs = comm.Get_size()

# Call COVID data
covid_raw_data_manager = CovidRawDataManager()
statewise_cases_df = covid_raw_data_manager.generate_statewise_history_data(
    states_df_label="cases"
)
# df = EventDetection(df)
"""
exclude_states = [
    "Alaska",
    "Hawaii",
    "Guam",
    "Puerto Rico",
    "Virgin Islands",
    "Northern Mariana Islands",
    "American Samoa",
]
df = df.loc[:, ~df.columns.isin(exclude_states)]
"""
df = (statewise_cases_df.diff().rolling("7D").sum().diff() > 0).dropna()

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

# Run
RUN_GRAPH = True
RUN_COMMUNITY_DETECTION = True
RUN_HIST = False

result_path = "results"
os.makedirs(result_path, exist_ok=True)
threshold = 0.2
node_size_scale = 0.15

# Task split per rank
tasks = []
for year in pd.DatetimeIndex(df.index).year.unique().sort_values():
    for month in pd.DatetimeIndex(df.index).month.unique().sort_values():
        tasks.append((year, month))
tasks_idx = np.array_split(np.arange(len(tasks)), nprocs)[rank]
for idx in tqdm(tasks_idx, position=rank):
    year, month = tasks[idx]
    ydf = df[df.index.year.isin([year])]
    mdf = ydf[ydf.index.month.isin([month])]

    directional_entropy = te(mdf, disable_progbar=True)

    # Get Population for node size
    new_case_df = statewise_cases_df.diff().dropna()
    y_new_case_df = new_case_df[new_case_df.index.year.isin([year])]
    m_new_case_df = y_new_case_df[y_new_case_df.index.month.isin([month])]

    if RUN_GRAPH:
        import networkx as nx
        import yaml

        with open("assets/StatesAbbreviation.yaml", "r") as file:
            states_abbreviation = yaml.load(file, Loader=yaml.Loader)

        """
        # Node size scales with new cases
        node_size = {}
        for state in enumerate(df.columns.to_list()):
            node_size[state] = np.sqrt(m_new_case_df[state].sum()+1)

        graph = Network(directional_entropy,
                        node_position = state_xy,
                        node_name = states_abbreviation,
                        node_size = node_size,
                        )
        """

        G_1 = nx.DiGraph()
        n_state = df.shape[1]

        edge_scale = lambda x: (x) ** 2

        # Nodes
        pos = {}
        new_cases = []
        for idx, state in enumerate(df.columns.to_list()):
            pos[idx] = state_xy[state]
            new_cases.append(np.sqrt(m_new_case_df[state].sum() + 1) * node_size_scale)

        # Edges
        edgelist = []
        for i in range(n_state):
            istr = df.columns[i]
            for j in range(i + 1, n_state):
                jstr = df.columns[j]
                diff = directional_entropy[i, j] - directional_entropy[j, i]
                weight = np.abs(diff)
                if weight < threshold:
                    continue
                if diff > 0:  # influence i->j
                    edgelist.append([i, j, weight])
                else:  # influence j-> i
                    edgelist.append([j, i, weight])
        plt.figure()
        G_1.add_weighted_edges_from(edgelist)
        edges = G_1.edges()
        weights = [edge_scale(G_1[u][v]["weight"]) for u, v in edges]
        node_size = [new_cases[k] for k in dict(G_1.degree).keys()]
        nx.draw(G_1, pos, width=weights, node_size=node_size)
        for node, (x, y) in pos.items():
            plt.text(x, y, states_abbreviation[df.columns[node]])
        plt.title(f"Covid19 Entropy Transfer {year}/{month}")
        # plt.xlim([xmin-10, xmax+10])
        # plt.ylim([ymin-10, ymax+10])
        plt.savefig(os.path.join(result_path, f"connectivity_{year}_{month:02d}.png"))
        plt.close()

        # Community Detection
        if RUN_COMMUNITY_DETECTION:
            import igraph as ig
            import leidenalg as la

            try:
                pageranks = pagerank(G_1)
            except NameError:
                continue

            h = ig.Graph.from_networkx(G_1)
            partitions = la.find_partition(
                h, la.ModularityVertexPartition, max_comm_size=4
            )
            # partitions = la.find_partition(h, la.SignificanceVertexPartition, max_comm_size=4)
            # partitions = la.find_partition(h, la.SurpriseVertexPartition, max_comm_size=4)
            print(len(partitions))
            colors = plt.cm.rainbow(np.linspace(0.1, 0.9, len(partitions)))

            node_color = []
            for node in G_1:
                color = [0.0, 0.0, 0.0, 1.0]
                for idx, p in enumerate(partitions):
                    if node in p:
                        color = colors[idx]
                        break
                node_color.append(color)

            plt.figure()
            node_size = [pageranks[k] * 5000 for k in dict(G_1.degree).keys()]
            nx.draw(G_1, pos, node_color=node_color, width=weights, node_size=node_size)
            for node, (x, y) in pos.items():
                plt.text(x, y, states_abbreviation[df.columns[node]])
            plt.title(f"Pagerank {year}/{month}")
            # plt.show()
            # plt.xlim([xmin-10, xmax+10])
            # plt.ylim([ymin-10, ymax+10])
            plt.savefig(os.path.join(result_path, f"pageranks_{year}_{month:02d}.png"))
            plt.close()

    if RUN_HIST:
        plt.figure()
        diff = directional_entropy - directional_entropy.T
        plt.hist(diff.ravel(), bins=20)
        plt.savefig(
            os.path.join(result_path, f"entropy_histogram_{year}_{month:02d}.png")
        )
        plt.close()
