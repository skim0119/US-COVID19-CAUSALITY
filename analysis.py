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

# Call COVID data
covid_raw_data_manager = CovidRawDataManager()
statewise_cases_df = covid_raw_data_manager.generate_statewise_history_data(
    states_df_label="cases"
)
# df = EventDetection(df)
df = (statewise_cases_df.diff().rolling("7D").sum().diff() > 0).dropna()
exclude_states = [
    "Alaska",
    "Hawaii",
    "Guam",
    "Puerto Rico",
    "Virgin Islands",
    "Northern Mariana Islands",
    "American Samoa",
]
states_abbreviation = {
    'Alaska':'AK',
    'Alabama':'AL',
    'Arkansas':'AR' ,
    'Arizona':'AZ' ,
    'California':'CA' ,
    'Colorado':'CO' ,
    'Connecticut':'CT' ,
    'District of Columbia':'DC' ,
    'Delaware':'DE' ,
    'Florida':'FL' ,
    'Georgia':'GA' ,
    'Hawaii':'HI' ,
    'Iowa':'IA' ,
    'Idaho':'ID' ,
    'Illinois':'IL' ,
    'Indiana':'IN' ,
    'Kansas':'KS' ,
    'Kentucky':'KY' ,
    'Louisiana':'LA' ,
    'Massachusetts':'MA' ,
    'Maryland':'MD' ,
    'Maine':'ME' ,
    'Michigan':'MI' ,
    'Minnesota':'MN' ,
    'Missouri':'MO' ,
    'Mississippi':'MS' ,
    'Montana':'MT' ,
    'North Carolina':'NC' ,
    'North Dakota':'ND' ,
    'Nebraska':'NE' ,
    'New Hampshire':'NH' ,
    'New Jersey':'NJ' ,
    'New Mexico':'NM' ,
    'Nevada':'NV' ,
    'New York':'NY' ,
    'Ohio':'OH' ,
    'Oklahoma':'OK' ,
    'Oregon':'OR' ,
    'Pennsylvania':'PA' ,
    'Rhode Island':'RI' ,
    'South Carolina':'SC' ,
    'South Dakota':'SD' ,
    'Tennessee':'TN' ,
    'Texas':'TX' ,
    'Utah':'UT' ,
    'Virginia':'VA' ,
    'Vermont':'VT' ,
    'Washington':'WA' ,
    'Wisconsin':'WI' ,
    'West Virginia':'WV' ,
    'Wyoming':'WY' 
}

df = df.loc[:, ~df.columns.isin(exclude_states)]

# Call Geography Data
geography_data_manager = GeographyDataManager()
state_xy = geography_data_manager.generate_us_state_coordinates()

xmin, xmax, ymin, ymax = 0,0,0,0
for x, y in state_xy.values():
    xmin = min(x, xmin)
    xmax = max(x, xmax)
    ymin = min(y, ymin)
    ymax = max(y, ymax)

# Compute transfer entropy
te = TransferEntropy(3, 3, 16)
# directional_entropy = te(df)

# Run

RUN_GRAPH = True
RUN_HIST = True

threshold = 0.200
edge_scale = lambda x: (x) ** 2
node_size_scale = 0.08
result_path = "results"
os.makedirs(result_path, exist_ok=True)

ROOT = 0
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
nprocs = comm.Get_size()

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

        G_1 = nx.DiGraph()
        n_state = df.shape[1]

        # Nodes
        pos = {}
        new_cases = []
        for idx, state in enumerate(df.columns.to_list()):
            pos[idx] = state_xy[state]
            new_cases.append(np.sqrt(m_new_case_df[state].sum()+1) * node_size_scale)

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
        #plt.xlim([xmin-10, xmax+10])
        #plt.ylim([ymin-10, ymax+10])
        plt.savefig(os.path.join(result_path, f"connectivity_{year}_{month:02d}.png"))

    if RUN_HIST:
        plt.figure()
        diff = directional_entropy - directional_entropy.T
        plt.hist(diff.ravel(), bins=20)
        plt.savefig(
            os.path.join(result_path, f"entropy_histogram_{year}_{month:02d}.png")
        )
