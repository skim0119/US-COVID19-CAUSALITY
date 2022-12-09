import numpy as np
import networkx as nx


class Network:
    def __init__(
        self,
        adjacency_matrix,
        node_position,
        node_name,
        node_size,
        threshold: float = 0.200,
        node_size_scale: float = 0.08,
    ):
        self.node_position = node_position
        self.node_name = node_name
        self.node_size = node_size

        self.threshold = threshold
        self.node_size_scale = node_size_scale

        self.network = self._draw_graph(adjacency_matrix)

    def _draw_graph(self, adjacency_matrix):
        G_1 = nx.DiGraph()
        n_state = len(adjacency_matrix)
        states = df.columns.to_list()

        # Nodes
        coords = {}
        new_cases = []
        for idx, state in enumerate(df.columns.to_list()):
            coords[idx] = self.node_position[state]
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
        G_1.add_weighted_edges_from(edgelist)
        edges = G_1.edges()
        weights = [edge_scale(G_1[u][v]["weight"]) for u, v in edges]
        node_size = [new_cases[k] for k in dict(G_1.degree).keys()]

        plt.figure()
        nx.draw(G_1, coords, width=weights, node_size=node_size)
        for node, (x, y) in coords.items():
            plt.text(x, y, states_abbreviation[df.columns[node]])
        plt.title(f"Covid19 Entropy Transfer {year}/{month}")
        # plt.xlim([xmin-10, xmax+10])
        # plt.ylim([ymin-10, ymax+10])
        plt.savefig(os.path.join(result_path, f"connectivity_{year}_{month:02d}.png"))

    def create_directed_graph_from_adjacency_matrix(adjacency_matrix):
        """
        Given adjacency matrix, create nx graph.
        """

        edge_scale = lambda x: (x) ** 2

        G_1 = nx.DiGraph()
        n_state = df.shape[1]

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

    def save_graph(self, path):
        pass


def pagerank(
    G,
    alpha=0.85,
    personalization=None,
    max_iter=100,
    tol=1.0e-6,
    nstart=None,
    weight="weight",
    dangling=None,
):
    """Return the PageRank of the nodes in the graph.

    PageRank computes a ranking of the nodes in the graph G based on
    the structure of the incoming links. It was originally designed as
    an algorithm to rank web pages.

    Parameters
    ----------
    G : graph
    A NetworkX graph. Undirected graphs will be converted to a directed
    graph with two directed edges for each undirected edge.

    alpha : float, optional
    Damping parameter for PageRank, default=0.85.

    personalization: dict, optional
    The "personalization vector" consisting of a dictionary with a
    key for every graph node and nonzero personalization value for each node.
    By default, a uniform distribution is used.

    max_iter : integer, optional
    Maximum number of iterations in power method eigenvalue solver.

    tol : float, optional
    Error tolerance used to check convergence in power method solver.

    nstart : dictionary, optional
    Starting value of PageRank iteration for each node.

    weight : key, optional
    Edge data key to use as weight. If None weights are set to 1.

    dangling: dict, optional
    The outedges to be assigned to any "dangling" nodes, i.e., nodes without
    any outedges. The dict key is the node the outedge points to and the dict
    value is the weight of that outedge. By default, dangling nodes are given
    outedges according to the personalization vector (uniform if not
    specified). This must be selected to result in an irreducible transition
    matrix (see notes under google_matrix). It may be common to have the
    dangling dict to be the same as the personalization dict.

    Returns
    -------
    pagerank : dictionary
    Dictionary of nodes with PageRank as value

    Notes
    -----
    The eigenvector calculation is done by the power iteration method
    and has no guarantee of convergence. The iteration will stop
    after max_iter iterations or an error tolerance of
    number_of_nodes(G)*tol has been reached.

    The PageRank algorithm was designed for directed graphs but this
    algorithm does not check if the input graph is directed and will
    execute on undirected graphs by converting each edge in the
    directed graph to two edges.


    """
    if len(G) == 0:
        return {}

    if not G.is_directed():
        D = G.to_directed()
    else:
        D = G

    # Create a copy in (right) stochastic form
    W = nx.stochastic_graph(D, weight=weight)
    N = W.number_of_nodes()

    # Choose fixed starting vector if not given
    if nstart is None:
        x = dict.fromkeys(W, 1.0 / N)
    else:
        # Normalized nstart vector
        s = float(sum(nstart.values()))
        x = dict((k, v / s) for k, v in nstart.items())

    if personalization is None:

        # Assign uniform personalization vector if not given
        p = dict.fromkeys(W, 1.0 / N)
    else:
        missing = set(G) - set(personalization)
        if missing:
            raise NetworkXError(
                "Personalization dictionary "
                "must have a value for every node. "
                "Missing nodes %s" % missing
            )
        s = float(sum(personalization.values()))
        p = dict((k, v / s) for k, v in personalization.items())

    if dangling is None:

        # Use personalization vector if dangling vector not specified
        dangling_weights = p
    else:
        missing = set(G) - set(dangling)
        if missing:
            raise NetworkXError(
                "Dangling node dictionary "
                "must have a value for every node. "
                "Missing nodes %s" % missing
            )
        s = float(sum(dangling.values()))
        dangling_weights = dict((k, v / s) for k, v in dangling.items())
    dangling_nodes = [n for n in W if W.out_degree(n, weight=weight) == 0.0]

    # power iteration: make up to max_iter iterations
    for _ in range(max_iter):
        xlast = x
        x = dict.fromkeys(xlast.keys(), 0)
        danglesum = alpha * sum(xlast[n] for n in dangling_nodes)
        for n in x:

            # this matrix multiply looks odd because it is
            # doing a left multiply x^T=xlast^T*W
            for nbr in W[n]:
                x[nbr] += alpha * xlast[n] * W[n][nbr][weight]
            x[n] += danglesum * dangling_weights[n] + (1.0 - alpha) * p[n]

        # check convergence, l1 norm
        err = sum([abs(x[n] - xlast[n]) for n in x])
        if err < N * tol:
            return x
    raise NameError(
        "pagerank: power iteration failed to converge " "in %d iterations." % max_iter
    )
