"""Graph analysis helpers for the Facebook ego-network.

Loading the edge list, basic structural statistics, approximate edge betweenness (to find the
bridges between communities), and the two community-detection methods with a partition
comparison.
"""

import os
import random
from collections import Counter

import numpy as np
import pandas as pd
import networkx as nx
import community as community_louvain                     # python-louvain
from networkx.algorithms.community import label_propagation_communities
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score


def load_graph(path):
    """Read the SNAP edge list into an undirected NetworkX graph (node ids kept as ints)."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Could not find {path} — see data/README.md.")
    return nx.read_edgelist(path, nodetype=int)


def basic_stats(G):
    """Return the core structural metrics: size, density, average degree, clustering, connectedness."""
    n, m = G.number_of_nodes(), G.number_of_edges()
    return pd.Series({
        "Nodes |V|": n,
        "Edges |E|": m,
        "Density": round(nx.density(G), 6),
        "Average degree": round(2 * m / n, 2),            # each edge touches two nodes
        "Connected": nx.is_connected(G),
        "Average clustering coefficient": round(nx.average_clustering(G), 3),
    }, name="value")


def compute_edge_betweenness(G, k=400, seed=0):
    """Approximate edge betweenness centrality using k sampled source nodes (exact is too slow)."""
    return nx.edge_betweenness_centrality(G, k=k, seed=seed, normalized=True)


def run_louvain(G, seed=0, resolution=1.0):
    """Run Louvain modularity optimisation; return (partition dict, modularity Q, sorted sizes)."""
    partition = community_louvain.best_partition(G, random_state=seed, resolution=resolution)
    Q = community_louvain.modularity(partition, G)
    sizes = sorted(Counter(partition.values()).values(), reverse=True)
    return partition, Q, sizes


def run_label_propagation(G, seed=0):
    """Run Label Propagation; return (partition dict, modularity Q, sorted sizes)."""
    random.seed(seed)                                     # tie-breaking uses the random module
    communities = list(label_propagation_communities(G))
    partition = {node: i for i, com in enumerate(communities) for node in com}
    Q = community_louvain.modularity(partition, G)        # same Q function, so the two are comparable
    sizes = sorted((len(c) for c in communities), reverse=True)
    return partition, Q, sizes


def compare_partitions(part_a, part_b, G):
    """Compare two node->community labellings with NMI and ARI (both 0 = unrelated, 1 = identical)."""
    nodes = sorted(G.nodes())                             # fixed order so the labels line up
    labels_a = [part_a[n] for n in nodes]
    labels_b = [part_b[n] for n in nodes]
    return (normalized_mutual_info_score(labels_a, labels_b),
            adjusted_rand_score(labels_a, labels_b))
