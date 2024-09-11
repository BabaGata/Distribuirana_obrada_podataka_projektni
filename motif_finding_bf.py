import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import itertools


import pycuda.autoinit
import pycuda.driver as drv
import numpy as np
from pycuda.compiler import SourceModule

mod = SourceModule(open("hello.cu").read())


def find_network_motifs_from_edges(edges, motif_size):
    # Create a graph from the list of edges
    graph = nx.Graph(edges)

    # Generate all subgraphs of the desired motif size
    subgraphs = list(nx.enumerate_all_cliques(graph))
    
    motifs = []

    for subgraph in subgraphs:
        if len(subgraph) >= motif_size:
            motifs.append(graph.subgraph(subgraph))

    return motifs


def is_subgraph_normal(graph, subgraph):
    """
    Check if `subgraph` is a subgraph of `graph`.
    """
    # print("is subgraph", subgraph)
    return all(edge in graph for edge in subgraph)

def is_subgraph_cuda(graph, subgraph):
    """
    Check if `subgraph` is a subgraph of `graph`.
    """
    is_subgraph = mod.get_function("isSubgraph")

    # Convert "graph" from networkx graph to numpy array
    graph = nx.Graph(graph)
    graph = nx.adjacency_matrix(graph).todense()
    graph = np.array(graph, dtype=np.int32)

    # Convert "subgraph" from networkx graph to numpy array
    subgraph = nx.Graph(subgraph)
    subgraph = nx.adjacency_matrix(subgraph).todense()
    subgraph = np.array(subgraph, dtype=np.int32)

    #print("graph", graph)
    #print("subgraph", subgraph)

    # CUDA function equivalent to Python expression "all(edge in graph for edge in subgraph)"
    result = np.zeros(subgraph.shape, dtype=np.int32)
    is_subgraph(drv.Out(result), drv.In(graph), drv.In(subgraph), np.int32(5), np.int32(5), block=(1,1,1), grid=(1,1))
    
    #print("result", result)
    return result[0][0] == 1


    #return result.all()

def is_motif(subgraph, motif_candidates, subgraf_func):
    """
    Check if a `subgraph` is a motif by comparing it to a list of `motif_candidates`.
    """
    for candidate in motif_candidates:
        if subgraf_func(candidate, subgraph):
            return True
    return False

def generate_subgraphs(edges, motif_size):
    """
    Generate all subgraphs of a specific size from a list of edges.
    """
    subgraphs = []

    for subset in itertools.combinations(edges, motif_size):
        subgraph = set(subset)
        subgraphs.append(subgraph)

    return subgraphs



def find_network_motifs(edges, motif_size, subgraf_func):
    motifs = []
    motif_candidates = generate_subgraphs(edges, motif_size)

    for candidate in motif_candidates:
        is_candidate_motif = True
        for edge in candidate:
            # Remove one edge at a time and check if it's still a motif
            temp_candidate = candidate.copy()
            temp_candidate.remove(edge)


            if not is_motif(temp_candidate, motif_candidates, subgraf_func):
                is_candidate_motif = False
                break

        if is_candidate_motif:
            motifs.append(candidate)

    return motifs