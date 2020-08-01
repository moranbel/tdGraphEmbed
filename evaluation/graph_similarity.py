import pickle
import networkx as nx
import pandas as pd
from scipy import spatial


def graph2graph_similarity(graphs_vectors):
    '''

    :param graphs_vectors: dict, key: timestep, value- graph vector
    :return:
    '''
    similarity_matrix = pd.DataFrame(columns = graphs_vectors.keys(), index = graphs_vectors.keys())
    for t1, g1 in graphs_vectors.items():
        # g1 = g1.mean(axis=0)
        for t2, g2 in graphs_vectors.items():
            # g2 = g2.mean(axis=0)
            sim = spatial.distance.cosine(g1, g2)
            similarity_matrix.loc[t1][t2] = sim
    return similarity_matrix


def graph2graph_mcs(graphs: list):
    '''
    :param graphs: list of networkx graphs
    :return: graph2graph similarity matrix
    '''
    # MCS (most common graph) measure implementation
    similarity_matrix = pd.DataFrame(columns = graphs.keys(), index = graphs.keys())
    for t1, g1 in graphs.items():
        for t2, g2 in graphs.items():
            mcs = getMCS(g1, g2)
            similarity_matrix.loc[t1][t2] = mcs
    return similarity_matrix


def getMCS(graph1, graph2):
    '''

    :param graph1:
    :param graph2:
    :return: MCS (most common graph) measure
    '''
    matching_graph = nx.Graph()

    for n1, n2, attr in graph2.edges(data = True):
        if graph1.has_edge(n1, n2):
            matching_graph.add_edge(n1, n2, weight = 1)

    graphs = list(nx.connected_component_subgraphs(matching_graph))

    mcs_length = 0
    mcs_graph = nx.Graph()
    for i, graph in enumerate(graphs):
        if len(graph.nodes()) > mcs_length:
            mcs_length = len(graph.nodes())
            mcs_graph = graph

    return len(mcs_graph.nodes()) / len(graph1.nodes())


if __name__ == "__main__":
    with open('data/slashdot/slashdot_monthly_dynamic.pkl', 'rb') as f:
        graphs = pickle.load(f)
    similarity_matrix = graph2graph_mcs(graphs)
    print(similarity_matrix)