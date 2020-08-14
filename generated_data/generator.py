import numpy as np
import matplotlib.pyplot as plt
import random
import pandas as pd
import holoviews as hv

from generated_data.LFR_benchmark import LFR_benchmark_graph
from generated_data.laplacian_eigenmaps import LaplacianEigenmaps
from generated_data.viaualize import plot_dynamic_embedding
from tdGraphEmbed.model import TdGraphEmbed

hv.extension('bokeh')
random.seed(10)
np.random.seed(10)

n = 1000
tau1 = 2.5
tau2 = 1.5
mu = 0.2
timesteps = 20
anomaly_times = [13, 29, 41, 72, 98]


def draw_degree_dist(g):
    degree_sequence = sorted([d for n, d in g.degree()], reverse = True)  # degree sequence
    # print "Degree sequence", degree_sequence
    plt.hist(degree_sequence, bins = 100, range = (0, 200))


def draw_temporal_communities(dynamic_graph_series):
    nodes_pos_list = []
    for i in range(len(dynamic_graph_series)):
        static_embedding = LaplacianEigenmaps(2)
        nodes_pos_list.append(static_embedding.learn_embedding(dynamic_graph_series[i][0]))
        draw_degree_dist(dynamic_graph_series[i][0])
    plt = plot_dynamic_embedding(nodes_pos_list, dynamic_graph_series)
    plt.show()
    plt.savefig('generated_data/communities_visualization.png')


def plot_regular_graph_stat(dynamic_graph_series, anomaly_times):
    degree = [np.mean(list(dict(g[0].degree()).values())) for g in dynamic_graph_series]
    edges = np.array([g[0].number_of_edges() for g in dynamic_graph_series])
    nodes = np.array([g[0].number_of_nodes() for g in dynamic_graph_series])
    density = (edges / nodes)
    df = pd.DataFrame({'time': range(0, timesteps), 'average_degree': degree, 'density': density})
    curve1 = hv.Curve(data = df, xdims = 'time', vdims = ['average_degree']).opts(width = 700, height = 500,
                                                                                  fontsize = 20)
    if anomaly_times is not None:
        scat_anoms = hv.Scatter(data = df[df['time'].isin(anomaly_times)], xdims = 'time',
                                vdims = ['average_degree']).opts(
            width = 700, height = 500, size = 10, fill_color = "red")
        curve1 = curve1 * scat_anoms
    curve2 = hv.Curve(data = df, xdims = 'time', vdims = ['density']).opts(width = 700, height = 500, fontsize = 20)
    if anomaly_times is not None:
        scat_anoms = hv.Scatter(data = df[df['time'].isin(anomaly_times)], xdims = 'time', vdims = ['density']).opts(
            width = 700, height = 500, size = 10, fill_color = "red")
        curve2 = curve2 * scat_anoms
    curve = curve1 + curve2
    renderer = hv.renderer('bokeh')
    renderer.save(curve, 'generated_data/general_stats')


def temporal_LFR_anomalies(n, tau1 , tau2, mu, timesteps, anomaly_times):
    '''

    :param n: int, number of nodes
    :param tau1:  float. Power law exponent for the degree distribution of the created
        graph. This value must be strictly greater than one.
    :param tau2: float. Power law exponent for the community size distribution in the
        created graph. This value must be strictly greater than one.
    :param mu: float. Fraction of intra-community edges incident to each node. This
        value must be in the interval [0, 1].
    :param timesteps: int,  number od time steps
    :param anomaly_times: list. list of anomaly times indices
    :return:
    '''
    G, communities = LFR_benchmark_graph(n, tau1, tau2, mu, average_degree = 30,
                                        min_community = 450, seed = 10)
    graphs = [G.copy()]
    node_community = np.zeros(n)
    for c in range(len(communities)):
        node_community[list(communities[c])] = int(c)
    nodes_comunities = [node_community]
    normal_day_distribution = np.random.normal(1, 0)
    abnormal_day_distribution = np.random.normal(20, 3)
    perturbations = [[]]
    dyn_change_nodes = [[]]
    for t in range(1, timesteps):
        community_id = random.choice(range(len(communities)))
        print('Step %d' % t)
        if t in anomaly_times:
            num_nodes_to_purturb = max(int(abnormal_day_distribution), 0)
        else:
            num_nodes_to_purturb = max(int(normal_day_distribution), 0)

        print("Migrating Nodes")
        nodes = [i for i in range(n) if node_community[i] == community_id]
        nodes_to_purturb = random.sample(nodes, num_nodes_to_purturb)
        perturbations.append(nodes_to_purturb)
        dyn_change_nodes.append(nodes_to_purturb)

        for node in nodes_to_purturb:
            new_community = random.choice(list(set(range(len(communities))) - set([community_id])))
            print('Node %d change from community %d to %d' % (node,
                                                              node_community[node],
                                                              new_community))
            node_community[node] = new_community
        for c in range(len(communities)):
            communities[c] = set(np.where(node_community == c)[0])
        G, _ = LFR_benchmark_graph(n, tau1, tau2, mu, average_degree = 30,
                                        min_community = 450, seed = 10, communities = communities)
        graphs.append(G.copy())
        nodes_comunities.append(node_community)
    return zip(graphs, nodes_comunities, perturbations, dyn_change_nodes)


if __name__ == '__main__':
    dynamic_graph_series = list(temporal_LFR_anomalies(n, tau1, tau2, mu, timesteps, anomaly_times))
    draw_temporal_communities(dynamic_graph_series)
    plot_regular_graph_stat(dynamic_graph_series, anomaly_times)

    graphs = {t: g[0] for t, g in zip(range(len(dynamic_graph_series)), dynamic_graph_series)}
    model= TdGraphEmbed(dataset_name = "generated_data")
    documents = model.get_documents_from_graph(graphs)
    model.run_doc2vec(documents)
