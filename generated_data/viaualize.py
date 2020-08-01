from sklearn.manifold import TSNE
from matplotlib import pyplot as plt
import networkx as nx


def plot_dynamic_embedding(nodes_pos_list, dynamic_graph_series):
    length = len(dynamic_graph_series)
    node_num, dimension = nodes_pos_list[0][0].shape

    if dimension > 2:
        print("Embedding dimension greater than 2, using tSNE to reduce it to 2")
        model = TSNE(n_components = 2, random_state = 42)
        nodes_pos_list = [model.fit_transform(X) for X in nodes_pos_list]

    pos = 1
    for t in range(length):
        # print(t)
        plt.subplots_adjust(wspace = 0.2, hspace = 0.4)
        ax = plt.subplot(4, 5, pos)
        pos += 1

        plot_single_step(nodes_pos_list[t],
                         dynamic_graph_series[t],
                         dynamic_graph_series[t],
                         dynamic_graph_series[t][2])
        ax.set_title(f't= {t}', fontsize = 'xx-small')
        for item in ([ax.xaxis.label, ax.yaxis.label] +
                     ax.get_xticklabels() + ax.get_yticklabels()):
            item.set_fontsize(6)

    return plt


def plot_single_step(node_pos, graph_info, graph_info_next, changed_node):
    node_colors = get_node_color(graph_info_next[1])
    node_num, embedding_dimension = node_pos[0].shape
    pos = {}
    for i in range(node_num):
        pos[i] = node_pos[0][i, :]
    unchanged_nodes = list(set(range(node_num)) - set(changed_node))

    nodes_draw = nx.draw_networkx_nodes(graph_info[0],
                                        pos,
                                        nodelist = unchanged_nodes,
                                        node_color = [node_colors[p] for p in unchanged_nodes],
                                        node_size = 40,
                                        with_labels = False)
    nodes_draw.set_edgecolor('w')

    nodes_draw = nx.draw_networkx_nodes(graph_info[0],
                                        pos,
                                        nodelist = changed_node,
                                        node_color = 'r',
                                        node_size = 40,
                                        with_labels = False)
    if nodes_draw is not None:
        nodes_draw.set_edgecolor('k')


def get_node_color(node_community):
    # cnames = [item[0] for item in matplotlib.colors.cnames.items()]
    cnames = ['navy', 'yellow']
    node_colors = [cnames[int(c)] for c in node_community]
    return node_colors
