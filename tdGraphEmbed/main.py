import pandas as pd

from tdGraphEmbed.model import TdGraphEmbed
from tdGraphEmbed.temporal_graph import TemporalGraph

if __name__ == "__main__":
    path = r"data/facebook/facebook-wall.txt"
    df = pd.read_table(path, sep = '\t', header = None)
    df.columns = ['source', 'target', 'time']
    temporal_g = TemporalGraph(data = df, time_granularity = 'months')
    graphs = temporal_g.get_temporal_graphs(min_degree = 10)
    model = TdGraphEmbed(dataset_name = "facebook")
    documents = model.get_documents_from_graph(graphs)
    model.run_doc2vec(documents)
    graph_vectors = model.get_embeddings()
