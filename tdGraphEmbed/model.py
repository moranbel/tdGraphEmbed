import json
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from node2vec import Node2Vec


class TdGraphEmbed():
    def __init__(self, dataset_name):
        with open("tdGraphEmbed/config.json", 'rb') as f:
            config = json.load(f)
        self.walk_length = config["walk_length"]
        self.num_walks = config["num_walks"]
        self.p = config["p"]
        self.q = config["q"]
        self.graph_vector_size = config["graph_vector_size"]
        self.window = config["window"]
        self.epochs = config["epochs"]
        self.dataset_name =dataset_name

    def get_documents_from_graph(self, graphs):
        documents = []
        for time in graphs.keys():
            node2vec = Node2Vec(graphs[time], walk_length = self.walk_length, num_walks= self.num_walks,
                                p = self.p, q = self.q, weight_key = 'weight')
            walks = node2vec.walks
            walks = [[str(word) for word in walk] for walk in walks]
            documents.append([TaggedDocument(doc, [time]) for doc in walks])
        documents = sum(documents, [])
        return documents

    def run_doc2vec(self,documents):
        alpha = 0.025
        model = Doc2Vec(vector_size = self.graph_vector_size, alpha = alpha, window = self.window, min_alpha = 0.025,
                        min_count = 1)
        model.build_vocab(documents)
        for epoch in range(self.epochs):
            print('iteration {0}'.format(epoch))
            model.train(documents, total_examples = model.corpus_count, epochs = model.iter)
            model.alpha -= 0.002
            model.min_alpha = model.alpha
        model.save(f"trained_models/{self.dataset_name}.model")
        print("Model Saved")
