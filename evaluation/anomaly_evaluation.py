from scipy import spatial
from scipy.stats import spearmanr
import holoviews as hv
from gensim.models.doc2vec import Doc2Vec
import numpy as np
import pandas as pd
import matplotlib
from sklearn.manifold import TSNE

matplotlib.use('tkagg')
import datetime

hv.extension('bokeh')


def tsne_visualize(path_model_results, anoms_dates = None):
    '''

    :param path_model_results: path to tdGraphEmbed model
    :param anoms_dates: list of date times
    :return:
    '''
    model = Doc2Vec.load(path_model_results)
    tsne = TSNE(n_components = 2, verbose = 1, perplexity = 10, n_iter = 500)
    days = [datetime.datetime.strftime(day, '%d-%m-%Y') for day in sorted(model.docvecs.doctags.keys())]
    doc_vecs = model.docvecs.doctag_syn0
    doc_vecs = doc_vecs[np.argsort([model.docvecs.index_to_doctag(i) for i in range(0, doc_vecs.shape[0])])]
    tsne_results = tsne.fit_transform(doc_vecs)
    tsne_df = pd.DataFrame({'x': tsne_results[:, 0], 'y': tsne_results[:, 1], 'day': days})

    tsne_scat_all = hv.Scatter(data = tsne_df, xdims = 'x', vdims = ['y', 'day']).opts(width = 1000, height = 600,
                                                                                       size = 20, tools = ['hover'])
    if anoms_dates is not None:
        anoms_dates = [datetime.datetime.strftime(day, '%d-%m-%Y') for day in anoms_dates]
        tsne_scat_anoms = hv.Scatter(data = tsne_df[tsne_df['day'].isin(anoms_dates)], xdims = 'x',
                                     vdims = ['y', 'day']).opts(width = 1000, height = 600, size = 20,
                                                                tools = ['hover'], fill_color = "red")
        tsne_scat_all = tsne_scat_all * tsne_scat_anoms
    renderer = hv.renderer('bokeh')
    renderer.save(tsne_scat_all, 'evaluation/g2vec_tsne')


def evaluate_anomalies(embs_vectors, days, anoms, google = None):
    '''

    :param embs_vectors: temporal graph vectors for each time step. numpy array of shape (number of timesteps, graph vector dimension size)
    :param days: list of datetime of all graph's times
    :param anoms: list of anomalies times
    :param google: google trend data in case we have
    :return:
    '''
    measures_df = pd.DataFrame(columns = ['K', 'Recall', 'Precision'])
    ks = [5, 10]
    dist = np.array([spatial.distance.cosine(embs_vectors[i + 1], embs_vectors[i])
                     for i in range(1, len(embs_vectors) - 1)])
    for k in ks:
        top_k = (-dist).argsort()[:k] + 1
        top_k = np.array(days)[top_k]
        tp = np.sum([1 if anom in top_k else 0 for anom in anoms])
        recall_val = tp / len(anoms)
        precision_val = tp / k
        measures_df = measures_df.append({'K': k, 'Recall': recall_val, 'Precision': precision_val},
                                         ignore_index = True)

        if google:
            corr, pval = spearmanr(dist, google.squeeze()[:-1])
            print(f'{corr}, {pval}')
    print(measures_df)


if __name__ == "__main__":
    model_path = ""
    model = Doc2Vec.load(model_path)
    doc_vecs = model.docvecs.doctag_syn0
    doc_vecs = doc_vecs[np.argsort([model.docvecs.index_to_doctag(i) for i in range(0, doc_vecs.shape[0])])]

    anoms = [datetime.date(2019, 3, 17), datetime.date(2019, 3, 31),
             datetime.date(2019, 4, 14), datetime.date(2019, 4, 28)]

    days = sorted(model.docvecs.doctags.keys())
    days = [day.date() for day in days]
    evaluate_anomalies(doc_vecs, days = days, anoms = anoms)
    tsne_visualize(model_path, anoms)
