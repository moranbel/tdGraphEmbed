import scipy.stats as stats
import numpy as np

def precision_at_k(predicted, real, k):
    precision=[]
    for t,graph in predicted.iterrows():
        top_predicted = graph.sort_values().index[1:k+1]
        top_real = real.loc[t].sort_values(ascending = False).index[1:k+1]
        precision.append(len(set(top_predicted).intersection(set(top_real)))/k)
    return np.mean(precision)

def spearman_ranking(predicted, gt):
    taus = []
    for t, graph in predicted.iterrows():
        predicted_rank = np.argsort(graph.values)
        real_rank = np.argsort(gt.loc[t].values)[::-1]
        tau, p_value = stats.spearmanr(predicted_rank, real_rank)
        taus.append(tau)
    return np.mean(taus)

def kendalltau_ranking(predicted, gt):
    taus=[]
    for t, graph in predicted.iterrows():
        predicted_rank = np.argsort(graph.values)
        real_rank = np.argsort(gt.loc[t].values)[::-1]
        tau, p_value = stats.kendalltau(predicted_rank, real_rank)
        taus.append(tau)
    return np.mean(taus)