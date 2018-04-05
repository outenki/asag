import numpy as np
from sklearn.metrics import cohen_kappa_score, f1_score

def RMSE(score, pred):
    rmse = np.sqrt(((pred - score) ** 2).mean())
    return rmse

def weightedF1(score, pred):
    wf1 = f1_score(score, pred, labels=None, pos_label=1, average='weighted', sample_weight=None)
    return wf1

def QWK(score, pred):
    qwk = cohen_kappa_score(score, pred, labels=None, weights='quadratic', sample_weight=None)
    return qwk
