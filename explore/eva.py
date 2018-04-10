import numpy as np
from sklearn.metrics import cohen_kappa_score, f1_score, confusion_matrix
import matplotlib
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import itertools



def RMSE(score, pred):
    '''
    return RMSE.
    '''
    rmse = np.sqrt(((pred - score) ** 2).mean())
    return rmse

def weightedF1(score, pred):
    '''
    return weighted F1 score
    '''
    wf1 = f1_score(score, pred, labels=None, pos_label=1, average='weighted', sample_weight=None)
    return wf1

def QWK(score, pred):
    '''
    Return QWK score
    '''
    qwk = cohen_kappa_score(score, pred, labels=None, weights='quadratic', sample_weight=None)
    return qwk

def conf_mat(score, pred):
    '''
    return confusion_matrix.
      T\P | label0 | label1|...
    label0| _______|_______|...
    label1| _______|_______|...
    label2| _______|_______|...
      ... |
    '''
    return confusion_matrix(score, pred)

def plot_confusion_matrix(cm, classes, save_path, normalize=False,
                          title='Confusion matrix', cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.clf()
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    # plt.plot(cm)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(save_path,  bbox_inches='tight')
