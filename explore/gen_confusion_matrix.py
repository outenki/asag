import argparse
import matplotlib.pyplot as plt

import os

import numpy as np
from eva import conf_mat, plot_confusion_matrix

PARSER = argparse.ArgumentParser()
PARSER.add_argument('-p', '--feature', dest='target_path', type=str, required=True,
                    help='Target path, the parent dir of results of each question')
ARGS = PARSER.parse_args()

QUE_ID = list(os.listdir(ARGS.target_path))
print('qids: ', QUE_ID)
for qid in QUE_ID:
    path_q = '%s/%s' % (ARGS.target_path, qid)
    if not os.path.isdir(path_q):
        print('Skip %s' % path_q)
        continue
    print('Processing %s' % path_q)
    file_pred = '%s/pred.txt' % path_q
    if not os.path.exists(file_pred):
        print('Skip %s which doesn\'t exist' % file_pred)
        continue
    file_cm = '%s/cm.txt' % path_q
    mat_true_pred = np.loadtxt(file_pred, skiprows=0, delimiter='\t')[:, [2, 3]].astype(int)
    true, pred = mat_true_pred[:, 0], mat_true_pred[:, 1]
    cm = conf_mat(true, pred)
    print(cm)
    np.savetxt(file_cm, cm, fmt='%d', delimiter='\t')

    # draw fig
    classes = np.array(range(int(max(true)+1)))
    fig_cm = '%s/cm.png' % path_q
    fig_cm_nm = '%s/cm_nm.png' % path_q
    plot_confusion_matrix(cm, classes, fig_cm, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues)
    plot_confusion_matrix(cm, classes, fig_cm_nm, normalize=True, title='Confusion matrix', cmap=plt.cm.Blues)


# read pred and true
