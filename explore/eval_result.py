'''
Evaluate results with QWK, RMSE, wF1 and confusion amtrix
@outenki
@2018.5
'''

import argparse
import numpy as np
from utils_eval import RMSE, weightedF1, QWK, conf_mat, plot_confusion_matrix
import matplotlib.pyplot as plt
plt.switch_backend('agg')

parser = argparse.ArgumentParser()
parser.add_argument('-f', '--input_file_name', dest='filename', type=str, required=True, help='Name of the result file.')
parser.add_argument('-t', '--lines_title', dest='skip', type=int, required=True, help='Number of lines of titles. Will skip them when reading in the results data')
parser.add_argument('-p', '--input_path', dest='pathname', type=str, required=True, help='The path to result file.')
parser.add_argument('-sc', '--column_of_score', dest='column_s', type=int, required=True, help='Column of golden score. Starts from 0')
parser.add_argument('-pr', '--column_of_predict', dest='column_p', type=int, required=True, help='Column of predicted score')
args = parser.parse_args()

print('loading results...')
print('file:', '%s/%s' % (args.pathname, args.filename))
print('usecols:', (args.column_s, args.column_p))
results = np.loadtxt('%s/%s' % (args.pathname, args.filename), skiprows=args.skip, dtype=str, delimiter='\t', usecols=(args.column_s, args.column_p))
# results = np.loadtxt('%s/%s' % (args.pathname, args.filename), skiprows=args.skip, dtype=str, delimiter='\t')
print(results.shape)
print(results)

results_float = results.astype(float)
scores_float = results_float[:, 0]
preds_float = results_float[:, 1]

results_int = np.rint(results_float).astype(int)
scores_int = results_int[:, 0]
preds_int = results_int[:, 1]

cm = conf_mat(scores_int, preds_int)
f_cm = "%s/cm.txt" % args.pathname
np.savetxt(f_cm, cm, fmt='%d', delimiter='\t')
classes = list(range(int(np.max(scores_int))+1))
plot_confusion_matrix(cm, classes, args.pathname, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues)

rmse = RMSE(scores_float, preds_float)
qwk = QWK(scores_int, preds_int)
wf1 = weightedF1(scores_int, preds_int)
f_eva = '%s/eval.txt' % args.pathname
with open(f_eva, 'w') as fe:
    fe.write('RMSE\t {}\n'.format(rmse))
    fe.write('QWK\t {}\n'.format(qwk))
    fe.write('wF1\t {}\n'.format(wf1))
