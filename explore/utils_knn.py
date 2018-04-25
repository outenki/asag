'''
Run training_testing progress for ASAG.
@outenki
@2018.4
'''
import os
import pickle
from utils_asag import check_c_path, generate_training_test_data_f

import matplotlib.pyplot as plt
plt.switch_backend('agg')

import logging
from eva import RMSE, weightedF1, QWK, conf_mat, plot_confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import operator
import feature_format as F

LOGGER = logging.getLogger(__name__)

def run_knn(file_data, f_vocab, path_out, que_id, train_ratio=0.8, training_scale=0, n_neighbors=0, weights='distance', algorithm='auto'):
    '''
    Run SVR on feature-value data.
    1. Rread in data from file_data (TSV file)
    2. Generate training data and test data based on the train_ratio parameter
    3. Train SVR models with appointed traiing scale
    4. Test on test data
    '''
    LOGGER.info("processing %s" % que_id)
    path_out = '%s/%s' % (path_out, que_id)
    check_c_path(path_out)
    model_name = 'KNN'

    # read vocab pickle file
    LOGGER.info('Q{}: Reading vocab file ...'.format(que_id))
    with open(f_vocab, 'rb') as fv:
        vocab = pickle.load(fv)
    LOGGER.info('Q{}: Reading vocab file ... DONE'.format(que_id))
    items = sorted(vocab.items(), key=operator.itemgetter(1))
    tokens = list(zip(*items))[0]
    LOGGER.info('Q{}: Size of vocab : {}'.format(que_id, len(items)))
    with open('{}/tokens'.format(path_out), 'w') as ft:
        for t in tokens:
            ft.write('{}\n'.format(t))
    LOGGER.info('Q{}: Tokens :{}'.format(que_id, tokens[:10]))

    LOGGER.info('Q{}: Initialize training data and test data from {} ...'.format(que_id, file_data))
    print('to generate training test data: %s' % file_data)
    data_train, data_test = generate_training_test_data_f(file_data, train_ratio=train_ratio)
    print('to generate training test data: done')

    # get features and scores from training data
    scale = len(data_train)
    print('scale of data_train:', scale)
    if training_scale > 0:
        scale = training_scale
    X = np.array(list(map(lambda r:r.split(','), data_train[:,F.pos_fea]))).astype(float)[:scale]
    y = np.rint(data_train[:,F.pos_score][:scale].astype(float))

    LOGGER.info('Q{}:Initialize {} ...'.format(que_id, model_name))
    model = KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights, algorithm = algorithm)

    LOGGER.info('Q{}: Training {} ...'.format(que_id, model_name))
    LOGGER.info('Q{}: Training scale: {}'.format(que_id, len(X)))

    # Train
    model.fit(X, y)

    # get features from test data
    X = np.array(list(map(lambda r:r.split(','), data_test[:,F.pos_fea]))).astype(float)
    ans = data_test[:,F.pos_ans]
    LOGGER.info('Q{}: Answers[:3]: {}'.format(que_id, ans[:3]))

    # Test
    LOGGER.info('Q{}: Predicting test data ...'.format(que_id))
    pred = model.predict(X)
    score = data_test[:,F.pos_score].astype(float)
    abs_diff = np.abs(pred - score)

    # Probabilities
    LOGGER.info('Q{}: Computing probabilities ...'.format(que_id))
    labels = sorted(list(set(y.astype(str))))
    prob = model.predict_proba(X)

    # Nearest Neighbors
    LOGGER.info('Q{}: Collecting NN ...'.format(que_id))
    dist, ind = model.kneighbors(X, n_neighbors, True)
    dist = dist.astype(str)
    nn = []
    row, col = dist.shape
    for r in range(row):
        row_nn = []
        for c in range(col):
            idx = ind[r, c] 
            aid = data_train[idx, F.pos_aid]
            s = data_train[idx, F.pos_score]
            a = str(data_train[idx, F.pos_ans]).strip()
            d = dist[r, c]
            col_nn = '{}\t{}\t{}\t{}'.format(aid, s, d, a)
            row_nn.append(col_nn)
        nn.append(row_nn) 
    nn = np.array(nn)

    LOGGER.info('Q{}: Generating report ...'.format(que_id))
    aid_qid = data_test[:,[F.pos_aid, F.pos_qid]] 
    results = np.column_stack((aid_qid, score, ans, pred.astype(str), abs_diff.astype(str), prob, nn))

    LOGGER.info('Q{}: Output results ...'.format(que_id))
    f_pred = '%s/pred.txt' % path_out
    f_eval = '%s/eval.txt' % path_out
    LOGGER.info('Q{}: \tOutput pred to: {}'.format(que_id, f_pred))
    title = 'AnswerID\tQuetionID\tScore\tAnswer\tPredict\tAbsDiff\t%s\t%s\n' % ('\t'.join(labels), ('AID\tScore\tDistance\tNN\t' * n_neighbors).rstrip())
    np.savetxt(f_pred, results, fmt='%s', delimiter='\t', newline='\n', header=title, footer='', comments='# ')
    LOGGER.info('Q{}: \tOutput pred to: {} DONE!'.format(que_id, f_pred))
    
    # Evaluation with RMSE, QWK and wF1 score
    LOGGER.info('Q{}: \tOutput evaluation to: {}'.format(que_id, f_eval))
    score_float = data_test[:,2].astype(float)
    pred_float = pred.astype(float)

    LOGGER.info('Q{}: \tTrue labels: {}'.format(que_id, set(score)))
    LOGGER.info('Q{}: \tPredicted labels: {}'.format(que_id, set(pred)))

    # Generate confusion matrix
    LOGGER.info('Q{}: \tGenerating confusion matrix'.format(que_id))
    cm = conf_mat(score, pred)
    f_cm = '%s/cm.txt' % path_out
    np.savetxt(f_cm, cm, fmt='%d', delimiter='\t')
    classes = list(range(int(np.max(score))+1))
    save_path = '%s/cm.png' % path_out
    LOGGER.info('Q{}: \tConfusion_matrix: {}'.format(que_id, save_path))
    plot_confusion_matrix(cm, classes, save_path, normalize=False,
                          title='Confusion matrix', cmap=plt.cm.Blues)
    save_path = '%s/cm_nm.png' % path_out
    LOGGER.info('Q{}: \tNormalized confusion_matrix: {}'.format(que_id, save_path))
    plot_confusion_matrix(cm, classes, save_path, normalize=True, 
                          title='Confusion matrix', cmap=plt.cm.Blues)

    LOGGER.info('Q{}: \tEvaluating ...'.format(que_id))
    rmse = RMSE(score_float, pred_float)
    LOGGER.info('Q{}: \trmse: {}'.format(que_id, rmse))
    qwk = QWK(score, pred)
    LOGGER.info('Q{}: \tqwk: {}'.format(que_id, qwk))
    wf1 = weightedF1(score, pred)
    LOGGER.info('Q{}: \twf1: {}'.format(que_id, wf1))
    LOGGER.info('Q{}: QWK: {}, RMSE: {}, wF1: {}'.format(que_id, qwk, rmse, wf1))
    with open(f_eval, 'w') as fe:
        fe.write('RMSE\t {}\n'.format(rmse))
        fe.write('QWK\t {}\n'.format(qwk))
        fe.write('wF1\t {}\n'.format(wf1))
    LOGGER.info('Q{}: Done.'.format(que_id))

def run_knn_on_list(qids, feature_path, feature_name, vocab_name, path_out, train_ratio, training_scale, n_neighbors, weight, algorithm):
    for qid in qids:
        if not os.path.isdir('%s/%s' % (feature_path, qid)):
            LOGGER.info('Q{}/{} is not dir. Skip it.'.format(feature_path, qid))
            continue
        file_data = '%s/%s/%s' % (feature_path, qid, feature_name)
        file_vocab = '%s/%s/%s' % (feature_path, qid, vocab_name)
        run_knn(file_data = file_data, f_vocab = file_vocab, path_out = path_out, que_id = qid,
                train_ratio = train_ratio, training_scale = training_scale, n_neighbors=n_neighbors,
                weights=weight, algorithm=algorithm)
