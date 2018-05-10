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
from sklearn.svm import SVR, SVC
import numpy as np
import operator

LOGGER = logging.getLogger(__name__)

def run_svm(kernel, file_data, f_vocab, path_out, que_id, train_ratio=0.8, rint=True, training_scale=0, epsilon=0.1, penalty=1.0, normalize=False, classify=False):
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
    model_name = 'SVC' if classify else 'SVR'

    # read vocab pickle file
    LOGGER.info('Q{}: Reading vocab file ...'.format(que_id))
    tokens = ['num_char', 'num_words', 'num_commas', 'num_apost', 'num_ending_punct', 'len_word_avg', 'num_bad_pos', 'prop_bad_pos', 'num_words_in_prompt', 'prop_words_in_prompt', 'num_synonym', 'prop_synonym', 'count_1_2_gram_unstemmed', 'count_1_2_gram_stemmed']

    if os.path.exists(f_vocab):
        try:
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
        except:
            print("Failed to read tokens from vocab file!")
            LOGGER.info('Q{}: Failed to read tokens from vocab file!'.format(que_id))

    LOGGER.info('Q{}: Initialize training data and test data from {} ...'.format(que_id, file_data))
    print('to generate training test data: %s' % file_data)
    data_train, data_test = generate_training_test_data_f(file_data, train_ratio=train_ratio)
    print('to generate training test data: done')

    # get features and scores from training data
    scale = len(data_train)
    print('scale of data_train:', scale)
    if training_scale > 0:
        scale = training_scale
    X = np.array(list(map(lambda r:r.split(','), data_train[:,3]))).astype(float)[:scale]
    print('X:', X[2])
    y = data_train[:,2][:scale].astype(float)
    print('Y:', y[2])

    LOGGER.info('Q{}:Initialize {} ...'.format(que_id, model_name))
    if classify:
        y = np.rint(y)
        y[y>=1] = 1 # convert the labels to 0/1
        model = SVC(C=penalty, kernel=kernel)
    else:
        model = SVR(C=penalty, epsilon=epsilon, kernel=kernel)

    LOGGER.info('Q{}: Training {} ...'.format(que_id, model_name))
    LOGGER.info('Q{}: Training scale: {}'.format(que_id, len(X)))
    print('p:', penalty)
    print('e:', epsilon)
    print('k:', kernel)

    # Train
    model.fit(X, y)

    print(X.shape)
    print(y.shape)
    # get features from test data
    LOGGER.info('Q{}: Predicting test data ...'.format(que_id))
    X = np.array(list(map(lambda r:r.split(','), data_test[:,3]))).astype(float)
    ans = data_test[:,4]
    LOGGER.info('Q{}: Answers[:3]: {}'.format(que_id, ans[:3]))

    # Test
    y = model.predict(X)
    score = data_test[:,2].astype(float)
    if classify:
        score[score>0] = 1
    if normalize:
        v_max, v_min = max(score), min(score)
        y = y.clip(v_min, v_max)
    abs_diff = np.abs(y - score)
    results = np.column_stack((data_test[:,[0,1]],score, ans, y.astype(str), abs_diff.astype(str)))

    # read weights of features
    weight_token = model.coef_
    LOGGER.info('Q{}: Weights :{}'.format(que_id, weight_token.astype(str)))
    item_weight = list(zip(tokens, weight_token[0]))

    LOGGER.info('Q{}: Output results ...'.format(que_id))
    f_pred = '%s/pred.txt' % path_out
    f_weight = '%s/weight.txt' % path_out
    f_eval = '%s/eval.txt' % path_out
    LOGGER.info('Q{}: \tOutput pred to: {}'.format(que_id, f_pred))
    title = 'AnswerID\tQuetionID\tScore\tAnswer\tPredict\tAbsDiff\n'
    np.savetxt(f_pred, results, fmt='%s', delimiter='\t', newline='\n', header=title, footer='', comments='# ')
    LOGGER.info('Q{}: \tOutput pred to: {} DONE!'.format(que_id, f_pred))
    LOGGER.info('Q{}: \tOutput weights to: {}'.format(que_id, f_weight))
    title = 'Token\tWeight\n'
    np.savetxt(f_weight, item_weight, fmt='%s', delimiter='\t', newline='\n', header=title, footer='', comments='# ')
    LOGGER.info('Q{}: \tOutput weights to: {} DONE!'.format(que_id, f_weight))
    
    # Evaluation with RMSE, QWK and wF1 score
    LOGGER.info('Q{}: \tOutput evaluation to: {}'.format(que_id, f_eval))
    score_float = data_test[:,2].astype(float)
    pred_float = y
    if rint:
        score_int = np.rint(score_float)
        if classify:
            score_int[score_int>0] = 1  # convert the gold label for evaluation of SVC
        pred_int = np.rint(y)
    else:
        score_int = score_float.astype(int)
        pred_int = y.astype(int)

    LOGGER.info('Q{}: \tTrue labels: {}'.format(que_id, set(score_int)))
    LOGGER.info('Q{}: \tPredicted labels: {}'.format(que_id, set(pred_int)))

    # Generate confusion matrix
    LOGGER.info('Q{}: \tGenerating confusion matrix'.format(que_id))
    cm = conf_mat(score_int, pred_int)
    f_cm = '%s/cm.txt' % path_out
    np.savetxt(f_cm, cm, fmt='%d', delimiter='\t')
    classes = list(range(int(np.max(score_int))+1))
    save_path = '%s/cm.png' % path_out
    LOGGER.info('Q{}: \tConfusion_matrix: {}'.format(que_id, save_path))
    plot_confusion_matrix(cm, classes, save_path, normalize=False,
                          title='Confusion matrix', cmap=plt.cm.Blues)
    save_path = '%s/cm_nm.png' % path_out
    LOGGER.info('Q{}: \tNormalized confusion_matrix: {}'.format(que_id, save_path))
    plot_confusion_matrix(cm, classes, save_path, normalize=True, 
                          title='Confusion matrix', cmap=plt.cm.Blues)



    rmse = RMSE(score_float, pred_float)
    LOGGER.info('Q{}: \trmse: {}'.format(que_id, rmse))
    qwk = QWK(score_int, pred_int)
    LOGGER.info('Q{}: \tqwk: {}'.format(que_id, qwk))
    wf1 = weightedF1(score_int, pred_int)
    LOGGER.info('Q{}: \twf1: {}'.format(que_id, wf1))
    LOGGER.info('Q{}: QWK: {}, RMSE: {}, wF1: {}'.format(que_id, qwk, rmse, wf1))
    with open(f_eval, 'w') as fe:
        fe.write('RMSE\t {}\n'.format(rmse))
        fe.write('QWK\t {}\n'.format(qwk))
        fe.write('wF1\t {}\n'.format(wf1))
    LOGGER.info('Q{}: Done.'.format(que_id))


# def run_svc_svr(kernel, file_data, f_vocab, path_out, que_id, train_ratio=0.8, rint=True, training_scale=0, epsilon=0.1, penalty=1.0, normalize=False):
def run_svr_on_list(qids, feature_path, feature_name, vocab_name, kernel, path_out, train_ratio,
        rint, training_scale, epsilon, penalty, normalize, classify):
    for qid in qids:
        if not os.path.isdir('%s/%s' % (feature_path, qid)):
            LOGGER.info('Q{}/{} is not dir. Skip it.'.format(feature_path, qid))
            continue
        file_data = '%s/%s/%s' % (feature_path, qid, feature_name)
        file_vocab = '%s/%s/%s' % (feature_path, qid, vocab_name)
        # def run_svr(kernel, file_data, f_vocab, path_out, que_id, train_ratio=0.8, 
        #               rint=True, training_scale=0, epsilon=0.1, penalty=1.0, normalize=False):
        # def run_svr(kernel, file_data, f_vocab, path_out, que_id, train_ratio=0.8, rint=True, 
        #               training_scale=0, epsilon=0.1, penalty=1.0, normalize=False):
        run_svm(kernel, file_data, file_vocab, path_out, qid, train_ratio, rint, training_scale, epsilon, penalty, normalize, classify)
        
