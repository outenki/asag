import nltk
import os
import string
import time
import numpy as np
from sklearn import neighbors
from config import *

# # Paths
# SCRIPT_PATH = os.path.dirname(os.path.abspath(__file__))
# # DATA_PATH = SCRIPT_PATH + "/../data/ShortAnswerGrading_v2.0/data"
# DATA_PATH = SCRIPT_PATH + "/../data/sciEntsBank/train"
# # DATA_PATH = SCRIPT_PATH + "/../data/sciEntsBank/test-unseen-questions"
# # DATA_PATH = SCRIPT_PATH + "/../data/sciEntsBank/test-unseen-answers"
# # DATA_PATH = SCRIPT_PATH + "/../data/sciEntsBank/test-unseen-domains"
#
# # RESULTS_PATH = SCRIPT_PATH + "/../results_sag"
# RESULTS_PATH = SCRIPT_PATH + "/../results_semi_train"
# # RESULTS_PATH = SCRIPT_PATH + "/../results_semi_uq"
# # RESULTS_PATH = SCRIPT_PATH + "/../results_semi_ua"
# # RESULTS_PATH = SCRIPT_PATH + "/../results_semi_ud"
# RAW_PATH = DATA_PATH + "/raw"
# RAW_PATH_STU = DATA_PATH + "/raw/ans_stu"


def cur_time():
    return time.strftime('%Y%m%d%H%M%S', time.localtime(time.time()))
def get_tokens(text):
    lower = text.lower()
    remove_punctuation_map = dict((ord(char), None) for char in string.punctuation)
    no_punctuation = lower.translate(remove_punctuation_map)
    tokens = nltk.word_tokenize(no_punctuation)
    return tokens
def read_tokens_answer(answer):
    # Answers are starts with answer id
    # Remove answer id first before extract tokens
    answer = answer[answer.find(' ') + 1:]
    return set(get_tokens(answer))

def read_tokens_answers(que_id, ref = True):
    '''
    Read answers under one question and return a tuple of tokens.
    The length of tuple of tokens is used as the length of BOW.
    :param fn_question:
    question id, it should be same to the file name.
    :return:
    '''
    token_set = set()
    if ref:
        # read reference answer
        with open(RAW_PATH + "/answers", errors="ignore") as f_ref:
            for answer in f_ref.readlines():
                if answer.startswith(que_id):
                    token_set = token_set.union(read_tokens_answer(answer))
                    break

    # read student answers
    with open(RAW_PATH_STU + "/" + que_id, "r", errors="ignore") as f_ans_raw:
        try:
            for answer in f_ans_raw.readlines():
                token_set = token_set.union(read_tokens_answer(answer))
        except:
            print("error:", answer)
    assert token_set
    return token_set

def generate_bow_features(ref = True):
    for que_id in os.listdir(RAW_PATH_STU):
        print(que_id)
        # if que_id in ['questions', 'answers', 'count']:
            # skip this files
            # continue

        # generate bow features
        tokens_all = tuple(read_tokens_answers(que_id, ref))

        with open(RESULTS_PATH+"/features_bow/" + que_id, "wt", encoding='utf-8', errors="ignore") as f_fea,\
            open(RAW_PATH_STU + "/" + que_id, "r", encoding='utf-8', errors="ignore") as f_ans:
            for answer in f_ans.readlines():
                tokens_answer = read_tokens_answer(answer)
                bow = [1] * len(tokens_all)
                for i in range(len(tokens_all)):
                    # print(tokens_all[i])
                    bow[i] = 1 if tokens_all[i] in tokens_answer else 0
                print(*bow, file=f_fea, sep=',')
                # print(bow)

def read_training_data(feature_path):
    '''
    Read features and labels for training. This function will read all the features
    and scores of each answer for each question.
    :param feature_path: path/of/feature/files/.
    :return: A dict with structure as below
    # data_dic = {
    #   '1.1':{
    #       'truth': array(n*1)
    #       'features': array(n*30)
    #       'diff': array(n*30)
    #   }
    # }
    '''
    scores_truth_path = DATA_PATH + '/scores/'
    que_ids = os.listdir(feature_path)
    data_dict = {}
    for que_id in que_ids:
        data_dict[que_id] = {}
        with open(feature_path + que_id, 'r') as ff, \
                open(scores_truth_path + que_id + '/ave') as fs, \
                open(RAW_PATH + "/answers", "r", errors="ignore") as f_raw_r, \
                open(RAW_PATH + "/questions", "r", errors="ignore") as f_raw_q, \
                open(RAW_PATH_STU + "/" + que_id, "r", errors="ignore") as f_raw_s, \
                open(scores_truth_path + que_id + '/diff') as fd:
            scores_truth = np.array(list(map(np.float64, fs.readlines())))
            diff = np.array(list(map(np.float64, fd.readlines())))
            features = list(map(lambda s: s.split(','), ff.readlines()))
            features = np.array(list(map(lambda l: list(map(np.float64, l)), features)))
            raw_r, raw_q, raw_s = '', '', []
            for s in f_raw_q.readlines():
                if s.startswith(que_id):
                    raw_q = s
                    break

            for s in f_raw_r.readlines():
                if s.startswith(que_id):
                    raw_r = s
                    break

            raw_s = np.array(list(map(lambda s:s.strip(), f_raw_s.readlines())))

            data_dict[que_id]['scores_truth'] = scores_truth
            data_dict[que_id]['features'] = features
            data_dict[que_id]['diff'] = diff
            data_dict[que_id]['question'] = raw_q.strip()
            data_dict[que_id]['ans_ref'] = raw_r.strip()
            data_dict[que_id]['ans_stu'] = raw_s
    return data_dict

def run_knn_question_wise(fn, feature_type, reliable, n_neighbors, weight, p=2, training_scale = 0):
    '''
    Run knn algorithm using all other answers under the same question as training data.
    :param fn: File name to save the results.
    :param feature_type: For now it may be one of 'bow', 'g', 'b' or 'gb'.
    :param reliable:
        When `reliable` is True, answers whose score is with diff over 2 will
        be removed from training data
    :param n_neighbors: Parameter for KNN. The number neighbors.
    :param weight:
        Weight function used in prediction. Possible values:
        ‘uniform’ : uniform weights. All points in each neighborhood are weighted equally.
        ‘distance’ : weight points by the inverse of their distance. in this case,
            closer neighbors of a query point will have a greater influence than neighbors
            which are further away.
        [callable] : a user-defined function which accepts an array of distances,
            and returns an array of the same shape containing the weights.
    :return: None
    '''
    feature_path = RESULTS_PATH + '/features_{}/'.format(feature_type)
    data_dict = read_training_data(feature_path)
    # fn = fn +  '.' + feature_type + '.' +  cur_time()
    fn = '{}.{}.{}.{}.{}.{}.{}.{}'.format(feature_type, fn, n_neighbors, p,
                                          'reliable' if reliable else 'unreliable', weight, training_scale, cur_time())
    result_path = RESULTS_PATH + '/results/' + fn
    if not os.path.exists(result_path):
        os.mkdir(result_path)

    with open(result_path + '/result', 'w') as fr:
        for que_id in data_dict:
            for i in range(len(data_dict[que_id]['scores_truth'])):
                # i refers an answer
                # Train knn for each answer with all other answers

                # remove unreliable training data
                array_filter = data_dict[que_id]['diff'] < 3 if reliable else np.array(
                    [True] * len(data_dict[que_id]['diff']))
                # remove current answer (to be predicted)
                array_filter[i] = False

                scores_truth = data_dict[que_id]['scores_truth'][array_filter]
                features = data_dict[que_id]['features'][array_filter]

                X = features
                Y = scores_truth
                Y = (Y * 2).astype(int)
                score_truth_i = data_dict[que_id]['scores_truth'][i]
                feature_i = data_dict[que_id]['features'][i:i + 1]
                if n_neighbors > len(X):
                    n_neighbors = len(X)
                clf = neighbors.KNeighborsClassifier(n_neighbors, weights=weight)
                clf.fit(X, Y)

                # predict
                score = clf.predict(feature_i) / 2
                error = score_truth_i - score[0]
                error_abs = abs(error)
                error_round = round(error_abs)
                question = data_dict[que_id]["question"]
                ans_ref = data_dict[que_id]["ans_ref"]
                ans_stu = data_dict[que_id]["ans_stu"][i]
                print('score of {}.{}: {}: {}: {}: {}: {}: {}: {}: {}'.format(que_id, i + 1, score[0], score_truth_i, error,
                                                                  error_abs, error_round, question, ans_ref, ans_stu))
                print('score of {}.{}: {}: {}: {}: {}: {}: {}: {}: {}'.format(que_id, i + 1, score[0], score_truth_i, error,
                                                                  error_abs, error_round, question, ans_ref, ans_stu), file=fr)

if __name__ == '__main__':
    # generate_bow_features()
    # for k in [5, 10, 20, 30]:
    run_knn_question_wise('knn.qwise', 'bow', True, n_neighbors=5, weight='uniform', p=2, training_scale=0)
    run_knn_question_wise('knn.qwise', 'bow', True, n_neighbors=5, weight='distance', p=2, training_scale=0)

