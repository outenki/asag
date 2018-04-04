from config import *
from math import ceil
import numpy as np
from basic_util import clean_text, token_strip, distance_of_couples, read_w2v
from itertools import groupby

BINS = 10
def generate_feature_for_distance(text, w2v, dim):
    tokens = token_strip(clean_text(text), NLP)
    fea_tokens = np.array([w2v.get(t, np.array([0]*dim)) for t in tokens])
    return fea_tokens.mean(axis=0)

def distance_answers_of_question(q_id, w2v, dim):
    score_dst = [0] * 6
    score_std = [0] * 6
    score_flag = [0] * 6
    with open(RAW_PATH_STU + '/' + q_id, 'r', errors='ignore') as f_ans,\
        open(DATA_PATH+'/scores/' + q_id + '/ave', 'r') as f_score:
        answers = f_ans.readlines()
        scores = f_score.readlines()
        scores = list(map(float, scores))
        scores = list(map(ceil, scores))
        features = [generate_feature_for_distance(a, w2v, dim) for a in answers]
        score_feature = list(zip(scores, features))
        score_feature.sort(key=lambda x:x[0])
        for s, item in groupby(score_feature, key=lambda x:x[0]):
            feas = list(zip(*item))[1]
            dists, mean, std, flag = distance_of_couples(feas, 'cos')
            score_dst[s] = mean
            score_std[s] = std
            score_flag[s] = flag
    return score_dst, score_std, score_flag

if __name__ == '__main__':
    file_list = os.listdir(RAW_PATH_STU)
    print('reading w2v... ')
    w2v, dim = read_w2v(W2V_PATH+'/'+W2V_FILE)
    # w2v, dim = {'a':np.array([1,2,3])}, 3
    print('done')
    print(file_list)
    dsts = []
    stds = []
    with open(RESULTS_PATH + '/score_distance_cos.txt', 'w') as f_sd:
        for q_id in file_list:
            print(q_id)
            dst, std, flag = distance_answers_of_question(q_id, w2v, dim)
            dst, std, flag = list(map(str, dst)), list(map(str,std)), list(map(str,flag))
            f_sd.write("{}\t{}\t{}\t{}\n".format(q_id, '\t'.join(dst), '\t'.join(std), '\t'.join(flag)))