from config import *
from basic_util import calculate_F1
from math import sqrt
from sklearn.metrics import mean_squared_error
import numpy as np
def count_error(fn):
    # with open(fn, 'r') as fe, open('error.count.txt', 'w') as ec,\
    #         open('error_abs.count.txt', 'w') as eac,\
    #         open('error_round.count.txt', 'w') as erc:
    result_path = RESULTS_PATH + '/results/' + fn
    wf1 = calculate_F1(result_path + '/result.txt')
    with open(result_path + '/result.txt' , 'r', errors='ignore') as fe, \
            open(result_path + '/errors.txt', 'w') as fo:
        svr_all = map(lambda line: line.split('\t'), fe.readlines())
        _, score, truth, error, error_abs, *_ = zip(*svr_all)
        count = len(error_abs)
        score = map(float, score)
        truth = map(float, truth)
        error = map(float, error)
        error_abs = map(float, error_abs)
        # error_round = map(float, error_round)
        rms = sqrt(mean_squared_error(list(score), list(truth)))

        def count_hist(error_hist, echo=False):
            k = list(np.arange(-4.5, 5.1, 0.5))
            v = [0] * 20
            hist = dict(zip(k, v))
            for e in error_hist:
                for k in hist:
                    if e <= k:
                        hist[k] += 1
                        # break
            return hist
            # for k,v in hist.items():
            #     f.write("{}\t{}\n".format(k, v))
            # f.write('{}\t{}\n'.format(count, sum(hist.values())))

        d_error = count_hist(error)
        d_error_abs = count_hist(error_abs)
        # d_error_round = count_hist(error_round)
        errors = zip(d_error.keys(), d_error.values(), d_error_abs.values())
        # errors = d_error_abs.values()
        # for item in d_error_abs.items():
        #     fo.write('{}\t{}\n'.format(item[0], item[1]))
        errors = map(lambda line: '\t'.join(map(str, line)) + '\n', errors)
        fo.writelines(errors)
        fo.write('{}\t{}\t{}\n'.format(count, sum(d_error.values()), sum(d_error_abs.values())))
        fo.write('RMSE\t{}\n'.format(rms))
        fo.write('wF1: \t{}\n'.format(wf1))

if __name__ == '__main__':
    file_list = os.listdir(RESULTS_PATH + '/results')
    print(file_list)
    for f in file_list:
        if f.startswith('.'):
            continue
        if not os.path.exists(RESULTS_PATH + '/results/' + f + '/result.txt'):
            continue
        # try:
        print(f)
            # if f.startswith('gb.kmenas'):
            #     print(f)
        count_error(f)
        # except:
        #     print("FAILED!")
