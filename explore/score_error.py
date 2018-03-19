from config import *
from basic_util import read_confusion_data
from itertools import groupby
import numpy as np
if __name__ == '__main__':
    file_list = os.listdir(RESULTS_PATH + '/results')
    print(file_list)
    for f in file_list:
        result_path = RESULTS_PATH + '/results/' + f
        if f.startswith('.'):
            continue
        if not os.path.exists(result_path + '/result.txt'):
            continue
        print(f)
        pres, exps = read_confusion_data(result_path + '/result.txt')
        pres = np.array(pres)/2
        exps = np.ceil(np.array(exps)/2)

        errs = list(abs(exps-pres))
        scos = list(exps)
        score_error = sorted(zip(scos, errs))
        score_mean_error = []
        for s, se in groupby(score_error, key=lambda x:x[0]):
            se = list(se)
            se = np.array(list(se))
            es = list(zip(*se))[1]
            print(s, es)
            e = np.mean(es)
            score_mean_error.append((s, e))


        with open(result_path + '/score_error.txt', 'w') as f_res:
            for s, e in score_mean_error:
                f_res.write('{}\t{}\n'.format(s, e))
