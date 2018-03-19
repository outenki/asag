from config import *
from math import ceil, floor
import numpy as np

def get_error_idx(err_abs):
    if err_abs == 0:
        return 0
    else:
        for e in range(1, SCORE_LEVELS):
            if err_abs <= e:
                return  e

def count_confidence_error(fn):
    # with open(fn, 'r') as fe, open('error.count.txt', 'w') as ec,\
    #         open('error_abs.count.txt', 'w') as eac,\
    #         open('error_round.count.txt', 'w') as erc:
    result_path = RESULTS_PATH + '/results/' + fn
    conf_err_count = []
    len_conf_err = ceil(1/CONFIDENCE_STEP)
    for i in range(len_conf_err):
        conf_err_count.append([0]*SCORE_LEVELS)


    with open(result_path + '/result.txt' , 'r') as f_res, \
            open(result_path + '/result_confidence.txt', 'w') as f_resc, \
            open(result_path + '/error_confidence_dstr.txt', 'w') as f_count, \
            open(result_path + '/error_confidence_dstr_per.txt', 'w') as f_count_per, \
            open(result_path + '/error_confidence_flat.txt', 'w') as f_count_flat, \
            open(result_path + '/error_confidence_flat_per.txt', 'w') as f_count_flat_per:
        results = f_res.readlines()
        distance = np.array(list(map(lambda x:float(x.strip().split('\t')[11]), results)))

        # normalize distance to [0~1]
        max_distance = max(distance)
        distance = distance / max_distance
        confidence = 1 - distance

        err_abs_list = []
        for i in range(len(results)):
            # add the confidence info to a new results file.
            result_confidence = '{}\t{}\n'.format(results[i].strip(), confidence[i])
            f_resc.write(result_confidence)

            idx_conf = floor(confidence[i] / CONFIDENCE_STEP)
            if idx_conf >= len_conf_err:
                idx_conf -= 1

            err_abs = float(results[i].split('\t')[4])
            err_abs_list.append(err_abs)
            idx_err = get_error_idx(err_abs)

            conf_err_count[idx_conf][idx_err] += 1

        conf_level = 0
        for line in conf_err_count:
            conf_level += CONFIDENCE_STEP
            num_ans = sum(line)
            f_count.write('{:.2f}\t{}\t{}\n'.format(conf_level, num_ans, '\t'.join(map(str, line))))
            f_count_per.write('{:.2f}\t{:.3f}\t{}\n'.format(conf_level, num_ans / len(results), '\t'.join(map(lambda x:format(x/num_ans if num_ans > 0 else 0, '0.3f'), line) )))

        conf_err_abs_list = sorted(zip(confidence, err_abs_list))
        conf_err_count = []
        conf_err_value = []
        for i in range(len_conf_err):
            conf_err_count.append([0] * SCORE_LEVELS)
            conf_err_value.append([])
        flat_step = len(conf_err_abs_list) / len_conf_err

        for i in range(len(conf_err_abs_list)):
            idx_conf = int(i / flat_step)
            idx_err = get_error_idx(conf_err_abs_list[i][1])
            conf_err_count[idx_conf][idx_err] += 1
            conf_err_value[idx_conf].append(conf_err_abs_list[i][1])

        print(conf_err_count)
        for i in range(0, len(conf_err_count)):
            line = conf_err_count[i]
            conf_level = int((i) * flat_step)
            print(conf_level)
            # print(conf_level)
            cur_conf = conf_err_abs_list[conf_level][0]
            num_ans = sum(line)
            mean_error = 0 if len(conf_err_value[i]) == 0 else sum(conf_err_value[i])/len(conf_err_value[i])
            f_count_flat.write('{:.1f}\t{}\t{}\t{}\n'.format(cur_conf, num_ans, '\t'.join(map(str, line)),  mean_error))
            f_count_flat_per.write('{:.3f}\t{:.2f}\t{}\t{}\n'.format(cur_conf, num_ans / len(results),
                                                                     '\t'.join(map(lambda x:format(x/num_ans if num_ans > 0 else 0, '0.3f'), line) ), mean_error))


if __name__ == '__main__':
    file_list = os.listdir(RESULTS_PATH + '/results')
    print(file_list)
    for f in file_list:
        if f.startswith('.'):
            continue
        if not os.path.exists(RESULTS_PATH + '/results/' + f + '/result.txt'):
            continue
        if not f.startswith('knn'):
            continue
        # try:
        print(f)
            # if f.startswith('gb.kmenas'):
            #     print(f)
        count_confidence_error(f)
        # except:
        #     print("FAILED!")
