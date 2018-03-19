from config import *
from math import ceil, floor
import numpy as np
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt

def get_error_idx(err_abs):
    if err_abs == 0:
        return 0
    else:
        for e in range(1, SCORE_LEVELS):
            if err_abs <= e:
                return e

def get_histogram(conf_err:list, err_levels:int):
    hist = [0] * err_levels
    for c, e in conf_err:
        hist[ceil(e)] += 1
    return np.array(hist)

def count_confidence_error(fn):
    # with open(fn, 'r') as fe, open('error.count.txt', 'w') as ec,\
    #         open('error_abs.count.txt', 'w') as eac,\
    #         open('error_round.count.txt', 'w') as erc:
    result_path = RESULTS_PATH + '/results/' + fn
    conf_err_count = []
    err_abs_list = []
    conf_levels = CONFIDENCE_LEVELS
    conf_window_width = CONFIDENCE_WINDOW_WIDTH

    for i in range(conf_levels):
        conf_err_count.append([0] * SCORE_LEVELS)

    with open(result_path + '/result.txt', 'r') as f_res:
        results = f_res.readlines()

    distance = np.array(list(map(lambda x: float(x.strip().split('\t')[11]), results)))
    # normalize distance to [0~1]
    max_distance = max(distance)
    distance = distance / max_distance
    confidence = 1 - distance

    with open(result_path + '/result_confidence.txt', 'w') as f_resc:
        for i in range(len(results)):
            # add the confidence info to a new results file.
            result_confidence = '{}\t{}\n'.format(results[i].strip(), confidence[i])
            f_resc.write(result_confidence)

            # get the errors
            err_abs = float(results[i].split('\t')[4])
            err_abs_list.append(err_abs)

    assert len(err_abs_list) == len(confidence)
    conf_err_abs_list = np.array(sorted(zip(list(confidence), err_abs_list)))
    conf_step = int(len(err_abs_list) / conf_levels)

    with open(result_path + '/error_confidence_flat.txt', 'w') as f_count_flat, \
        open(result_path + '/error_confidence_flat_per.txt', 'w') as f_count_flat_per:
        f_count_flat.write('{}\t{}\t{}\n'.format('count', 'confidence_from', 'confidence_to'))
        for i in range(0, len(conf_err_abs_list), conf_step):
            cur_data = conf_err_abs_list[i:i+conf_window_width]
            count = len(cur_data)
            cur_hist = get_histogram(cur_data, SCORE_LEVELS)
            cur_hist_per = cur_hist / len(cur_data) if len(cur_data) > 0 else 0
            conf_from, conf_to = cur_data[0][0], cur_data[-1][0]

            mean_error = sum(cur_data[:,1]) / len(cur_data) if len(cur_data) > 0 else 0
            f_count_flat.write('{}\t{:.3f}\t{:.3f}\t{}\t{}\n'.format(count, conf_from, conf_to,
                                                                 '\t'.join(map(str, cur_hist)), mean_error))
            f_count_flat_per.write('{}\t{:.3f}\t{:.3f}\t{}\t{}\n'.format(count, conf_from, conf_to,
                                                                    '\t'.join(map(str, cur_hist_per)), mean_error))
            if count < conf_window_width:
                break


def drawHistogram(fn):
    with open(fn, 'r') as f_conf:
        f_conf.readline()
        data = f_conf.readlines()
    data = np.array(list(map(lambda l:l.split('\t'), data)))
    data = data.astype(float)
    x_conf = data[:, 1]
    y_err = np.transpose(data[:, 3:-1])
    x_mean_err = data[:, -1]

    n_bars = SCORE_LEVELS;
    fig, ax = plt.subplots()
    btm = np.zeros(len(x_conf))
    y = np.ones(len(x_conf))
    x = np.array(list(range(len(x_conf))))
    x = x.astype(float)/10
    plt.bar(x, y, label='err={}'.format(0))
    plt.xticks(x_conf)

    # for e, c in enumerate(y_err):
    #     print('x:', x_conf)
    #     print('y:', c)
    #
    #     plt.bar(x_conf, y, label='err={}'.format(e), bottom=btm)
    #     btm += c

    # x = [1, 2, 3, 4]
    # y1 = np.array([2.0, 3.1, 2, 3])
    # y2 = np.array([3, 1, 5, 3])
    # y3 = np.array([1, 2, 1, 3])
    # plt.bar(x, y1, color='green', label='y1')
    # plt.bar(x, y2, bottom=y1, color='red', label='y2')
    # plt.bar(x, y3, bottom=y1 + y2, color='blue', label='y3')

    opacity = 0.4
    # rects1 = plt.bar(index, means_men, bar_width, alpha=opacity, color='b', label='Men')
    # rects2 = plt.bar(index + bar_width, means_women, bar_width, alpha=opacity, color='r', label='Women')

    plt.xlabel('Confidence')
    plt.ylabel('Proportion of Errors')
    # plt.xticks(index + bar_width, ('A', 'B', 'C', 'D', 'E'))
    # plt.ylim(0, 40);
    plt.legend()

    # plt.tight_layout()
    plt.show()

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
        # count_confidence_error(f)
        drawHistogram(RESULTS_PATH+ '/results/'+f+'/error_confidence_flat.txt')
        # except:
        #     print("FAILED!")
