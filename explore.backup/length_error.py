from config import *
from math import ceil
import numpy as np

BINS = 10
def count_length_error(fn):
    result_path = RESULTS_PATH + '/results/' + fn
    conf_err_count = []
    with open(result_path + '/result.txt' , 'r') as f_res, \
            open(result_path + '/error_length.txt', 'w') as f_el, \
            open(result_path + '/length_error_flat.txt', 'w') as f_le, \
            open(result_path + '/length_error_dst.txt', 'w') as f_led:
        results = f_res.readlines()
        error_length  = []
        for result in results:
            result = result.split('\t')
            # print((result[5]), (result[14))
            error_length.append((float(result[4]), float(result[13])))

        errors, length = zip(*error_length)
        print(max(length))


        # error-length
        max_error = int(max(errors))
        err_len_count = []
        for i in range(max_error+1):
            err_len_count.append([])

        for e, l in error_length:
            idx = 0 if e == 0 else ceil(e)
            err_len_count[idx].append(l)

        for i in range(max_error+1):
            e = np.array(err_len_count[i])
            ans_count = len(e)
            mean_length = 0 if ans_count == 0 else np.mean(e)
            std_length = 0 if ans_count == 0 else np.std(e)
            f_el.write('{:.2f}\t{}\t{:.2f}\t{:.2f}\n'.format(i, ans_count, mean_length, std_length))

        error_length = sorted(error_length, key=lambda x:x[1])

        len_err_count_flat = []
        print(error_length[-1])
        for i in range(BINS):
            len_err_count_flat.append([])
        len_step = 1 if len(length) <=BINS else len(length)/BINS
        for i in range(len(error_length)):
            idx_len_err = int(i/len_step)
            len_err_count_flat[idx_len_err].append(error_length[i][0])
        for i in range(len(len_err_count_flat)):
            e = np.array(len_err_count_flat[i])
            ans_count = len(e)
            err_mean = 0 if  ans_count== 0 else np.mean(e)
            err_std = 0 if ans_count==0 else np.std(e)
            idx_len = int(i * len_step)
            cur_len = error_length[idx_len][1]
            f_le.write('{:.2f}\t{}\t{:.2f}\t{:.2f}\n'.format(cur_len, ans_count, err_mean, err_std))

        len_step = 1 if len(length) <= BINS else max(length)/BINS
        len_err_count_dst = []
        for i in range(BINS):
            len_err_count_dst.append([])
        for i in range(len(error_length)):
            e, l = error_length[i]
            idx_len = int(l/len_step)

            if idx_len >= BINS:
                idx_len = BINS - 1
            len_err_count_dst[idx_len].append(e)
        for i in range(len(len_err_count_dst)):
            e = len_err_count_dst[i]
            cur_len = (i+1)*len_step
            ans_count = len(e)
            err_mean = 0 if ans_count == 0 else np.mean(e)
            err_std = 0 if ans_count == 0 else np.std(e)
            f_led.write('{:.2f}\t{}\t{:.2f}\t{:.2f}\n'.format(cur_len, ans_count, err_mean, err_std))




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
        count_length_error(f)
        # except:
        #     print("FAILED!")
