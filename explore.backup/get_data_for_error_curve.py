import os
SCRIPT_PATH = os.path.dirname(os.path.abspath(__file__))
# RESULTS_PATH = SCRIPT_PATH + "/../results_semi_train"
# RESULTS_PATH = SCRIPT_PATH + "/../results_semi_ud"
# RESULTS_PATH = SCRIPT_PATH + "/../results_kaggle_train"
RESULTS_PATH = SCRIPT_PATH + "/../results_sag"
CURVE_PATH = RESULTS_PATH + "/curves"
OUTPUT_FILE = 'error_curve.txt'

def read_data(path_name, title_type):
    def get_scale(fn):
        return fn.split('.')[-2]
    def get_p(fn):
        split = fn.split('.')
        if len(split) == 8:
            return split[3]
        else:
            return split[3]+"." + split[4]
    def get_k(fn):
        return fn.split('.')[2]
    def get_name(fn):
        return fn[:-14]
    get_type = {
        'scale': get_scale,
        'p': get_p,
        'k': get_k,
        'name': get_name
    }
    data_path = CURVE_PATH + '/' + path_name
    with open(data_path + '/' + OUTPUT_FILE, 'w') as fo:
        first_line = [' ']
        data = [['0.0', '0.5', '1.0', '1.5', '2.0', '2.5', '3.0', '3.5', '4.0', '4.5', '5.0']]
        for fn in os.listdir(data_path):

            if os.path.isfile(data_path + '/' + fn):
                continue
            first_line.append(get_type[title_type](fn))
            with open(data_path + '/' + fn + '/errors.txt', 'r') as fe:
                ctnt = fe.readlines()[:-2]
                ctnt = list(map(lambda s: s.split('\t')[2], ctnt))[9:]
                data.append(ctnt)
        data = zip(*data)
        data = list(map(lambda l:'\t'.join(l)+'\n', data))
        fo.write('\t'.join(first_line) + '\n')
        fo.writelines(data)

read_data('w2v.error_curve.knnr.cos.svr.qwise.171222', 'name')