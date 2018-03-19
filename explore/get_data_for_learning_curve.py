import os
SCRIPT_PATH = os.path.dirname(os.path.abspath(__file__))
RESULTS_PATH = SCRIPT_PATH + "/../results_semi_train"
# RESULTS_PATH = SCRIPT_PATH + "/../results_kaggle_train"
# RESULTS_PATH = SCRIPT_PATH + "/../results_sag"
CURVE_PATH = RESULTS_PATH + "/curves"

OUTPUT_FILE = 'learning_curve.txt'
def read_data(path_name):
    data_path = CURVE_PATH + '/' + path_name
    if not os.path.exists(data_path):
        os.mkdir(data_path)
    with open(data_path + '/' + OUTPUT_FILE, 'w') as fo:
        data = []
        for fn in os.listdir(data_path):
            if os.path.isfile(data_path + '/' + fn):
                continue
            scale = fn.split('.')[-2]
            with open(data_path + '/' + fn + '/errors.txt', 'r') as fe:
                rmse = fe.readlines()[-1].split('\t')[-1]
            data.append(scale + '\t' + rmse )
        fo.writelines(data)
read_data('bow_1gram.learning_curve.knn.171117')
# read_data('gb.learning_curve.kmeans.171107')
# os.chdir(RESULTS_PATH + '/results')
# print(os.listdir('.'))
# for p in os.listdir('.'):
#     if p.startswith('gb.kmenas'):
#         print(p)
#         read_data(p)