from asag_utils import *

if __name__ == '__main__':
    # print(cur_time())
    # for i in range(1,6):
    # score_answer(fn_prefix='knnc',
    #              reliable=True,
    #              feature='infersent',
    #              model='knnc',
    #              model_params={'n_neighbors':5, 'weights':"distance"},
    #              qwise=True,
    #              training_scale=0)
    #
    # score_answer(fn_prefix='cosc',
    #              reliable=True,
    #              feature='infersent',
    #              model='cosc',
    #              model_params={'n_neighbors':5, 'dist_func':'l2'},
    #              qwise=True,
    #              training_scale=0)

    for s in range(10, 100, 10):
        score_answer(fn_prefix='svr_linear',
                     reliable=True,
                     feature='infersent',
                     model='svr',
                     model_params={'kernel': "linear"},
                     qwise=True,
                     training_scale=s)

    # score_answer(fn_prefix='svr_linear',
    #              reliable=True,
    #              feature='bow_1gram',
    #              model='svr',
    #              model_params={'kernel': "linear"},
    #              qwise=True,
    #              training_scale=0)
