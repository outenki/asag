from config import *
from basic_util import plot_confusion_matrix, read_confusion_data, draw_confusion_matrix
from sklearn.metrics import confusion_matrix
WAYS = 11
labels = [str(i/2) for i in range(WAYS)]

if __name__ == '__main__':
    file_list = os.listdir(RESULTS_PATH + '/results')
    print(file_list)
    for f in file_list:
        if f.startswith('.'):
            continue
        if not os.path.exists(RESULTS_PATH + '/results/' + f + '/result.txt'):
            continue
        print(f)
        pres, exps = read_confusion_data(RESULTS_PATH + '/results/' + f + '/result.txt')
        pres = pres + list(range(WAYS))
        exps = exps + list(range(WAYS))
        print('pres:', pres)
        print('exps:', exps)
        # print('len_pres:', len(pres))
        # print('set_pres:', set(pres))
        # print('set_exps:', set(exps))
        print()
        data = confusion_matrix(exps, pres)
        for i in range(WAYS):
            data[i][i] -= 1
        plot_confusion_matrix(cm=data, classes=labels, path_name=RESULTS_PATH+'/results/'+f+'/cm.png', normalize=True)
        # draw_confusion_matrix(data, labels, RESULTS_PATH+'/results/'+f+'/cm1.png')