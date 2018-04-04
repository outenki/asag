from scipy import spatial
# from config import NLP
import itertools

import string
from pylab import *
import os
import errno


def cur_time():
    return time.strftime('%Y%m%d%H%M%S', time.localtime(time.time()))


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


def token_strip(string, instance_nlp):
    doc = instance_nlp(clean_text(string))
    # return [instance_lemmatizer(doc[i].string.strip(), doc[i].pos)[0] for i in range(len(doc))]
    return [doc[i].string.strip() for i in range(len(doc)) if doc[i].string.strip()]

def token_lemma(text, instance_nlp, instance_lemmatizer):
    doc = instance_nlp(clean_text(text))
    return [instance_lemmatizer(doc[i].string.strip(), doc[i].pos)[0] for i in range(len(doc)) if doc[i].string.strip()]
    # return [doc[i].string.strip() for i in range(len(doc)) if doc[i].string.strip()]


def clean_text(text):
    lower = text.lower()
    remove_punctuation_map = dict((ord(char), '') for char in string.punctuation)
    return lower.translate(remove_punctuation_map)


def draw_confusion_matrix(data: '2 dimension matrix', labels, path_name, show=False):
    norm_data = []  # values of data are normalized to 0~1
    for r in data:
        sum_r = sum(r)
        norm_r = list(map(lambda x: x / sum_r, r))
        norm_data.append(norm_r)

    fig = plt.figure()
    plt.clf()
    ax = fig.add_subplot(111)
    res = ax.imshow(np.array(norm_data), cmap=plt.cm.gray,
                    interpolation='nearest')
    fig.colorbar(res)
    width = len(norm_data)
    height = len(norm_data[0])

    _, labels = plt.xticks(range(width), labels[:width])
    for t in labels:
        t.set_rotation(60)
    plt.yticks(range(height), labels[:height])

    if show: plt.show()
    plt.savefig(path_name, format='png')

def read_confusion_data(file_name):
    pres, exps = [], []
    with open(file_name, 'r') as f:
        for line in f:
            _, pre, exp, *_ = line.split('\t')
            pre = round(float(pre)*2)
            if pre > 10: pre = 10
            if pre < 0: pre = 0
            exp = round(float(exp)*2)
            pres.append(pre)
            exps.append(exp)
    return pres, exps

def plot_confusion_matrix(cm, classes, path_name,
                          normalize=False,
                          title='Result Distribution',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        sum_by_row = cm.sum(axis=1)
        for i in range(len(sum_by_row)):
            if sum_by_row[i] == 0:
                sum_by_row[i] = 1
        cm = cm.astype('float') / sum_by_row[:, np.newaxis]
        # cm = cm.astype('float') / cm.sum()
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    # print(cm)
    plt.clf()
    plt.figure(figsize=(8, 7), )
    plt.subplots_adjust(left=0.1, right=1, top=0.9, bottom=0.15)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = (cm.max()) / 2.
    print("thresh:", cm.max(), thresh)
    # print(cm[1][1])
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j]*100, fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    # plt.tight_layout()
    plt.ylabel('True label', fontsize=14)
    plt.xlabel('Predicted label', fontsize=14)
    plt.savefig(path_name, format='png')
    # plt.show()

def calculate_F1(file_name):
    f1, found_correct, found, truth = {}, {}, {}, {}

    with open(file_name, 'r', errors='ignore') as f:
        lines = f.readlines()
        n = len(lines)
        for line in lines:
            _, pre, exp, *_ = line.split('\t')
            pre = round(float(pre))
            if pre > 5: pre = 5
            if pre < 0: pre = 0
            exp = round(float(exp))
            found[pre] = found.get(pre, 0) + 1
            truth[exp] = truth.get(exp, 0) + 1
            if pre == exp:
                found_correct[exp] = found_correct.get(exp, 0) + 1
    for score in truth:
        if score not in found:
            precision = 0
        else:
            precision = found_correct.get(score, 0) / found[score]
        recall = found_correct.get(score, 0) / truth[score]
        if precision * recall == 0:
            f1[score] = 0
        else:
            f1[score] = 2 * precision * recall / (recall + precision)
    wf1 = 1/n * sum([truth[score] * f1.get(score, 0) for score in truth])
    return wf1

def text_weight_color(text, weights):
    html = []
    text = text.lower()
    tokens = text.strip().split(' ')
    for token in tokens:
        token = token.strip()
        if not token:
            continue
        weight = float(weights.get(clean_text(token), 0))
        b, r = 0, 0
        if weight < 0:
            r = 255
        if weight > 0:
            b = 255
        html.append("<span style='border-radius:5px;background-color:rgba({},0,{},{})'>{}</span>".format(r, b, abs(weight), token))
    return '<div>' + ' '.join(html) + '</div>'

def read_weight_from_string(text):
    return dict([(item[0], item[1]) for item in text.split(',')])

def read_w2v(f_w2c):
    print("reading w2v file...", end='')
    w2v = dict()
    with open(f_w2c, 'r', encoding='utf-8') as f:
        for line in f:
            data = line.split()
            w2v[data[0]] = np.array(list(map(float, data[1:])))
    with open(f_w2c, 'r') as f:
        d_vec = len(f.readline().split()) - 1
    return w2v, d_vec


def distance_of_couple(arr1, arr2, distance='euclidean'):
    if distance == 'euclidean':
        diff = arr1 - arr2
        return sqrt(diff.dot(diff))
    if distance == 'cos':
        return spatial.distance.cosine(arr1, arr2)

def overlap(text1, text2):
    tokens1 = set(token_strip(clean_text(text1),NLP))
    tokens2 = set(token_strip(clean_text(text2), NLP))
    tokens_u = tokens1 | tokens2
    tokens_i = tokens1 & tokens2
    if len(tokens_u) == 0:
        return 0
    return len(tokens_i)/len(tokens_u)

def overlap_of_couples(text_array):
    distances = []
    num_text = len(text_array)
    for i in range(num_text):
        for j in range(i+1, num_text):
            distances.append(1-overlap(text_array[i], text_array[j]))
    distances = np.array(distances)
    if len(distances) > 0:
        flag = 1
    else:
        flag = 0
    if len(distances) > 0:
        return np.array(distances), np.mean(distances), np.std(distances), flag
    else:
        return np.array([]), 0, 0, 0



def distance_of_couples(fea_array, distance_type):
    distances = []
    num_fea = len(fea_array)
    for i in range(num_fea):
        for j in range(i+1, num_fea):
            distances.append(distance_of_couple(fea_array[i], fea_array[j], distance_type))
    distances = np.array(distances)
    if len(distances) > 0:
        flag = 1
    else:
        flag = 0
    if len(distances) > 0:
        return np.array(distances), np.mean(distances), np.std(distances), flag
    else:
        return np.array([]), 0, 0, 0

def check_dir(dir_path):
    if not os.path.exists(dir_path):
        try:
            os.mkdir(dir_path)
        except OSError as exc:
            if exc.errno == errno.EEXIST and os.path.isdir(dir_path):
                pass
            else:
                raise
