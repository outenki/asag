from functools import reduce
from typing import Optional, IO

import numpy as np

from sklearn import neighbors
from gensim.models import Word2Vec
import progressbar
from scipy import spatial
import autograd
import spacy
from spacy.lang.en import LEMMA_INDEX, LEMMA_EXC, LEMMA_RULES
from random import shuffle

from config import *
from basic_util import *
from autograd.misc.optimizers import adam
import multiprocessing



np.set_printoptions(threshold=np.nan)


def cos_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """
    Calculate cosine between two vector
    :param vec1:
    :param vec2: vectors
    :return: float between (0, 1)
    """
    norm1 = autograd.numpy.linalg.norm(vec1)
    norm2 = autograd.numpy.linalg.norm(vec2)
    if norm1 == 0 or norm2 == 0:
        return 1
    return (1.0 + autograd.numpy.dot(vec1, vec2) / (norm1 * norm2)) / 2


def similarity_func(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """
    Calculate distance between two sentences.
    :param vec1:
    :param vec2: vectors
    :return: a value between 0 and 1
    """
    distance = cos_similarity(vec1, vec2)
    return distance


def read_w2v(fname_w2v: str) -> (dict, int):
    """
    Read word2vec model from a file.
    The data of file is strings, each line being one record
    :param fname_w2v: file name
    :return: A dict instance and
    """
    w2v = dict()
    print("reading w2v file...", end='')
    with open(fname_w2v, 'r', encoding='utf-8') as f:
        for line in f:
            data = line.split()
            w2v[data[0]] = np.array(list(map(float, data[1:])))

    # Dimension of word vector.
    with open(fname_w2v, 'r') as f:
        d_vec = len(f.readline().split()) - 1
    return w2v, d_vec


class Feature:
    def __init__(self, w2v: dict = None, vec_dimension: int = None, instance_nlp=None, instance_lemmatizer=None,
                 vali: 'position from the end where the validation data starts'=0):
        self.__scores_list = []  # Scores of training data
        self.__weights_dict = {}  # Weights of each words
        self.__voc_list = []  # Vocabulary of training data. Weight of words not in this list is 0.
        self.__w2v_dict = dict()  # Trained word2vec instance
        self.__sent_words_list = []  # Stemmed words list of each answer.
        self.d_vec = 100  # Dimension of vector from w2v
        self.__vec_sent_words_list = []  # Vector version of self.__sent_words
        self.__most_similar = dict()  # key: (

        # some initialization
        self.__nlp = spacy.load('en') if not instance_nlp else instance_nlp
        self.__lemmatizer = instance_lemmatizer if instance_lemmatizer else spacy.lemmatizer.Lemmatizer(
            LEMMA_INDEX, LEMMA_EXC, LEMMA_RULES)
        self.__validation_data = None  # a list of answer couples:[((sent1, score1), (sent2, score2)), (), (),...]
        self.__training_data = None  # a list of answer couples:[((sent1, score1), (sent2, score2)), (), (),...]
        self.__cur = 0  # index of training data that is used for current epoch
        self.__step = 1  # For each epoch, the weight will be updated using training_data[cur:step]
        self.__vali = vali
        if w2v:
            self.__w2v_dict = w2v
            self.d_vec = vec_dimension

        self.__echo = False

    def write_weight(self, fn):
        string = ','.join(["{}:{}".format(k, w) for (k, w) in self.__weights_dict.items()]) + '\n'
        fn.write(string)

    def write_voc(self, fn):
        string = ' '.join(self.__voc_list) + '\n'
        fn.write(string)

    def __similarity_words(self, w1, w2):
        if not w1 or not w2 :
            return 1
        if w1 not in self.__w2v_dict:
            # print("Word '{}' is not in w2v!".format(w1))
            return 0
        if w2 not in self.__w2v_dict:
            # print("Word '{}' is not in w2v!".format(w2))
            return 0
        return 1 - spatial.distance.cosine(self.__w2v_dict[w1], self.__w2v_dict[w2])

    def __calculate_most_similarites(self, limit = 11):
        for word in self.__voc_list:
            self.__most_similar[word] = sorted([(w, self.__similarity_words(word, w)) for w in self.__voc_list], reverse=True, key=lambda pair:pair[1])[1:1 + limit]

    def write_mosti_similar(self, fn):
        for word in self.__most_similar:
            string = '{} {}'.format(word, ','.join(['{}:{}'.format(w, s) for (w, s) in self.__most_similar[word]])) + '\n'
            fn.write(string)


    def __sent2vec(self, word_list, weights_dict=None):
        """
        Generate sentence vectors as mean of weighted summary of word vectors.
        :param word_list: iterable variable of words in the sentence
        :param weights_dict:
        :return: a vector with dimension of self.d_vec (same to w2v)
        """
        if not weights_dict:
            weights_dict = self.__weights_dict
        vec_sent = np.zeros(self.d_vec)
        for word in word_list:
            if word not in weights_dict:
                # if word:
                #     print("Can't find word '{}' in weight dict.".format(word))
                continue
            if word not in self.__w2v_dict:
                # if word:
                #     print("Can't find word '{}' in Word2vec.".format(word))
                continue
            vec_sent += weights_dict.get(word, 0) * self.__w2v_dict[word]
        return vec_sent / len(word_list)

    def __loss_func_minibatch(self, weights_list, words_lists: 'sentence' = None):
        """
        Calculate loss with weights. The loss is based on the similarity betwen sentences.
        Answers with same score should very similar to each other, and those with different score
        should be far away from each other in vector space.
        To do it better, the distance between each pair of answer should be related to the difference
        between their scores.
        Every time this function will use a part of of data, and save the current position.

        :param weights_list: A list of weights of words in vocabulary.
        :param words_lists: Sentences as a list of tokens. It is usually used for validation. If `None` is
            passed in, then self.__training_data will be used.
        :return:
        """
        loss = []
        # Generate weight dict for words in vocabulary
        weights_dict = dict(zip(self.__voc_list, weights_list))

        # Generate vector of answers
        if not words_lists:
            loss_data = self.__training_data[self.__cur: self.__cur + self.__step]
        else:
            loss_data = words_lists #[self.__cur: self.__cur + self.__step]
        self.__cur = (self.__cur + self.__step) % len(self.__training_data)

        for data1, data2 in loss_data:
            vec1 = self.__sent2vec(data1[0], weights_dict)
            score1 = data1[1]
            vec2 = self.__sent2vec(data2[0], weights_dict)
            score2 = data2[1]
            similarity = similarity_func(vec1, vec2)
            loss.append(1 - similarity if score1 == score2 else similarity)

            # loss.append(abs(1 - similarity - abs(score2-score1)/5))
        loss = sum(loss) / len(loss) #+ sum(abs(np.array(weights_list))) / len(weights_list)
        # print('loss:', loss)
        return loss

    def __loss_func_full(self, weights_list, words_lists=None):
        """
        Same to __loss_func_minipatch. Use all the data in words_lists (no validation data).
        :param weights_list: A list of weights of words in vocabulary.
        :param words_lists: Sentences
        :return:
        """
        loss = []
        weights_dict = dict(zip(self.__voc_list, weights_list))

        # Generate vector of answers
        if not words_lists:
            words_lists = self.__sent_words_list
        vector_ans = [self.__sent2vec(ans, weights_dict) for ans in words_lists]

        for i in range(len(vector_ans)):
            ws1 = vector_ans[i]
            for j in range(i, len(vector_ans)):
                ws2 = vector_ans[j]
                similarity = similarity_func(ws1, ws2)
                loss.append(1 - similarity if self.__scores_list[i] == self.__scores_list[j] else similarity)
        loss = sum(loss) / len(loss)
        # print('loss:', loss)
        return loss

    def __batch_indices(self, iter):
        idx = iter * self.__step % len(self.__training_data)
        return slice(idx * self.__step, (idx + 1) * self.__step)

    def __objective(self, weights, iter):
        idx = self.__batch_indices(iter)
        loss = self.__loss_func_minibatch(weights, self.__training_data[idx])
        # print("{}th weights: {}".format(iter, sorted(weights)[:11]))

        if self.__echo: print("{}th loss:\t{}".format(iter, loss))
        return loss

    def __validation(self, weight_list):
        return self.__loss_func_minibatch(weight_list, self.__validation_data)

    def fit_simple(self, answers_raw, scores_raw):
        """
        Prepare for generating vectors for answers.
        Simple method
        * Generate vocabulary
        * Train word2vec instance if necessary
        * Calculate weight for each word
        :param answers_raw: A list of raw data of answers
        :param scores_raw: A list of scores as training data. The length need to be
                        the same with with the list of answers
        :return: None
        """

        assert len(answers_raw) == len(scores_raw)

        self.__scores_list = list(map(float, scores_raw[:]))
        self.__sent_words_list = [set(token_strip(str_ans, self.__nlp, self.__lemmatizer) for str_ans in answers_raw)]
        assert len(self.__sent_words_list) == len(answers_raw) == len(self.__scores_list)

        # calculate weights for each word in vocabulary
        self.__voc_list = list(reduce(lambda x, y: x | y, self.__sent_words_list))

        l_answers = len(answers_raw)
        for i, j in [(i, j) for i in range(l_answers) for j in range(l_answers)]:
            if i == j:
                continue
            if scores_raw[i] == scores_raw[j]:
                for word in self.__sent_words_list[i] & self.__sent_words_list[j]:
                    self.__weights_dict[word] = self.__weights_dict.get(word, 0) + 1
                for word in self.__sent_words_list[i] ^ self.__sent_words_list[j]:
                    self.__weights_dict[word] = self.__weights_dict.get(word, 0) - 1

            if scores_raw[i] != scores_raw[j]:
                for word in self.__sent_words_list[i] & self.__sent_words_list[j]:
                    self.__weights_dict[word] = self.__weights_dict.get(word, 0) - 1
                for word in self.__sent_words_list[i] ^ self.__sent_words_list[j]:
                    self.__weights_dict[word] = self.__weights_dict.get(word, 0) + 1

            for w in self.__weights_dict:
                self.__weights_dict[w] = sigmoid(self.__weights_dict[w])

        # print(sorted(self.__weights.items(), key=lambda d:d[1], reverse=True)[:11])

        if not self.__w2v_dict:
            # No w2v model is provided, then train a new one use current vocabulary
            voc = [token_strip(s, self.__nlp, self.__lemmatizer) for s in answers_raw]
            self.__w2v_dict = Word2Vec(voc, min_count=1)

    def fit(self, answers_raw, scores_raw, que_id = '', threshold_loss=0.35, threshold_epochs=10000, echo = False):
        """
        Prepare for generating vectors for answers.
        * Generate vocabulary
        * Train word2vec instance if necessary
        * Calculate weight for each word
        * Generate a word-weight dictionary as self._weight_dict
        :param answers_raw: A list of raw data of answers
        :param scores_raw: A list of scores as training data. The length need to be
                        the same with with the list of answers
        :param threshold_loss: Threshold of loss. Training of weight will stop when the loss is less than threshold.
        :param threshold_epochs: Threshold of epochs. The training of weight will stop after threshold_epochs times.
        :return: None
        """
        self.__echo = echo
        assert len(answers_raw) == len(scores_raw)
        self.__scores_list = list(map(float, scores_raw))

        # Tokenize each answer with lemmatization
        # print("Tokenizing...")
        # for ans in answers_raw:
        #     doc = self.__nlp(ans)
        #     words_of_ans.append(self.__lemmatizer(doc[i].string, doc[i].pos) for i in range(len(doc)))
        self.__sent_words_list = np.array([token_strip(ans, self.__nlp) for ans in answers_raw])
        # print("sent words list:", self.__sent_words_list)
        assert len(self.__sent_words_list) == len(answers_raw)

        # Generate training data and validation data
        if echo: print("Generating training data and validation data...")
        sent_words = self.__sent_words_list[:]
        sent_score = self.__scores_list[:]
        sent_data_ = list(zip(sent_words, sent_score))
        l_sent_data = len(sent_data_)

        # Generate answer pairs and shuffle them
        sent_data_pair = [(sent_data_[i], sent_data_[j]) for i in range(l_sent_data) for j in range(i, l_sent_data)]
        shuffle(sent_data_pair)
        l_sent_data = len(sent_data_pair)

        # Get training data and validation data
        edge = int((1-self.__vali) * l_sent_data)
        self.__training_data = sent_data_pair[:edge]
        self.__validation_data = sent_data_pair[edge:]

        def validation_fun(params, iter, gradient):
            if self.__echo: print('{}th vali:\t{}'.format(iter, self.__loss_func_minibatch(params, self.__validation_data)))

        self.__cur = 0
        self.__step = int(0.25 * len(self.__training_data))
        # print("Training step:{}/{}".format(self.__step, len(self.__training_data)))

        # Generate vocabulary based on all training data
        # print("Generateing vocabulary...")

        self.__voc_list = list(set(reduce(lambda x, y: set(x) | set(y), self.__sent_words_list)))
        # print("Voc list:", self.__voc_list)
        # print("Size of vocabular:", len(self.__voc_list))

        # calculate weights for each word in vocabulary
        # print("Training weights for words...")

        # print("\nQue_id: ", que_id)
        param_under_weight = 10
        weights = np.ones(len(self.__voc_list)) / param_under_weight
        # weights = np.array([random.random()/50 for i in range(len(self.__voc_list))])
        # print('Random parameter')
        # grad_desent = autograd.grad(self.__loss_func_minibatch)
        # print('Lemma')
        # print('init_weight({}):'.format(param_under_weight), sorted(weights)[:10])
        # epochs = 0
        # loss = threshold_loss + 1

        objective_grad = autograd.grad(self.__objective)
        num_epochs = 300
        step_size = 0.001
        weights = adam(objective_grad, weights, step_size=step_size, num_iters=num_epochs, callback=validation_fun)
        # while loss > threshold_loss and epochs < threshold_epochs:
        #     gradient = grad_desent(weights)
        #     weights -= alpha * gradient
        #     loss = self.__loss_func_minibatch(weights, self.__validation_data)
        #     epochs += 1
        #     print('loss in {}th epoch: {}'.format(epochs, loss))
        #     print('weights:', sorted(weights)[:11])
        #     print('gradient:', sorted(gradient)[:10])
        #     print()
        # loss = self.__loss_func_minibatch(weights, self.__validation_data)
        # print(loss)
        self.__weights_dict = dict(zip(self.__voc_list, weights))
        self.__calculate_most_similarites()
        # print(sorted(self.__weights_dict.items(), key=lambda d: d[1], reverse=True)[:11])
        # print("Fit done")

    def feature(self, sent):
        return self.__sent2vec(sent)



def generate_features_sent2vec(fname_w2v, instance_nlp, instance_lemmatizer, q_list = None):
    feature_path = RESULTS_PATH + "/features_sent2vec/"
    weight_path = RESULTS_PATH + "/word_weights"
    voc_path = RESULTS_PATH + "/vocabulary"
    word_similar_path = RESULTS_PATH + "/similar_words"
    if not os.path.exists(feature_path):
        os.makedirs(feature_path)

    print("Reading in w2c dic...")
    w2v, d_vec = read_w2v(fname_w2v)

    if not q_list:
        q_list = sorted(os.listdir(RAW_PATH_STU))

    for que_id in q_list:
        print("\n" + que_id)
        ans_ref = ''
        # generate bow features
        with open(RAW_PATH_STU + "/" + que_id, 'r', errors="ignore") as f_ans, \
                open(DATA_PATH + '/scores/{}/ave'.format(que_id), 'r') as f_score, \
                open(voc_path + "/" + que_id, "w") as f_voc, \
                open(feature_path + "/" + que_id, 'w') as f_fea:
            with open(RAW_PATH + '/answers', 'r', errors='ignore') as f_ref:
                for r in f_ref:
                    if r.startswith(que_id):
                        ans_ref = r
            raw_answers = f_ans.readlines()
            raw_answers.append(ans_ref)
            for i in range(len(raw_answers)):
                raw_answers[i] = ' '.join(raw_answers[i].split(' ')[1:])
            arr_ans = np.array(raw_answers)
            scores = f_score.readlines()
            if len(scores) < len(raw_answers):
                scores.append(str(SCORE_LEVELS-1))
            scores = np.array(scores)
            # bar = progressbar.ProgressBar(max_value=len(arr_ans))  # progressbar
            # bar_i = 0  # progressbar
            for i in range(len(arr_ans)):
                print('{}.{}'.format(que_id, i+1))
                data_filter = np.array([True] * len(arr_ans))
                data_filter[i] = False

                fea = Feature(w2v, d_vec, instance_nlp, instance_lemmatizer)
                print("Fitting...", end='')
                fea.fit(arr_ans[data_filter], scores[data_filter],que_id = que_id)
                print("done")
                print("Writing weights...", end='')
                with open("{}/{}.{}".format(weight_path, que_id, i+1), "w") as f_weight:
                    fea.write_weight(f_weight)
                print("done")
                print("Writing vocabulary...", end='')
                fea.write_voc(f_voc)
                print("done")

                print("Writing similarities...", end='')
                with open(word_similar_path + "/" + '{}.{}'.format(que_id, i+1), 'w') as f_sim:
                    fea.write_mosti_similar(f_sim)
                print("done")

                print("Generating...", end='')
                feature = fea.feature(token_strip(arr_ans[i], instance_nlp))

                # print('Feature:', *feature, sep=',')
                print(*feature, file=f_fea, sep=',')
                print("done")
                # bar.update(bar_i)  # progressbar
                # bar_i += 1  # progressbar


# def generate(que_id, feature_path, w2v, d_vec, instance_nlp, instance_lemmatizer):
#     print("\n" + que_id)
#     # generate bow features
#     with open(RAW_PATH_STU + "/" + que_id, 'r', errors="ignore") as f_ans, \
#             open(DATA_PATH + '/scores/{}/ave'.format(que_id), 'r') as f_score, \
#             open(feature_path + "/" + que_id, 'w') as f_fea:  # type: Optional[IO[str]]
#         raw_answers = f_ans.readlines()
#         for i in range(len(raw_answers)):
#             raw_answers[i] = ''.join(raw_answers[i].split(' ')[1:])
#         arr_ans = np.array(raw_answers)
#         scores = np.array(f_score.readlines())
#         # bar = progressbar.ProgressBar(max_value=len(arr_ans))  # progressbar
#         # bar_i = 0  # progressbar
#         for i in range(len(arr_ans)):
#             print('{}.{}'.format(que_id, i + 1))
#             data_filter = np.array([True] * len(arr_ans))
#             data_filter[i] = False
#
#             fea = Feature(w2v, d_vec, instance_nlp, instance_lemmatizer)
#             print("Fitting...", end='')
#             fea.fit(arr_ans[data_filter], scores[data_filter], que_id=que_id)
#             print("done")
#             print("Generating...", end='')
#             feature = fea.feature(token_lemma(arr_ans[i], instance_nlp, instance_lemmatizer))
#             print("done")
#             print('Feature:', *feature, sep=',')
#             print(*feature, file=f_fea, sep=',')
#
# def generate_features_sent2vec_multi(fname_w2v, instance_nlp, instance_lemmatizer):
#     feature_path = RESULTS_PATH + "/features_sent2vec/"
#     if not os.path.exists(feature_path):
#         os.makedirs(feature_path)
#
#     print("Reading in w2c dic...")
#     w2v, d_vec = read_w2v(fname_w2v)
#
#     cores = multiprocessing.cpu_count()
#     pool = multiprocessing.Pool(processes=cores)
#     nargs = [(q, feature_path, w2v, d_vec, instance_nlp, instance_lemmatizer) for q in sorted(os.listdir(RAW_PATH_STU))]
#     pool.map(generate, nargs)
#


def weight_test(instance_nlp, instance_lemmatizer):

    test_list = ['11.3', '1.5', '10.2', '11.5', '2.5']
    # test_list = ['FaultFinding-BULB_C_VOLTAGE_EXPLAIN_WHY1', 'FaultFinding-BULB_ONLY_EXPLAIN_WHY6',  'FaultFinding-OTHER_TERMINAL_STATE_EXPLAIN_Q']
    # test_list = ['2.5']
    print("Reading word2vec model from files:")
    f_w2v = W2V_PATH + "/" + W2V_FILE

    w2v, d_vec = read_w2v(f_w2v)
    for que_id in test_list:
        with open(DATA_PATH + "/raw/ans_stu/" + que_id, 'r') as fq, \
                open(DATA_PATH + "/scores/" + que_id + "/ave", 'r') as fs:

            # f_w2v = RESULTS_PATH + "/models_w2v/w2v_all"
            # w2v, d_vec = Word2Vec.load(f_w2v), 100
            print(que_id)
            raw_answers = fq.readlines()
            for i in range(len(raw_answers)):
                raw_answers[i] = ' '.join(raw_answers[i].split(' ')[1:])
                print(raw_answers[i])
            raw_scores = fs.readlines()
            g = Feature(w2v, d_vec, instance_nlp=instance_nlp, instance_lemmatizer=instance_lemmatizer)
            g.fit(raw_answers, raw_scores, que_id = que_id, echo=True)


def generate_feature_for_sentence(text, que_id, ans_id, w2v, dim):
    # read weights of words
    with open('{}/word_weights/{}'.format(RESULTS_PATH, que_id), 'r') as f_weights:
        dict_lines = f_weights.readlines()
        dict_string = dict_lines[ans_id-1]
        weight_dict = dict([(item.split(':')[0], float(item.split(':')[1])) for item in dict_string.split(',')])

    def sent2vec(word_list, weights_dict, w2v):
        """
        Generate sentence vectors as mean of weighted summary of word vectors.
        :param word_list: iterable variable of words in the sentence
        :param weights_dict:
        :return: a vector with dimension of self.d_vec (same to w2v)
        """

        vec_sent = np.zeros(dim)
        for word in word_list:
            if word not in weights_dict:
                continue
            if word not in w2v:
                continue
            weight = weights_dict.get(word, 0)
            vec = w2v[word]
            vec_sent += weight* vec
        return vec_sent / len(word_list)

    feature = sent2vec(token_strip(text, NLP), weight_dict, w2v)
    print(*feature, sep=',')



if __name__ == '__main__':
    # run_procerpron_learning()
    # read_training_data("/features_bow_1gram/")

    # training w2v


    # w2v_train_file = RAW_PATH + '/all'
    file_w2v = W2V_PATH + "/" + W2V_FILE
    # weight_test(instance_nlp=nlp, instance_lemmatizer=lemmatizer)
    q_list = sorted(os.listdir(RAW_PATH_STU))
    # generate_features_sent2vec(file_w2v, nlp, lemmatizer, ['1.1'])
    generate_features_sent2vec(file_w2v, NLP, LEMMATIZER, ['4.5'])
    # generate_features_sent2vec(file_w2v, nlp, lemmatizer)
    # generate_features_sent2vec_multi(file_w2v, nlp, lemmatizer)
