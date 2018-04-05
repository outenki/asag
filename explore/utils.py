from scipy import spatial
import numpy as np
import nltk
from nltk.util import ngrams
import spacy
import time
import os

def check_c_path(path):
    '''
    check if path exist or not.
    If not, creat it.
    :param path: string of path.
    '''
    if not os.path.exists(path):
        os.makedirs(path)

def cur_time():
    '''
    Return current time stamp in type of string.
    '''
    return time.strftime('%Y%m%d%H%M%S', time.localtime(time.time()))

def tokenize(nlp, text, rm_punct, ngram):
    '''
    Tokenize the input text.
    1. Lower all the characters.
    2. Tokenize using spacy with default rules.
    3. Remove punctuation (optional)
    :param nlp: Instance of spacy 
    :param text: Input text.
    :param ngram: Ngram.
    :return: Strings of tokens
    >>> tokenize(spacy.load('en'), u'a, b, c', True, 1)
    [(u'a',), (u'b',), (u'c',)]
    >>> tokenize(spacy.load('en'), u'a, b, c', True, 2)
    [(u'a', u'b'), (u'b', u'c')]
    >>> tokenize(spacy.load('en'), u'a, b, c', False, 1)
    [(u'a',), (u',',), (u'b',), (u',',), (u'c',)]
    '''
    text = str.lower(text.strip())
    tokens = nlp(text)
    if rm_punct:
        tokens = list(filter(lambda t:not t.is_punct, tokens))
    tokens_text = [t.text for t in tokens]
    n_gram_tokens = ngrams(tokens_text, ngram)
    return list(n_gram_tokens)

def generate_training_test_data_f(fn, train_ratio=0.8):
    '''
    Split data to training data and test data.
    :param fn: tsv file storing the data
    :param train_ratio: Ratio of training data. 0.8 as default.
    :return: Training data and test data in form of `AID\tQID\tSCORE\tFEATURE`
    '''
    data = np.loadtxt(fn, skiprows=1, dtype=str)
    # with open(fn, 'r') as f_data:
    #     f_data.readline()
    #     data = np.array(f_data.readlines())
    leng = len(data)
    edge = int(leng * train_ratio)
    return data[:edge], data[edge:]

if __name__ == '__main__':
    import doctest
    doctest.testmod(verbose=True)
