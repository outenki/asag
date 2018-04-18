import numpy as np
from nltk.util import ngrams
import time
import os
import sys
import logging

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


def tokenize(*, nlp, text, rm_punct, ngram, rm_stop, lemma):
    '''
    Tokenize the input text.
    1. Lower all the characters.
    2. Tokenize using spacy with default rules.
    3. Remove punctuation (optional)
    :param *: Force to use keyword parameter
    :param nlp: Instance of spacy 
    :param text: Input text.
    :param ngram: Ngram.
    :param rm_stop: Flag if rm stopwords.
    :param lemmatizer: Flag if use the base form of tokens
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
    if rm_stop:
        tokens = list(filter(lambda t:not t.is_stop, tokens))
    if lemma:
        tokens_text = list(map(lambda t: t.lemma_))
    else:
        tokens_text = [t.text for t in tokens]
    n_gram_tokens = ngrams(tokens_text, ngram)
    return list(n_gram_tokens)

def generate_training_test_data_f(fn, train_ratio=0.8):
    '''
    Split data to training data and test data.
    :param fn: tsv file storing the data
    :param train_ratio: Ratio of training data. 0.8 as default.
    :return: Training data and test data in form of `AID\tQID\tSCORE\tFEATURE\tANSWER`
    '''
    data = np.loadtxt(fn, skiprows=1, dtype=str, delimiter='\t')
    # with open(fn, 'r') as f_data:
    #     f_data.readline()
    #     data = np.array(f_data.readlines())
    leng = len(data)
    edge = int(leng * train_ratio)
    return data[:edge], data[edge:]

def generate_weight_html_text(nlp, text, weight):
    """
    convert text to html with visualizing.
    """
    words = tokenize(nlp, text, False, 1)
    words = [w[0] for w in words]
    html = ''
    for wo in words:
        wei = weight.get(wo, 0)
        html += "&nbsp;<span style='border-radius:5px;background-color:rgba(255,0,0,{})'>{}</span>".format(abs(wei), wo)
    return html


def genertate_weight_html_pred(nlp, pred, weight, title):
    tplt_style = '''
    <style>
        .c{
            text-align: center;
        }
        table, th, td
        {
            border: 1px solid ;
        }
        table
        {
            border-collapse:collapse;
            width:100%;
            word-break:break-all; 
            word-wrap:break-all;
        }
    </style>

    '''

    tbody = ''
    for aid, qid, gold, ans, pred, diff in pred:
        ans = generate_weight_html_text(nlp, ans, weight)
        tbody += "<tr><td class='c'>{}</td><td class='c'>{}</td><td> {}</td><td class='c'>{}</td><td class='c'>{}</td><td class='c'>{}</td></tr>".format(aid, qid, ans, gold, pred, diff)

    body = """
        <table><tr><td class='c'>ans_id</td><td class='c'>que_id</td><td class='c' style='width:70%'>answer</td><td class='c'>gold</td><td class='c'>pred</td><td class='c'>diff</td></tr>
        {}
        </table>
    """.format(tbody)

    html = """
    <html>
    <head>
        <title>{}</title>
        {}
    <head>
    <body>
        {}
    </body>
    </html>
    """.format(title, tplt_style, body)
    return html
    


def print_args(args, path=None):
    '''
    Print args to log file
    '''
    if path:
        output_file = open(path, 'w')
    logger = logging.getLogger(__name__)
    logger.info("Arguments:")
    args.command = ' '.join(sys.argv)
    items = vars(args)
    for key in sorted(items.keys(), key=lambda s: s.lower()):
        value = items[key]
        if not value:
            value = "None"
        logger.info("  " + key + ": " + str(items[key]))
        if path is not None:
            output_file.write("  " + key + ": " + str(items[key]) + "\n")
    if path:
        output_file.close()
    del args.command

def gen_word_weights(ngram_weights):
    """
    Convert weights of n gram tokens to weights of words (1-gram).
    Weight of a word is calculated by sum up weights of tokens containing the word.
    :param ngram_weights: Dict with tuples of n-gram tokens as keys and weights as values
    :return: Dict with words as keys and weights as values.
    >>> gen_word_weights({('a', 'b'): 1, ('b', 'c'): 2})
    >>> {'a': 1, 'b': 3, 'c': 2}
    """
    dict_weight = {}
    for tokens in ngram_weights:
        weight = ngram_weights[tokens]
        for t in tokens:
            try:
                dict_weight[t] += weight
            except KeyError:
                dict_weight[t] = weight
    dict_weight[''] = 0
    weight_max = max(dict_weight.values())
    for k in dict_weight:
        # dict_weight[k] = dict_weight[k] * 1
        dict_weight[k] = dict_weight[k] / float(weight_max)
    print(sorted(dict_weight.values(), reverse=True)[:10])
    return dict_weight


if __name__ == '__main__':
    import doctest
    doctest.testmod(verbose=True)

