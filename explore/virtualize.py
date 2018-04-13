import os
import argparse
import numpy as np
import spacy

import utils_asag as UA

PARSER = argparse.ArgumentParser()
PARSER.add_argument('-p', '--prediction', dest='pred', type=str, metavar='Prediction file', required=True, help="Txt file of prediction. It is saved by numpy.savetxt()")
PARSER.add_argument('-w', '--token_weight', dest='weight', type=str, metavar='Token weights file', required=True, help="File in form of '(token1, token2, ...)'\tweight")

ARGS = PARSER.parse_args()
NLP = spacy.load('en')

print('Processing %s' % ARGS.pred)
path, fname = os.path.split(ARGS.pred)
title, ext = os.path.splitext(fname)

# Generate weights for words
weights = np.loadtxt(ARGS.weight, dtype=str, delimiter='\t')
rm_list = ['(', ')', "'", ' ']
token_weights = dict()
for tokens, weight in weights:
    for t in rm_list:
        tokens = tokens.replace(t, '')
    tokens = tuple(tokens.split(','))
    token_weights[tokens] = abs(float(weight))

word_weights = UA.gen_word_weights(token_weights)

pred = np.loadtxt(ARGS.pred, dtype=str, delimiter='\t')
# sort by diff
pred = np.array(sorted(pred, key=lambda r:float(r[-1]), reverse=True))
html = UA.genertate_weight_html_pred(NLP, pred, word_weights, title) 

with open('%s/%s.html' % (path, title), 'w') as fh:
    fh.write(html)

