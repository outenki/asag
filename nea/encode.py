#!/usr/bin/env python
import os
import keras.backend as K
import argparse
import logging
import numpy as np
from nea.models import create_model
# from nea.encoder import create_model
import pickle as pk
from sklearn.metrics import cohen_kappa_score as QWK
from keras.utils.vis_utils import plot_model 
from keras.preprocessing import sequence
import nea.asap_reader as dataset
from keras.models import load_model, model_from_json
from nea.my_layers import MeanOverTime
from nea.optimizers import get_optimizer

parser = argparse.ArgumentParser() 
parser.add_argument('-pmt', '--prompt-id', dest='prompt_id', type=int, required=True, help='Prompt ID') 
parser.add_argument('-lf', '--load-from', dest='path_load', type=str, required=True, help='Paht to files to load')
parser.add_argument('-o', '--output', dest='output', type=str, required=True, help='Path to output folder.')
parser.add_argument('-d', '--input', dest='data_path', type=str, required=True, help='Path to input dataset.')
parser.add_argument('-e', '--evaluate', dest='evaluate', type=bool, required=True, help='If evaluate is set, the encoder model would output scores and give an evaluation. Otherwise the embedding will be output.')
args = parser.parse_args()

out_dir = args.output
if not os.path.exists(out_dir):
    os.makedirs(out_dir)
train_path = '%s/%d/train.tsv' % (args.data_path, args.prompt_id)
dev_path = '%s/%d/dev.tsv' % (args.data_path, args.prompt_id)
test_path = '%s/%d/test.tsv' % (args.data_path, args.prompt_id)

# set logger
logger = logging.getLogger()
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fh = logging.FileHandler('%s/log.txt' % out_dir)
fh.setLevel(logging.INFO)
fh.setFormatter(formatter)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
ch.setFormatter(formatter)
logger.addHandler(fh)
logger.addHandler(ch)


# files to load
f_vocab = os.path.join(args.path_load, 'vocab.pkl')
f_args = os.path.join(args.path_load, 'args.pkl')
f_h5_best_model_cb = os.path.join(args.path_load, 'best_model_cb.h5')
f_h5_best_weights_cb = os.path.join(args.path_load, 'best_weights_cb.h5')
f_h5_best_model = os.path.join(args.path_load, 'best_model.h5')
f_h5_best_weights = os.path.join(args.path_load, 'best_model.h5')
f_json_best_model = os.path.join(args.path_load, 'best_model.json')
f_pkl_best_model = os.path.join(args.path_load, 'best_model.pkl')

logger.info('Load vocab from %s', f_vocab)
with open(f_vocab, 'rb') as fpk:
    vocab = pk.load(fpk)
logger.info('Load args from %s', f_vocab)
with open(f_args, 'rb') as fpk:
    pre_args = pk.load(fpk)

logger.info('Reading data from %s', args.data_path)
# test_data: Id      EssaySet        EssayText       Score

# test_data = np.loadtxt(args.data_file, dtype=str, delimiter='\t', skiprows=1, usecols=[0,1,2,3])
# test_x = test_data[:, 2]
# test_y = test_data[:, 3].astype(float)

'''
test_x, test_y, prmt_ids, maxlen_x = read_dataset(
        file_path = args.data_file,
        prompt_id = args.prompt_id,
        vocab= vocab,
        maxlen = pre_args.maxlen,
        tokenize_text = True,
        to_lower = True,
        )

'''
(train_x, train_y, train_pmt), (dev_x, dev_y, dev_pmt), (test_x, test_y, test_pmt), vocab, vocab_size, overal_maxlen, num_outputs = dataset.get_data(
        (train_path, dev_path, test_path),
        pre_args.prompt_id, pre_args.vocab_size, pre_args.maxlen,
        tokenize_text=True, to_lower=True, sort_by_len=False, vocab_path=pre_args.vocab_path)

train_y = np.array(train_y, dtype=K.floatx())
dev_y = np.array(dev_y, dtype=K.floatx())
test_y = np.array(test_y, dtype=K.floatx())
if args.prompt_id != None:
    train_pmt = np.array(train_pmt, dtype='int32')
    dev_pmt = np.array(dev_pmt, dtype='int32')
    test_pmt = np.array(test_pmt, dtype='int32')

test_x = sequence.pad_sequences(test_x, overal_maxlen)
dev_x = sequence.pad_sequences(dev_x, overal_maxlen)

dev_y_org = np.array(dev_y).astype(dataset.get_ref_dtype())
test_y_org = np.array(test_y).astype(dataset.get_ref_dtype())

train_y = dataset.get_model_friendly_scores(train_y, train_pmt)
dev_y = dataset.get_model_friendly_scores(dev_y, dev_pmt)
test_y = dataset.get_model_friendly_scores(test_y, test_pmt)

loss = 'mse'
optimizer = get_optimizer(pre_args)

logger.info('Loading model saved by checkpoint from %s', f_h5_best_model_cb)
model_load_cb = load_model(f_h5_best_model_cb, custom_objects={'MeanOverTime': MeanOverTime()})
model_load_cb.summary()
test_pred = model_load_cb.predict(test_x).squeeze() * 3
qwk = QWK(test_y_org.astype(int), np.rint(test_pred).astype(int), labels=None, weights='quadratic', sample_weight=None)
logger.info('QWK: %f\n', qwk)
np.savetxt(out_dir + '/test_pred_model_load_cb.txt', test_pred, fmt='%.4f')

# Load best model
logger.info('Loading weights saved by checkpoint from %s', f_h5_best_weights_cb)
model_weight_cb = create_model(pre_args, train_y.mean(axis=0), vocab)
model_weight_cb.compile(optimizer = optimizer, loss = loss)
model_weight_cb.load_weights(f_h5_best_weights_cb)
model_weight_cb.summary()
test_pred = model_weight_cb.predict(test_x).squeeze() * 3
qwk = QWK(test_y_org.astype(int), np.rint(test_pred).astype(int), labels=None, weights='quadratic', sample_weight=None)
logger.info('QWK: %f\n', qwk)
np.savetxt(out_dir + '/test_pred_model_weights_cb.txt', test_pred, fmt='%.4f')

logger.info('Loading model with json from %s and h5 from %s', f_json_best_model, f_h5_best_weights)
json_model = open(f_json_best_model).read()
model_json = model_from_json(json_model, custom_objects={'MeanOverTime': MeanOverTime})
model_json.summary()
model_json.compile(loss=loss, optimizer = optimizer)
model_json.load_weights(f_h5_best_weights)
test_pred = model_json.predict(test_x).squeeze() * 3
qwk = QWK(test_y_org.astype(int), np.rint(test_pred).astype(int), labels=None, weights='quadratic', sample_weight=None)
logger.info('QWK: %f\n', qwk)
np.savetxt(out_dir + '/test_pred_model_json.txt', test_pred, fmt='%.4f')

logger.info('Loading model with pkl from %s', f_pkl_best_model)
with open(f_pkl_best_model, 'rb') as fpk:
    model_pkl = pk.load(fpk)
    model_pkl.summary()
test_pred = model_pkl.predict(test_x).squeeze() * 3
qwk = QWK(test_y_org.astype(int), np.rint(test_pred).astype(int), labels=None, weights='quadratic', sample_weight=None)
logger.info('QWK: %f\n', qwk)
np.savetxt(out_dir + '/test_pred_model_json.txt', test_pred, fmt='%.4f')
