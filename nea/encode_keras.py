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
from keras.models import load_model
from nea.my_layers import MeanOverTime
from nea.optimizers import get_optimizer

parser = argparse.ArgumentParser() 
parser.add_argument('-pmt', '--prompt-id', dest='prompt_id', type=int, required=True, help='Prompt ID') 
parser.add_argument('-laf', '--load-args-from', dest='args_file', type=str, required=True, help='File of arguments for training model') 
parser.add_argument('-lmf', '--load-model-from', dest='trained_model_file', type=str, required=True, help='Path to pretrained model for encoder') 
parser.add_argument('-lwf', '--load-weight-from', dest='trained_weight_file', type=str, required=True, help='Path to pretrained weight for encoder') 
parser.add_argument('-lvf', '--load-vocab-from', dest='vocab_file', type=str, required=True, help='Path to pickle file of vocab') 
parser.add_argument('-o', '--output', dest='output', type=str, required=True, help='Path to output folder.')
parser.add_argument('-d', '--input', dest='data_path', type=str, required=True, help='Path to input dataset.')
parser.add_argument('-e', '--evaluate', dest='evaluate', type=bool, required=True, help='If evaluate is set, the encoder model would output scores and give an evaluation. Otherwise the embedding will be output.')
args = parser.parse_args()

output = args.output
if not os.path.exists(output):
    os.makedirs(output)
train_path = '%s/%d/train.tsv' % (args.data_path, args.prompt_id)
dev_path = '%s/%d/dev.tsv' % (args.data_path, args.prompt_id)
test_path = '%s/%d/test.tsv' % (args.data_path, args.prompt_id)

# set logger
logger = logging.getLogger()
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fh = logging.FileHandler('%s/log.txt' % output)
fh.setLevel(logging.INFO)
fh.setFormatter(formatter)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
ch.setFormatter(formatter)
logger.addHandler(fh)
logger.addHandler(ch)

logger.info('Load vocab from %s', args.vocab_file)
with open(args.vocab_file, 'rb') as fpk:
    vocab = pk.load(fpk)
with open(args.args_file, 'rb') as fpk:
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

# load best model
logger.info('Loading weight from %s', args.trained_weight_file)
encoder_weight = create_model(args=pre_args, initial_mean_value=test_y.mean(axis=0), vocab=vocab)
encoder_weight.compile(optimizer = optimizer, loss = loss)
encoder_weight.load_weights(args.trained_weight_file, by_name=True)
encoder_weight.save(output + '/model_encoder_weight.h5', overwrite=True)

logger.info('Loading weight from %s', args.trained_weight_file)
encoder_load = load_model(args.trained_model_file, custom_objects={'MeanOverTime': MeanOverTime()})
encoder_load.save_weights(output + '/weight_encoder_load.h5', overwrite=True)
encoder_load.save(output + '/model_encoder_load.h5', overwrite=True)

plot_file_weight = '%s/model_weight.png' % output
plot_file_load = '%s/model_load.png' % output
logger.info('Plot model to %s', plot_file_weight)
plot_model(encoder_weight, to_file = plot_file_weight,  show_shapes = True, show_layer_names=True)
plot_model(encoder_load, to_file = plot_file_load,  show_shapes = True, show_layer_names=True)

np.savetxt(output + '/test_x.txt', test_x, fmt='%d')
logger.info("Evaluate model:")
test_pred_weight = encoder_weight.predict(test_x, batch_size = 180).squeeze() * 3
test_pred_load = encoder_load.predict(test_x, batch_size = 180).squeeze() * 3
np.savetxt(output + '/test_pred_weight.txt', test_pred_weight, fmt='%.4f')
np.savetxt(output + '/test_pred_load.txt', test_pred_load, fmt='%.4f')

np.savetxt(output + '/test_y_org.txt', test_y_org, fmt='%.4f')
np.savetxt(output + '/test_y.txt', test_y, fmt='%.4f')

test_pred_weight = np.rint(test_pred_weight)
test_pred_load = np.rint(test_pred_load)
qwk_weight = QWK(test_y_org.astype(int), test_pred_weight.astype(int), labels=None, weights='quadratic', sample_weight=None)
qwk_load = QWK(test_y_org.astype(int), test_pred_load.astype(int), labels=None, weights='quadratic', sample_weight=None)
logger.info('QWK_weight: %f', qwk_weight)
logger.info('QWK_LOAD: %f', qwk_load)

# # Try to load from the file again
# logger.info('Load encoder same to encoder_load again and save the pred to %s', output + '/test_pred_load_2.txt')
# encoder_load = load_model(args.trained_model_file, custom_objects={'MeanOverTime': MeanOverTime()})
# test_pred_load = encoder.predict(test_x, batch_size = 180).squeeze() * 3
# np.savetxt(output + '/test_pred_load_2.txt', test_pred_load, fmt='%.4f')
# 
# # Try to save encoder_load to file and laod it again 
# logger.info('Save the model encoder_load  and load the file again and save the pred to %s', output + '/test_pred_save_load.txt') 
# encoder_load.save(output+'/best_model2.h5')
# encoder_load = load_model(output+'/best_model2.h5', custom_objects={'MeanOverTime': MeanOverTime()})
# test_pred_load = encoder.predict(test_x, batch_size = 180).squeeze() * 3
# np.savetxt(output + '/test_pred_save_load.txt', test_pred_load, fmt='%.4f')
# 
# # Try to save encoder_load with model.model.save() to file and load it again 
# logger.info('Save the encoder_load and load the file again and save the pred to %s', output + '/test_pred_save2_load.txt') 
# encoder_load.model.save(output+'/best_model_save2.h5')
# encoder_load = load_model(output+'/best_model_save2.h5', custom_objects={'MeanOverTime': MeanOverTime()})
# test_pred_load = encoder.predict(test_x, batch_size = 180).squeeze() * 3
# np.savetxt(output + '/test_pred_save2_load.txt', test_pred_load, fmt='%.4f')
