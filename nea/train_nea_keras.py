#!/usr/bin/env python
from nea.my_layers import MeanOverTime
from sklearn.metrics import cohen_kappa_score as QWK
import os
import argparse
import logging
import numpy as np
import nea.utils as U
import pickle as pk
# from nea.asap_evaluator import Evaluator
import nea.asap_reader as dataset
from keras.preprocessing import sequence
import keras
from keras.models import load_model

logger = logging.getLogger(__name__)

###############################################################################################################################
## Parse arguments
#

parser = argparse.ArgumentParser()
parser.add_argument("-tr", "--train", dest="train_path", type=str, metavar='<str>', required=True, help="The path to the training set")
parser.add_argument("-tu", "--tune", dest="dev_path", type=str, metavar='<str>', required=True, help="The path to the development set")
parser.add_argument("-ts", "--test", dest="test_path", type=str, metavar='<str>', required=True, help="The path to the test set")
parser.add_argument("-o", "--out-dir", dest="out_dir_path", type=str, metavar='<str>', required=True, help="The path to the output directory")
parser.add_argument("-p", "--prompt", dest="prompt_id", type=int, metavar='<int>', default=0, required=False, help="Promp ID for ASAP dataset. '0' means all prompts.")
parser.add_argument("-t", "--type", dest="model_type", type=str, metavar='<str>', default='regp', help="Model type (reg|regp|breg|bregp) (default=regp)")
parser.add_argument("-u", "--rec-unit", dest="recurrent_unit", type=str, metavar='<str>', default='lstm', help="Recurrent unit type (lstm|gru|simple) (default=lstm)")
parser.add_argument("-a", "--algorithm", dest="algorithm", type=str, metavar='<str>', default='rmsprop', help="Optimization algorithm (rmsprop|sgd|adagrad|adadelta|adam|adamax) (default=rmsprop)")
parser.add_argument("-l", "--loss", dest="loss", type=str, metavar='<str>', default='mse', help="Loss function (mse|mae) (default=mse)")
parser.add_argument("-e", "--embdim", dest="emb_dim", type=int, metavar='<int>', default=50, help="Embeddings dimension (default=50)")
parser.add_argument("-c", "--cnndim", dest="cnn_dim", type=int, metavar='<int>', default=0, help="CNN output dimension. '0' means no CNN layer (default=0)")
parser.add_argument("-w", "--cnnwin", dest="cnn_window_size", type=int, metavar='<int>', default=3, help="CNN window size. (default=3)")
parser.add_argument("-r", "--rnndim", dest="rnn_dim", type=int, metavar='<int>', default=300, help="RNN dimension. '0' means no RNN layer (default=300)")
parser.add_argument("-b", "--batch-size", dest="batch_size", type=int, metavar='<int>', default=32, help="Batch size (default=32)")
parser.add_argument("-v", "--vocab-size", dest="vocab_size", type=int, metavar='<int>', default=4000, help="Vocab size (default=4000)")
parser.add_argument("--aggregation", dest="aggregation", type=str, metavar='<str>', default='mot', help="The aggregation method for regp and bregp types (mot|attsum|attmean) (default=mot)")
parser.add_argument("--dropout", dest="dropout_prob", type=float, metavar='<float>', default=0.5, help="The dropout probability. To disable, give a negative number (default=0.5)")
parser.add_argument("--vocab-path", dest="vocab_path", type=str, metavar='<str>', help="(Optional) The path to the existing vocab file (*.pkl)")
parser.add_argument("--skip-init-bias", dest="skip_init_bias", action='store_true', help="Skip initialization of the last layer bias")
parser.add_argument("--emb", dest="emb_path", type=str, metavar='<str>', help="The path to the word embeddings file (Word2Vec format)")
parser.add_argument("--epochs", dest="epochs", type=int, metavar='<int>', default=50, help="Number of epochs (default=50)")
parser.add_argument("--maxlen", dest="maxlen", type=int, metavar='<int>', default=0, help="Maximum allowed number of words during training. '0' means no limit (default=0)")
parser.add_argument("--seed", dest="seed", type=int, metavar='<int>', default=1234, help="Random seed (default=1234)")
args = parser.parse_args()
out_dir = args.out_dir_path

with open(out_dir + '/args.pkl', 'wb') as args_file:
    pk.dump(args, args_file)

U.mkdir_p(out_dir + '/preds')
U.set_logger(out_dir)
U.print_args(args)

assert args.model_type in {'reg', 'regp', 'breg', 'bregp'}
assert args.algorithm in {'rmsprop', 'sgd', 'adagrad', 'adadelta', 'adam', 'adamax'}
assert args.loss in {'mse', 'mae'}
assert args.recurrent_unit in {'lstm', 'gru', 'simple'}
assert args.aggregation in {'mot', 'attsum', 'attmean'}

if args.seed > 0:
    np.random.seed(args.seed)

if not args.prompt_id:
    args.prompt_id = 0

###############################################################################################################################
## Prepare data
#


# data_x is a list of lists
(train_x, train_y, train_pmt), (dev_x, dev_y, dev_pmt), (test_x, test_y, test_pmt), vocab, vocab_size, overal_maxlen, num_outputs = dataset.get_data(
    (args.train_path, args.dev_path, args.test_path), args.prompt_id, args.vocab_size, args.maxlen, tokenize_text=True, to_lower=True, sort_by_len=False, vocab_path=args.vocab_path)

# Dump vocab
with open(out_dir + '/vocab.pkl', 'wb') as vocab_file:
    pk.dump(vocab, vocab_file)

# Pad sequences for mini-batch processing
if args.model_type in {'breg', 'bregp'}:
    assert args.rnn_dim > 0
    assert args.recurrent_unit == 'lstm'
    train_x = sequence.pad_sequences(train_x, maxlen=overal_maxlen)
    dev_x = sequence.pad_sequences(dev_x, maxlen=overal_maxlen)
    test_x = sequence.pad_sequences(test_x, maxlen=overal_maxlen)
else:
    train_x = sequence.pad_sequences(train_x)
    dev_x = sequence.pad_sequences(dev_x)
    test_x = sequence.pad_sequences(test_x)

###############################################################################################################################
## Some statistics
#

import keras.backend as K

train_y = np.array(train_y, dtype=K.floatx())
dev_y = np.array(dev_y, dtype=K.floatx())
test_y = np.array(test_y, dtype=K.floatx())

if args.prompt_id != None:
    train_pmt = np.array(train_pmt, dtype='int32')
    dev_pmt = np.array(dev_pmt, dtype='int32')
    test_pmt = np.array(test_pmt, dtype='int32')

# mfe: most frequent elements
# bincounts: dict objects. each counter stores the frequency of each elements for each column.
# Not so useful here. No idea if anywhere else use this function (!!!!!TO CONFIRM!!!!!)
# Here is used to count the distribution of scores.
bincounts, mfe_list = U.bincounts(train_y)
with open('%s/bincounts.txt' % out_dir, 'w') as output_file:
    for bincount in bincounts:
        output_file.write(str(bincount) + '\n')

logger.info('Statistics:')
# !!!!! This part is unused !!!!!
train_mean = train_y.mean(axis=0)
train_std = train_y.std(axis=0)
dev_mean = dev_y.mean(axis=0)
dev_std = dev_y.std(axis=0)
test_mean = test_y.mean(axis=0)
test_std = test_y.std(axis=0)
logger.info('  train_y mean: %s, stdev: %s, MFC: %s' % (str(train_mean), str(train_std), str(mfe_list)))
# !!!!! This part is unused !!!!!

logger.info('  train_x shape: ' + str(np.array(train_x).shape))
logger.info('  dev_x shape:   ' + str(np.array(dev_x).shape))
logger.info('  test_x shape:  ' + str(np.array(test_x).shape))

logger.info('  train_y shape: ' + str(train_y.shape))
logger.info('  dev_y shape:   ' + str(dev_y.shape))
logger.info('  test_y shape:  ' + str(test_y.shape))


# We need the dev and test sets in the original scale for evaluation
dev_y_org = dev_y.astype(dataset.get_ref_dtype())
test_y_org = test_y.astype(dataset.get_ref_dtype())

# Convert scores to boundary of [0 1] for training and evaluation (loss calculation)
train_y = dataset.get_model_friendly_scores(train_y, train_pmt)
dev_y = dataset.get_model_friendly_scores(dev_y, dev_pmt)
test_y = dataset.get_model_friendly_scores(test_y, test_pmt)

###############################################################################################################################
## Optimizaer algorithm
#

from nea.optimizers import get_optimizer

optimizer = get_optimizer(args)

###############################################################################################################################
## Building model
#

from nea.models import create_model
if args.loss == 'mse':
    loss = 'mean_squared_error'
    metric = 'mean_absolute_error'
else:
    loss = 'mean_absolute_error'
    metric = 'mean_squared_error'

model = create_model(args, train_y.mean(axis=0), vocab)
# model = create_model(args, vocab)
# model = multi_gpu_model(model, gpus=2)
model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])

###############################################################################################################################
## Plotting model
#
from keras.utils.vis_utils import plot_model
plot_model(model, to_file = out_dir + '/model.png', show_shapes=True, show_layer_names=True)

###############################################################################################################################
## Save model architecture
#

logger.info('Saving model architecture')
with open(out_dir + '/model_arch.json', 'w') as arch:
    arch.write(model.to_json(indent=2))
logger.info('  Done')

f_log = os.path.join(out_dir, 'train.log')
tb_cb = keras.callbacks.TensorBoard(log_dir=f_log, histogram_freq=1)
cpw_cb = keras.callbacks.ModelCheckpoint(filepath = os.path.join(out_dir,'best_weight_cb.h5'), monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=True, mode='auto')
cpm_cb = keras.callbacks.ModelCheckpoint(filepath = os.path.join(out_dir,'best_model_cb.h5'), monitor='val_loss', verbose=1, save_best_only=True, mode='auto')
es_cb = keras.callbacks.EarlyStopping(monitor='val_loss', patience=2, verbose=1, mode='auto')
cbks = [tb_cb, cpw_cb, cpm_cb, es_cb]

history = model.fit(train_x, train_y, batch_size=args.batch_size, epochs=args.epochs, verbose=1, callbacks=cbks, validation_data=(dev_x, dev_y), shuffle=True)

# Load best model
logger.info('Loading weights from %s', out_dir + '/best_model_weight_cb.h5')
model_weight = create_model(args, train_y.mean(axis=0), vocab)
model_weight.compile(optimizer = optimizer, loss = loss)
model_weight.load_weights(out_dir + '/best_model_weight_cb.h5', by_name=True)
model_weight.model.save(out_dir+'/best_model.h5', overwrite=True)

logger.info('Loading model from %s', out_dir+'/best_model_cb.h5')
model_load = load_model(out_dir + '/best_model_cb.h5', custom_objects={'MeanOverTime': MeanOverTime()})
model_load.save_weights(out_dir + '/best_model_weights.h5')

np.savetxt(out_dir + '/test_x.txt', test_x, fmt='%d')
np.savetxt(out_dir + '/test_y_org.txt', test_y_org, fmt='%.4f')
np.savetxt(out_dir + '/test_y.txt', test_y, fmt='%.4f')

logger.info('Evaluate model_load:')
# score, accu = model_load.evaluate(test_x, test_y, verbose=1)
test_pred = model_load.predict(test_x).squeeze() * 3
qwk = QWK(test_y_org.astype(int), np.rint(test_pred).astype(int), labels=None, weights='quadratic', sample_weight=None)
np.savetxt(out_dir + '/test_pred_model_load.txt', test_pred, fmt='%.4f')
# logger.info('\tSCROE: %f', score)
# logger.info('\tACCURACYT: %f', accu)
logger.info('\tQWK: %f', qwk)

logger.info('Evaluate model_weight:')
# score, accu = model_weight.evaluate(test_x, test_y, verbose=1)
test_pred = model_weight.predict(test_x).squeeze() * 3
qwk = QWK(test_y_org.astype(int), np.rint(test_pred).astype(int), labels=None, weights='quadratic', sample_weight=None)
np.savetxt(out_dir + '/test_pred_model_weight.txt', test_pred, fmt='%.4f')
# logger.info('\tSCROE: %f', score)
# logger.info('\tACCURACYT: %f', accu)
logger.info('\tQWK: %f', qwk)
