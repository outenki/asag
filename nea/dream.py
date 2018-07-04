#!/usr/bin/env python
from sklearn.metrics import cohen_kappa_score as QWK
from keras.utils.vis_utils import plot_model
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
from nea.models import create_model_nea, create_model_mn_with_coded_data
# from nea.my_layers import MeanOverTime
import ipdb
from keras.models import Model

logger = logging.getLogger(__name__)

###############################################################################################################################
## Parse arguments
#

parser = argparse.ArgumentParser()
parser.add_argument('-dp','--data-path', dest='data_path', type=str, required=True, help="The path to data files, including train.tsv, dev.tsv and test.tsv")
parser.add_argument("-o", "--out-dir", dest="out_dir_path", type=str, metavar='<str>', required=True, help="The path to the output directory")
parser.add_argument("-p", "--prompt", dest="prompt_id", type=int, metavar='<int>', default=0, required=False, help="Promp ID for ASAP dataset. '0' means all prompts.")
parser.add_argument("-t", "--type", dest="model_type", type=str, metavar='<str>', default='regp', help="Model type (reg|regp|breg|bregp) (default=regp)")
parser.add_argument("-u", "--rec-unit", dest="recurrent_unit", type=str, metavar='<str>', default='lstm', help="Recurrent unit type (lstm|gru|simple) (default=lstm)")
parser.add_argument("-oa", "--optimizer-algorithm", dest="optimizer_algorithm", type=str, metavar='<str>', default='rmsprop', help="Optimization algorithm (rmsprop|sgd|adagrad|adadelta|adam|adamax) (default=rmsprop)")
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

if not os.path.exists(out_dir):
    os.makedirs(out_dir)

with open(out_dir + '/args.pkl', 'wb') as args_file:
    pk.dump(args, args_file)

U.mkdir_p(out_dir + '/preds')
U.set_logger(out_dir)
U.print_args(args)

f_encoded_mem = os.path.join(out_dir, 'encoded_mem.txt')
f_encoded_train = os.path.join(out_dir, 'encoded_train.txt')
f_encoded_dev = os.path.join(out_dir, 'encoded_dev.txt')
f_encoded_test = os.path.join(out_dir, 'encoded_test.txt')

assert args.model_type in {'reg', 'regp', 'breg', 'bregp'}
assert args.optimizer_algorithm in {'rmsprop', 'sgd', 'adagrad', 'adadelta', 'adam', 'adamax'}
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
(train_x, train_y, train_pmt), (dev_x, dev_y, dev_pmt), (test_x, test_y, test_pmt), memories, vocab, overal_maxlen = dataset.get_data(
    args.data_path, args.prompt_id, args.vocab_size, args.maxlen, to_lower=True, vocab_path=args.vocab_path)

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
    memories = sequence.pad_sequences(memories, maxlen=overal_maxlen)
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
logger.info('  memory shape: ' + str(np.array(memories).shape))

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

np.savetxt(out_dir + '/test_x.txt', test_x, fmt='%d')
np.savetxt(out_dir + '/test_y_org.txt', test_y_org, fmt='%.4f')
np.savetxt(out_dir + '/test_y.txt', test_y, fmt='%.4f')
###############################################################################################################################
## Optimizaer algorithm
from nea.optimizers import get_optimizer
optimizer_nea = get_optimizer(args.optimizer_algorithm)
optimizer_mn = get_optimizer(args.optimizer_algorithm)

###############################################################################################################################
## Building model and train with answers in training data

if args.loss == 'mse':
    loss = 'mean_squared_error'
else:
    loss = 'mean_absolute_error'

# ''' Open this when you need not to train the nea model
model_nea = create_model_nea(args, train_y.mean(axis=0), vocab)
model_nea.compile(loss=loss, optimizer=optimizer_nea, metrics=['accuracy'])

###############################################################################################################################
## Plotting model
logger.info('Plotting model_nea to %s...', out_dir + '/model_nea.png')
plot_model(model_nea, to_file = out_dir + '/model_nea.png', show_shapes=True, show_layer_names=True)

###############################################################################################################################

# Train the model and save the best automatically
f_h5_best_weights_nea_cb = os.path.join(out_dir, 'best_weights_nea.h5')
f_log = os.path.join(out_dir, 'train.log')
cpw_cb = keras.callbacks.ModelCheckpoint(filepath = f_h5_best_weights_nea_cb, monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=True, mode='auto')
es_cb = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, verbose=1, mode='auto')
cbks = [cpw_cb, es_cb]

logger.info('Training ...')
model_nea.fit(train_x, train_y, batch_size=args.batch_size, epochs=args.epochs, verbose=1, callbacks=cbks, validation_data=(dev_x, dev_y), shuffle=True)

# Load the best model_nea
logger.info('Loading weights saved by checkpoint from %s', f_h5_best_weights_nea_cb)
best_nea = model_nea
best_nea.load_weights(f_h5_best_weights_nea_cb)
logger.info("Architecture of the best tuned NEA model:")
best_nea.summary()

test_pred = best_nea.predict(test_x).squeeze() * 3
qwk = QWK(test_y_org.astype(int), np.rint(test_pred).astype(int), labels=None, weights='quadratic', sample_weight=None)
logger.info('QWK of best trained nea: %f\n', qwk)
output_pred = os.path.join(out_dir, 'test_pred_best_nea.txt')
logger.info('Save the prediction to : %s', output_pred)
np.savetxt(output_pred, test_pred, fmt='%.4f')

logger.info('Create encoder...\n')
encoder = Model(inputs = model_nea.input, outputs = model_nea.get_layer('mean_over_time_1').output)
logger.info("Architecture of the best tuned encoder:")
encoder.summary()


# encode answers and memories
logger.info("Encoding answers and memories ...")
encoded_mem = encoder.predict(memories)
encoded_train_ans = encoder.predict(train_x)
encoded_dev_ans = encoder.predict(dev_x)
encoded_test_ans = encoder.predict(test_x)
logger.info('Shape of encoded_mem: {}'.format(encoded_mem.shape))

logger.info('Saving encoded memories to %s', f_encoded_mem)
np.savetxt(f_encoded_mem, encoded_mem, delimiter='\t')
logger.info('Saving encoded training data to %s', f_encoded_train)
np.savetxt(f_encoded_train, encoded_train_ans, delimiter='\t')
logger.info('Saving encoded test data to %s', f_encoded_test)
np.savetxt(f_encoded_test, encoded_test_ans, delimiter='\t')
logger.info('Saving encoded dev data to %s', f_encoded_dev)
np.savetxt(f_encoded_dev, encoded_dev_ans, delimiter='\t')

# Open this when you need not to train the nea model ''' 

# Load the pre-encoded data
logger.info('Loading encoded memories from %s', f_encoded_mem)
encoded_mem = np.loadtxt(f_encoded_mem, dtype=float, delimiter='\t')
logger.info('Loading encoded training data from %s', f_encoded_train)
encoded_train_ans = np.loadtxt(f_encoded_train, dtype=float, delimiter='\t')
logger.info('Loading encoded test data from %s', f_encoded_test)
encoded_test_ans = np.loadtxt(f_encoded_test, dtype=float, delimiter='\t')
logger.info('Loading encoded dev data from %s', f_encoded_mem)
encoded_dev_ans = np.loadtxt(f_encoded_dev, dtype=float, delimiter='\t')

# Create Memory Network model
logger.info('Create MN mode ...')
# create MN model with encoded sentences as inputs
model_mn = create_model_mn_with_coded_data(encoded_mem.shape[1], len(encoded_mem))
logger.info('Compile ...')
model_mn.compile(optimizer=optimizer_mn, loss=loss, metrics=['accuracy'])
logger.info('Plotting model_mn to file %s ...', out_dir+'/model_mn.png')
plot_model(model_mn, to_file = out_dir + '/model_mn.png', show_shapes=True, show_layer_names=True)

######################### Prepare training data for memory network: [answer, memories, scores] ######################### 
train_x_mn = [encoded_train_ans]
for i in range(encoded_mem.shape[0]):
    train_x_mn.append(np.array([encoded_mem[i]] * len(encoded_train_ans)))

dev_x_mn = [encoded_dev_ans]
for i in range(encoded_mem.shape[0]):
    dev_x_mn.append(np.array([encoded_mem[i]] * len(encoded_dev_ans)))

test_x_mn = [encoded_test_ans]
for i in range(encoded_mem.shape[0]):
    test_x_mn.append(np.array([encoded_mem[i]] * len(encoded_test_ans)))

logger.info('Size of training data for MN model: %d', len(train_x_mn[0]))
logger.info('Size of test data for MN model: %d', len(test_x_mn[0]))
logger.info('Size of dev data for MN model: %d', len(dev_x_mn[0]))

######################### Train the MN model ######################### 
f_h5_best_weights_cb = os.path.join(out_dir, 'best_weights_mn.h5')
logger.info('The best tuned model will be saved to %s', f_h5_best_weights_cb)
cpm_cb = keras.callbacks.ModelCheckpoint(filepath = f_h5_best_weights_cb, monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=True, mode='auto')
es_cb = keras.callbacks.EarlyStopping(monitor='val_loss', patience=30, verbose=1, mode='auto')
cbks = [cpm_cb, ]
logger.info('Training MN model...')
model_mn.fit(train_x_mn, train_y, batch_size=args.batch_size, epochs=args.epochs, verbose=1, callbacks=cbks, validation_data=(dev_x_mn, dev_y), shuffle=True)

######################### Load the best tuned MN model ######################### 
logger.info('Loading the best tuned weight from %s ...', f_h5_best_weights_cb)
best_mn = model_mn
best_mn.load_weights(f_h5_best_weights_cb)

######################### Evaluate the best tuned MN model ######################### 
test_pred = best_mn.predict(test_x_mn).squeeze() * 3
qwk = QWK(test_y_org.astype(int), np.rint(test_pred).astype(int), labels=None, weights='quadratic', sample_weight=None)
with open(os.path.join(out_dir, 'qwk_mn.txt'), 'w') as f:
    f.write('%d\n' % qwk)
logger.info('QWK of best trained MN: %f\n', qwk)
# ipdb.set_trace()
output_pred = os.path.join(out_dir, 'test_pred_best_mn.txt')
logger.info('Save the prediction to : %s', output_pred)
np.savetxt(output_pred, test_pred, fmt='%.4f')
