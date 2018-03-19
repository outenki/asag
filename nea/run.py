#!/usr/bin/env python

import os
import argparse
from multiprocessing import Process, Pool

parser = argparse.ArgumentParser()
parser.add_argument('-d', '--data_path', dest='data_path', required=True, help='Path to data')
parser.add_argument('-p', '--process', dest='num_process', type=int, required=True, help='Number of multiprocess')
parser.add_argument('-g', '--gpu', dest='gpu', required=True, help='Code of GPU device')
args = parser.parse_args()

# Data path and output path
path_data = args.data_path
if path_data.endswith('/'):
    path_data = path_data[:-1]
path_out = path_data + '_result'
if not os.path.exists(path_out):
    try:
        os.mkdir(path_out)
    except Exception as e:
        print 'Failed to create output path: %s', e

def run_shell(path_i, path_o):
    cmd="MKL_THREADING_LAYER=GNU KERAS_BACKEND='theano' THEANO_FLAGS='device=cuda%s,floatX=float32' python train_nea.py -tr %s/train.tsv -tu %s/dev.tsv -ts %s/test.tsv -p 0 -o %s" % (args.gpu, path_i, path_i, path_i, path_o)
    os.system(cmd)

def run_on_list(path_data, path_in, path_out):
    for p in path_in:
        run_shell('%s/%s' % (path_data, p), '%s/%s' %(path_out, p))

answer_path = sorted(os.listdir(path_data))
print answer_path
pool = Pool()
for i in range(args.num_process):
    pool.apply_async(run_on_list, args=(path_data, answer_path[i::args.num_process], path_out))
pool.close()
pool.join()

