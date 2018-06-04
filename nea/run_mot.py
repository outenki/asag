#!/usr/bin/env python

import os
import argparse
from multiprocessing import Pool
from datetime import datetime

parser = argparse.ArgumentParser()
parser.add_argument('-d', '--data_path', dest='data_path', required=True, help='Path to data, it is expected to contain subdirs of real input data(for each question e.g.)')
parser.add_argument('-p', '--process', dest='num_process', type=int, required=True, help='Number of multiprocess')
parser.add_argument('-g', '--gpu', dest='gpu', required=True, help='Code of GPU device')
parser.add_argument('-e', '--emb', dest='emb', required=False, default='', help='w2v file')
parser.add_argument('-dim', '--emb_dim', dest='dim', type=int, required=False, default=50, help='w2v file')
args = parser.parse_args()

# Data path and output path
dt = datetime.now()
path_data = args.data_path
if path_data.endswith('/'):
    path_data = path_data[:-1]
path_out = path_data + '_result_mot' + dt.strftime('%Y%m%d%H%M%S')
if not os.path.exists(path_out):
    try:
        os.mkdir(path_out)
    except Exception as e:
        print('Failed to create output path: %s' % e)

def run_shell(path_i, path_o, emb, pmt_id):
    print('processing %s' % path_i)
    cmd="MKL_THREADING_LAYER=GNU KERAS_BACKEND='theano' THEANO_FLAGS='device=cuda%s,floatX=float32' python train_nea.py -tr %s/train.tsv -tu %s/dev.tsv -ts %s/test.tsv -p %s -o %s -e %d -c 0 -r 250 -t breg --skip-init-bias --aggregation mot" % (args.gpu, path_i, path_i, path_i, pmt_id, path_o, args.dim)
    if emb:
        cmd += ' --emb %s' % emb
    print('cmd: %s' % cmd)
    os.system(cmd)

def run_on_list(path_data, path_in, path_out):
    # path_data/path_in is the fullpath of input data path
    # this function will run over all the subdir in path_data
    # and generate the results using the subdir's name
    for p in path_in:
        run_shell('%s/%s' % (path_data, p), '%s/%s' %(path_out, p), args.emb, p)

answer_path = sorted(os.listdir(path_data))
print(answer_path)
print(args.dim)
# run_on_list(path_data, answer_path, path_out)
pool = Pool()
for i in range(args.num_process):
    pool.apply_async(run_on_list, args=(path_data, answer_path[i::args.num_process], path_out))
pool.close()
pool.join()

