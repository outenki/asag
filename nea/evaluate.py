#!/usr/bin/env python

# Read results of nea, and generate result file.
# 1. Read test.tsv from data path
# 2. From test.tsv read the answer id, prmt id, answer
# 3. From data/scores/ave read golden scores based on answer id
# 4. From data/raw read the questions and reference
# 5. The format of result file is:
#     score of X\tres\tgolden\tdiff\tdiff_abs\tdiff_round\tque\tref\tans

import os
import argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('-p', '--predict', dest='predict_path', type=str, required=True, help='Path of predicted results')
parser.add_argument('-t', '--tsv', dest='tsv_file', type=str, required=True, help='Path to data-file.tsv')
parser.add_argument('-td', '--path_id_tsv', dest='path_id_tsv', type=str, required=True, help='Path to tsv data file.tsv')
parser.add_argument('-i', '--index', dest='index_file', type=str, required=True, help='Index file from prmpt_id to que_id')
parser.add_argument('-rd', '--path_raw_data', dest='path_raw_data', type=str, required=True, help='Path to meta data')
parser.add_argument('-o', '--output', dest='output', type=str, required=True, help='Path to output dir')

args=parser.parse_args()
path_pred = args.predict_path
path_raw_data = args.path_raw_data
rslt_list = sorted(os.listdir(path_pred))
path_id_tsv = args.path_id_tsv

# read tsv file
with open(args.tsv_file, 'r') as ft:
    ft.readline()
    tsv = ft.readlines()

# read index file
with open(args.index_file, 'r') as fi:
    index = fi.readlines()

# read reference
with open('%s/raw/answers' % path_raw_data, 'r') as fr:
    refs = fr.readlines()
# read questions
with open('%s/raw/questions' %  path_raw_data, 'r') as fq:
    ques = fq.readlines()


data = []
for rslt_fold in rslt_list:
    path_rslt = '%s/%s/' % (path_pred, rslt_fold)
    # read que_id
    prmpt = int(rslt_fold.split('_')[0])
    que_id = index[int(prmpt-1)].split('\t')[1].strip()
    
    # read ref and que
    que = ''
    for q in ques:
        if q.startswith(que_id):
            que = q.strip()
            break
    ref = ''
    for r in refs:
        if r.startswith(que_id):
            ref = r.strip()
            break

    # read result
    with open('%s/preds/test_pred_-1.txt' % path_rslt, 'r') as fs:
        preds = list(map(lambda x:x.strip(), fs.readlines()))
        preds = np.array(preds).astype(np.float)

    # read golden scores
    with open('%s/preds/test_ref.txt' % path_rslt, 'r') as fr:
        scores = list(map(lambda x:x.strip(), fr.readlines()))
        scores = np.array(scores).astype(np.float)

    # read answers
    with open('%s/%s/test.tsv' % (path_id_tsv, rslt_fold), 'r') as ft:
        ft.readline()
        ans = ft.readlines()
        ans = list(map(lambda x:x.split('\t')[2], ans))
        
    diff = scores - preds
    diff_abs = np.abs(diff)
    diff_round = np.round(diff_abs)
    
    # generate result data
    for i, s in enumerate(preds):
        data.append('%s\t%f\t%f\t%f\t%f\t%f\t%s\t%s\t%s\n' % (que_id, preds[i], scores[i], diff[i], diff_abs[i], diff_round[i], que, ref, ans[i].strip()))


os.mkdir(args.output)
with open('%s/%s'%(args.output,'result.txt'), 'w') as f:
    f.writelines(data)
