#!/usr/bin/env python
'''
Take tsv file as input, in format of :
'essay_id\tessay_set\tessay\trater1_domain1\trater2_domain1\trater3_domain1\tdomain1_score\trater1_domain2\trater2_domain2\tdomain2_score\trater1_trait1\trater1_trait2\trater1_trait3\trater1_trait4\trater1_trait5\trater1_trait6\trater2_trait1\trater2_trait2\trater2_trait3\trater2_trait4\trater2_trait5\trater2_trait6\trater3_trait1\trater3_trait2\trater3_trait3\trater3_trait4\trater3_trait5\trater3_trait6\n'
Generate id files ( train_id.txt, dev_id.txt and test_id.txt), which will be input of preprocess_asap.py to generate data for nea model
'''

import argparse
import os
from itertools import groupby
from random import shuffle

parser = argparse.ArgumentParser()

parser.add_argument('-i', '--input', dest='input', type=str, metavar='Input file', required=True, help='Name of input file')
parser.add_argument('-o', '--output', dest='output', type=str, metavar='Data name', required=True, help='Name of data, used to name the output folder')
parser.add_argument('-qa', '--question_or_answer', dest='qa', type=str, metavar='Generate data by question or by answer.', required=False, default='q', help="If 'answer' is set, data will be generated for leave-one-out, generating data for each answer. 'q' as default")
parser.add_argument('-rt', '--ratio_train', dest='ratio_train', type=float, metavar='Ratio of train data', required=False, default=0.8, help='Ratio of training data among the dataset, and the left will be used as dev and test data. 0.8 as default')
parser.add_argument('-ts', '--training_scale', dest='training_scale', type=int, metavar='Training scale', required=False, default=1.0, help='0~1, control size of training data. 1 as default.')

args = parser.parse_args()
output = args.output
ratio_train = args.ratio_train
scale_train = args.training_scale
if not os.path.exists(output):
    os.mkdir(output)

# read from tsv files
with open(args.input, 'r') as fi:
    fi.readline()
    lines = list(map(lambda x:x.split('\t'), fi.readlines()))

# sort by prmpt id
lines.sort(key=lambda x:x[1])
data_dict = dict()
for prmpt, data in groupby(lines, key=lambda x:x[1]):
    data_dict[prmpt] = list(data)

for prmpt in sorted(data_dict.keys()):
    # generate id files for each prmpt/question
    print('generate for %s' % prmpt)
    id_answers = data_dict[prmpt]
    ids = list(map(lambda x:x[0], id_answers))

    # generate for each question
    if args.qa == 'q':
        # seperate to training data, dev data and test data
        len_data = len(ids)
        range_train = int(len_data * ratio_train)
        range_dev = range_test = int(len_data * ( 1 - ratio_train ) / 2)
        
        if args.training_scale <= 1:
            range_train *= args.training_scale
        else:
            range_train = args.training_scale
        range_train = int(range_train)
        train_ids = ids[:int(range_train)] # scale the training data
        dev_ids = ids[range_train:range_train+range_dev]
        test_ids = ids[-range_test:]
        print(train_ids)
        print(dev_ids)
        print(test_ids)

        # create folder
        ans_path = "%s/%s" % (output, prmpt)
        if not os.path.exists(ans_path):
            os.makedirs(ans_path)
        with open(ans_path + '/test_ids.txt', 'w') as fn:
            fn.writelines('\n'.join(test_ids)+'\n')
        with open(ans_path + '/train_ids.txt', 'w') as fn:
            fn.writelines('\n'.join(train_ids)+'\n')
        with open(ans_path + '/dev_ids.txt', 'w') as fn:
            fn.writelines('\n'.join(dev_ids)+'\n')
        
    else: # generate for each answer
        for i, idx in enumerate(ids):
            # generate a dir for each answer
            ans_path = "%s/%s_%s" % (output, prmpt, i+1)
            if not os.path.exists(ans_path):
                os.makedirs(ans_path)
                # os.mkdir(ans_path)
            # generate train_id.tsv, dev_id.tsv and test_id.tsv for each answer
            test_id = idx
            train_dev_id = ids[:i] + ids[i+1:]
            # shuffle(train_dev_id)
            edge_train_dev = int(len(train_dev_id) * ratio_train)
            train_ids = train_dev_id[:edge_train_dev]
            dev_ids = train_dev_id[edge_train_dev:]
            with open(ans_path + '/test_ids.txt', 'w') as fn:
                fn.write(test_id + '\n')
            with open(ans_path + '/train_ids.txt', 'w') as fn:
                fn.writelines('\n'.join(train_ids)+'\n')
            with open(ans_path + '/dev_ids.txt', 'w') as fn:
                fn.writelines('\n'.join(dev_ids)+'\n')

