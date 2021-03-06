import operator
import numpy as snp
import argparse
import logging
import os
import spacy
from utils import cur_time, tokenize
from bow import *

from itertools import groupby

parser = argparse.ArgumentParser()
parser.add_argument('-ft', '--feature', dest='fea_type', type=str, metavar='Feature type', required=True, help="Feature type (bow|)")
parser.add_argument('-n', '--ngram', dest='ngram', type=int, metavar='N-gram bow', required=True, help="n-gram for bow feature")
parser.add_argument('-q', '--question', dest='que_id', type=int, metavar='Question ID', required=True, help="Question id or prompt id. If it is set, only features for answers to the specific question will be generated. If it i set to minus, features will be generated for each question.")
parser.add_argument('-o', '--output_path', dest='output_path', type=str, metavar='Output Path', required=True, help="Path to store generated feature files")
parser.add_argument('-i', '--input_file', dest='input_file', type=str, metavar='Input TSV file', required=True, help="TSV file of raw data. ans_id, que_id, raw data etc. are expected in this file")
parser.add_argument('-ap', '--ans_pos', dest='ans_pos', type=int, metavar="Position of answer in each row", required=True, help="Position of answer in each row. Starts from 0")
parser.add_argument('-vf', '--vocab_file', dest='vocab_file', type=str, metavar="Pregenerated vocabulary pickle file", required=False, default='', help="If this parameter is not none, the program will try to read the pickle file to load the vocabulary. If the file doesn't exist, the vocabulary will be created based on the input file and write to the appointed pickel file.")
args = parser.parse_args()

out_dir = args.output_path
if not os.path.exists(out_dir):
    os.makedirs(out_dir)

logger = logging.getLogger(__name__)
logging.basicConfig(filename='%s/gen_fea.log' % out_dir, level=logging.INFO)

NLP = spacy.load('en')
POS_AID = 0
POS_QID = 1
POS_SCORE = 2

# read records from tsv file
with open(args.input_file, 'r', encoding='utf-8') as f_tsv:
    logger.info('Reading file: %s' % args.input_file)
    f_tsv.readline()    # The first line is titles
    records = list(map(lambda line:line.split('\t'), f_tsv.readlines()))
    logger.info("\tsize of data: %d" % len(records))

    titles = 'AID\tQID\tScore\tFeature\n'
    logger.info("\tTitles: %s" % titles.strip())

    # sort and group records by que_id
    records.sort(key=operator.itemgetter(POS_QID))
    qid_records = groupby(records, key=operator.itemgetter(POS_QID))
    if int(args.que_id) < 0:
        # generate feature for everyquestion, 
        # and the output filename is created based on the que_id
        for qid, rec in qid_records:
            rec = list(rec)
            logger.info("Processing question %s" % qid)
            features = gen_bow_for_records(rec, pos_ans=args.ans_pos, ngram=args.ngram)
            with open('%s/%s_%s' % (out_dir, args.fea_type, qid), 'w') as f_out:
                f_out.write(titles)
                f_out.writelines(features)
    else:
        # only generate feature for appointed que_id
        for qid, rec in qid_records:
            if qid == str(args.que_id):
                logger.info("Processing question %s" % qid)
                features = gen_bow_for_records(rec, pos_ans=args.ans_pos, ngram=args.ngram)
                with open('%s/%s_%s' % (out_dir, args.fea_type, qid), 'w') as f_out:
                    f_out.write(titles)
                    f_out.writelines(features)
    logger.info("Done!")
