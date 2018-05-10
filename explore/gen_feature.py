import numpy as np
import operator
import argparse
import logging
import os
import spacy
import utils_basic as UB
from utils_asag import check_c_path, Tokenizer
from utils_bow import BOW
from utils_pandi import Pandi
# import utils_bow as B
import data_format as D
import pickle

from itertools import groupby

parser = argparse.ArgumentParser()
parser.add_argument('-ft', '--feature', dest='fea_type', type=str, metavar='Feature type', required=True, help="Feature type (bow|pandi)")
parser.add_argument('-n', '--ngram', dest='ngram', type=int, metavar='N-gram bow', required=True, help="n-gram for bow feature")
parser.add_argument('-q', '--question', dest='que_id', type=int, metavar='Question ID', required=True, help="Question id or prompt id. If it is set, only features for answers to the specific question will be generated. If it i set to minus, features will be generated for each question.")
parser.add_argument('-qf', '--question_file', dest='que_file', type=str, metavar='Question File', required=True, help="TSV file of questions. ID, promoptID, questionText are expected in this file.")
parser.add_argument('-o', '--output_path', dest='output_path', type=str, metavar='Output Path', required=True, help="Path to store generated feature files")
parser.add_argument('-i', '--input_file', dest='input_file', type=str, metavar='Input TSV file', required=True, help="TSV file of raw data. ans_id, que_id, raw data etc. are expected in this file")
# parser.add_argument('-ap', '--ans_pos', dest='ans_pos', type=int, metavar="Position of answer in each row", required=True, help="Position of answer in each row. Starts from 0")
parser.add_argument('-vf', '--vocab_file', dest='vocab_file', type=str, metavar="Pregenerated vocabulary pickle file", required=False, default='', help="If this parameter is not none, the program will try to read the pickle file to load the vocabulary. If the file doesn't exist, the vocabulary will be created based on the input file and write to the appointed pickel file.")
parser.add_argument('-vs', '--vocab_size', dest='vocab_size', type=int, metavar="Size of vocabulary", required=False, default=0, help="Necessary for BOW feature. If 0 is set, all the tokens will be counted in as vocabulary. Default set as 0.")
parser.add_argument('-rp', '--rm_punct', dest='rm_punct', type=UB.str2bool, metavar="Flat of removal of punctuation", required=True, help="If this flag is set, punctuation will be removed before the generation of n-grams")
parser.add_argument('-rs', '--rm_stop', dest='rm_stop', type=UB.str2bool, metavar="Flag of removal of stopwords", required=True, help="If this flag is set, stop words will be removed before the generation of n-grams")
parser.add_argument('-lm', '--lemma', dest='lemma', type=UB.str2bool, metavar="Flag of lemmatization", required=True, help="If this flag is set, words will be lemmatized before the generation of n-grams")
parser.add_argument('-tf', '--term_frequency', dest='tf', type=UB.str2bool, metavar="Flag of term frequency", required=False, default='',  help="Necessary for BOW feature. If this flag is set, tf will be used as part of weight of terms. Defaultly set as False")
parser.add_argument('-idf', '--inverse_doc_frequency', dest='idf', type=UB.str2bool, metavar="Flag of inverse doc frequency", required=False, default='', help="If this flag is set, idf will be used as part of weight of terms. Defaultly set as False")
parser.add_argument('-pst', '--pandi_synonym_threshold', dest='pst', type=float, metavar="Threshold of similarity of synonymy", required=False, default=0.7, help="Necessary for Pandi feature. Words whose similarity is less than -pst will be considered as synonyms.")

args = parser.parse_args()

out_dir = args.output_path
if not os.path.exists(out_dir):
    os.makedirs(out_dir)

logger = logging.getLogger(__name__)
logging.basicConfig(filename='%s/gen_fea.log' % out_dir, level=logging.INFO)

logger.info('Loading spacy.en...')
nlp = spacy.load('en')

logger.info('Initialize Tokenizer...')

tokenizer = Tokenizer(ngram = args.ngram, rm_punct = args.rm_punct, rm_stop = args.rm_stop, lemma = args.lemma, nlp=nlp)
if args.fea_type == 'bow':
    generator = BOW(tokenizer=tokenizer, voc_size = args.vocab_size, tf = args.tf, idf=args.idf) 
elif args.fea_type == 'pandi':
    generator = Pandi(tokenizer=tokenizer, synonym_threshold=args.pst)

# read records from tsv file 
logger.info('Reading input file: %s' % args.input_file)
records_ans = np.loadtxt(args.input_file, dtype=str, delimiter='\t', skiprows=1, encoding='utf-8')
records_ans = list(records_ans)
logger.info("\tsize of data: %d" % len(records_ans))
logger.info("\te.g.: %s", '\t'.join(records_ans[0]))

logger.info('Reading question file: %s' % args.que_file)
records_que = np.loadtxt(args.que_file, dtype=str, delimiter='\t', skiprows=1, encoding='utf-8')
logger.info("\tsize of data: %d" % len(records_que))
logger.info("\te.g.: %s", '\t'.join(records_que[0]))
dict_que = dict()
for rec_que in records_que:
    qid = rec_que[D.que_pos_qid]
    dict_que[qid] = rec_que

titles = 'AID\tQID\tScore\tFeature\tAnswer\n'

# sort and group records by que_id
records_ans.sort(key=operator.itemgetter(D.ans_pos_qid))
qid_records = groupby(records_ans, key=operator.itemgetter(D.ans_pos_qid))
if int(args.que_id) < 0:
    # generate feature for everyquestion, 
    # and the output filename is created based on the que_id
    for qid, rec in qid_records:
        path_q = '%s/%s' % (out_dir, qid)
        check_c_path(path_q)
        records_ans = list(rec)
        record_prompt = dict_que[qid]
        logger.info("Processing question %s" % qid)
        print("Processing question %s" % qid)
        features = generator.gen_fea_for_records(records_ans, record_prompt, path_save_token=path_q)
        with open('%s/%s/%s.fea' % (out_dir, qid, args.fea_type), 'w') as f_out:
            f_out.write(titles)
            f_out.writelines(features)
        if args.fea_type == 'bow':
            with open('%s/vocab' % path_q, 'wb') as f:
                pickle.dump(generator.vocab, f)
else:
    # only generate feature for appointed que_id
    for qid, rec in qid_records:
        if qid == str(args.que_id):
            path_q = '%s/%s' % (out_dir, qid)
            check_c_path(path_q)
            records_ans = list(rec)
            record_prompt = dict_que[qid]
            logger.info("Processing question %s" % qid)
            logger.info("Input vocab_size: %d", qid)
            features = generator.gen_fea_for_records(records_ans, record_prompt, path_save_token=path_q)
            with open('%s/%s/%s.fea' % (out_dir, qid, args.fea_type), 'w') as f_out:
                f_out.write(titles)
                f_out.writelines(features)
            if args.fea_type == 'bow':
                with open('%s/vocab' % path_q, 'wb') as f:
                    pickle.dump(generator.vocab, f)
logger.info("Done!")

# with open(args.input_file, 'r', encoding='utf-8') as f_tsv, open(args.que_file, 'r', encoding='utf-8') as q_tsv:
#     logger.info('Reading input file: %s' % args.input_file)
#     logger.info('Reading question file: %s' % args.que_file)
#     f_tsv.readline()    # The first line is title
#     records = list(map(lambda line:line.split('\t'), f_tsv.readlines()))
#     logger.info("\tsize of data: %d" % len(records))
#     logger.info("\te.g.: %s", '\t'.join(records[0]))
# 
#     titles = 'AID\tQID\tScore\tFeature\tAnswer\n'
#     logger.info("\tTitles: %s" % titles.strip())
# 
#     # sort and group records by que_id
#     records.sort(key=operator.itemgetter(D.pos_qid))
#     qid_records = groupby(records, key=operator.itemgetter(D.pos_qid))
#     if int(args.que_id) < 0:
#         # generate feature for everyquestion, 
#         # and the output filename is created based on the que_id
#         for qid, rec in qid_records:
#             path_q = '%s/%s' % (out_dir, qid)
#             check_c_path(path_q)
#             rec = list(rec)
#             logger.info("Processing question %s" % qid)
#             print("Processing question %s" % qid)
#             features = bow.gen_bow_for_records(rec, ngram=args.ngram, path_save_token=path_q)
#             with open('%s/%s/%s.fea' % (out_dir, qid, args.fea_type), 'w') as f_out:
#                 f_out.write(titles)
#                 f_out.writelines(features)
#             with open('%s/vocab' % path_q, 'wb') as f:
#                 pickle.dump(bow.vocab, f)
#     else:
#         # only generate feature for appointed que_id
#         for qid, rec in qid_records:
#             if qid == str(args.que_id):
#                 path_q = '%s/%s' % (out_dir, qid)
#                 check_c_path(path_q)
#                 logger.info("Processing question %s" % qid)
#                 logger.info("Input vocab_size: %d", qid)
#                 features = bow.gen_bow_for_records(rec, ngram=args.ngram, path_save_token=path_q)
#                 with open('%s/%s/%s.fea' % (out_dir, qid, args.fea_type), 'w') as f_out:
#                     f_out.write(titles)
#                     f_out.writelines(features)
#                 with open('%s/vocab' % path_q, 'wb') as f:
#                     pickle.dump(bow.vocab, f)
#     logger.info("Done!")
