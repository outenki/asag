import os
import spacy
from spacy.lang.en import LEMMA_INDEX, LEMMA_EXC, LEMMA_RULES

LEVENSHTEIN = 3

CONFIDENCE_STEP = 0.05
CONFIDENCE_LEVELS = 20
CONFIDENCE_WINDOW_WIDTH = 300
SCORE_LEVELS = 6

# Paths
SCRIPT_PATH = os.path.dirname(os.path.abspath(__file__))
# DATA_PATH = SCRIPT_PATH + "/../data/kaggle_train"
# RESULTS_PATH = SCRIPT_PATH + "/../results_kaggle"
DATA_PATH = SCRIPT_PATH + "/../data/XCSD_6Ways/data"
RESULTS_PATH = SCRIPT_PATH + "/../results_xcsd_6ways"
# DATA_PATH = SCRIPT_PATH + "/../data/XCSD_2Ways/data"
# DATA_PATH = SCRIPT_PATH + "/../data/beetle_2Ways/all"
# DATA_PATH = SCRIPT_PATH + "/../data/sciEntsBank_2ways/all"
# DATA_PATH = SCRIPT_PATH + "/../data/sciEntsBank_5ways/train"
# DATA_PATH = SCRIPT_PATH + "/../data/kaggle_train"
# DATA_PATH = SCRIPT_PATH + "/../data/sciEntsBank/test-unseen-questions"
# DATA_PATH = SCRIPT_PATH + "/../data/sciEntsBank/test-unseen-answers"
# DATA_PATH = SCRIPT_PATH + "/../data/sciEntsBank/test-unseen-domains"
# RESULTS_PATH = SCRIPT_PATH + "/../results_xcsd_2ways"
# RESULTS_PATH = SCRIPT_PATH + "/../results_sciEntsBank_2ways/all"
# RESULTS_PATH = SCRIPT_PATH + "/../results_sciEntsBank_5ways/train"
# RESULTS_PATH = SCRIPT_PATH + "/../results_beetle_2Ways"
# RESULTS_PATH = SCRIPT_PATH + "/../results_semi_train"
# RESULTS_PATH = SCRIPT_PATH + "/../results_kaggle_train"
# RESULTS_PATH = SCRIPT_PATH + "/../results_semi_uq"
# RESULTS_PATH = SCRIPT_PATH + "/../results_semi_ua"
# RESULTS_PATH = SCRIPT_PATH + "/../results_semi_ud"

RAW_PATH = DATA_PATH + "/raw"
RAW_PATH_STU = DATA_PATH + "/raw/ans_stu"
W2V_PATH = SCRIPT_PATH + '/../data/glove.6B'
W2V_FILE = 'glove.6B.300d.txt'
# 
WEIGHTS_PATH = RESULTS_PATH + "/word_weights"
WEIGHT_SVR_PATH = RESULTS_PATH + "/word_weight_svr"
# NLP = spacy.load('en')
# # tokenizer = spacy.tokenizer.Tokenizer(nlp.vocab)
# LEMMATIZER = spacy.lemmatizer.Lemmatizer(LEMMA_INDEX, LEMMA_EXC, LEMMA_RULES)
