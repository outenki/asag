from __future__ import unicode_literals
'''
Generate features proposed from Phandi et al, 2015
FeatureType         Description
------------------------------------------
Length              Number of characters
                    Number of words
                    Number of commas
                    Number of apostrophes
                    Number of sentence ending punctuation symbols
                    Average word length
POS                 Number of bad POS n-grams
                    Number of bad POS n-grams divided by the total number of words in the essay

**TODO: NEED SOME MORE WORK FOR KAGGLE DATASET FOR PROMPT FEATURES**
Prompt              Number of words words in the essay that appears in the prompt 
                    Number of words in the essay that appears in the prompt divided by the total number of words in the essay
                    Number of words in the essay which is a word or a synonym of a word that appears in the prompt
                    Number of words in the essay which is a word or a synonym of a word that appears in the prompt 
                        divided by the total number of words in the essay
BOW                 Count of useful unigrams and bigrams (unstemmed)
                    Count of stemmed and spell corrected useful unigrams and bigrams
'''

import numpy as np
import logging
import data_format as D
from scipy import stats
from utils_asag import Tokenizer
logger = logging.getLogger(__name__)

class Vocab:
    '''
    Generate unigram and bigram for an answer
    '''
    unigram_unstemmed = list()
    bigram_unstemmed = list()
    unigram_stemmed= list()
    bigram_stemmed= list()
    prompt_id = 0
    ans_id = 0
    score = 0
    def __init__(self, aid, qid, score, text, nlp):
        self.ans_id, self.prompt_id, self.score = aid, qid, score
        tokenizer = Tokenizer(ngram=1, rm_punct=False, rm_stop=False, lemma=False, nlp=nlp)
        self.unigram_unstemmed = tokenizer.ngram_words(text)
        tokenizer.ngram = 2
        self.bigram_unstemmed = tokenizer.ngram_words(text)
        tokenizer.lemma = True
        self.bigram_stemmed= tokenizer.ngram_words(text)
        tokenizer.ngram = 1
        self.unigram_stemmed= tokenizer.ngram_words(text)



class Phandi:
    def __init__(self, pos_aid=D.ans_pos_id, pos_qid=D.ans_pos_qid, pos_score=D.ans_pos_score, pos_ans=D.ans_pos_ans, pos_que=D.que_pos_que, nlp=None, tokenizer=None, synonym_threshold = 0.7, save_path = ''):
        '''
        Initialize parameters for BOW class.
        :param pos_*: position of answer_id(aid), question_id(qid) and score in the input file.
        '''
        self.pos_aid, self.pos_qid, self.pos_score, self.pos_ans, self.pos_que = pos_aid, pos_qid, pos_score, pos_ans, pos_que
        self.tokenizer = tokenizer
        self.nlp = nlp

        self.ending_punct = {'.', '!', '?' }
        self.save_path = save_path

    def vocab_from_tokens(self, tokens_list, save_path):
        '''
        Generate vocabulary from tokens. The tokens will be sorted by frequency.
        :param tokens_list: list of list (tokens).
        :return: A dict with token as keys and index as values
        >>> vocab([['a','b','c', 'b'],['b','c','a','d']])
        {'a': 0, 'b': 1, 'c': 2, 'd': 3}
        '''
        vocab_dict = dict()
        # frequency of tokens
        logger.info('Generating vocabulary ...')
        for ts in tokens_list:
            for t in ts:
                try:
                    vocab_dict[t] += 1
                except KeyError:
                    vocab_dict[t] = 1

        # idf of tokens
        logger.info('Computing idf ...')
        self.vocab_idf = dict()
        D = float(len(tokens_list))
        for t in vocab_dict:
            df = len(list(filter(lambda ts:t in ts, tokens_list)))
            self.vocab_idf[t] = np.log10(D/df)

        # sort vocab by frequency
        logger.info('Computing frequency ...')
        vocab_sorted = sorted(vocab_dict.items(), key=lambda i: (i[1], i[0]), reverse=True)
        if self.voc_size > 0:
            # only the {size}th most frequent tokens are counted in as vocabulary.
            vocab_sorted = vocab_sorted[:self.voc_size]

        with open('%s/token_freq.txt' % save_path, 'w') as ft:
            for item, freq in vocab_sorted:
                ft.write('{}\t{}\n'.format(item, freq))

        # sort vocab by idf
        vocab_sorted = sorted(self.vocab_idf.items(), key=lambda i: (i[1], i[0]), reverse=True)
        if self.voc_size > 0:
            # only the {size}th most frequent tokens are counted in as vocabulary.
            vocab_sorted = vocab_sorted[:self.voc_size]

        with open('%s/token_idf.txt' % save_path, 'w') as ft:
            for item, idf in vocab_sorted:
                ft.write('{}\t{}\n'.format(item, idf))

        self.vocab = {item[0]:idx for (idx, item) in enumerate(vocab_sorted)}

    def generate_fea_for_tokens(self, tokens_ans, tokens_prompt, vocab, useful_ngrams):
        '''
        Generate features proposed from Pandi et al, 2015

        FeatureType         Description
        ------------------------------------------
        Length              Number of characters
                            Number of words
                            Number of commas
                            Number of apostrophes
                            Number of sentence ending punctuation symbols
                            Average word length
        POS                 Number of bad POS n-grams
                            Number of bad POS n-grams divided by the total number of words in the essay

        **TODO: NEED SOME MORE WORK FOR KAGGLE DATASET FOR PROMPT FEATURES**
        Prompt              Number of words words in the essay that appears in the prompt 
                            Number of words in the essay that appears in the prompt divided by the total number of words in the essay
                            Number of words in the essay which is a word or a synonym of a word that appears in the prompt
                            Number of words in the essay which is a word or a synonym of a word that appears in the prompt 
                                divided by the total number of words in the essay
        BOW                 Count of useful unigrams and bigrams (unstemmed)
                            Count of stemmed and spell corrected useful unigrams and bigrams

        :param tokens_ans: Tokenized text of an answer
        :param tokens_prompt: Tokenized text of the question
        :return: 1-D numpy array 
        '''

        ### Length ###
        num_char = sum([len(t.text) for t in tokens_ans]) - 1 + len(tokens_ans)  # Including the number of spaces
        num_words = len(list(filter(lambda t: not t.is_punct, tokens_ans)))
        num_commas = len(list(filter(lambda t: t.text==',', tokens_ans)))
        num_apost = len(list(filter(lambda t: t.text=="'", tokens_ans)))
        num_ending_punct = len(list(filter(lambda t: t.text in self.ending_punct, tokens_ans))) 
        len_word_avg = sum([len(t.text) for t in tokens_ans if not t.is_punct]) / num_words

        ### POS ###
        num_bad_pos = len(list(filter(lambda t: t.pos_ == 'NOUN', tokens_ans)))
        prop_bad_pos = num_bad_pos / num_words
        
        ### Prompt ###
        vocab_prompt = {t.text for t in tokens_prompt if not t.is_punct}
        num_words_in_prompt = len(list(filter(lambda t:t.text in vocab_prompt, tokens_ans)))
        prop_words_in_prompt = num_words_in_prompt / num_words
        num_synonym = 0
        for ta in tokens_ans:
            for tp in tokens_prompt:
                if ta.similarity(tp):
                    num_synonym += 1

        prop_synonym = num_synonym / num_words

        ### BOW ###
        count_1_2_gram_unstemmed = np.zeros(201)
        for ng in vocab.unigram_unstemmed:
            if ng in useful_ngrams['unigram_unstemmed']:
                idx = useful_ngrams['unigram_unstemmed'][ng]
                count_1_2_gram_unstemmed[idx] += 1

        for ng in vocab.bigram_unstemmed:
            if ng in useful_ngrams['bigram_unstemmed']:
                idx = useful_ngrams['bigram_unstemmed'][ng]
                count_1_2_gram_unstemmed[idx] += 1

        count_1_2_gram_stemmed = np.zeros(201)
        for ng in vocab.unigram_stemmed:
            if ng in useful_ngrams['unigram_stemmed']:
                idx = useful_ngrams['unigram_stemmed'][ng]
                count_1_2_gram_stemmed[idx] += 1

        for ng in vocab.bigram_stemmed:
            if ng in useful_ngrams['bigram_stemmed']:
                idx = useful_ngrams['bigram_stemmed'][ng]
                count_1_2_gram_stemmed[idx] += 1

        fea = np.array([num_char, num_words, num_commas, num_apost, num_ending_punct, len_word_avg, 
                num_bad_pos, prop_bad_pos, num_words_in_prompt, prop_words_in_prompt, num_synonym, prop_synonym])
        return np.concatenate([fea, count_1_2_gram_unstemmed, count_1_2_gram_stemmed])
        
    def gen_fea_for_records(self, records_ans, record_prompt, path_save_token):
        '''
        Generate feature for answers to a question from a list of records.
        The records are expected to be a list starting with: [AnswerId, QuestionId, Score, ...]
        :param records_ans, records_prompt: Lists of records of answers and questions read from a tsv file.
        :return: numpy array as the features
        '''
        # Generate useful unigrams and bigrams
        unigram_unstemmed = set()
        bigram_unstemmed = set()
        unigram_stemmed = set()
        bigram_stemmed = set()

        tokens_ans_list = []
        vocabs = []
        scores = []
        for items in records_ans:
            ans_id = items[self.pos_aid]
            que_id = items[self.pos_qid]
            score = items[self.pos_score]
            scores.append(float(score))
            text = items[self.pos_ans]
            # def __init__(self, aid, qid, score, text, nlp):
            vocab = Vocab(aid = ans_id, qid=que_id, score=score, text=text, nlp=self.nlp)
            vocabs.append(vocab)
            unigram_unstemmed |= set(vocab.unigram_unstemmed)
            bigram_unstemmed |= set(vocab.bigram_unstemmed)
            unigram_stemmed |= set(vocab.unigram_stemmed)
            bigram_stemmed |= set(vocab.bigram_stemmed)

            nt = self.tokenizer.nlp(str(text))
            #nt = self.tokenizer.nlp(u'%s' % str(text))
            tokens_ans_list.append(nt)

        score_ave = np.mean(scores)
        # For each ngram, generate the 2x2 matrix for Fisher test
        #                             |good|bad|
        #                          in| 100 | 2 |
        #                         out| 15  | 3 |

        # generate a ngram-pvalue dict
        def pvalue_ngrams(ngrams):
            pvalue_ngram = dict()
            for ng in ngrams:
                mat = np.zeros((2,2))
                for vocab in vocabs:
                    is_in = ng in vocab.unigram_unstemmed
                    is_good = float(vocab.score) >= score_ave
                    if is_in and is_good:
                        mat[0, 0] += 1
                    elif is_in and not is_good:
                        mat[0, 1] += 1
                    elif not is_in and is_good:
                        mat[1, 0] += 1
                    elif not is_in and not is_good:
                        mat[1, 1] += 1
                _, pvalue = stats.fisher_exact(mat)
                pvalue_ngram[ng] = pvalue
            return pvalue_ngram

        print('Fisher testing...')
        pvalue_unigram_unstemmed = pvalue_ngrams(unigram_unstemmed)
        pvalue_bigram_unstemmed = pvalue_ngrams(bigram_unstemmed)
        pvalue_unigram_stemmed = pvalue_ngrams(unigram_stemmed)
        pvalue_bigram_stemmed = pvalue_ngrams(bigram_stemmed)
                    
        print('Get the top 200 useful ngrams')
        useful_unigram_unstemmed = list(map(lambda x:x[0], sorted(pvalue_unigram_unstemmed.items(), key=lambda x:x[1], reverse=True)[:201]))
        useful_bigram_unstemmed = list(map(lambda x:x[0], sorted(pvalue_bigram_unstemmed.items(), key=lambda x:x[1], reverse=True)[:201]))
        useful_unigram_stemmed = list(map(lambda x:x[0], sorted(pvalue_unigram_stemmed.items(), key=lambda x:x[1], reverse=True)[:201]))
        useful_bigram_stemmed = list(map(lambda x:x[0], sorted(pvalue_bigram_stemmed.items(), key=lambda x:x[1], reverse=True)[:201]))

        enum_useful_unigram_unstemmed = dict([(ng, idx) for idx, ng in enumerate(useful_unigram_unstemmed)])
        enum_useful_bigram_unstemmed = dict([(ng, idx) for idx, ng in enumerate(useful_bigram_unstemmed)])
        enum_useful_unigram_stemmed = dict([(ng, idx) for idx, ng in enumerate(useful_unigram_stemmed)])
        enum_useful_bigram_stemmed = dict([(ng, idx) for idx, ng in enumerate(useful_bigram_stemmed)])

        useful_ngrams = {
                'unigram_unstemmed': enum_useful_unigram_unstemmed,
                'bigram_unstemmed': enum_useful_bigram_unstemmed,
                'unigram_stemmed': enum_useful_unigram_stemmed,
                'bigram_stemmed': enum_useful_bigram_stemmed
                }

        que = str(record_prompt[self.pos_que])
        qid = str(record_prompt[self.pos_qid])
        # tokens_prompt = self.tokenizer.nlp(text=u'%s' % que)
        tokens_prompt = self.tokenizer.nlp(text=que)
        print('tokens_prompt:',tokens_prompt)
        
        save_path = '%s/%s' % (self.save_path, qid)
        for k in useful_ngrams:
            save_path_useful = '%s/%s.useful' % (save_path, k)
            with open(save_path_useful, 'w') as fu:
                for ng in useful_ngrams[k]:
                    fu.write(str(ng) + '\n')

        features = []
        # Generate feature for each answer
        logger.info("\tGenerating features for que %s \n" % record_prompt[1]) 

        for items, token, vocab in zip(records_ans, tokens_ans_list, vocabs):
            ans_id = items[self.pos_aid]
            que_id = items[self.pos_qid]
            score = items[self.pos_score]
            answer = items[self.pos_ans]
            fea = ','.join(self.generate_fea_for_tokens(token, tokens_prompt, vocab, useful_ngrams).astype(str))
            features.append('{aid}\t{qid}\t{score}\t{fea}\t{ans}\n'.format(aid=ans_id, qid=que_id, score=score, fea=fea, ans=answer.strip()))
        return features

if __name__ == '__main__':
    import doctest
    doctest.testmod(verbose=True)
