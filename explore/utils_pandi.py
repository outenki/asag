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
'''

import numpy as np
import spacy
import logging
import data_format as D
logger = logging.getLogger(__name__)
NLP = spacy.load('en')


class Pandi:
    def __init__(self, pos_aid=D.ans_pos_id, pos_qid=D.ans_pos_qid, pos_score=D.ans_pos_score, pos_ans=D.ans_pos_ans, pos_que=D.que_pos_que, tokenizer=None, synonym_threshold = 0.7):
        '''
        Initialize parameters for BOW class.
        :param pos_*: position of answer_id(aid), question_id(qid) and score in the input file.
        '''
        self.pos_aid, self.pos_qid, self.pos_score, self.pos_ans, self.pos_que = pos_aid, pos_qid, pos_score, pos_ans, pos_que
        self.tokenizer = tokenizer

        self.ending_punct = {'.', '!', '?' }

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

    def generate_fea_for_tokens(self, tokens_ans, tokens_prompt):
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
        count_1_2_gram_unstemmed = 0
        count_1_2_gram_stemmed = 0

        return np.array([num_char, num_words, num_commas, num_apost, num_ending_punct, len_word_avg, 
                num_bad_pos, prop_bad_pos, num_words_in_prompt, prop_words_in_prompt, num_synonym, prop_synonym,
                count_1_2_gram_unstemmed, count_1_2_gram_stemmed])
        
    def gen_fea_for_records(self, records_ans, record_prompt, path_save_token):
        '''
        Generate feature for answers to a question from a list of records.
        The records are expected to be a list starting with: [AnswerId, QuestionId, Score, ...]
        :param records_ans, records_prompt: Lists of records of answers and questions read from a tsv file.
        :return: numpy array as the features
        '''
        tokens_ans_list = []
        for items in records_ans:
            ans = items[self.pos_ans]
            nt = self.tokenizer.nlp(str(ans))
            tokens_ans_list.append(nt)

        que = str(record_prompt[self.pos_que])
        tokens_prompt = self.tokenizer.nlp(text=que)
        print('tokens_prompt:',tokens_prompt)
        
        features = []
        # Generate feature for each answer
        logger.info("\tGenerating features for que %s \n" % record_prompt[1]) 

        for items, token in zip(records_ans, tokens_ans_list):
            ans_id = items[self.pos_aid]
            que_id = items[self.pos_qid]
            score = items[self.pos_score]
            answer = items[self.pos_ans]
            fea = ','.join(self.generate_fea_for_tokens(token, tokens_prompt).astype(str))
            features.append('{aid}\t{qid}\t{score}\t{fea}\t{ans}\t{token}\n'.format(aid=ans_id, qid=que_id, score=score, fea=fea, ans=answer.strip(), token=token))
        return features

if __name__ == '__main__':
    import doctest
    doctest.testmod(verbose=True)
