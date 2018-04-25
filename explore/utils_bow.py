import numpy as np
import spacy
import logging
import data_format as D
logger = logging.getLogger(__name__)
NLP = spacy.load('en')

class BOW:
    def __init__(self, pos_aid=D.pos_aid, pos_qid=D.pos_qid, pos_score=D.pos_score, pos_ans=D.pos_ans, voc_size=0, tokenizer=None):
        '''
        Initialize parameters for BOW class.
        :param pos_*: position of answer_id(aid), question_id(qid) and score in the input file.
        :param size: size of vocab. All the tokens will be counted in if 0 is set.
        '''
        self.pos_aid, self.pos_qid, self.pos_score, self.pos_ans, self.voc_size = pos_aid, pos_qid, pos_score, pos_ans, voc_size
        self.tokenizer = tokenizer

    def vocab_from_tokens(self, token_list, save_path):
        '''
        Generate vocabulary from tokens. The tokens will be sorted by frequency.
        :param token_list: list of list (tokens).
        :return: A dict with token as keys and index as values
        >>> vocab([['a','b','c', 'b'],['b','c','a','d']])
        {'a': 0, 'b': 1, 'c': 2, 'd': 3}
        '''
        vocab_dict = dict()
        for ts in token_list:
            for t in ts:
                try:
                    vocab_dict[t] += 1
                except KeyError:
                    vocab_dict[t] = 1

        # sort vocab by frequency
        vocab_sorted = sorted(vocab_dict.items(), key=lambda i: (i[1], i[0]), reverse=True)
        if self.voc_size > 0:
            # only the {size}th most frequent tokens are counted in as vocabulary.
            vocab_sorted = vocab_sorted[:self.voc_size]
        with open('%s/token_freq.txt' % save_path, 'w') as ft:
            for item, freq in vocab_sorted:
                ft.write('{}\t{}\n'.format(item, freq))

        return {item[0]:idx for (idx, item) in enumerate(vocab_sorted)}

# def generate_bow_feature_from_text(nlp, vocab, text, ngram, rm_punct, rm_stop, lemma):
#     '''
#     Generate ngram bow feature vector for input text
#     :param nlp: Instance of spacy model
#     :param vocab: Dict with token tuple as key and index as value
#     :param text: Input text
#     :param ngram: n-gram
#     :return: 1-D numpy array with same size to vocab.
#     >>> generate_bow_feature(spacy.load('en'), {(u'a',):0, (u'b',):1, (u'c',): 3}, u'a b', 1)
#     array([1., 1., 0.])
#     '''
#     # tokenize text
#     bow = np.zeros(len(vocab))
#     tokens = U.tokenize(nlp=nlp, text=text, rm_punct=rm_punct, ngram=ngram, rm_stop=rm_stop, lemma=lemma)
#     for t in tokens:
#         bow[vocab[t]] = 1
#     return bow

    def generate_bow_feature_from_tokens(self, vocab, tokens):
        '''
        Generate ngram bow feature vector for tokenized text
        :param vocab: Dict with token tuple as key and index as value
        :param tokens: Tokenized text
        :return: 1-D numpy array with same size to vocab.
        >>> generate_bow_feature(spacy.load('en'), {(u'a',):0, (u'b',):1, (u'c',): 3}, [u'a', u'b'])
        array([1., 1., 0.])
        '''
        # print('vocab:', vocab)
        # print('tokens:', tokens)
        bow = np.zeros(len(vocab))
        assert max(vocab.values()) + 1 == len(bow)
        for t in tokens:
            if t in vocab:
                bow[vocab[t]] = 1
        return bow

    def gen_bow_for_records(self, records, ngram, path_save_token):
        '''
        Generate feature for answers to a question from a list of records.
        The records are expected to be a list starting with: [AnswerId, QuestionId, Score, ...]
        1. Read all the answers
        2. Generate vocab with answers
        3. Generate feature for each answer
            3.1 tokenize the answer
            3.2 Generate n-gram
            3.3 Set featue values as 0 or 1 in an array
        :param records: A list of records read from a tsv file.
        :param ans_pos: Appoint the position of answer in the record.
        :param ngram: Generate n-gram bow feature
        :param vocab_size: Size of vocabulary that will be generated. 
                        All the tokens will be counted in when 0 is set.
        :return:
            A list of results in form of AnswerID\tQuestionID\tScore\tFeature
            Used vocabulary
        '''
        # generate vocab
        logger.info("Generating vocab ...")
        token_list = []
        for items in records:
            ans = items[self.pos_ans]
            nt = self.tokenizer.tokenize(text=ans)
            token_list.append(nt)
        vocab = self.vocab_from_tokens(token_list, path_save_token)
        logger.info("\tVocab_size: %d", self.voc_size)
        logger.info("\tsize of vocab: %d", len(vocab))
        features = []
        # Generate feature for each answer
        logger.info("\tGenerating features ...")
        # for aid, qid, sco, t in zip(ans_ids, que_ids, scores, token_list):
        for items, token in zip(records, token_list):
            ans_id = items[self.pos_aid]
            que_id = items[self.pos_qid]
            score = items[self.pos_score]
            answer = items[self.pos_ans]
            fea = ','.join(self.generate_bow_feature_from_tokens(vocab, token).astype(str))
            features.append('{aid}\t{qid}\t{score}\t{fea}\t{ans}\n'.format(aid=ans_id, qid=que_id, score=score, fea=fea, ans=answer.strip()))
        return features, vocab

if __name__ == '__main__':
    import doctest
    doctest.testmod(verbose=True)
