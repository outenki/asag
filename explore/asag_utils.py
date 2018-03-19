from nltk.corpus import wordnet as wn
from nltk.corpus import wordnet_ic as wic
from basic_util import distance_of_couples
from functools import reduce
import collections

from math import sqrt, ceil
import re
import math
import numpy as np

np.set_printoptions(threshold=np.nan)

# from munkres import Munkres
from nltk.metrics import *
import nltk
import string
import os
import time
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import neighbors
from nltk.stem.porter import PorterStemmer

from sklearn.svm import SVR
from gensim.models import Word2Vec
# import progressbar

from scipy import spatial
# import autograd
import spacy
from spacy.lang.en import LEMMA_INDEX, LEMMA_EXC, LEMMA_RULES
import random
from config import *
from basic_util import token_lemma, clean_text
from itertools import groupby


LEVENSHTEIN = 3



class Sentence:
    def __init__(self, graph_text, ans_id=''):
        """
        Generate sub-graph for each node of input graph.
        graph_text:
            1.1 nn(problem:NNS:3, High:NNP:1)|||nn(problem:NNS:3, risk:NN:2)|||nsubj(address:NN:5, problem:NNS:3)|||...
            |||ccomp(show:VB:8, program:VBN:17)|||<STOP>
        Properties:
            self.question_num:
                Int. Corresponding question number of current sentence (q or a)
            self.edges:
                List of tuplles. [(relate, governor, dependent),...].
                Relate, governor and dependent are strings.
                Relate and governor are nodes: word:POS:index such like 'problem:NNS:3'
            self.nodes:
                List of string. [node1, node2, ...]
            self.subgraph:
                Dict.
                Keys: nodes
                Vals: list of set of nodes. Vals[0~3] stands for 4 levels of subgraphs.
        """
        tmp = graph_text.split(' ')
        self.id, self.edges_text = tmp[0], ' '.join(tmp[1:])
        self.id = self.id + "." + ans_id  # Question id + answer id when there is an answer id
        self.edges = [self._read_edges(edge_text) for edge_text in self.edges_text.split('|||')][:-1]
        self.nodes = self._read_nodes
        self.subgraph = {}
        for node in self.nodes:
            self.subgraph[node] = [self._find_sub_graph(node, level) for level in range(4)]
        self.words = self._read_words()
        self.words_with_demoting = self.words
        self.similarity = {}  # cache for similarity, in case one similarity be calculated for times

    def _read_words(self):
        return list(map(lambda n: n.split(':')[0], self.nodes))

    def question_demoting(self, words_que):
        words_que = set(words_que)
        self.words_with_demoting = [word for word in self.words if word not in words_que]
        if len(self.words_with_demoting) == 0:
            print('WARNING: No words left after question demoting')
        return self.words_with_demoting

    @staticmethod
    def _read_edges(edge_text):
        """
        Convert relate(governor, dependent) to (relate, governor, dependent)
        nn(problem:NNS:3, High:NNP:1) --> tuple(nn, problem:NNS:3, High:NNP:1)
        The meta data should not contain characters of '(' or ')'. Replace them if it does.
        """
        edge = tuple(edge_text.replace('(', ', ').replace(')', '').split(', '))
        return edge

    @property
    def _read_nodes(self):
        """
        Read nodes of dependency graph from edges.
        :return: A set of nodes in tuple
        """
        nodes = set()
        for e in self.edges:
            nodes.add(e[1])
            nodes.add(e[2])
        return tuple(nodes)

    def _find_sub_graph(self, node, level):
        """
            node: int, position of the node in the sentence.
            level: 0~3, level of subgraph
                0 : All edge types may be followed
                1 : All edge types except for subject types, ADVCL, PURPCL, APPOS, PARATAXIS, ABBREV, TMOD, and CONJ
                2 : All edge types except for those in N1 plus object/complement types, PREP, and RCMOD
                3 : No edge types may be followed (This set is the single starting node x)
        """
        edge1 = {'advcl', 'purpcl', 'appos', 'parataxis', 'abbrev', 'tmod', 'conj'}
        edge2 = edge1 | {'sub', 'nsub', 'csub', 'obj', 'dobj', 'iobj', 'pobj', 'comp', 'ccomp',
                         'xcomp', 'acomp', 'prep', 'rcmod'}

        nodes = set()
        from_nodes = {node}
        edges = set(self.edges)
        if 0 == level:
            while from_nodes:
                to_nodes = set()
                for e in edges:
                    r, gov, dep = e
                    if gov in from_nodes and dep not in nodes | from_nodes:
                        to_nodes.add(dep)
                nodes |= from_nodes
                from_nodes = to_nodes

        elif 1 == level:
            while from_nodes:
                to_nodes = set()
                for r, gov, dep in edges:
                    if r not in edge1 and gov in from_nodes and dep not in nodes | from_nodes:
                        to_nodes.add(dep)
                nodes |= from_nodes
                from_nodes = to_nodes

        elif 2 == level:
            while from_nodes:
                to_nodes = set()
                for r, gov, dep in edges:
                    if r not in edge2 and gov in from_nodes and dep not in nodes | from_nodes:
                        to_nodes.add(dep)
                nodes |= from_nodes
                from_nodes = to_nodes

        elif 3 == level:
            nodes.add(node)
        return nodes


def cur_time():
    return time.strftime('%Y%m%d%H%M%S', time.localtime(time.time()))


def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def norm_of_vec(v):
    norm = np.dot(v, v)
    if type(norm) == np.float64:
        return np.sqrt(norm)
    return sqrt(norm._value)
    # return np.linalg.norm(v)

def cos_distance(vec1, vec2):
    # if not isinstance(vec1, np.ndarray): vec1 = vec1._value
    # if not isinstance(vec2, np.ndarray): vec2 = vec2._value
    # norm1 = norm_of_vec(vec1)
    # norm2 = norm_of_vec(vec2)
    norm1 = autograd.numpy.linalg.norm(vec1)
    norm2 = autograd.numpy.linalg.norm(vec2)
    if norm1 == 0 or norm2 == 0:
        return 1
    return (1.0 - autograd.numpy.dot(vec1, vec2) / (norm1 * norm2))/2

def overlap(syn1, syn2):
    def1 = set([word for word in re.split(r'[^a-zA-Z]', syn1.definition()) if word])
    def2 = set([word for word in re.split(r'[^a-zA-Z]', syn2.definition()) if word])
    return len(def1 & def2) / len(def1 | def2)


def similarity_nodes(func, node_stu, node_ins, cache, ic=None):
    """
    Calculate one similarity between two words and save it to cache dict to avoid
    repeat calculation.
    :param func:
        Similarity function from wordnet
    :param node_stu:
        node of student answer in form of problem:NNS:3
    :param node_ins:
        node of reference answer in form of problem:NNS:3
    :param cache:
        Dictionary to save calculated similarities.
    :param ic:
        parameter for some of similarity function of WordNet.
    :return:
        Real number between 0 to 1.
    """

    word1, word2 = node_ins.split(":")[0], node_stu.split(":")[0]
    c_key = word1 + ',' + word2
    if c_key in cache:
        return cache[c_key]

    if func.__name__ == 'lch_similarity':
        sims = [func(s1, s2) if s1.pos() == s2.pos() else 0 for s1 in wn.synsets(word1) if
                s1.name().split('.')[0] == word1 for s2 in wn.synsets(word2) if s2.name().split('.')[0] == word2]
    elif func.__name__ in {'res_similarity', 'lin_similarity', 'jcn_similarity'}:
        sims = [func(s1, s2, ic) if s1.pos() == s2.pos() else 0 for s1 in wn.synsets(word1) if
                s1.name().split('.')[0] == word1 for s2 in wn.synsets(word2) if s2.name().split('.')[0] == word2]
    else:
        sims = [func(s1, s2) for s1 in wn.synsets(word1) if s1.name().split('.')[0] == word1 for s2 in wn.synsets(word2)
                if s2.name().split('.')[0] == word2]
    sims = list(filter(lambda x: x, sims))
    if not sims:
        sim = 0
        pass
    else:
        sim = max(sims)
    cache[c_key] = sim
    return sim


def similarity_subgraph(func, nodes_stu, nodes_ins, cache, ic=None):
    """
    Return wordnet similarity between two set of nodes.
    This function calculate similarity between each pair of nodes and return the largest one.
    fun:
        similarity function from wordnet
    nodes_stu, nodes_ins:
        set of nodes of answers from students and instructor
    :rtype: int
    """
    sims = []
    for node_stu, node_ins in [(ns, ni) for ns in nodes_stu for ni in nodes_ins]:
        sim = similarity_nodes(func, node_stu, node_ins, cache, ic=ic)
        sims.append(sim)
    sim = max(sims) if sims else 0
    return sim


def knowledge_based_feature_between_sentence_8(nodes_stu, nodes_ins, cache):
    """
    8 dimension knowledge based features between two sentences.
    Used for generating 30 dimension BOW features.
    Each feature between two answer is the largest one among all
    of features calculated between every pair of nodes from each subgraph.
    :param nodes_ins:
        node of reference answer
    :param nodes_stu:
        node of student answer
    :type cache: Dict
        A 8-dimension list of knowledge based features for the input pair
    """
    f_shortest_path = similarity_subgraph(wn.path_similarity, nodes_stu, nodes_ins, cache)
    f_lch = similarity_subgraph(wn.lch_similarity, nodes_stu, nodes_ins, cache)
    f_wup = similarity_subgraph(wn.wup_similarity, nodes_stu, nodes_ins, cache)
    f_res = similarity_subgraph(wn.res_similarity, nodes_stu, nodes_ins, cache)
    f_lin = similarity_subgraph(wn.lin_similarity, nodes_stu, nodes_ins, cache)
    f_jcn = similarity_subgraph(wn.jcn_similarity, nodes_stu, nodes_ins, cache)
    f_lesk = similarity_subgraph(overlap, nodes_stu, nodes_ins, cache)
    f_hso = 1  # TODO: Update the algorithm

    return [f_shortest_path, f_lch, f_wup, f_res, f_lin, f_jcn, f_lesk, f_hso]


def semantic_similarity_between_subgraph_9(nodes_stu, nodes_ins, cache, ic):
    """
    Subgraph-level similarity. Used for alignment matching score.

    subgraph_stu, subgraph_ins:
        Set of nodes of subgraph of student and instructor answers in form of 'word:pos:los'
    return:
        a 9-dimension list of knowledge based features for the input pair
    Each feature between two answer is the largest one among all of features
    calculated between every pair of nodes from each subgraph.
    """
    f_shortest_path = similarity_subgraph(wn.path_similarity, nodes_stu, nodes_ins, cache)
    f_lch = similarity_subgraph(wn.lch_similarity, nodes_stu, nodes_ins, cache)
    f_wup = similarity_subgraph(wn.wup_similarity, nodes_stu, nodes_ins, cache)
    f_res = similarity_subgraph(wn.res_similarity, nodes_stu, nodes_ins, cache, ic=ic)
    f_lin = similarity_subgraph(wn.lin_similarity, nodes_stu, nodes_ins, cache, ic=ic)
    f_jcn = similarity_subgraph(wn.jcn_similarity, nodes_stu, nodes_ins, cache, ic=ic)
    f_lesk = similarity_subgraph(overlap, nodes_stu, nodes_ins, cache)
    f_hso = 1  # TODO: Update the algorithm
    f_lsa = 1  # TODO: Update the algorithm
    return [f_shortest_path, f_lch, f_wup, f_res, f_lin, f_jcn, f_lesk, f_hso, f_lsa]


def phi(node_stu, ans_stu, node_ins, ans_ins, cache, ic):
    """
    Generate 68-d features for node-pair <node_stu, node_ins>
    node_stu, node_ins: nodes of answers
    ans_stu, ans_ins: Object of Sentence generated from dependence graph of answers
    """
    # for the knowledge-based measures, we use the maximum semantic similarity
    # - for each open-class word - that can be obtained by pairing
    # it up with individual open-class words in the second input text.
    subgraphs_stu = ans_stu.subgraph[node_stu]  # subgraphs of node_stu
    subgraphs_ins = ans_ins.subgraph[node_ins]  # subgraphs of node_ins
    features_68 = []
    for i in range(4):
        subgraph_stu = subgraphs_stu[i]
        subgraph_ins = subgraphs_ins[i]
        features_68.extend(semantic_similarity_between_subgraph_9(subgraph_stu, subgraph_ins, cache, ic))
    features_68.extend(lexico_syntactic_features_32(node_stu, ans_stu, node_ins, ans_ins, cache))
    return np.array(features_68)


def lexico_syntactic_features_32(node_stu, ans_stu, node_ins, ans_ins, cache):
    """
    lexico syntactic features (32 dimensions)
    This features are for N3, meaning just for the single node.
    """
    try:
        word_stu, pos_stu, loc_stu = node_stu.split(':')

    except:
        print("!!!!!stu!!!!!")
        print(node_stu)
        exit(-1)

    try:
        word_ins, pos_ins, loc_ins = node_ins.split(':')
    except:
        print("!!!!ins!!!!")
        print(node_ins)
        exit(-1)

    feature_32 = []
    c_key = ans_stu.id + ',' + node_stu + ',' + node_ins
    # RootMatch: 5d / Is a ROOT node matched to: ROOT, N, V, JJ, or Other
    if c_key not in cache:
        cache[c_key] = {}

    if 'root_match' in cache[c_key]:
        f_root_match = cache[c_key]['root_match']
    else:
        f_root_match = [0, 0, 0, 0, 0]
        if pos_stu == 'ROOT':
            if pos_ins == 'ROOT':
                f_root_match[0] = 1
            elif pos_ins.startswith('NN'):
                f_root_match[1] = 1
            elif pos_ins.startswith('VB'):
                f_root_match[2] = 1
            elif pos_ins.startswith('JJ'):
                f_root_match[3] = 1
            else:
                f_root_match[4] = 1
    cache[c_key]['root_match'] = f_root_match
    feature_32.extend(f_root_match)

    # Lexical: 3d / Exact match, Stemmed match, close Levenshtein match
    if 'lexical' in cache[c_key]:
        f_lexical = cache[c_key]['lexical']
    else:
        st = PorterStemmer()
        f_lexical = [0, 0, 0]
        if word_ins == word_stu:
            f_lexical[0] = 1
        if st.stem(word_ins) == st.stem(word_stu):
            f_lexical[1] = 1
        if edit_distance(word_ins, word_stu) < LEVENSHTEIN:
            f_lexical[2] = 1
    cache[c_key]['lexical'] = f_lexical
    feature_32.extend(f_lexical)

    # POS Match: 2d / Exact POS match, Coarse POS match
    if 'pos_match' in cache[c_key]:
        f_pos_match = cache[c_key]['pos_match']
    else:
        f_pos_match = [0, 0]
        if pos_stu == pos_ins:
            f_pos_match[0] = 1
        if pos_stu.startswith(pos_ins) or pos_ins.startswith(pos_stu):
            f_pos_match[1] = 1
    cache[c_key]['pos_match'] = f_pos_match
    feature_32.extend(f_pos_match)

    # POS Pair: 8d / Specific X-Y POS matches found
    # POS:
    #   CC: Coordinating Conjunctions
    #   NN: Common Nouns
    #   PRP: PRONOUN
    #   VB: VERB
    #   JJ: ADJECTIVE
    #   RB: ADVERB
    #   IN: Prepositions and Subordinating Conjunctions
    #   UH: INTERJECTION
    pos_8 = ['CC', 'NN', 'PRP', 'VB', 'JJ', 'RB', 'IN', 'UH']
    if 'pos_pair' in cache[c_key]:
        f_pos_pair = cache[c_key]['ontological']
    else:
        f_pos_pair = [0, ] * 8
        pos_stu, pos_ins = node_stu.split(':')[1], node_ins.split(':')[1]
        for i in range(8):
            if pos_stu.startswith(pos_8[i]) and pos_ins.startswith(pos_8[i]):
                f_pos_pair[i] = 1

    feature_32.extend(f_pos_pair)

    # Ontological: 4d / WordNet relationships: synonymy, antonymy, hypernymy, hyponymy
    if 'ontological' in cache[c_key]:
        f_ontological = cache[c_key]['ontological']
    else:
        f_ontological = [0, ] * 4
        st = PorterStemmer()
        synsets_s = wn.synsets(st.stem(word_stu))
        stemmed_word_ins = st.stem(word_ins)
        for synset_s in synsets_s:
            if synset_s.name().split('.')[0] == stemmed_word_ins:
                f_ontological[0] = 1

            antos = synset_s.lemmas()[0].antonyms()
            for anto in antos:
                if anto.name() == stemmed_word_ins:
                    f_ontological[1] = 1

            hypos = synset_s.hyponyms()
            for hypo in hypos:
                if hypo.name().split('.')[0] == stemmed_word_ins:
                    f_ontological[2] = 1
    cache[c_key]['ontological'] = f_ontological
    feature_32.extend(f_ontological)

    # RoleBased: 3d / Has as a child - subject, object, verb
    if 'role_based' in cache[c_key]:
        f_role_based = cache[c_key]['ontological']
    else:
        f_role_based = [0, ] * 3

        def rolebased(node, ans, role):
            for e in ans.edges:
                if e[1] == node:
                    if role in {'sub', 'obj'} and e[0].endswith(role):
                        return True
                    if role == 'verb' and e[2].split(':')[1].startswith('VB'):
                        return True
            return False

        if rolebased(node_stu, ans_stu, 'sub') and rolebased(node_ins, ans_ins, 'sub'):
            f_role_based[0] = 1
        if rolebased(node_stu, ans_stu, 'obj') and rolebased(node_ins, ans_ins, 'obj'):
            f_role_based[1] = 1
        if rolebased(node_stu, ans_stu, 'verb') and rolebased(node_ins, ans_ins, 'verb'):
            f_role_based[2] = 1
    feature_32.extend(f_role_based)

    # VerbSubject: 3d / Both are verbs and neither, one, or both have a subject child
    if 'verb_subject' in cache[c_key]:
        f_verb_subject = cache[c_key]['verb_subject']
    else:
        f_verb_subject = [0, ] * 3
        v = 0
        if pos_stu.startswith('VB') and pos_ins.startswith('VB'):
            for edge_s in ans_stu.edges:
                if edge_s[-1].endswith('sub'):
                    v += 1
            for edge_i in ans_ins.edges:
                if edge_i[-1].endswith('sub'):
                    v += 1
        f_verb_subject[v] = 1
    cache[c_key]['verb_subject'] = f_verb_subject
    feature_32.extend(f_verb_subject)

    # VerbObject: 3d / Both are verbs and neither, one, or both have an object child
    if 'verb_object' in cache[c_key]:
        f_verb_object = cache[c_key]['verb_object']
    else:
        f_verb_object = [0, ] * 3
        v = 0
        if pos_stu.startswith('VB') and pos_ins.startswith('VB'):
            for edge_s in ans_stu.edges:
                if edge_s[-1].endswith('obj'):
                    v += 1
            for edge_i in ans_ins.edges:
                if edge_i[-1].endswith('obj'):
                    v += 1
        f_verb_object[v] = 1
    cache[c_key]['verb_object'] = f_verb_object
    feature_32.extend(f_verb_object)

    # Bias: 1d / A value of 1 for all vectors
    f_bias = 1
    feature_32.append(f_bias)
    return np.array(feature_32)


def perceptron_train(cache, ic, epochs=50, fn_ans='answers', fn_que='questions'):
    """
    Train vector *w* for node-level maching score
    Read training data in from data/annotations
    Generate a dic for each answer as training data:
        {
            '1.1.19':{
                'ans_stu': ans_stu, # Sentence Object
                'ans_ins': ans_ins, # Sentence Object
                'que': que,         # Sentence Object
                'labels': {
                    'node_ins,node_stu': label,
                        ...
                    },
                    ...
                }
            },
            {
            '1.2':...
            }, ...
        }
    Run perceptron training.
    fn_ans and fn_que:
        File names of answers and questions.
    """
    training_data = {}
    path_parse_file = DATA_PATH + '/parses/'
    path_data_file = DATA_PATH + '/annotations/'
    file_list = os.listdir(path_data_file)
    for fn in file_list:
        if fn.endswith('.pl'):
            continue
        parse_fn, parse_ln = os.path.splitext(fn)  # get file name and line
        # number of graph_text (start
        # from 0)
        parse_ln = int(parse_ln[1:])  # '.n' --> int(n)
        training_data[fn] = {
            'que': None,
            'ans_stu': None,
            'ans_ins': None,
            'labels': {},
        }

        # Read question
        print('Reading file:', path_parse_file + fn_que)
        with open(path_parse_file + fn_que, 'r') as f:
            while True:
                line = f.readline()
                if not line:
                    print("ERROR: Over range when read question " + fn)
                    break
                if line.startswith(parse_fn):
                    break
        print('Generate Sentence Obejct for question: ', line)
        training_data[fn]['que'] = Sentence(line)

        # Read instructor answer
        print('Reading file:', path_parse_file + fn_ans)
        with open(path_parse_file + fn_ans, 'r') as f:
            while True:
                line = f.readline()
                if not line:
                    print("ERROR: Over range when read instructor answer " + fn)
                    break
                if line.startswith(parse_fn):
                    break
        print('Generate Sentence Obejct for instructor answer: ', line)
        training_data[fn]['ans_ins'] = Sentence(line)
        training_data[fn]['ans_ins'].question_demoting(training_data[fn]['que'].words)

        # Read student answer
        print('Reading file:', path_parse_file + parse_fn)
        with open(path_parse_file + parse_fn, 'r') as f:
            for i in range(parse_ln):
                line = f.readline()
                if not line:
                    break
            if line:  # in case of EOF
                line = f.readline()
            else:
                print("ERROR: Over range when read student answer " + fn)
        print('Generate Sentence Obejct for student answer: ', line, str(parse_ln))
        training_data[fn]['ans_stu'] = Sentence(line, str(parse_ln))
        training_data[fn]['ans_stu'].question_demoting(training_data[fn]['que'].words)

        # Read labels
        # TODO: ROOT:ROOT:* will be skiped for the reason that the similarity between ROOT node can't be calculated.
        print('Reading file:', path_data_file + fn)
        with open(path_data_file + fn, 'r') as f:
            while True:
                line = f.readline()
                if not line or line.strip().startswith(';Edge'):
                    break
                if line.strip().startswith('#') or not line.strip() or line.strip().startswith(';'):
                    continue
                label, node_ins, node_stu = line.strip().split('\t')
                node_ins, node_stu = ':'.join(node_ins.split(':')[1:]), ':'.join(node_stu.split(':')[1:])
                if node_ins.startswith('ROOT:ROOT') or node_stu.startswith('ROOT:ROOT'):
                    continue
                # The same node from parse and from annotations are different
                nodes = node_ins + ',' + node_stu
                training_data[fn]['labels'][nodes] = -1 if float(label) == 0 else 1
                # training_data[fn]['labels'][nodes] = float(label)-1
    print('Training data of perceptron: ', training_data)

    # perceptron training
    print('Start to train...')
    w_perceptron = np.array([0.0] * 68)
    w_avg = np.array([0.0] * 68)
    n = 0
    # pbar = progressbar.ProgressBar(max_value=epochs)
    for _i in range(epochs):
        # pbar.update(_i)
        # print('The {} th epochs...'.format(_i))
        for ans in training_data:

            ans_stu = training_data[ans]['ans_stu']
            ans_ins = training_data[ans]['ans_ins']
            # print("\nTraining with answer ", ans)

            for node_pair in training_data[ans]['labels']:
                # start = time.time()
                # time1 = time.time()
                node_ins, node_stu = node_pair.split(',')
                label = training_data[ans]['labels'][node_pair]
                # time2 = time.time()
                v_phi = phi(node_stu, ans_stu, node_ins, ans_ins, cache, ic)
                # time3 = time.time()
                f_value = w_perceptron.dot(v_phi)
                # time4 = time.time()
                if f_value == 0 and label != 0 or f_value != 0 and label == 0 or f_value * label < 0:
                    w_perceptron += np.float64(label) * v_phi
                # time5 = time.time()
                w_avg += w_perceptron
                n += 1
    pbar.finish()
    # print('Perceptron train finished.')
    return w_avg / n


def alignment(ans_stu, ans_ins, que, w_phi, cache, ic, transform=0):
    # This results is in an optimal matching, not a mapping, so that an individual node is
    # associated with at most one node in the other answer
    # 3 transforms:
    # 1. Normalize a matching : Divide the total alignment score by the number of nodes
    #   in the instructor answer to avoid longer answers getting higher score.
    # 2. Scales the node matching score by multiplying it with the idf of the instructor
    #   answer node. replace f(x_i, x_s) with idf(x_i)*f(x_i, x_s)
    # 3. Remove any words in the question from both the instructor answer and the student answer.
    #
    # ans_stu, ans_ins, sgs_que:
    #   subgraphs of students' answer, instructor answer and question.
    #   sgs_que is question used for answer demoting for transform 3.
    # transform:
    #   0: 000b / No transform will be done.
    #   1: 001b / Apply transform(1)
    #   2: 010b / Apply transform(2)
    #   4: 100b / Apply transform(3)
    #   ...
    #   7: 111b / Apply all transforms

    # generate a nxn cost matrix
    # x deriction: nodes of student answer
    # y deriction: nodes of instructor answer
    # Same to Hungarian algorithm.
    # print("Alignment start!, transform = ", transform)
    nodes_stu, nodes_ins, nodes_que = ans_stu.nodes, ans_ins.nodes, que.nodes
    if 0b011 | transform == 0b111:  # Normalize 3
        # print("Normlize 3 DONE!")
        st = PorterStemmer()

        def not_que_node(node):
            for node_que in nodes_que:
                if st.stem(node.split(':')[0]) == st.stem(node_que.split(':')[0]):
                    return False
            return True

        nodes_stu = list(filter(not_que_node, nodes_stu))
        nodes_ins = list(filter(not_que_node, nodes_ins))
    if min(len(nodes_stu), len(nodes_ins)) == 0:
        return 0
    size = max(len(nodes_stu), len(nodes_ins))
    matrix = [[0] * size] * size
    normalize_1 = len(nodes_ins)
    for (i, s) in [(i, s) for i in range(len(nodes_ins)) for s in range(len(nodes_stu))]:
        matrix[i][s] = 1 / (w_phi.dot(phi(nodes_stu[s], ans_stu, nodes_ins[i], ans_ins, cache, ic)) + 1)
        if 0b110 | transform == 0b111:  # Normalize 1
            # print("Normlize 1 DONE!")
            matrix[i][s] /= normalize_1
        if 0b101 | transform == 0b111:  # Normalize 2
            # print("Normlize 2 DONE!")
            matrix[i][s] *= 1  # TODO: update it to idf value
    m = Munkres()

    indexes = m.compute(matrix)
    sim_scores = [matrix[r][c] for (r, c) in indexes]
    # print('Alighment Scores: {}~{}'.format(max(sim_scores),  min(sim_scores)))
    alignment_score = sum(sim_scores)

    return alignment_score


def tf_idf_weight_answer_v(ans_stu, ans_ins, answer_demoting=False):
    def tokenize(sen_text):
        tokens = nltk.word_tokenize(sen_text)
        stems = []
        for item in tokens:
            stems.append(PorterStemmer().stem(item))
        return stems

    rm_dic = dict((ord(p), None) for p in string.punctuation)

    path = './docs'
    token_dict = {}
    for dirpath, dirs, files in os.walk(path):
        for f in files:
            fname = os.path.join(dirpath, f)
            print("fname=", fname)
            with open(fname) as pearl:
                text = pearl.read()
                token_dict[f] = text.lower().translate(rm_dic)

    tfidf_vector = TfidfVectorizer(tokenizer=tokenize, stop_words='english', sublinear_tf=False)
    tfidf = tfidf_vector.fit_transform(token_dict.values())

    if answer_demoting:
        answer_pair = ' '.join(ans_stu.words_with_demoting) + ' ' + ' '.join(ans_ins.words_with_demotin)
    else:
        answer_pair = ' '.join(ans_stu.words) + ' ' + ' '.join(ans_ins.words)
    tfidf_values = tfidf.transform([answer_pair])
    return [tfidf[0, col] for col in tfidf_values.nonzero()[1]]


def get_tokens(text, gram_n, char_gram):
    lower = text.lower()
    # remove_punctuation_map = dict((ord(char), None) for char in string.punctuation)
    # no_punctuation = lower.translate(remove_punctuation_map)
    # tokens = list(no_punctuation) if char_gram else nltk.word_tokenize(no_punctuation
    tokens = token_lemma(lower, NLP, LEMMATIZER)
    return list(nltk.ngrams(tokens, gram_n))


def read_voc(text):
    lower = text.lower()
    remove_punctuation_map = dict((ord(char), None) for char in string.punctuation)
    no_punctuation = lower.translate(remove_punctuation_map)
    tokens = nltk.word_tokenize(no_punctuation)
    return tokens


def read_tokens_answer(answer, gram_n, char_gram):
    # Answers are starts with answer id
    # Remove answer id first before extract tokens
    answer = answer[answer.find(' ') + 1:]
    tokens = get_tokens(answer, gram_n=gram_n, char_gram=char_gram)
    return sorted(tokens)


def read_tokens_answers(que_id, gram_n, ref, char_gram):
    '''
    Read all the tokens of answers under a question with id of que_id. Tokens consist of n-gram tuples.
    :param que_id:
        question id
    :param gram_n:
        number of grams
    :param ref:
        If ref is True, reference answer is needed and will be read in.
    :param char_gram:
        If char_gram is True, character n-gram will be applied.
    :param query_id:
        answers with idx in query_id will be skipped when generate vocabulary
    :return:
        An sorted list of token set.
    '''
    token_set = set()
    if ref:
        # read reference answer
        with open(RAW_PATH + "/answers", errors="ignore") as f_ref:
            for answer in f_ref.readlines():
                if answer.startswith(que_id):
                    token_set = token_set.union(read_tokens_answer(answer, gram_n=gram_n, char_gram=char_gram))
                    break

    # read student answers
    with open(RAW_PATH_STU + "/" + que_id, "r", errors="ignore") as f_ans_raw:
        try:
            for answer in f_ans_raw.readlines():
                token_set = token_set.union(read_tokens_answer(answer, gram_n=gram_n, char_gram=char_gram))
        except:
            print("error:", answer)
    assert token_set
    return sorted(list(token_set))


def generate_features_bow(grams_n_list, ref, char_gram):
    '''
    generate n-gram features for BOW
    :param grams_n_list:
        A list of number as n-gram. If the length of list is greater than 1, then the feature will be
        extended with all the n-gram features like [ ... 1-gram ..., ... 2-gram ..., ..., ... n-gram ...]
    :param ref:
        When ref is True, reference answer will be read as one of the training data. Leave it as False when
        there's no reference answer.
    :param char_gram:
        Character n-gram features will be generated when char_gram is True
    :return:
        None. The featrues will be written to files named with n-gram
    '''
    # stemmer = PorterStemmer()
    for que_id in sorted(os.listdir(RAW_PATH_STU)):
        print("\n" + que_id)

        # generate bow features
        if char_gram:
            feature_path = RESULTS_PATH + "/features_bow_{}gram_char/".format("-".join(map(str, grams_n_list)))
        else:
            feature_path = RESULTS_PATH + "/features_bow_{}gram/".format("-".join(map(str, grams_n_list)))
        if not os.path.exists(feature_path):
            os.makedirs(feature_path)


        # Read n-gram set from answers of question with id of que_id.
        with open(feature_path + "/" + que_id, "wt", encoding='utf-8', errors="ignore") as f_fea,  \
            open(RAW_PATH_STU + "/" + que_id, 'r', encoding='utf-8', errors='ignore') as f_ans:
            voc_que = list()
            ans_list = f_ans.readlines()

            #generate vocabulary for each question
            # with open('{}/vocabulary_svr/bow_{}'.format(RESULTS_PATH, que_id, i + 1), 'w', encoding='utf-8',
            #           errors="ignore") as f_voc:
            #     for gram in grams_n_list:
            #         voc = set()
            #         for ans in ans_list:
            #             tokens = set(read_tokens_answer(ans, gram, char_gram))
            #             # voc = {1}
            #             voc = voc.union(tokens)
            #         voc_que[i][gram] = sorted(list(voc))
            #         f_voc.write('\t'.join([','.join(t) for t in voc_que[i][gram]]) + '\t')

            #bar = progressbar.ProgressBar(max_value=len(ans_list))
            #bar_i = 0
            for i in range(len(ans_list)):
                voc_que.append({})
                # for each answer generate vocabulary
                with open('{}/vocabulary_svr/bow_{}.{}'.format(RESULTS_PATH, que_id, i + 1), 'w', encoding='utf-8', errors="ignore") as f_voc:
                    for gram in grams_n_list:
                        voc = set()
                        # for ans in ans_list[:i] + ans_list[i+1:]:
                        for ans in ans_list:
                            tokens = set(read_tokens_answer(ans, gram, char_gram))
                            # voc = {1}
                            voc = voc.union(tokens)
                        voc_que[i][gram] = sorted(list(voc))
                        f_voc.write('\t'.join([','.join(t) for t in voc_que[i][gram]])+'\t')

                # generate features for each answer
                features = []
                for gram in grams_n_list:
                    tokens_answer = set(read_tokens_answer(ans_list[i], gram, False))
                    bow = [1] * len(voc_que[i][gram])
                    for j in range(len(voc_que[i][gram])):
                        bow[j] = 1 if voc_que[i][gram][j] in tokens_answer else 0
                    features.extend(bow)

                print(*features, file=f_fea, sep=',')
                # bar.update(bar_i)
                # bar_i += 1



        #
        # with open(feature_path + "/bow_{}".format(que_id), "wt", encoding='utf-8',
        #           errors="ignore") as f_bow:
        #     for gram in grams_n_list:
        #         tokens_que[gram] = tuple(read_tokens_answers(que_id, gram_n=gram, ref=ref, char_gram=char_gram))
        #         f_bow.write("\t".join(map(','.join, tokens_que[gram])) + "\t")
        #
        # with open(feature_path + "/" + que_id, "wt", encoding='utf-8', errors="ignore") as f_fea, \
        #         open(RAW_PATH_STU + "/" + que_id, "r", encoding='utf-8', errors="ignore") as f_ans:
        #     f_ans_lines = f_ans.readlines()
        #     bar = progressbar.ProgressBar(max_value=len(f_ans_lines))
        #     bar_i = 0
        #     for answer in f_ans_lines:
        #         features = []
        #         for gram in grams_n_list:
        #             tokens_answer = set(token_lemma(answer, NLP, LEMMATIZER))
        #             bow = [1] * len(voc_que[gram])
        #             for i in range(len(voc_que[gram])):
        #                 bow[i] = 1 if voc_que[i][gram][i] in tokens_answer else 0
        #             features.extend(bow)
        #
        #         print(*features, file=f_fea, sep=',')
        #         bar.update(bar_i)
        #         bar_i += 1


def generate_feature_g(ans_stu, ans_ins, que, w_phi, cache, ic):
    """
    Generate feature psi_G for each student answer.
    Input:
        ans_stu, ans_ins, que: Sentence objects of student/instructor answers and question
        w: vector for calculating node-level matching for alignment
    Output:
        A list of feature (30-dimension feature vector)
    """

    # feature vector for SVM and SVMRANK
    # 8 knowledge based measures of semantic similarity + 2 corpus based measures
    # +1 tf*idf weights ==> 11 dimension feature vector

    # psi_G
    # contains the eight alignment scores found by applying the three transformations in the graph alignment stage.
    psi_g_8 = [alignment(ans_stu, ans_ins, que, w_phi, cache, ic, transform=i) for i in range(8)]
    return psi_g_8


def generate_feature_b(ans_stu, ans_ins, que, w_phi, cache, ic):
    """
    Generate feature psi_B for each student answer.
    Input:
        ans_stu, ans_ins, que: Sentence objects of student/instructor answers and question
        w: vector for calculating node-level matching for alignment
    Output:
        A list of feature (30-dimension feature vector)
    """
    # feature vector for SVM and SVMRANK
    # 8 knowledge based measures of semantic similarity + 2 corpus based measures
    # +1 tf*idf weights ==> 11 dimension feature vector

    psi_b_kbfa_8 = knowledge_based_feature_between_sentence_8(ans_stu.words, ans_ins.words, cache)
    psi_b_la = 1  # TODO: lsa bewteen two sentence?
    psi_b_ea = 1  # TODO: esa bewteen two sentence?
    # psi_b_ti = tf_idf_weight_answer_v(ans_stu.words, ans_ins.words)
    psi_b_ti = 1  # TODO: esa bewteen two sentence?

    psi_b_11_without_demoting = psi_b_kbfa_8
    psi_b_11_without_demoting.append(psi_b_la)
    psi_b_11_without_demoting.append(psi_b_ea)
    psi_b_11_without_demoting.append(psi_b_ti)

    psi_b_kbfa_8 = knowledge_based_feature_between_sentence_8(ans_stu.words_with_demoting, ans_ins.words_with_demoting,
                                                              cache)
    psi_b_la = 1  # TODO: lsa between two sentence?
    psi_b_ea = 1  # TODO: esa between two sentence?
    # psi_b_ti = tf_idf_weight_answer_v(ans_stu.words_with_demoting, ans_ins.words_with_demoting)
    psi_b_ti = 1

    psi_b_11_with_demoting = psi_b_kbfa_8
    psi_b_11_with_demoting.append(psi_b_la)
    psi_b_11_with_demoting.append(psi_b_ea)
    psi_b_11_with_demoting.append(psi_b_ti)

    features_22 = psi_b_11_with_demoting + psi_b_11_without_demoting
    print('features_22: ', features_22)
    return features_22


def generate_feature(ans_stu, ans_ins, que, w_phi, cache, ic):
    """
    Generate feature for each student answer.
    Input:
        ans_stu, ans_ins, que: Sentence objects of student/instructor answers and question
        w: vector for calculating node-level matching for alignment
    Output:
        A list of feature (30-dimension feature vector)
    """

    psi_b_kbfa_8 = knowledge_based_feature_between_sentence_8(ans_stu.words, ans_ins.words, cache)
    psi_b_la = 1  # TODO: lsa between two sentence?
    psi_b_ea = 1  # TODO: esa between two sentence?
    # psi_b_ti = tf_idf_weight_answer_v(ans_stu.words, ans_ins.words)
    psi_b_ti = 1  # TODO: esa between two sentence?

    psi_b_11_without_demoting = psi_b_kbfa_8
    psi_b_11_without_demoting.append(psi_b_la)
    psi_b_11_without_demoting.append(psi_b_ea)
    psi_b_11_without_demoting.append(psi_b_ti)

    psi_b_kbfa_8 = knowledge_based_feature_between_sentence_8(ans_stu.words_with_demoting, ans_ins.words_with_demoting,
                                                              cache)
    psi_b_la = 1  # TODO: lsa between two sentence?
    psi_b_ea = 1  # TODO: esa between two sentence?
    # psi_b_ti = tf_idf_weight_answer_v(ans_stu.words_with_demoting, ans_ins.words_with_demoting)
    psi_b_ti = 1

    psi_b_11_with_demoting = psi_b_kbfa_8
    psi_b_11_with_demoting.append(psi_b_la)
    psi_b_11_with_demoting.append(psi_b_ea)
    psi_b_11_with_demoting.append(psi_b_ti)

    # psi_G
    # contains the eight alignment scores found by applying the three transformations in the graph alignment stage.
    psi_g_8 = [alignment(ans_stu, ans_ins, que, w_phi, cache, ic, transform=i) for i in range(8)]
    features_30 = psi_g_8
    features_30.extend(psi_b_11_with_demoting)
    features_30.extend(psi_b_11_without_demoting)
    # print('Features:', features_30)
    return features_30


def generate_features(que_id, w_phi, cache, ic, feature_type, fn_ans_ins='answers', fn_que='questions'):
    """
    Input:
        A parse file of dependence graph. One student answer each line.
    Output:
        A feature file. One feature vector of an answer for each line.
        Dimensions are seperated by space
    que_id: String
        File name of student answers. 1.1, 1.2, ..., etc.
        The que_id will be used to locate the answer and question files.
        It must be the NO. of q/a.
    """
    path_fn_ans_stu = DATA_PATH + '/parses/' + que_id
    path_fn_ans_ins = DATA_PATH + '/parses/' + fn_ans_ins
    path_fn_que = DATA_PATH + '/parses/' + fn_que
    print("On processing: " + path_fn_ans_stu)
    print("Instructor file is: " + path_fn_ans_ins)
    ans_ins, ans_stu_s, que = None, None, None

    # Read the instructor answers based on the input number
    print('Reading file:', path_fn_ans_ins)
    with open(path_fn_ans_ins, 'r') as f_ans_ins:
        while True:
            ans_ins_text = f_ans_ins.readline()
            if not ans_ins_text:
                break
            if ans_ins_text.startswith(que_id):
                ans_ins = Sentence(ans_ins_text)
                break

    # Read the question based on the input number
    print('Reading file:', path_fn_que)
    with open(path_fn_que, 'r') as f_que:
        while True:
            que_text = f_que.readline()
            if not que_text:
                break
            if que_text.startswith(que_id):
                que = Sentence(que_text)
                break

    # Read student answers
    ans_stu_s = []
    print('Reading file:', path_fn_ans_stu)
    with open(path_fn_ans_stu, 'r') as f_ans_stu:
        aid = 0
        while True:
            ans_stu_text = f_ans_stu.readline()

            if not ans_stu_text:
                break
            if not ans_stu_text.startswith(que_id):
                continue
            aid += 1
            ans_stu = Sentence(ans_stu_text, str(aid))
            ans_stu.question_demoting(que.words)
            ans_stu_s.append(ans_stu)

    # Generate features for SVMRank
    # w is trained by a subset of answers used for calculating the node-to-node
    # score
    # Also tf-idf vector need to be trained in advance.
    if not (ans_stu_s and ans_ins and que):
        return -1
    feature_path = RESULTS_PATH + '/features_' + feature_type
    if not os.path.exists(feature_path):
        os.mkdir(feature_path)
    with open(feature_path + '/' + que_id,
              'wt') as f:  # , open(sys.path[0]+'/../data/scores/'+que_id+'/ave') as fs:
        for ans_stu in ans_stu_s:
            if feature_type == 'b':
                feature = generate_feature_b(ans_stu, ans_ins, que, w_phi, cache, ic)
            if feature_type == 'g':
                feature = generate_feature_g(ans_stu, ans_ins, que, w_phi, cache, ic)
            else:
                feature = generate_feature(ans_stu, ans_ins, que, w_phi, cache, ic)

            print(','.join(map(str, feature)), file=f)


def run_procerpron_learning():
    ic = wic.ic('ic-bnc.dat')
    similarity_cache = {}
    # epochs = 10
    for epochs in [50]:
        w = perceptron_train(similarity_cache, ic, epochs)
        print('w: ', ','.join(map(str, w)))
        with open('w' + str(epochs), 'w') as f:
            print(','.join(map(str, w)), file=f)


def run_gen_features(qids='all', fn_w='w', feature_type='gb'):
    fw = RESULTS_PATH + '/' + fn_w
    with open(fw, 'r') as f:
        w_string = f.readline()
        print('w: ', w_string)
    w_phi = np.array(list(map(np.float64, w_string.split(','))))
    similarity_cache = {}
    ic = wic.ic('ic-bnc.dat')
    path = DATA_PATH + '/scores/'
    if qids == 'all':
        qids = os.listdir(path)
    # qids = ['LF_33b']
    print(qids)
    for qid in qids:
        generate_features(qid, w_phi, similarity_cache, ic, feature_type)


def read_training_data(feature_path, raw_path=RAW_PATH, score_path=DATA_PATH + '/scores/', include_ref=False):
    id_que = os.listdir(feature_path)
    record = list()
    for i in id_que:
        with open(feature_path + '/' + i, 'r') as ff, \
                open(score_path + '/' + i + '/ave') as fs, \
                open(raw_path + "/answers", "r", errors="ignore") as f_raw_r, \
                open(raw_path + "/questions", "r", errors="ignore") as f_raw_q, \
                open(raw_path + "/ans_stu/" + i, "r", errors="ignore") as f_raw_s, \
                open(score_path + "/" + i + '/diff') as fd:
            scores_truth = np.array(list(map(np.float64, fs.readlines())))
            diff = np.array(list(map(np.float64, fd.readlines())))
            features = list(map(lambda s: s.split(','), ff.readlines()))
            features = (list(map(lambda l: np.array(list(map(np.float64, l))), features)))
            raw_r, raw_q, raw_s = '', '', []

            for s in f_raw_q.readlines():
                if s.startswith(i):
                    raw_q = s
                    break

            for s in f_raw_r.readlines():
                if s.startswith(i):
                    raw_r = s
                    break

            id_q = [i] * len(features)
            id_s = list(range(1, len(features) + 1))

            raw_stu = np.array(list(map(lambda s: s.strip(), f_raw_s.readlines())))
            raw_que = [raw_q] * len(features)
            raw_ref = [raw_r] * len(features)

            recode_i = list(zip(id_q, raw_que, id_s, raw_stu, raw_ref, features, scores_truth, diff))
            record.extend(recode_i)

    TrainingData = collections.namedtuple('TrainingData', 'id id_que que id_ans ans ref feature score diff')
    ret = TrainingData(list(range(len(record))), *list(map(np.array, zip(*record))))
    # print(ret.id, ret.id_que, ret.stu)
    return ret


def score_answer(fn_prefix, reliable, feature, model, model_params, qwise, training_scale):
    fn_params = ['{}_{}'.format(k, v) for k, v in model_params.items()]
    fn = '{}.{}.{}.{}.{}.scale{}.{}'.format(fn_prefix, feature, 'reliable' if reliable else 'unreliable',
                                    'qwise' if reliable else 'unqwise', ".".join(fn_params), training_scale,cur_time())

    result_path = RESULTS_PATH + '/results/' + fn
    if not os.path.exists(result_path):
        os.mkdir(result_path)

    # Initialize the model
    if 'knnc' == model:
        runner = neighbors.KNeighborsClassifier(**model_params)
    elif 'knnr' == model:
        runner = neighbors.KNeighborsRegressor(**model_params)
    elif 'svr' == model:
        runner = SVR(**model_params)
    elif 'cos' == model:
        runner = CosineKNN(**model_params)
    elif 'cosc' == model:
        runner = CosineKNNC(**model_params)
    # Read training data
    training_data = read_training_data(RESULTS_PATH + "/features_" + feature)

    n_data = len(training_data.id)
    with open(result_path + '/result.txt', 'w') as fr:
        for i in training_data.id:
            filter_arr = list()
            if qwise:
                filter_qwise = np.array(training_data.id_que) == training_data.id_que[i]
                filter_arr.append(filter_qwise)
            if reliable:
                filter_arr.append(np.array(training_data.diff) < 3)
            filter_rm = [True] * n_data
            filter_rm[i] = False
            filter_arr.append(filter_rm)

            filter_arr = np.array(list(map(lambda f: reduce(lambda x, y: x and y, f), zip(*filter_arr))))

            scores_truth = training_data.score[filter_arr]
            features = training_data.feature[filter_arr]
            no_of_answers = training_data.id_ans[filter_arr]
            id_ques = training_data.id_que[filter_arr]

            X = features[:training_scale] if training_scale > 0 else features
            X = np.vstack(X)
            Y = scores_truth[:training_scale] if training_scale > 0 else scores_truth
            Y = (Y * 2).astype(int)
            score_truth_i = training_data.score[i]
            feature_i = training_data.feature[i]
            # training
            runner.fit(X, Y)
            # predict
            score = runner.predict(np.array([feature_i])) / 2

            error = score_truth_i - score[0]
            error_abs = abs(error)
            error_round = round(error_abs)
            question = training_data.que[i].strip()
            ans_ref = training_data.ref[i].strip()
            ans_stu = training_data.ans[i].strip()
            que_id = training_data.id_que[i]
            ans_id = training_data.id_ans[i]

            len_ref = len(list(filter(lambda x:x, clean_text(ans_ref).split(' '))))
            len_stu = len(list(filter(lambda x:x, clean_text(ans_stu).split(' '))))
            if 'knnc' == model or 'knnr' == model:
                distance_of_neighbors, no_of_neighbors = runner.kneighbors(np.array([feature_i]),
                                                                           model_params['n_neighbors'])
                # runner.predict_proba()
                # Find the N.O. of nearest answers by features
                n_s = ['{}.{}'.format(training_data.id_que[i], no_of_answers[no]) for no in no_of_neighbors[0]] # ans_id
                t_s = [str(Y[no] / 2) for no in no_of_neighbors[0]] # scores of neighbors
                d_s = distance_of_neighbors[0]  # distance
                neighbor_scores = [Y[no]/2 for no in no_of_neighbors[0]]
                mean_neighbor_scores = sum(neighbor_scores) / len(neighbor_scores)
                mean_distance = sum(d_s)/len(d_s)
                print('score of {}.{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}'.format(
                    que_id, ans_id,
                    score[0],
                    score_truth_i,
                    error,
                    error_abs,
                    error_round,
                    question,
                    ans_ref,
                    ans_stu))
                print('score of {qid}.{aid}\t{pred}\t{truth}\t{err}\t{err_abs}\t{que}\t{ref}\t{stu}\t{id_nei}\t \
                    {score_nei}\t{mean_score_nei}\t{mean_dist}\t{len_ref}\t{len_stu}'.format(
                    qid=que_id, aid=ans_id, pred=score[0], truth=score_truth_i, err=error, err_abs=error_abs,
                    que=question, ref=ans_ref, stu=ans_stu, id_nei=','.join(n_s), score_nei=','.join(t_s),
                    mean_score_nei=mean_neighbor_scores, mean_dist=mean_distance, len_ref=len_ref, len_stu=len_stu),
                    file=fr)
                # with open(result_path + '/features.txt', 'a') as f_features:
                #     print('X of {}.{}:'.format(que_id, ans_id),  X, file=f_features)
            elif 'svr' == model:
                # if model_params['kernel'] == 'linear':
                #     # generate word_weights_dict
                #     with open('{}/vocabulary_svr/bow_{}.{}'.format(RESULTS_PATH, que_id, ans_id ), 'r', errors="ignore") as f_voc, \
                #         open('{}/word_weights_svr/{}.{}'.format(RESULTS_PATH, que_id, ans_id ), 'w') as f_weight:
                #         voc = f_voc.readline().strip().split('\t')
                #         weight = runner.coef_
                #         weight_dict = dict(zip(voc, weight[0]))
                #         string = ','.join(["{}:{}".format(k, w) for (k, w) in weight_dict.items()])
                #         f_weight.write(string)

                print('score of {}.{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}'.format(
                    que_id, ans_id, score[0], score_truth_i,
                    error,
                    error_abs, error_round, question, ans_ref,
                    ans_stu))
                print('score of {}.{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}'.format(
                    que_id, ans_id, score[0], score_truth_i,
                    error,
                    error_abs, error_round, question, ans_ref, ans_stu),
                    file=fr)

            elif 'cos' == model or 'cosc' == model:
                idx, dist = runner.k_nearest(np.array([feature_i]), model_params['n_neighbors'])
                n_s = ['{}.{}'.format(training_data.id_que[i], no_of_answers[no]) for no in idx]
                t_s = [Y[no] / 2 for no in idx]
                print('score of {}.{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}'.format(
                    que_id, ans_id, score[0], score_truth_i,
                    error,
                    error_abs, error_round, question, ans_ref,
                    ans_stu))
                print('score of {}.{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}'.format(
                    que_id, ans_id, score[0], score_truth_i,
                    error,
                    error_abs, error_round, question, ans_ref, ans_stu, *n_s, *t_s),
                    file=fr)

def read_w2v(f_w2v):
    w2v = dict()
    print("reading w2v file...", end='')
    with open(f_w2v, 'r', encoding='utf-8') as f:
        for line in f:
            data = line.split()
            w2v[data[0]] = np.array(list(map(float, data[1:])))
    with open(f_w2v, 'r') as f:
        d_vec = len(f.readline().split()) - 1
    return w2v, d_vec
class Feature:
    def __init__(self, w2v=None, d_vec=None, nlp=None, lemmatizer=None, vali = 0.1):
        self.__scores = []  # Scores of training data
        self.__weights = {}  # Weights of each words
        self.__voc = []  # Vocabulary of training data. Weight of words not in this list is 0.
        self.__w2v = dict()  # Trained word2vec instance
        self.__sent_words = []  # Stemmed words list of each answer.
        self.d_vec = 100  # Dimension of vector from w2v
        self.__vec_sent_words = []  # Vector version of self.__sent_words

        self.__nlp = spacy.load('en') if not nlp else nlp
        self.__lemmatizer = lemmatizer if lemmatizer else spacy.lemmatizer.Lemmatizer(LEMMA_INDEX, LEMMA_EXC,
                                                                                      LEMMA_RULES)
        self.__validation_data = None   # a list of answer couples:[((sent1, score1), (sent2, score2)), (), (),...]
        self.__training_data = None     # a list of answer couples:[((sent1, score1), (sent2, score2)), (), (),...]
        self.__cur = 0  # index of training data that is used for current epoch
        self.__step = 1 # For each epoch, the weight will be updated using training_data[cur:step]
        if not w2v:
            # self.__w2v = Word2Vec.load(f_w2v)
            self.__read_w2v(f_w2v)
        else:
            self.__w2v = w2v
            self.d_vec = d_vec


    def __read_w2v(self, f_w2c):
        print("reading w2v file...", end='')
        with open(f_w2c, 'r', encoding='utf-8') as f:
            for line in f:
                data = line.split()
                self.__w2v[data[0]] = np.array(list(map(float, data[1:])))
        with open(f_w2c, 'r') as f:
            self.d_vec = len(f.readline().split()) - 1
        print("done")

    def __sent2vec(self, sent, weights=None):
        '''
            generate sentence vectors based on vector of words
            :param sent: iterable variable of words in the sentence
            :return: a vector with dimension of self.d_vec (same to w2v)
            '''
        if not weights:
            weights = self.__weights
        vec_sent = np.zeros(self.d_vec)
        for word in sent:
            if word not in weights:
                print("Can't find word '{}' in weight dict.".format(word))
                continue
            if word not in self.__w2v:
                print("Can't find word '{}' in Word2Vec".format(word))

                continue
            # vec_word = self.__w2v[word]
            vec_sent += weights.get(word, 0) * self.__w2v[word]
        return vec_sent/len(sent)

    def __distance_func(self, s1, s2):
        '''
        hyper function
        :param weights: dict(word:weight)
        :param s1:
        :param s2: set of words of sentence1 and sentence2
        :return: a value between 0 and 1
        '''

        # return similarity between two sentence
        # return spatial.distance.cosine(s1, s2)
        distance = cos_distance(s1, s2)
        return distance

    def __loss_func_mini(self, weights, sents = None):
        loss = []
        weights_dict = dict(zip(self.__voc, weights))
        # Generate vector of answers
        if not sents:
            loss_data = self.__training_data[self.__cur: self.__cur + self.__step]
        else:
            loss_data = sents[self.__cur: self.__cur + self.__step]
        self.__cur = (self.__cur + self.__step) % len(self.__training_data)
        loss = []
        for data1, data2 in loss_data:
            vec1 = self.__sent2vec(data1[0], weights_dict)
            score1 = data1[1]
            vec2 = self.__sent2vec(data2[0], weights_dict)
            score2 = data2[1]
            distance = self.__distance_func(vec1, vec2)
            # if not isinstance(vec1, np.ndarray): vec1 = vec1._value
            # if not isinstance(vec2, np.ndarray): vec2 = vec2._value
            # if sum(vec1) == 0 or sum(vec2) == 0:
            #     distance = 0
            # else:
            #     distance = spatial.distance.cosine(vec1, vec2)
            # loss += distance if score1 == score2 else 1 - distance
            loss.append(distance if score1 == score2 else 1 - distance)
            # loss.append(abs(distance - abs(score2-score1)/5))
        # vector_ans = [self.__sent2vec(ans[0], weights_dict) for ans in training_data]
        # for i in range(len(vector_ans)):
        #     ws1 = vector_ans[i]
        #     for j in range(i, len(vector_ans)):
        #         ws2 = vector_ans[j]
        #         distance = self.__distance_func(ws1, ws2)
        #         loss.append(distance if self.__scores[i] == self.__scores[j] else 1 - distance)
        # loss = loss / len(training_data)
        loss = sum(loss) / len(loss)
        # print('loss:', loss)
        return loss

    def __loss_func(self, weights, sents = None):
        loss = []
        weights_dict = dict(zip(self.__voc, weights))

        # Generate vector of answers
        if not sents:
            sents = self.__sent_words
        vector_ans = [self.__sent2vec(ans, weights_dict) for ans in sents]
        for i in range(len(vector_ans)):
            ws1 = vector_ans[i]
            for j in range(i, len(vector_ans)):
                ws2 = vector_ans[j]
                distance = self.__distance_func(ws1, ws2)
                loss.append(distance if self.__scores[i] == self.__scores[j] else 1 - distance)
        loss = sum(loss) / len(loss)
        # print('loss:', loss)
        return loss

    #
    #
    # def __loss_func_part(self, weights):
    #     loss = []
    #     weights_dict = dict(zip(self.__voc, weights))
    #     # Generate vector of answers
    #     vector_ans = [self.__sent2vec(ans, weights_dict) for ans in training_data]
    #     for i in range(len(vector_ans)):
    #         ws1 = vector_ans[i]
    #         for j in range(i, len(vector_ans)):
    #             ws2 = vector_ans[j]
    #             distance = self.__distance_func(ws1, ws2)
    #             loss.append(distance if self.__scores[i] == self.__scores[j] else 1 - distance)
    #     loss = loss / len(loss)
    #     print('loss:', loss)
    #     return loss

    def __token_lemma(self, string):
        doc = self.__nlp(string)
        return [self.__lemmatizer(doc[i].string, doc[i].pos)[0] for i in range(len(doc)) if doc[i].pos_ != u'PUNCT']

    def fit_simple(self, answers, scores):
        '''
        Prepare for generating vectors for answers.
        Simple method
        * Generate vocabulary
        * Train word2vec instance if necessary
        * Calculate weight for each word
        :param answers: A list of raw data of answers
        :param scores: A list of scores as training data. The length need to be
                        the same with with the list of answers
        :return: None
        '''

        assert len(answers) == len(scores)
        # weight = [0] * len(words)
        # self.weight = dict(zip(words, weight))

        # Generate bag of words for each sentence
        self.__scores = scores[:]
        words = []
        st = PorterStemmer()
        for a in answers:
            words.append(set(read_voc(text=a, stemmer=st)))
        self.__sent_words = words[:]
        assert len(words) == len(answers)

        # calculate weights for each word in vocabulary
        self.__voc = list(reduce(lambda x, y: x | y, words))
        l_answers = len(answers)
        for i, j in [(i, j) for i in range(l_answers) for j in range(l_answers)]:
            if i == j:
                continue
            # TODO: The algorithm to weight words need to be updated
            if scores[i] == scores[j]:
                for word in words[i] & words[j]:
                    self.__weights[word] = self.__weights.get(word, 0) + 1
                for word in words[i] ^ words[j]:
                    self.__weights[word] = self.__weights.get(word, 0) - 1

            if scores[i] != scores[j]:
                for word in words[i] & words[j]:
                    self.__weights[word] = self.__weights.get(word, 0) - 1
                for word in words[i] ^ words[j]:
                    self.__weights[word] = self.__weights.get(word, 0) + 1

            for w in self.__weights:
                self.__weights[w] = sigmoid(self.__weights[w])

        # print(sorted(self.__weights.items(), key=lambda d:d[1], reverse=True)[:11])

        if not self.__w2v:
            # No w2v model is provided, then train a new one use current vocabulary
            voc = [nltk.word_tokenize(s) for s in answers]
            self.__w2v = Word2Vec(voc, min_count=1)

            # Generate vectors for query answer
            # for words in self.sent_words:
            #     self.vec_sent.append(self.sent2vec(words))

            # training knn model

    def fit(self, answers, scores, lemmatizer):
        '''
        Prepare for generating vectors for answers.
        * Generate vocabulary
        * Train word2vec instance if necessary
        * Calculate weight for each word
        :param answers: A list of raw data of answers
        :param scores: A list of scores as training data. The length need to be
                        the same with with the list of answers
        :param f_w2v: path/to/file of an instance of word2vec model. If not provided, a new one will be trained
                        with the vocabulary
        :return: None
        '''

        assert len(answers) == len(scores)

        self.__scores = list(map(float, scores))

        # Tokenize each answer with lemmatization
        print("Tokenizing...")
        if self.__w2v:
            # if word2vec model is provided, the vector of answer words will be generated at the same time
            words_of_ans = []
            word_vectors_of_ans = []
            for ans in answers:
                doc = self.__nlp(ans)
                ws = [lemmatizer(doc[i].string, doc[i].pos)[0] for i in range(len(doc)) if doc[i].pos_ != u'PUNCT']
                words_of_ans.append(ws)
                vs = [self.__w2v[w] if w in self.__w2v else np.zeros(self.d_vec) for w in ws]
                word_vectors_of_ans.append(vs)
            self.__sent_words = words_of_ans
            self.__vec_sent_words = word_vectors_of_ans
        else:
            # if word2vec model is not provided, generate tokens first, train the word2vec model
            # with tokens, and generate the vectors of answer words at last
            words_of_ans = []
            word_vectors_of_ans = []
            for ans in answers:
                doc = self.__nlp(ans)
                ws = [lemmatizer(doc[i].string, doc[i].pos)[0] for i in range(len(doc)) if doc[i].pos_ != u'PUNCT']
                words_of_ans.append(ws)
                self.__sent_words = words_of_ans
            self.__w2v = Word2Vec(self.__sent_words, min_count=1)
            for ws in self.__sent_words:
                vs = [self.__w2v[w] if w in self.__w2v else np.zeros(self.d_vec) for w in ws]
                word_vectors_of_ans.append(vs)
            self.__vec_sent_words = word_vectors_of_ans
        assert len(words_of_ans) == len(answers)

        # Generate training data and validation data
        print("Generating training data and validation data...")
        sent_words = self.__sent_words[:]
        sent_score= self.__scores[:]
        sent_data = list(zip(sent_words, sent_score))
        l_sent_data = len(sent_data)

        sent_data = [(sent_data[i], sent_data[j]) for i in range(l_sent_data) for j in range(i,l_sent_data)]
        random.shuffle(sent_data)
        l_sent_data = len(sent_data)

        # take 90% of the data for training, and the left 10% for validation
        edge = int(0.9 * l_sent_data)
        self.__training_data = sent_data[:edge]
        self.__validation_data = sent_data[edge:]

        self.__cur = 0
        self.__step = int(1.0 * len(self.__training_data))
        print("Training step:", self.__step)

        # Generate vocabulary based on all training data
        print("Generateing vocabulary...")
        self.__voc = set(reduce(lambda x, y: set(x) | set(y), words_of_ans))
        print("Size of vocabular:", len(self.__voc))

        # calculate weights for each word in vocabulary
        print("Training weights for words...")
        weights = np.ones(len(self.__voc)) / 10
        # weights = np.array([random.random() for i in range(len(self.__voc))])
        grad_desent = autograd.grad(self.__loss_func)
        print('init_weight(10/0.03):', sorted(weights)[:10])
        epochs = 0
        while True:
            epochs += 1
            gradient = grad_desent(weights)

            weights -= 0.03 * gradient
            loss = self.__loss_func(weights, self.__validation_data)
            print('loss in {}th epoch: {}'.format(epochs, loss))
            print('weights:', sorted(weights)[:11])
            print('gradient:', sorted(gradient)[:10])
            # print()

        self.__weights = dict(zip(self.__voc, weights))
        print(sorted(self.__weights.items(), key=lambda d: d[1], reverse=True)[:11])
        print("Fit done")

    def feature(self, sent):
        # feature_file = RESULTS_PATH + "/features_weighted_bow/" + fn
        return self.__sent2vec(sent)


class CosineKNN():
    def __init__(self, n_neighbors=5, dist_func='cos'):
        self.n_neighbors = n_neighbors
        self.dist_func = None
        self.nearest = dict()
        if 'cos' == dist_func:
            self.dist_func = spatial.distance.cosine
        elif 'l2' == dist_func:
            def euclidean(x, y):
                return np.linalg.norm(x - y)

            self.dist_func = euclidean

    def fit(self, X, Y):
        self.x = X
        self.y = Y

    def k_nearest(self, vec, k):
        neighbors = []
        for idx in range(len(self.x)):
            vec_x = self.x[idx]
            distance = self.dist_func(vec, vec_x)
            neighbors.append((idx, distance))
        neighbors = sorted(neighbors, key=lambda t:t[1])[:k]
        neighbors = list(zip(*neighbors))
        return neighbors


    def predict(self, vecs):
        # compute cosine similarity
        # find top N largest ones
        # calculate score by average
        preds = []
        for vec in vecs:
            distance = []
            for v_x in self.x:
                distance.append(self.dist_func(v_x, vec))
            sim_score = zip(distance, self.y)
            neighbor_scores = list(zip(*sorted(sim_score)))[1][:self.n_neighbors]
            assert neighbors
            preds.append(sum(neighbor_scores) / len(neighbor_scores))
        return np.array(preds)

class CosineKNNC():
    def __init__(self, n_neighbors=5, dist_func='cos'):
        self.n_neighbors = n_neighbors
        self.dist_func = None
        self.nearest = dict()
        if 'cos' == dist_func:
            self.dist_func = spatial.distance.cosine
        elif 'l2' == dist_func:
            def euclidean(x, y):
                return np.linalg.norm(x - y)

            self.dist_func = euclidean

    def fit(self, X, Y):
        self.x = X
        self.y = Y

    def k_nearest(self, vec, k):
        neighbors = []
        for idx in range(len(self.x)):
            vec_x = self.x[idx]
            distance = self.dist_func(vec, vec_x)
            neighbors.append((idx, distance))
        neighbors = sorted(neighbors, key=lambda t:t[1])[:k]
        neighbors = list(zip(*neighbors))
        return neighbors


    def predict(self, vecs):
        # compute cosine similarity
        # find top N largest ones
        # calculate score by average
        preds = []
        for vec in vecs:
            distance = []
            for v_x in self.x:
                distance.append(self.dist_func(v_x, vec))
            distance_score = list(sorted(zip(distance, self.y)))
            count = {}
            dists = {}
            for dis, sco in distance_score:
                count[sco] = count.get(sco, 0) + 1
                if sco in dists:
                    dists[sco].append(dis)
                else:
                    dists[sco] = [sco,]
            for sco in dists:
                dists[sco] = sum(dists[sco])/len(dists[sco])
            score_count = sorted(count.items(), key=lambda x:x[1], reverse=True)
            res_score, res_count = score_count[0]
            for s, c in score_count[1:]:
                if c < res_count:
                    break
                elif dists[s] < dists[res_score]:
                    res_score, res_count = s, c
            preds.append(res_score)
        return np.array(preds)


def train_w2v(fn, model_name, tokenizer):
    with open(fn, 'r', errors='ignore') as f_sents:
        voca = []
        for l in f_sents:
            voca.append(l.split())
        model = Word2Vec(voca, min_count=5)
        model.save(RESULTS_PATH + "/models_w2v/" + model_name)

def score_distance(fn_prefix, reliable, feature, distance, qwise):
    fn = 'score_distance.{}.{}.{}.{}.{}.'.format(fn_prefix, feature, 'reliable' if reliable else 'unreliable',
                                    'qwise' if reliable else 'unqwise', cur_time())

    result_path = RESULTS_PATH + '/results/' + fn
    if not os.path.exists(result_path):
        os.mkdir(result_path)

    # Read data
    feature_path = RESULTS_PATH + "/features_" + feature
    score_path = DATA_PATH + '/scores/'
    ids_que = os.listdir(feature_path)
    distance_sum_all_table = np.array([0] * 6)
    distance_n_all_table = np.array([0] * 6)
    distance_all = []
    for i in range(6):
        distance_all.append(np.array([]))
    with open(result_path + '/score_distance_que.txt', 'w') as f_sd:
        for id_que in ids_que:
            with open(feature_path + '/' + id_que, 'r') as ff, \
                    open(score_path + '/' + id_que + '/ave') as fs:
                scores_truth = np.array(list(map(float, fs.readlines())))
                features = list(map(lambda s: s.split(','), ff.readlines()))
                features = (list(map(lambda l: np.array(list(map(np.float64, l))), features)))

                distance_table = ['0'] * 6
                score_vali = ['0'] * 6

                print('question '+ id_que)
                score_fea = list(zip(scores_truth, features))
                score_fea.sort(key=lambda x:x[0])
                for s, items in groupby(score_fea, key=lambda x:ceil(x[0])):
                    print('score: ',s)
                    feas = list(zip(*items))[1]
                    d_d, d_m, d_st, flag = distance_of_couples(feas, distance)
                    distance_table[s] = '{:.3f}'.format(d_m)
                    distance_all[s] = np.append(distance_all[s], d_d)
                    score_vali[s] = '1' if len(d_d) > 0 else '0'


                print('distance of {qid}\t{distance}\t{score_count}'.format(
                    qid=id_que, distance = '\t'.join(distance_table), score_count = '\t'.join(score_vali)), file=f_sd)

    with open(result_path + '/score_distance_all.txt', 'w') as f_sd:
        f_sd.write('{}\t{}\t{}\n'.format('score', 'mean_distance', 'std_deviation'))
        print(distance_all)
        for i in range(len(distance_all)):
            ds = distance_all[i]
            f_sd.write('{}\t{}\t{}\n'.format(i, np.mean(ds), np.std(ds) ))
        # distance_mean = distance_sum_all_table/distance_n_all_table
        # print(distance_sum_all_table)
        # print(distance_n_all_table)
        # for i in range(len(distance_n_all_table)):
        #     f_sd.write(str(i)+'\t')
        # f_sd.write('\n')
        # for d in distance_mean:
        #     f_sd.write('{:.3f}'.format(d)+'\t')



if __name__ == '__main__':
    pass
    # run_procerpron_learning()
    # read_training_data("/features_bow_1gram/")
    # generate_features_bow([1,], True, False)

    # training w2v
    # generate_features_sent2vec(f_w2v, nlp, lemmatizer)
