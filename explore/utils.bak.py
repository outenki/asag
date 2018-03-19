from nltk.corpus import wordnet as wn
from nltk.corpus import wordnet_ic as wic
from sklearn.metrics import mean_squared_error
from sklearn.cluster import KMeans

import collections

from math import sqrt
import re
import sys
import numpy as np
np.set_printoptions(threshold=np.nan)
from munkres import Munkres
from nltk.metrics import *
# from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
import string
import os
# import progressbar
import time
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import neighbors
from nltk.stem.porter import PorterStemmer

from sklearn.svm import SVR

import progressbar

LEVENSHTEIN = 3

# Paths
SCRIPT_PATH = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = SCRIPT_PATH + "/../data/ShortAnswerGrading_v2.0/data"
# DATA_PATH = SCRIPT_PATH + "/../data/sciEntsBank/train"
# DATA_PATH = SCRIPT_PATH + "/../data/kaggle_train"
# DATA_PATH = SCRIPT_PATH + "/../data/sciEntsBank/test-unseen-questions"
# DATA_PATH = SCRIPT_PATH + "/../data/sciEntsBank/test-unseen-answers"
# DATA_PATH = SCRIPT_PATH + "/../data/sciEntsBank/test-unseen-domains"
RESULTS_PATH = SCRIPT_PATH + "/../results_sag"
# RESULTS_PATH = SCRIPT_PATH + "/../results_semi_train"
# RESULTS_PATH = SCRIPT_PATH + "/../results_kaggle_train"
# RESULTS_PATH = SCRIPT_PATH + "/../results_semi_uq"
# RESULTS_PATH = SCRIPT_PATH + "/../results_semi_ua"
# RESULTS_PATH = SCRIPT_PATH + "/../results_semi_ud"

RAW_PATH = DATA_PATH + "/raw"
RAW_PATH_STU = DATA_PATH + "/raw/ans_stu"


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


# def similarity_between_nodes(fun, node_stu, ans_stu, node_ins, ans_ins, ic=None):
#     """
#     Calculate one similarity between two nodes (not subgraph) and save it to instructor answer to avoid
#     repeat calculation.
#     fun: Similarity function from wordnet
#     node_stu, node_ins: Nodes of dependence graph of answers from student and instructor
#     ans_stu, ans_ins: Sentence objects of answers from student and instructor
#     """
#     if ans_stu not in ans_ins.similarity:
#         ans_ins.similarity[ans_stu] = {}
#
#     if fun.__name__ in ans_ins.similarity[ans_stu]:
#         return ans_ins.similarity[ans_stu][fun.__name__]
#
#     word_stu, word_ins = node_ins.split(":")[0], node_stu.split(":")[0]
#     if fun.__name__ == 'lch_similarity':
#         sims = [fun(s1, s2) if s1.pos() == s2.pos() else 0 for s1 in wn.synsets(word_stu) for s2 in
#                 wn.synsets(word_ins)]
#     elif fun.__name__ in {'res_similarity', 'lin_similarity', 'jcn_similarity'}:
#         sims = [fun(s1, s2, ic) if s1.pos() == s2.pos() else 0 for s1 in wn.synsets(word_stu) for s2 in
#                 wn.synsets(word_ins)]
#     else:
#         sims = [fun(s1, s2) for s1 in wn.synsets(word_stu) for s2 in wn.synsets(word_ins)]
#     sims = list(filter(lambda x: x, sims))
#     if not sims:
#         # print('WARNING: The similarity of "{0}" between [{1}] and [{2}] is 0!'.format(fun.__name__, word1, word2))
#         sim = 0
#         pass
#     else:
#         sim = max(sims)
#     return sim


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
    pbar = progressbar.ProgressBar(max_value=epochs)
    for _i in range(epochs):
        pbar.update(_i)
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


def stem_tokens(tokens, stemmer):
    stemmed = set()
    for item in tokens:
        stemmed.add(stemmer.stem(item))
    return stemmed


def get_tokens(text, gram_n, char_gram, stemmer):
    lower = text.lower()
    remove_punctuation_map = dict((ord(char), None) for char in string.punctuation)
    no_punctuation = lower.translate(remove_punctuation_map)
    tokens = list(no_punctuation) if char_gram else nltk.word_tokenize(no_punctuation)
    return nltk.ngrams(stem_tokens(tokens, stemmer), gram_n)


def read_tokens_answer(answer, gram_n, char_gram, stemmer):
    # Answers are starts with answer id
    # Remove answer id first before extract tokens
    answer = answer[answer.find(' ') + 1:]
    return sorted(get_tokens(answer, gram_n=gram_n, char_gram=char_gram, stemmer=stemmer))


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
    :return:
        An sorted list of token set.
    '''
    stemmer = PorterStemmer()
    token_set = set()
    if ref:
        # read reference answer
        with open(RAW_PATH + "/answers", errors="ignore") as f_ref:
            for answer in f_ref.readlines():
                if answer.startswith(que_id):
                    token_set = token_set.union(read_tokens_answer(answer, gram_n=gram_n, char_gram=char_gram,
                                                                   stemmer=stemmer))
                    break

    # read student answers
    with open(RAW_PATH_STU + "/" + que_id, "r", errors="ignore") as f_ans_raw:
        try:
            for answer in f_ans_raw.readlines():
                token_set = token_set.union(read_tokens_answer(answer, gram_n=gram_n, char_gram=char_gram,
                                                               stemmer=stemmer))
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
    stemmer = PorterStemmer()
    for que_id in sorted(os.listdir(RAW_PATH_STU)):
        print("\n" + que_id)

        # generate bow features
        if char_gram:
            feature_path = RESULTS_PATH + "/features_bow_{}gram_char/".format("-".join(map(str, grams_n_list)))
        else:
            feature_path = RESULTS_PATH + "/features_bow_{}gram/".format("-".join(map(str, grams_n_list)))
        if not os.path.exists(feature_path):
            os.makedirs(feature_path)

        tokens_que = {}

        # Read n-gram set from answers of question with id of que_id.
        with open(feature_path + "/bow_{}".format(que_id), "wt", encoding='utf-8',
                  errors="ignore") as f_bow:
            for gram in grams_n_list:
                tokens_que[gram] = tuple(read_tokens_answers(que_id, gram_n=gram, ref=ref, char_gram=char_gram))
                f_bow.write("\t".join(map(','.join, tokens_que[gram])) + "\t")

        with open(feature_path + "/" + que_id, "wt", encoding='utf-8', errors="ignore") as f_fea, \
                open(RAW_PATH_STU + "/" + que_id, "r", encoding='utf-8', errors="ignore") as f_ans:
            f_ans_lines = f_ans.readlines()
            bar = progressbar.ProgressBar(max_value=len(f_ans_lines))
            bar_i = 0
            for answer in f_ans_lines:
                features = []
                for gram in grams_n_list:
                    # Read n-gram sef from an answer and generate bow feature based on tokens_que for it.
                    tokens_answer = set(read_tokens_answer(answer, gram_n=gram, char_gram=char_gram,
                                                           stemmer=stemmer))
                    bow = [1] * len(tokens_que[gram])
                    for i in range(len(tokens_que[gram])):
                        bow[i] = 1 if tokens_que[gram][i] in tokens_answer else 0
                    features.extend(bow)

                print(*features, file=f_fea, sep=',')
                bar.update(bar_i)
                bar_i += 1
                # print(bow)


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


def _read_training_data(feature_path, raw_path=RAW_PATH, score_path=DATA_PATH + '/scores/', include_ref=False):
    id_que = os.listdir(feature_path)
    record = list()
    for i in id_que:
        with open(feature_path + i, 'r') as ff, \
                open(score_path + '/' + i + '/ave') as fs, \
                open(raw_path + "/answers", "r", errors="ignore") as f_raw_r, \
                open(raw_path + "/questions", "r", errors="ignore") as f_raw_q, \
                open(raw_path + "/ans_stu/" + i, "r", errors="ignore") as f_raw_s, \
                open(score_path + "/" + i + '/diff') as fd:
            scores_truth = np.array(list(map(np.float64, fs.readlines())))
            diff = np.array(list(map(np.float64, fd.readlines())))
            features = list(map(lambda s: s.split(','), ff.readlines()))
            features = np.array(list(map(lambda l: list(map(np.float64, l)), features)))
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
    TrainingData = collections.namedtuple('TrainingData', 'id id_que que id_stu stu ref feature score diff')
    ret = TrainingData(list(range(len(record))), *list(zip(*record)))
    print(ret.id, ret.id_que, ret.stu)
    return ret


def read_training_data(feature_path):
    '''
    Read features and labels for training. This function will read all the features
    and scores of each answer for each question.
    :param feature_path: path/of/feature/files/.
    :return: A dict with structure as below
    # data_dic = {
    #   '1.1':{
    #       'truth': array(n*1)
    #       'features': array(n*30)
    #       'diff': array(n*30)
    #   }
    # }
    '''
    scores_truth_path = DATA_PATH + '/scores/'
    que_ids = os.listdir(feature_path)
    data_dict = {}
    for que_id in que_ids:
        data_dict[que_id] = {}
        with open(feature_path + que_id, 'r') as ff, \
                open(scores_truth_path + que_id + '/ave') as fs, \
                open(RAW_PATH + "/answers", "r", errors="ignore") as f_raw_r, \
                open(RAW_PATH + "/questions", "r", errors="ignore") as f_raw_q, \
                open(RAW_PATH_STU + "/" + que_id, "r", errors="ignore") as f_raw_s, \
                open(scores_truth_path + que_id + '/diff') as fd:
            scores_truth = np.array(list(map(np.float64, fs.readlines())))
            diff = np.array(list(map(np.float64, fd.readlines())))
            features = list(map(lambda s: s.split(','), ff.readlines()))
            features = np.array(list(map(lambda l: list(map(np.float64, l)), features)))
            raw_r, raw_q, raw_s = '', '', []

            for s in f_raw_q.readlines():
                if s.startswith(que_id):
                    raw_q = s
                    break

            for s in f_raw_r.readlines():
                if s.startswith(que_id):
                    raw_r = s
                    break

            raw_s = np.array(list(map(lambda s: s.strip(), f_raw_s.readlines())))

            data_dict[que_id]['scores_truth'] = scores_truth
            data_dict[que_id]['features'] = features
            data_dict[que_id]['diff'] = diff
            data_dict[que_id]['question'] = raw_q.strip()
            data_dict[que_id]['ans_ref'] = raw_r.strip()
            data_dict[que_id]['ans_stu'] = raw_s
    return data_dict


def run_svr(fn, feature_type, reliable, training_scale=0):
    # When `reliable` is True, answers whose score is with diff over 2 will be removed
    # from training data
    feature_path = RESULTS_PATH + '/features_{}/'.format(feature_type)
    data_dict = read_training_data(feature_path)
    fn = '{}.{}.{}.{}.{}'.format(feature_type, fn, 'reliable' if reliable else 'unreliable', training_scale, cur_time())
    result_path = RESULTS_PATH + '/results/' + fn
    if not os.path.exists(result_path):
        os.mkdir(result_path)

    with open(result_path + '/result.txt', 'w') as fr:
        for que_id in data_dict:
            for i in range(len(data_dict[que_id]['scores_truth'])):
                # i refers the answer to be scored
                # Train svr for each answer with all other answers
                scale = 0
                features_all = []
                scores_all = []
                for qid in data_dict:
                    array_filter = data_dict[qid]['diff'] < 3 if reliable else np.array(
                        [True] * len(data_dict[qid]['diff']))
                    if qid != que_id:
                        scores_truth = data_dict[qid]['scores_truth'][array_filter]
                        features = data_dict[qid]['features'][array_filter]
                    else:
                        array_filter[i] = False
                        scores_truth = data_dict[qid]['scores_truth'][array_filter]
                        features = data_dict[qid]['features'][array_filter]
                        # scores_truth = np.delete(data_dict[qid]['scores_truth'], i, 0)
                        # features = np.delete(data_dict[qid]['features'], i, 0)
                    features_all.append(np.array(features))
                    scores_all.append(np.array(scores_truth))
                    scale += len(features)
                    if scale >= training_scale > 0:
                        # print('scale: ', scale)
                        break
                X = np.concatenate(features_all)
                Y = np.concatenate(scores_all)
                score_truth_i = data_dict[que_id]['scores_truth'][i]
                feature_i = data_dict[que_id]['features'][i:i + 1]

                clf = SVR()
                clf.fit(X, Y)

                # predict
                score = clf.predict(feature_i)
                error = score_truth_i - score[0]
                error_abs = abs(error)
                error_round = round(error_abs)
                question = data_dict[que_id]["question"]
                ans_ref = data_dict[que_id]["ans_ref"]
                ans_stu = data_dict[que_id]["ans_stu"][i]
                print('score of {}.{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}'.format(que_id, i + 1, score[0], score_truth_i,
                                                                              error,
                                                                              error_abs, error_round, question, ans_ref,
                                                                              ans_stu))
                print('score of {}.{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}'.format(que_id, i + 1, score[0], score_truth_i,
                                                                              error,
                                                                              error_abs, error_round, question, ans_ref,
                                                                              ans_stu), file=fr)


def run_svr_question_wise(fn, feature_type, reliable, fn_bow=None, training_scale=0, kernel='rbf'):
    '''
    Train SVR model for each answer with all the other answers under the same question.
    When `reliable` is True, answers whose score is with diff over 2 will be removed
    from training data
    '''

    feature_path = RESULTS_PATH + '/features_{}/'.format(feature_type)
    if fn_bow:
        bow_path = RESULTS_PATH + '/' + fn_bow
        assert os.path.exists(bow_path)
    data_dict = read_training_data(feature_path)
    fn = '{}.{}.{}.{}.{}'.format(feature_type, fn, 'reliable' if reliable else 'unreliable', training_scale, cur_time())
    result_path = RESULTS_PATH + '/results/' + fn
    if not os.path.exists(result_path):
        os.mkdir(result_path)

    with open(result_path + '/result.txt', 'w') as fr, open(result_path + '/weights.txt', 'w') as fw:
        for que_id in data_dict:
            for i in range(len(data_dict[que_id]['scores_truth'])):
                # i refers an answer
                # Train svr for each answer with all other answers under the same question

                # remove unreliable training data
                scale = 0

                array_filter = data_dict[que_id]['diff'] < 3 if reliable else np.array(
                    [True] * len(data_dict[que_id]['diff']))
                # remove current answer (to be predicted)
                array_filter[i] = False

                scores_truth = data_dict[que_id]['scores_truth'][array_filter]
                features = data_dict[que_id]['features'][array_filter]
                X = features[:training_scale] if training_scale > 0 else features
                Y = scores_truth[:training_scale] if training_scale > 0 else scores_truth
                score_truth_i = data_dict[que_id]['scores_truth'][i]
                feature_i = data_dict[que_id]['features'][i:i + 1]
                clf = SVR(kernel=kernel)
                clf.fit(X, Y)
                if kernel == 'linear' and fn_bow:
                    weights = (clf.coef_)
                    with open(bow_path + '/bow_' + que_id, 'r') as f_bow:
                        bow = f_bow.readline().strip().split('\t')
                    word_weight = sorted(zip(bow, weights[0]), key=lambda x: x[1], reverse=True)
                    # print(word_weight)
                    print('{}.{}\t{}'.format(que_id, i + 1, word_weight), file=fw)
                # predict
                score = clf.predict(feature_i)
                error = score_truth_i - score[0]
                error_abs = abs(error)
                error_round = round(error_abs)
                question = data_dict[que_id]["question"]
                ans_ref = data_dict[que_id]["ans_ref"]
                ans_stu = data_dict[que_id]["ans_stu"][i]
                print('score of {}.{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}'.format(que_id, i + 1, score[0], score_truth_i,
                                                                              error,
                                                                              error_abs, error_round, question, ans_ref,
                                                                              ans_stu))
                print('score of {}.{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}'.format(que_id, i + 1, score[0], score_truth_i,
                                                                              error,
                                                                              error_abs, error_round, question, ans_ref,
                                                                              ans_stu), file=fr)


def run_knn(fn, feature_type, reliable, n_neighbors, weight, p=2, training_scale=0):
    '''
    Run knn algorithm using all other answers as training data.
    :param fn: File name to save the results.
    :param feature_type: For now it may be one of 'g', 'b' or 'gb'.
    :param reliable:
        When `reliable` is True, answers whose score is with diff over 2 will
        be removed from training data
    :param n_neighbors: Parameter for KNN. The number neighbors.
    :param weight:
        Weight function used in prediction. Possible values:
        ‘uniform’ : uniform weights. All points in each neighborhood are weighted equally.
        ‘distance’ : weight points by the inverse of their distance. in this case,
            closer neighbors of a query point will have a greater influence than neighbors
            which are further away.
        [callable] : a user-defined function which accepts an array of distances,
            and returns an array of the same shape containing the weights.
    :return: None
    '''

    feature_path = RESULTS_PATH + '/features_{}/'.format(feature_type)
    data_dict = read_training_data(feature_path)
    # fn = fn +  '.' + feature_type + '.' +  cur_time()
    fn = '{}.{}.{}.{}.{}.{}.{}.{}'.format(feature_type, fn, n_neighbors, p,
                                          'reliable' if reliable else 'unreliable', weight, training_scale, cur_time())
    result_path = RESULTS_PATH + '/results/' + fn
    if not os.path.exists(result_path):
        os.mkdir(result_path)

    with open(result_path + '/result.txt', 'w') as fr:
        for que_id in data_dict:
            for i in range(len(data_dict[que_id]['scores_truth'])):
                # i refers an student answer
                # Train knn for each answer with all other answers
                scale = 0
                features_all = []
                scores_all = []
                for qid in data_dict:
                    array_filter = data_dict[qid]['diff'] < 3 if reliable else np.array(
                        [True] * len(data_dict[qid]['diff']))
                    if qid != que_id:
                        scores_truth = data_dict[qid]['scores_truth'][array_filter]
                        features = data_dict[qid]['features'][array_filter]
                    else:
                        array_filter[i] = False
                        scores_truth = data_dict[qid]['scores_truth'][array_filter]
                        features = data_dict[qid]['features'][array_filter]
                        # scores_truth = np.delete(data_dict[qid]['scores_truth'], i, 0)
                        # features = np.delete(data_dict[qid]['features'], i, 0)
                    features_all.append(np.array(features))
                    scores_all.append(np.array(scores_truth))
                    scale += len(features)
                    if scale >= training_scale > 0:
                        print('scale: ', scale)
                        break
                X = np.concatenate(features_all)
                Y = np.concatenate(scores_all)
                Y = (Y * 2).astype(int)  # Here Y need to be int as a labels, that's why *2 is needed.
                # Y comes from average scores of two ground truth, so there're .5 scores.
                score_truth_i = data_dict[que_id]['scores_truth'][i]
                feature_i = data_dict[que_id]['features'][i:i + 1]
                if n_neighbors > len(X):
                    n_neighbors = len(X)
                clf = neighbors.KNeighborsClassifier(n_neighbors, weights=weight, p=p)
                clf.fit(X, Y)

                # predict
                score = clf.predict(feature_i) / 2
                error = score_truth_i - score[0]
                error_abs = abs(error)
                error_round = round(error_abs)
                question = data_dict[que_id]["question"]
                ans_ref = data_dict[que_id]["ans_ref"]
                ans_stu = data_dict[que_id]["ans_stu"][i]
                print('score of {}.{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}'.format(que_id, i + 1, score[0], score_truth_i,
                                                                              error,
                                                                              error_abs, error_round, question, ans_ref,
                                                                              ans_stu))
                print('score of {}.{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}'.format(que_id, i + 1, score[0], score_truth_i,
                                                                              error,
                                                                              error_abs, error_round, question, ans_ref,
                                                                              ans_stu), file=fr)


def run_kmeans(fn, feature_type, reliable, k, training_scale=0):
    '''
    Run knn algorithm using all other answers as training data.
    :param fn: File name to save the results.
    :param feature_type: For now it may be one of 'g', 'b' or 'gb'.
    :param reliable:
        When `reliable` is True, answers whose score is with diff over 2 will
        be removed from training data
    :param n_neighbors: Parameter for KNN. The number neighbors.
    :param weight:
        Weight function used in prediction. Possible values:
        ‘uniform’ : uniform weights. All points in each neighborhood are weighted equally.
        ‘distance’ : weight points by the inverse of their distance. in this case,
            closer neighbors of a query point will have a greater influence than neighbors
            which are further away.
        [callable] : a user-defined function which accepts an array of distances,
            and returns an array of the same shape containing the weights.
    :return: None
    '''

    feature_path = RESULTS_PATH + '/features_{}/'.format(feature_type)
    data_dict = read_training_data(feature_path)
    # fn = fn +  '.' + feature_type + '.' +  cur_time()
    fn = '{}.{}.{}.{}.{}.{}'.format(feature_type, fn, k, 'reliable' if reliable else 'unreliable', training_scale,
                                    cur_time())
    result_path = RESULTS_PATH + '/results/' + fn
    if not os.path.exists(result_path):
        os.mkdir(result_path)

    with open(result_path + '/result.txt', 'w') as fr:
        for que_id in data_dict:
            for i in range(len(data_dict[que_id]['scores_truth'])):
                # i refers an student answer
                # Train kmeans for each answer with all other answers
                scale = 0
                features_all = []
                scores_all = []
                for qid in data_dict:
                    array_filter = data_dict[qid]['diff'] < 3 if reliable else np.array(
                        [True] * len(data_dict[qid]['diff']))
                    if qid != que_id:
                        scores_truth = data_dict[qid]['scores_truth'][array_filter]
                        features = data_dict[qid]['features'][array_filter]
                    else:
                        array_filter[i] = False
                        scores_truth = data_dict[qid]['scores_truth'][array_filter]
                        features = data_dict[qid]['features'][array_filter]
                        # scores_truth = np.delete(data_dict[qid]['scores_truth'], i, 0)
                        # features = np.delete(data_dict[qid]['features'], i, 0)
                    features_all.append(np.array(features))
                    scores_all.append(np.array(scores_truth))
                    scale += len(features)
                    if scale >= training_scale > 0:
                        #     print('scale: ', scale)
                        break
                X = np.concatenate(features_all)
                Y = np.concatenate(scores_all)
                Y = (Y * 2).astype(int)

                score_truth_i = data_dict[que_id]['scores_truth'][i]
                feature_i = data_dict[que_id]['features'][i:i + 1]

                kmeans = KMeans(n_clusters=k, random_state=0).fit(X)
                label_i = kmeans.predict(feature_i)

                # connect labels to scores
                scores = Y[kmeans.labels_ == label_i[0]]
                # print('socres:', scores)
                # print('label_i', label)
                count = len(scores)
                score = sum(scores) / count / 2

                # dict_score_label, dict_label_score = {}, {}
                # for label in range(0, k):
                #     scores = Y[kmeans.labels_==label]
                #     count = len(scores)
                #     dict_score_label[]
                #     pass


                # predict
                # score = clf.predict(feature_i) / 2
                error = score_truth_i - score
                error_abs = abs(error)
                error_round = round(error_abs)
                question = data_dict[que_id]["question"]
                ans_ref = data_dict[que_id]["ans_ref"]
                ans_stu = data_dict[que_id]["ans_stu"][i]
                print('score of {}.{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}'.format(que_id, i + 1, score[0], score_truth_i,
                                                                              error,
                                                                              error_abs, error_round, question, ans_ref,
                                                                              ans_stu))
                print('score of {}.{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}'.format(que_id, i + 1, score[0], score_truth_i,
                                                                              error,
                                                                              error_abs, error_round, question, ans_ref,
                                                                              ans_stu), file=fr)


def run_kmeans_question_wise(fn, feature_type, reliable, k, training_scale=0):
    '''
    Run knn algorithm using all other answers as training data.
    :param fn: File name to save the results.
    :param feature_type: For now it may be one of 'g', 'b' or 'gb'.
    :param reliable:
        When `reliable` is True, answers whose score is with diff over 2 will
        be removed from training data
    :param n_neighbors: Parameter for KNN. The number neighbors.
    :param weight:
        Weight function used in prediction. Possible values:
        ‘uniform’ : uniform weights. All points in each neighborhood are weighted equally.
        ‘distance’ : weight points by the inverse of their distance. in this case,
            closer neighbors of a query point will have a greater influence than neighbors
            which are further away.
        [callable] : a user-defined function which accepts an array of distances,
            and returns an array of the same shape containing the weights.
    :return: None
    '''

    feature_path = RESULTS_PATH + '/features_{}/'.format(feature_type)
    data_dict = read_training_data(feature_path)
    # fn = fn +  '.' + feature_type + '.' +  cur_time()
    fn = '{}.{}.{}.{}.{}.{}'.format(feature_type, fn, k, 'reliable' if reliable else 'unreliable', training_scale,
                                    cur_time())
    result_path = RESULTS_PATH + '/results/' + fn
    if not os.path.exists(result_path):
        os.mkdir(result_path)

    with open(result_path + '/result.txt', 'w') as fr:
        for que_id in data_dict:
            for i in range(len(data_dict[que_id]['scores_truth'])):
                # i refers an answer
                # Train knn for each answer with all other answers

                # remove unreliable training data
                array_filter = data_dict[que_id]['diff'] < 3 if reliable else np.array(
                    [True] * len(data_dict[que_id]['diff']))
                # remove current answer (to be predicted)
                array_filter[i] = False

                scores_truth = data_dict[que_id]['scores_truth'][array_filter]
                features = data_dict[que_id]['features'][array_filter]

                X = features[:training_scale] if training_scale > 0 else features
                Y = scores_truth[:training_scale] if training_scale > 0 else scores_truth
                # Y = (Y * 2).astype(int)

                score_truth_i = data_dict[que_id]['scores_truth'][i]
                feature_i = data_dict[que_id]['features'][i:i + 1]

                kmeans = KMeans(n_clusters=k, random_state=0).fit(X)
                label_i = kmeans.predict(feature_i)

                # connect labels to scores
                scores = Y[kmeans.labels_ == label_i[0]]
                # print('socres:', scores)
                # print('label_i', label)
                count = len(scores)
                score = sum(scores) / count

                # dict_score_label, dict_label_score = {}, {}
                # for label in range(0, k):
                #     scores = Y[kmeans.labels_==label]
                #     count = len(scores)
                #     dict_score_label[]
                #     pass


                # predict
                # score = clf.predict(feature_i) / 2
                error = score_truth_i - score
                error_abs = abs(error)
                error_round = round(error_abs)
                question = data_dict[que_id]["question"]
                ans_ref = data_dict[que_id]["ans_ref"]
                ans_stu = data_dict[que_id]["ans_stu"][i]
                print('score of {}.{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}'.format(que_id, i + 1, score, score_truth_i,
                                                                              error,
                                                                              error_abs, error_round, question, ans_ref,
                                                                              ans_stu))
                print('score of {}.{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}'.format(que_id, i + 1, score, score_truth_i,
                                                                              error,
                                                                              error_abs, error_round, question, ans_ref,
                                                                              ans_stu), file=fr)


def run_knn_question_wise(fn, feature_type, reliable, n_neighbors, weight, p=2, training_scale=0, regression=False):
    '''
    Run knn algorithm using all other answers under the same question as training data.
    :param fn: File name to save the results.
    :param feature_type: For now it may be one of 'g', 'b' or 'gb'.
    :param reliable:
        When `reliable` is True, answers whose score is with diff over 2 will
        be removed from training data
    :param n_neighbors: Parameter for KNN. The number neighbors.
    :param weight:
        Weight function used in prediction. Possible values:
        ‘uniform’ : uniform weights. All points in each neighborhood are weighted equally.
        ‘distance’ : weight points by the inverse of their distance. in this case,
            closer neighbors of a query point will have a greater influence than neighbors
            which are further away.
        [callable] : a user-defined function which accepts an array of distances,
            and returns an array of the same shape containing the weights.
    :return: None
    '''
    feature_path = RESULTS_PATH + '/features_{}/'.format(feature_type)
    data_dict = read_training_data(feature_path)
    # fn = fn +  '.' + feature_type + '.' +  cur_time()
    fn = '{}.{}.{}.{}.{}.{}.{}.{}'.format(feature_type, fn, n_neighbors, p,
                                          'reliable' if reliable else 'unreliable', weight, training_scale, cur_time())
    result_path = RESULTS_PATH + '/results/' + fn
    if not os.path.exists(result_path):
        os.mkdir(result_path)

    with open(result_path + '/result.txt', 'w') as fr:
        for que_id in data_dict:
            for i in range(len(data_dict[que_id]['scores_truth'])):
                # i refers an answer
                # Train knn for each answer with all other answers

                # remove unreliable training data
                array_filter = data_dict[que_id]['diff'] < 3 if reliable else np.array(
                    [True] * len(data_dict[que_id]['diff']))
                # remove current answer (to be predicted)
                array_filter[i] = False

                scores_truth = data_dict[que_id]['scores_truth'][array_filter]
                features = data_dict[que_id]['features'][array_filter]
                no_of_answers = np.array(range(len(data_dict[que_id]['features'])))[array_filter]

                X = features[:training_scale] if training_scale > 0 else features
                Y = scores_truth[:training_scale] if training_scale > 0 else scores_truth
                Y = (Y * 2).astype(int)
                score_truth_i = data_dict[que_id]['scores_truth'][i]
                feature_i = data_dict[que_id]['features'][i:i + 1]
                if n_neighbors > len(X):
                    n_neighbors = len(X)
                if regression:
                    clf = neighbors.KNeighborsRegressor(n_neighbors, weights=weight)
                else:
                    clf = neighbors.KNeighborsClassifier(n_neighbors, weights=weight)
                clf.fit(X, Y)

                # predict
                score = clf.predict(feature_i) / 2
                distance_of_neighbors, no_of_neighbors = clf.kneighbors(feature_i, n_neighbors)

                # Find the N.O. of nearest answers by features
                n_s = ['{}.{}'.format(que_id, no_of_answers[no] + 1) for no in no_of_neighbors[0]]
                t_s = [Y[no] / 2 for no in no_of_neighbors[0]]
                d_s = distance_of_neighbors[0]
                # print(n_s, t_s, d_s)
                # neighbor_truth_distance = zip(n_s, t_s, d_s)
                # print("ntd:", list(neighbor_truth_distance))
                # print()
                # with open(result_path + '/neighbor_{}.{}.txt'.format(que_id, i), 'w') as f_neighbor:
                #     for n, s, d in neighbor_truth_distance:
                #         print(n,s,d)
                #         print('{}\t{}'.format(n, s, d), file=f_neighbor)
                # try:
                #     score = clf.predict(feature_i) / 2
                # except :
                #     print("error id:{}.{}".format(que_id, i+1))
                #     print("error: k=", n_neighbors)
                error = score_truth_i - score[0]
                error_abs = abs(error)
                error_round = round(error_abs)
                question = data_dict[que_id]["question"]
                ans_ref = data_dict[que_id]["ans_ref"]
                ans_stu = data_dict[que_id]["ans_stu"][i]
                print('score of {}.{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}'.format(que_id, i + 1, score[0], score_truth_i,
                                                                              error,
                                                                              error_abs, error_round, question, ans_ref,
                                                                              ans_stu))
                print('score of {}.{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}'.format(que_id, i + 1, score[0],
                                                                                          score_truth_i,
                                                                                          error,
                                                                                          error_abs, error_round,
                                                                                          question, ans_ref,
                                                                                          ans_stu, n_s, t_s, d_s),
                      file=fr)

                with open(result_path + '/features.txt', 'a') as f_features:
                    print('X of {}.{}:'.format(que_id, i+1),  X, file=f_features)

def score_question_wise(fn, clf, feature_type, reliable=True):
    '''
    Train SVR model for each answer with all the other answers under the same question.
    When `reliable` is True, answers whose score is with diff over 2 will be removed
    from training data
    '''

    feature_path = RESULTS_PATH + '/features_{}/'.format(feature_type)
    data_dict = read_training_data(feature_path)
    fn = RESULTS_PATH + '/' + fn + '.' + cur_time()
    print(fn)
    with open(fn, 'w') as fr:
        for que_id in data_dict:
            for i in range(len(data_dict[que_id]['scores_truth'])):
                # i refers an answer
                # Train knn for each answer with all other answers

                # remove unreliable training data
                array_filter = data_dict[que_id]['diff'] < 3 if reliable else np.array(
                    [True] * len(data_dict[que_id]['diff']))
                # remove current answer (to be predicted)
                array_filter[i] = False

                scores_truth = data_dict[que_id]['scores_truth'][array_filter]
                features = data_dict[que_id]['features'][array_filter]

                X = features
                Y = scores_truth
                Y = (Y * 2).astype(int)
                score_truth_i = data_dict[que_id]['scores_truth'][i]
                feature_i = data_dict[que_id]['features'][i:i + 1]
                clf.fit(X, Y)

                # predict
                score = clf.predict(feature_i) / 2
                error = score_truth_i - score[0]
                error_abs = abs(error)
                error_round = round(error_abs)
                print('score of {}.{}\t{}\t{}\t{}\t{}\t{}'.format(que_id, i + 1, score[0], score_truth_i, error,
                                                                  error_abs, error_round))
                print('score of {}.{}\t{}\t{}\t{}\t{}\t{}'.format(que_id, i + 1, score[0], score_truth_i, error,
                                                                  error_abs, error_round), file=fr)


def score(fn, clf, feature_type, reliable=True):
    # When `reliable` is True, answers whose score is with diff over 2 will be removed
    # from training data

    feature_path = RESULTS_PATH + '/features_{}/'.format(feature_type)
    data_dict = read_training_data(feature_path)

    with open(fn, 'w') as fr:
        for que_id in data_dict:
            for i in range(len(data_dict[que_id]['scores_truth'])):
                # i refers an answer

                # Train svr for each answer with all other answers
                features_all = []
                scores_all = []
                for qid in data_dict:
                    array_filter = data_dict[qid]['diff'] < 3 if reliable else np.array(
                        [True] * len(data_dict[qid]['diff']))
                    if qid != que_id:
                        scores_truth = data_dict[qid]['scores_truth'][array_filter]
                        features = data_dict[qid]['features'][array_filter]
                    else:
                        array_filter[i] = False
                        scores_truth = data_dict[qid]['scores_truth'][array_filter]
                        features = data_dict[qid]['features'][array_filter]
                    features_all.append(np.array(features))
                    scores_all.append(np.array(scores_truth))
                X = np.concatenate(features_all)
                Y = np.concatenate(scores_all)
                Y = (Y * 2).astype(int)
                score_truth_i = data_dict[que_id]['scores_truth'][i]
                feature_i = data_dict[que_id]['features'][i:i + 1]
                clf.fit(X, Y)

                # predict
                score = clf.predict(feature_i) / 2
                error = score_truth_i - score[0]
                error_abs = abs(error)
                error_round = round(error_abs)
                print('score of {}.{}\t{}\t{}\t{}\t{}\t{}'.format(que_id, i + 1, score[0], score_truth_i, error,
                                                                  error_abs, error_round))
                print('score of {}.{}\t{}\t{}\t{}\t{}\t{}'.format(que_id, i + 1, score[0], score_truth_i, error,
                                                                  error_abs, error_round), file=fr)


def score_svr_question_wise(fn='svr.all', reliable=True):
    clf = SVR()
    score_question_wise(fn, clf, reliable)


def score_knn(fn='knn.all', feature_type='gb', reliable=True, n_neighbors=10, weight='distance'):
    clf = neighbors.KNeighborsClassifier(n_neighbors, weights=weight)
    score(fn, clf, feature_type, reliable)


def score_knn_question_wise(fn='knn.all', feature_type='gb', reliable=True, n_neighbors=10, weight='distance'):
    clf = neighbors.KNeighborsClassifier(n_neighbors, weights=weight)
    score_question_wise(fn, clf, feature_type, reliable)


def __score_answer(fn_prefix, reliable, feature, model, model_params, feature_path, output_path, qwise, training_scale):
    fn_params = ['{}_{}'.format(k, v) for k, v in model_params.items()]
    fn = '{}.{}.{}.{}.{}.{}'.format(fn_prefix, feature, 'reliable' if reliable else 'unreliable',
                                    'qwise' if reliable else 'unqwise', ".".join(fn_params), cur_time())

    result_path = output_path + '/results/' + fn
    if not os.path.exists(output_path):
        os.mkdir(result_path)

    # Initialize the model
    if 'knnc' == model:
        runner = neighbors.KNeighborsClassifier(**model_params)
    elif 'knnr' == model:
        runner = neighbors.KNeighborsRegressor(**model_params)
    elif 'svr' == model:
        runner = SVR(**model_params)

    # Read training data
    data_dict = read_training_data(feature_path)

    with open(result_path + '/result.txt', 'w') as fr:
        for que_id in data_dict:
            for i in range(len(data_dict[que_id]['scores_truth'])):
                # i refers an answer
                # Train knn for each answer with all other answers
                # remove unreliable training data
                array_filter = data_dict[que_id]['diff'] < 3 if reliable else np.array(
                    [True] * len(data_dict[que_id]['diff']))

                if qwise:
                    # remove current answer (to be predicted)
                    array_filter[i] = False
                    scores_truth = data_dict[que_id]['scores_truth'][array_filter]
                    features = data_dict[que_id]['features'][array_filter]
                    no_of_answers = np.array(range(len(data_dict[que_id]['features'])))[array_filter]

                else:
                    features_all = []
                    scores_all = []
                    for qid in data_dict:
                        array_filter = data_dict[qid]['diff'] < 3 if reliable else np.array(
                            [True] * len(data_dict[qid]['diff']))
                        if qid != que_id:
                            scores_truth = data_dict[qid]['scores_truth'][array_filter]
                            features = data_dict[qid]['features'][array_filter]
                        else:
                            array_filter[i] = False
                            scores_truth = data_dict[qid]['scores_truth'][array_filter]
                            features = data_dict[qid]['features'][array_filter]
                            # scores_truth = np.delete(data_dict[qid]['scores_truth'], i, 0)
                            # features = np.delete(data_dict[qid]['features'], i, 0)
                        features_all.append(np.array(features))
                        scores_all.append(np.array(scores_truth))
                        no_of_answers = np.array(range(len(data_dict[que_id]['features'])))[array_filter]

                X = features[:training_scale] if training_scale > 0 else features
                Y = scores_truth[:training_scale] if training_scale > 0 else scores_truth
                Y = (Y * 2).astype(int)
                score_truth_i = data_dict[que_id]['scores_truth'][i]
                feature_i = data_dict[que_id]['features'][i:i + 1]
                if n_neighbors > len(X):
                    n_neighbors = len(X)
                if regression:
                    clf = neighbors.KNeighborsRegressor(n_neighbors, weights=weight)
                else:
                    clf = neighbors.KNeighborsClassifier(n_neighbors, weights=weight)
                clf.fit(X, Y)

                # predict
                score = clf.predict(feature_i) / 2
                distance_of_neighbors, no_of_neighbors = clf.kneighbors(feature_i, n_neighbors)

                # Find the N.O. of nearest answers by features
                n_s = ['{}.{}'.format(que_id, no_of_answers[no] + 1) for no in no_of_neighbors[0]]
                t_s = [Y[no] / 2 for no in no_of_neighbors[0]]
                d_s = distance_of_neighbors[0]
                # print(n_s, t_s, d_s)
                # neighbor_truth_distance = zip(n_s, t_s, d_s)
                # print("ntd:", list(neighbor_truth_distance))
                # print()
                # with open(result_path + '/neighbor_{}.{}.txt'.format(que_id, i), 'w') as f_neighbor:
                #     for n, s, d in neighbor_truth_distance:
                #         print(n,s,d)
                #         print('{}\t{}'.format(n, s, d), file=f_neighbor)
                # try:
                #     score = clf.predict(feature_i) / 2
                # except :
                #     print("error id:{}.{}".format(que_id, i+1))
                #     print("error: k=", n_neighbors)
                error = score_truth_i - score[0]
                error_abs = abs(error)
                error_round = round(error_abs)
                question = data_dict[que_id]["question"]
                ans_ref = data_dict[que_id]["ans_ref"]
                ans_stu = data_dict[que_id]["ans_stu"][i]
                print('score of {}.{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}'.format(que_id, i + 1, score[0], score_truth_i,
                                                                              error,
                                                                              error_abs, error_round, question, ans_ref,
                                                                              ans_stu))
                print('score of {}.{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}'.format(que_id, i + 1, score[0],
                                                                                          score_truth_i,
                                                                                          error,
                                                                                          error_abs, error_round,
                                                                                          question, ans_ref,
                                                                                          ans_stu, n_s, t_s, d_s),
                      file=fr)

                # Train the model
                # Predict


def score_answer(fn_prefix, reliable, feature, model, model_params, feature_path, output_path, qwise, training_scale):
    fn_params = ['{}_{}'.format(k, v) for k, v in model_params.items()]
    fn = '{}.{}.{}.{}.{}.{}'.format(fn_prefix, feature, 'reliable' if reliable else 'unreliable',
                                    'qwise' if reliable else 'unqwise', ".".join(fn_params), cur_time())

    result_path = output_path + '/results/' + fn
    if not os.path.exists(output_path):
        os.mkdir(result_path)

    # Initialize the model
    if 'knnc' == model:
        runner = neighbors.KNeighborsClassifier(**model_params)
    elif 'knnr' == model:
        runner = neighbors.KNeighborsRegressor(**model_params)
    elif 'svr' == model:
        runner = SVR(**model_params)

    # Read training data
    training_data = read_training_data(feature_path)

    n_data = len(training_data.id)
    with open(result_path + '/result.txt', 'w') as fr:
        for i in training_data.id:
            filter = list()
            if qwise:
                filter_qwise = training_data.id_que == training_data.id_que[i]
                filter.append(filter_qwise)
            if reliable:
                filter.append(training_data.diff < 3)
            filter_rm = [True] * n_data
            filter_rm[i] = False
            filter.append(filter_rm)

            filter = np.array(list(map(lambda f: reduce(lambda x, y: x and y, f), zip(*filter))))

            scores_truth = training_data.score[filter]
            features = training_data.feature[filter]
            no_of_answers = training_data.id_stu[filter]

            X = features[:training_scale] if training_scale > 0 else features
            Y = scores_truth[:training_scale] if training_scale > 0 else scores_truth
            Y = (Y * 2).astype(int)

            score_truth_i = training_data.score[i]
            feature_i = training_data.feature[i]

            # predict
            score = runner.predict(feature_i) / 2

            error = score_truth_i - score[0]
            error_abs = abs(error)
            error_round = round(error_abs)
            question = training_data.que
            ans_ref = training_data.ref
            ans_stu = training_data.stu
            que_id = training_data.id_que[i]

            if 'knnc' == model or 'knnr' == model:
                distance_of_neighbors, no_of_neighbors = runner.kneighbors(feature_i, model_params['n_features'])

                # Find the N.O. of nearest answers by features
                n_s = ['{}.{}'.format(training_data.id_que[i], no_of_answers[no] + 1) for no in no_of_neighbors[0]]
                t_s = [Y[no] / 2 for no in no_of_neighbors[0]]
                d_s = distance_of_neighbors[0]

                print('score of {}.{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}'.format(
                    que_id, i + 1,
                    score[0],
                    score_truth_i,
                    error,
                    error_abs,
                    error_round,
                    question,
                    ans_ref,
                    ans_stu))
                print('score of {}.{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}'.format(
                    que_id, i + 1, score[0],
                    score_truth_i,
                    error,
                    error_abs, error_round,
                    question, ans_ref,
                    ans_stu, n_s, t_s, d_s),
                    file=fr)

                # Train the model
                # Predict


def count_error(fn):
    # with open(fn, 'r') as fe, open('error.count.txt', 'w') as ec,\
    #         open('error_abs.count.txt', 'w') as eac,\
    #         open('error_round.count.txt', 'w') as erc:
    result_path = RESULTS_PATH + '/results/' + fn

    with open(result_path + '/result.txt', 'r') as fe, \
            open(result_path + '/errors', 'w') as fo:
        svr_all = map(lambda line: line.split(':'), fe.readlines())
        _, score, truth, error, error_abs, error_round = zip(*svr_all)
        count = len(error)
        score = map(float, score)
        truth = map(float, truth)
        error = map(float, error)
        error_abs = map(float, error_abs)
        error_round = map(float, error_round)
        rms = sqrt(mean_squared_error(list(score), list(truth)))

        def count_hist(error_hist, echo=False):
            k = list(np.arange(-4.5, 5.1, 0.5))
            v = [0] * 20
            hist = dict(zip(k, v))
            for e in error_hist:
                for k in hist:
                    if e <= k:
                        hist[k] += 1
                        break
            return hist
            # for k,v in hist.items():
            #     f.write("{}\t{}\n".format(k, v))
            # f.write('{}\t{}\n'.format(count, sum(hist.values())))

        d_error = count_hist(error)
        d_error_abs = count_hist(error_abs)
        d_error_round = count_hist(error_round)
        errors = zip(d_error.keys(), d_error.values(), d_error_abs.values(), d_error_round.values())
        errors = map(lambda line: '\t'.join(map(str, line)) + '\n', errors)
        fo.writelines(errors)
        fo.write('{}\t{}\t{}\t{}\n'.format(count, sum(d_error.values()), sum(d_error_abs.values()),
                                           sum(d_error_round.values())))
        fo.write('RMSE:' + str(rms) + '\n')


def remove_scores():
    cur_path = sys.path[0]
    scores_path = DATA_PATH + "/scores/"
    scores = os.listdir(scores_path)
    for score_path in scores:
        me = scores_path + score_path + "/me"
        other = scores_path + score_path + "/other"
        diff = scores_path + score_path + '/diff'
        print('me: ', me)
        print('other: ', other)
        with open(me, 'r') as fm, open(other, 'r') as fo, open(diff, 'w') as fd:
            score_me = np.array(list(map(float, fm.readlines())))
            score_other = np.array(list(map(float, fo.readlines())))
            fd.writelines('\n'.join(list(map(str, abs(score_me - score_other)))))


if __name__ == '__main__':
    # run_procerpron_learning()
    # read_training_data(RESULTS_PATH + "/features_bow_1gram/")
    # run_gen_features()
    # remove_scores()

    # print("\ngenerating feature: bow_1gram")
    # generate_features_bow(grams_n_list=[1], ref=True, char_gram=False)
    # generate_features_bow(grams_n_list=[2], ref=True, char_gram=False)

    # print("runing svr...")
    # run_svr_question_wise("svr_linear_qwise_stem", 'bow_1gram', True, fn_bow='bow_1gram', training_scale=0, kernel='linear')
    # run_svr_question_wise("svr_linear_qwise_stem", 'bow_2gram', True, fn_bow='bow_2gram', training_scale=0, kernel='linear')
    # run_svr_question_wise("svr_rbf_qwise_stem", 'bow_1gram', True, fn_bow='bow_1gram', training_scale=0, kernel='rbf')
    # run_svr_question_wise("svr_rbf_qwise_stem", 'bow_2gram', True, fn_bow='bow_2gram', training_scale=0, kernel='rbf')
    # for k in [5, 10, 20, 30]:
    #     print("runing knn with k of ",k)
    #     run_knn_question_wise("knn_qwise", feature_type="bow_2gram_char", reliable=False, n_neighbors=k, weight="uniform")
    #     run_knn_question_wise("knn_qwise", feature_type="bow_2gram_char", reliable=False, n_neighbors=k, weight="distance")

    # print("runing svr...")
    # run_svr_question_wise("svr_qwise", 'bow_3gram_char', True, 0)
    # for k in [5, 10, 20, 30]:
    #     print("runing knn with k of ",k)
    #     run_knn_question_wise("knn_qwise", feature_type="bow_3gram_char", reliable=False, n_neighbors=k, weight="uniform")
    #     run_knn_question_wise("knn_qwise", feature_type="bow_3gram_char", reliable=False, n_neighbors=k, weight="distance")
    #
    # print("runing svr...")
    # run_svr_question_wise("svr_qwise", 'bow_4gram_char', True, 0)
    # for k in [5, 10, 20, 30]:
    #     print("runing knn with k of ",k)
    #     run_knn_question_wise("knn_qwise", feature_type="bow_4gram_char", reliable=False, n_neighbors=k, weight="uniform")
    #     run_knn_question_wise("knn_qwise", feature_type="bow_4gram_char", reliable=False, n_neighbors=k, weight="distance")
    #
    # print("runing svr...")
    # run_svr_question_wise("svr_qwise", 'bow_5gram_char', True, 0)
    # for k in [5, 10, 20, 30]:
    #     print("runing knn with k of ",k)
    #     run_knn_question_wise("knn_qwise", feature_type="bow_5gram_char", reliable=False, n_neighbors=k, weight="uniform")
    #     run_knn_question_wise("knn_qwise", feature_type="bow_5gram_char", reliable=False, n_neighbors=k, weight="distance")
    run_knn_question_wise("knnc_qwise", feature_type="bow_1gram", reliable=True, n_neighbors=5, weight="distance", training_scale=0)
    # run_knn_question_wise("knnc_qwise", feature_type="bow_2gram", reliable=True, n_neighbors=5, weight="distance")
    # run_knn_question_wise("knng_qwise", feature_type="bow_1gram", reliable=True, n_neighbors=5, weight="distance", regression=True)
    # run_knn_question_wise("knng_qwise", feature_type="bow_2gram", reliable=True, n_neighbors=5, weight="distance", regression=True)
