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

