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
    fn = '{}.{}.{}.{}.{}.{}'.format(fn_prefix, feature, 'reliable' if reliable else 'unreliable',
                                    'qwise' if reliable else 'unqwise', ".".join(fn_params), cur_time())

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

    # Read training data
    training_data = read_training_data(RESULTS_PATH + "/features_" + feature)

    n_data = len(training_data.id)
    with open(result_path + '/result.txt', 'w') as fr:
        for i in training_data.id:
            filter = list()
            if qwise:
                filter_qwise = np.array(training_data.id_que) == training_data.id_que[i]
                filter.append(filter_qwise)
            if reliable:
                filter.append(np.array(training_data.diff) < 3)
            filter_rm = [True] * n_data
            filter_rm[i] = False
            filter.append(filter_rm)

            filter = np.array(list(map(lambda f: reduce(lambda x, y: x and y, f), zip(*filter))))

            scores_truth = training_data.score[filter]
            features = training_data.feature[filter]
            no_of_answers = training_data.id_ans[filter]
            id_ques = training_data.id_que[filter]

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

            if 'knnc' == model or 'knnr' == model:
                distance_of_neighbors, no_of_neighbors = runner.kneighbors(np.array([feature_i]),
                                                                           model_params['n_neighbors'])

                # Find the N.O. of nearest answers by features
                n_s = ['{}.{}'.format(training_data.id_que[i], no_of_answers[no]) for no in no_of_neighbors[0]]
                t_s = [Y[no] / 2 for no in no_of_neighbors[0]]
                d_s = distance_of_neighbors[0]

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
                print('score of {}.{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}'.format(
                    que_id, ans_id, score[0],
                    score_truth_i,
                    error,
                    error_abs, error_round,
                    question, ans_ref,
                    ans_stu, n_s, t_s),
                    file=fr)
                # with open(result_path + '/features.txt', 'a') as f_features:
                #     print('X of {}.{}:'.format(que_id, ans_id),  X, file=f_features)
            elif 'svr' == model:
                print('score of {}.{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}'.format(
                    que_id, i + 1, score[0], score_truth_i,
                    error,
                    error_abs, error_round, question, ans_ref,
                    ans_stu))
                print('score of {}.{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}'.format(
                    que_id, i + 1, score[0], score_truth_i,
                    error,
                    error_abs, error_round, question, ans_ref, ans_stu),
                    file=fr)

            elif 'cos' == model:
                print('score of {}.{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}'.format(
                    que_id, i + 1, score[0], score_truth_i,
                    error,
                    error_abs, error_round, question, ans_ref,
                    ans_stu))
                print('score of {}.{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}'.format(
                    que_id, i + 1, score[0], score_truth_i,
                    error,
                    error_abs, error_round, question, ans_ref, ans_stu),
                    file=fr)
