import data_utils
import numpy as np
from sklearn import cross_validation
from qwk import quadratic_weighted_kappa
import tensorflow as tf
from memn2n_kv import add_gradient_noise
import time
import os
import sys
import pandas as pd
import logging

# flags
tf.flags.DEFINE_float("epsilon", 0.1, "Epsilon value for Adam Optimizer.")
tf.flags.DEFINE_float("l2_lambda", 0.3, "Lambda for l2 loss.")
tf.flags.DEFINE_float("learning_rate", 0.002, "Learning rate")
tf.flags.DEFINE_float("max_grad_norm", 10.0, "Clip gradients to this norm.")
tf.flags.DEFINE_float("keep_prob", 0.8, "Keep probability for dropout")
tf.flags.DEFINE_integer("evaluation_interval", 1, "Evaluate and print results every x epochs")
tf.flags.DEFINE_integer("batch_size", 32, "Batch size for training.")
tf.flags.DEFINE_integer("feature_size", 100, "Feature size")
tf.flags.DEFINE_integer("num_samples", 1, "Number of samples selected from training for each score")
tf.flags.DEFINE_integer("hops", 3, "Number of hops in the Memory Network.")
tf.flags.DEFINE_integer("epochs", 100, "Number of epochs to train for.")
tf.flags.DEFINE_integer("embedding_size", 300, "Embedding size for embedding matrices.")
tf.flags.DEFINE_string("essay_set_id_file", 'prompts.txt', "file stroring the IDs of essay set, one id at each line")
tf.flags.DEFINE_integer("token_num", 6, "The number of token in glove (6, 42)")
tf.flags.DEFINE_boolean("gated_addressing", False, "Simple gated addressing")
tf.flags.DEFINE_boolean("allow_soft_placement", False, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")
tf.flags.DEFINE_string("output", '', "Log placement of ops on devices")
# hyper-parameters
FLAGS = tf.flags.FLAGS
# FLAGS._parse_flags()
FLAGS(sys.argv)

gated_addressing = FLAGS.gated_addressing
# essay_set_id = FLAGS.essay_set_id
batch_size = FLAGS.batch_size
embedding_size = FLAGS.embedding_size
feature_size = FLAGS.feature_size
l2_lambda = FLAGS.l2_lambda
hops = FLAGS.hops
reader = 'bow'
epochs = FLAGS.epochs
num_samples = FLAGS.num_samples
num_tokens = FLAGS.token_num
output = FLAGS.output

with open(FLAGS.essay_set_id_file, 'r') as f:
    essay_set_id_list = f.readlines()
    essay_set_id_list = np.array(essay_set_id_list, dtype=int)

test_batch_size = batch_size
random_state = 0
if gated_addressing:
    from memn2n_g_kv import MemN2N_KV
else:
    from memn2n_kv import MemN2N_KV
# print flags info
orig_stdout = sys.stdout

timestamp = time.strftime("%b_%d_%Y_%H_%M_%S", time.localtime())
if output == '':
    output = 'output_{}'.format(timestamp)
if not os.path.exists(output):
    os.makedirs(output)


# Setting logger
logger = logging.getLogger()
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

fh = logging.FileHandler('runs/{}/log.txt'.format(output))
fh.setLevel(logging.INFO)
fh.setFormatter(formatter)
ch = logging.StreamHandler()
ch.setFormatter(formatter)
ch.setLevel(logging.INFO)
logger.addHandler(fh)
logger.addHandler(ch)
# Setting logger done.


logger.info("Parameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    logger.info("\t{}={}".format(attr.upper(), value))

# print("\nParameters:")
# for attr, value in sorted(FLAGS.__flags.items()):
#     print("{}={}".format(attr.upper(), value))
# print("")

logger.info('Loading glove...')
# load glove
word_idx, word2vec = data_utils.load_glove(num_tokens, dim=embedding_size)

vocab_size = len(word_idx) + 1
# stat info on data set
logger.info('Glove loaded.')
for essay_set_id in essay_set_id_list:
    logger_in.info('\n------------------Runing on %d------------------' % essay_set_id)

    folder_name = '{}/essay_set_{}_nsample_{}'.format(output, essay_set_id, num_samples)
    out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", folder_name))
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # Setting logger
    logger_in = logging.getLogger()
    logger_in.setLevel(logging.INFO)

    fh_in = logging.FileHandler('{}/log.txt'.format(out_dir))
    fh_in.setLevel(logging.INFO)
    fh_in.setFormatter(formatter)
    ch_in = logging.StreamHandler()
    ch_in.setLevel(logging.INFO)
    ch_in.setFormatter(formatter)
    logger.addHandler(fh_in)
    logger.addHandler(ch_in)
    # Setting logger_in done.

    # save output to a file
    logger_in.info("Writing to {}\n".format(out_dir))

    with open(out_dir+'/params', 'w') as f:
        for attr, value in sorted(FLAGS.__flags.items()):
            f.write("{}={}".format(attr.upper(), value))
            f.write("\n")

    # hyper-parameters end here
    training_path = 'training_set_rel3.tsv'
    essay_list, resolved_scores, essay_id = data_utils.load_training_data(training_path, essay_set_id)

    max_score = max(resolved_scores)
    min_score = min(resolved_scores)
    if essay_set_id == 7:
        min_score, max_score = 0, 30
    elif essay_set_id == 8:
        min_score, max_score = 0, 60

    logger_in.info('max_score is {} \t min_score is {}\n'.format(max_score, min_score))
    with open(out_dir+'/params', 'a') as f:
        f.write('max_score is {} \t min_score is {} \n'.format(max_score, min_score))

    # include max score
    score_range = range(min_score, max_score+1)

    #word_idx, _ = data_utils.build_vocab(essay_list, vocab_limit)

    sent_size_list = list(map(len, [essay for essay in essay_list]))
    max_sent_size = max(sent_size_list)
    mean_sent_size = int(np.mean(list(map(len, essay_list))))

    logger_in.info('max sentence size: {} \nmean sentence size: {}\n'.format(max_sent_size, mean_sent_size))
    with open(out_dir+'/params', 'a') as f:
        f.write('max sentence size: {} \nmean sentence size: {}\n'.format(max_sent_size, mean_sent_size))

    logger_in.info('The length of score range is {}'.format(len(score_range)))
    E = data_utils.vectorize_data(essay_list, word_idx, max_sent_size)

    labeled_data = zip(E, resolved_scores, sent_size_list) # vector, score, length

    # split the data on the fly
    #trainE, testE, train_scores, test_scores, train_sent_sizes, test_sent_sizes = cross_validation.train_test_split(
    #    E, resolved_scores, sent_size_list, test_size=.2, random_state=random_state)

    #trainE, evalE, train_scores, eval_scores, train_sent_sizes, eval_sent_sizes = cross_validation.train_test_split(
    #    trainE, train_scores, train_sent_sizes, test_size=.1, random_state=random_state)
    # split the data on the fly
    trainE, testE, train_scores, test_scores, train_essay_id, test_essay_id = cross_validation.train_test_split(
        E, resolved_scores, essay_id, test_size=.2, random_state=random_state)

    memory = []
    memory_score = []
    memory_sent_size = []
    memory_essay_ids = []

    # pick sampled essay for each score
    for i in score_range:
        # test point: limit the number of samples in memory for 8
        for j in range(num_samples):
            if i in train_scores:
                score_idx = train_scores.index(i)
                score = train_scores.pop(score_idx)
                essay = trainE.pop(score_idx)
                sent_size = sent_size_list.pop(score_idx)
                memory.append(essay)
                memory_score.append(score)
                memory_essay_ids.append(train_essay_id.pop(score_idx))
                memory_sent_size.append(sent_size)
    memory_size = len(memory)
    trainE, evalE, train_scores, eval_scores, train_essay_id, eval_essay_id = cross_validation.train_test_split(
        trainE, train_scores, train_essay_id, test_size=.2)
    # convert score to one hot encoding
    train_scores_encoding = list(map(lambda x: score_range.index(x), train_scores))

    # data size
    n_train = len(trainE)
    n_test = len(testE)
    n_eval = len(evalE)

    logger_in.info('The size of training data: {}'.format(n_train))
    logger_in.info('The size of testing data: {}'.format(n_test))
    logger_in.info('The size of evaluation data: {}'.format(n_eval))
    with open(out_dir+'/params', 'a') as f:
        f.write('The size of training data: {}\n'.format(n_train))
        f.write('The size of testing data: {}\n'.format(n_test))
        f.write('The size of evaluation data: {}\n'.format(n_eval))
        f.write('\nEssay scores in memory:\n{}'.format(memory_score))
        f.write('\nEssay ids in memory:\n{}'.format(memory_essay_ids))
        f.write('\nEssay ids in training:\n{}'.format(train_essay_id))
        f.write('\nEssay ids in evaluation:\n{}'.format(eval_essay_id))
        f.write('\nEssay ids in testing:\n{}'.format(test_essay_id))

    batches = zip(range(0, n_train-batch_size, batch_size), range(batch_size, n_train, batch_size))
    batches = [(start, end) for start, end in batches]

    with tf.Graph().as_default():
        session_conf = tf.ConfigProto(
            allow_soft_placement=FLAGS.allow_soft_placement,
            log_device_placement=FLAGS.log_device_placement)
        session_conf.gpu_options.allow_growth = True

        global_step = tf.Variable(0, name="global_step", trainable=False)
        # decay learning rate
        starter_learning_rate = FLAGS.learning_rate
        learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step, 3000, 0.96, staircase=True)

        # test point
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, epsilon=FLAGS.epsilon)
        best_test_kappa_so_far = 0.0
        best_eval_kappa_so_far = 0.0
        with tf.Session(config=session_conf) as sess:
            model = MemN2N_KV(batch_size, vocab_size, max_sent_size, max_sent_size, memory_size,
                              memory_size, embedding_size, len(score_range), feature_size, hops, reader, l2_lambda)

            grads_and_vars = optimizer.compute_gradients(model.loss_op, aggregation_method=tf.AggregationMethod.EXPERIMENTAL_TREE)
            grads_and_vars = [(tf.clip_by_norm(g, FLAGS.max_grad_norm), v)
                              for g, v in grads_and_vars if g is not None]
            grads_and_vars = [(add_gradient_noise(g, 1e-3), v) for g, v in grads_and_vars]
            # test point
            #nil_grads_and_vars = []
            #for g, v in grads_and_vars:
            #    if v.name in model._nil_vars:
            #        nil_grads_and_vars.append((zero_nil_slot(g), v))
            #    else:
            #        nil_grads_and_vars.append((g, v))

            train_op = optimizer.apply_gradients(grads_and_vars, name="train_op", global_step=global_step)
            
            # sess.run(tf.initialize_all_variables(), feed_dict={model.w_placeholder: word2vec})
            sess.run(tf.global_variables_initializer(), feed_dict={model.w_placeholder: word2vec})

            saver = tf.train.Saver(tf.all_variables())

            def train_step(m, e, s, ma):
                start_time = time.time()
                feed_dict = {
                    model._query: e,
                    model._memory_key: m,
                    model._score_encoding: s,
                    model._mem_attention_encoding: ma,
                    model.keep_prob: FLAGS.keep_prob
                    #model.w_placeholder: word2vec
                }
                _, step, predict_op, cost = sess.run([train_op, global_step, model.predict_op, model.cost], feed_dict)
                end_time = time.time()
                time_spent = end_time - start_time
                return predict_op, cost, time_spent

            def test_step(e, m):
                feed_dict = {
                    model._query: e,
                    model._memory_key: m,
                    model.keep_prob: 1
                    #model.w_placeholder: word2vec
                }
                preds, mem_attention_probs = sess.run([model.predict_op, model.mem_attention_probs], feed_dict)
                return preds, mem_attention_probs

            for i in range(1, epochs+1):
                train_cost = 0
                total_time = 0
                np.random.shuffle(batches)
                for start, end in batches:
                    e = trainE[start:end]
                    s = train_scores_encoding[start:end]
                    s_num = train_scores[start:end]
                    #batched_memory = []
                    # batch sized memory
                    #for _ in range(len(e)):
                    #    batched_memory.append(memory)
                    mem_atten_encoding = []
                    for ite in s_num:
                        mem_encoding = np.zeros(memory_size)
                        for j_idx, j in enumerate(memory_score):
                            if j == ite:
                                mem_encoding[j_idx] = 1
                        mem_atten_encoding.append(mem_encoding)
                    batched_memory = [memory] * (end-start)
                    _, cost, time_spent = train_step(batched_memory, e, s, mem_atten_encoding)
                    total_time += time_spent
                    train_cost += cost
                logger_in.info('Finish epoch {}, total training cost is {}, time spent is {}'.format(i, train_cost, total_time))
                # evaluation
                if i % FLAGS.evaluation_interval == 0 or i == FLAGS.epochs:
                    # test on training data
                    train_preds = []
                    for start in range(0, n_train, test_batch_size):
                        end = min(n_train, start+test_batch_size)
                        
                        #batched_memory = []
                        #for _ in range(end-start):
                        #    batched_memory.append(memory)
                        batched_memory = [memory] * (end-start)
                        preds, _ = test_step(trainE[start:end], batched_memory)
                        for ite in preds:
                            train_preds.append(ite)
                    train_preds = np.add(train_preds, min_score)
                    #train_kappa_score = kappa(train_scores, train_preds, 'quadratic')
                    train_kappa_score = quadratic_weighted_kappa(
                        train_scores, train_preds, min_score, max_score)
                    # test on eval data
                    eval_preds = []
                    for start in range(0, n_eval, test_batch_size):
                        end = min(n_eval, start+test_batch_size)
                        
                        batched_memory = [memory] * (end-start)
                        preds, _ = test_step(evalE[start:end], batched_memory)
                        for ite in preds:
                            eval_preds.append(ite)

                    eval_preds = np.add(eval_preds, min_score)
                    eval_kappa_score = quadratic_weighted_kappa(eval_scores, eval_preds, min_score, max_score)

                    # test on test data
                    test_preds = []
                    test_atten_probs = []
                    for start in range(0, n_test, test_batch_size):
                        end = min(n_test, start+test_batch_size)
                        
                        batched_memory = [memory] * (end-start)
                        preds, mem_attention_probs = test_step(testE[start:end], batched_memory)
                        for ite in preds:
                            test_preds.append(ite)
                        for ite in mem_attention_probs:
                            test_atten_probs.append(ite)
                    test_preds = np.add(test_preds, min_score)
                    test_kappa_score = quadratic_weighted_kappa(test_scores, test_preds, min_score, max_score)
                    stat_dict = {'essay_id': test_essay_id, 'score': test_scores, 'pred_score': test_preds}
                    stat_df = pd.DataFrame(stat_dict)

                    # save the model if it gets best kappa
                    # if(test_kappa_score > best_test_kappa_so_far):
                    if(eval_kappa_score > best_eval_kappa_so_far):
                        best_eval_kappa_so_far = eval_kappa_score
                        best_test_kappa_so_far = test_kappa_score
                        # stats on test
                        stat_df.to_csv(out_dir+'/stat')
                        with open(out_dir+'/mem_atten', 'a') as f:
                            for idx, ite in enumerate(test_essay_id):
                                f.write('{}\n'.format(ite))
                                f.write('{}\n'.format(test_atten_probs[idx]))
                        #saver.save(sess, out_dir+'/checkpoints', global_step)
                    logger_in.info("Training kappa score = {}".format(train_kappa_score))
                    logger_in.info("Validation kappa score = {}".format(eval_kappa_score))
                    logger_in.info("Testing kappa score = {}".format(test_kappa_score))
                    logger_in.info("Best kappa score = {} at epoch {}".format(best_test_kappa_so_far, i))
                    with open(out_dir+'/eval', 'a') as f:
                        f.write("Training kappa score = {}\n".format(train_kappa_score))
                        f.write("Validation kappa score = {}\n".format(eval_kappa_score))
                        f.write("Testing kappa score = {}\n".format(test_kappa_score))
                        f.write("Best Testing kappa score so far = {}\n".format(best_test_kappa_so_far))
                        f.write('*'*10)
                        f.write('\n')
    #sys.stdout = orig_stdout
    #f.close()
