from config import *
import os
ans_list = sorted(os.listdir(RAW_PATH_STU))
formated_dir = RAW_PATH + '/formated'
if not os.path.exists(formated_dir):
    os.mkdir(formated_dir)

ans_id = 0
formated_answers = ['essay_id\tessay_set\tessay\trater1_domain1\trater2_domain1\trater3_domain1\tdomain1_score\t'
                    'rater1_domain2\trater2_domain2\tdomain2_score\trater1_trait1\trater1_trait2\trater1_trait3\t'
                    'rater1_trait4\trater1_trait5\trater1_trait6\trater2_trait1\trater2_trait2\trater2_trait3\t'
                    'rater2_trait4\trater2_trait5\trater2_trait6\trater3_trait1\trater3_trait2\trater3_trait3\t'
                    'rater3_trait4\trater3_trait5\trater3_trait6\n']

print(ans_list)
for que_id, ans_file in enumerate(ans_list):
    que_id += 1
    with open('{}/{}'.format(RAW_PATH_STU, ans_file), 'r', errors='ignore') as f_ans:
        answers = f_ans.readlines()
    with open('{}/scores/{}/ave'.format(DATA_PATH, ans_file), 'r') as f_scores:
        scores = f_scores.readlines()

    path_formated_answer = '{}/formated/ans_stu'.format(RAW_PATH)
    path_formated_score = '{}/formated/scores'.format(RAW_PATH)
    if not os.path.exists(path_formated_answer):
        os.mkdir(path_formated_answer)
    if not os.path.exists(path_formated_score):
        os.mkdir(path_formated_score)

    with open('{}/{}'.format(path_formated_answer, que_id), 'w') as f_stu:
        f_stu.writelines(answers)
    with open('{}/{}'.format(path_formated_score, que_id), 'w') as f_score:
        f_score.writelines(scores)

    for ans, score in zip(answers, scores):
        ans_id += 1
        ans = ans.strip()
        score = score.strip()
        formated_answer = '{ans_id}\t{que_id}\t{ans}\t{scores}'.format(
            ans_id=ans_id,
            que_id=que_id,
            ans=ans,
            scores='\t'.join([score]*25) + '\n'
        )

        formated_answers.append(formated_answer)

with open('{}/data.tsv'.format(formated_dir), 'w') as f_data:
    f_data.writelines(formated_answers)
