from config import *
from basic_util import text_weight_color, clean_text
from itertools import groupby
import os

'''
read results, generate a html to virtualize them.
'''

def virtualize_results(result_dir_name):
    tplt_style = '''
    <style>
        .c{
            text-align: center;
        }
        table, th, td
        {
            border: 1px solid ;
        }
        table
        {
            border-collapse:collapse;
            width:100%;
        }
    </style>
                
            '''
    tplt_tr_title = '''
        <tr>
            <td name='id'>ID</td>
            <td class=c name='role'>Neighbor</td>
            <td class=c name='score'>Score</td>
            <td name='content'>Sentence</td>
        </tr>
            '''
    tplt_tr = '''
        <tr>
            <td name='id' id={ans_id}>{ans_id}</td>
            <td class=c name='neighbor'>{neighbor}</td>
            <td class=c name='score'>{score}</td>
            <td name='sentence'>{sentence}</td>
        </tr>
            '''

    tplt_tbl_question = '''
        <table><tr><td class='c'>{que_id}</td><td>{question}</td><table>
    '''
    tplt_tbl_query = '''
        <table>
            <tr>
                <td class='c'>ID</td><td class='c'>{query_id}</td>
                <td class='c'>Truth</td><td class='c'>{truth:.1f}</td>
                <td class='c'>Prediction</td><td class='c'>{prediction:.1f}</td>
                <td class='c'>Confidence</td><td class='c'>{confidence:.3f}</td>
            </tr>
            <tr>
                <td colspan=8>{query_ans}</td>
            </tr>
        </table>
    '''
    with open(result_dir_name + '/result_confidence.txt', 'r', errors='ignore') as f_res:
        results = list(map(lambda x:x.split('\t'), f_res.readlines()))

    assert results

    # ==== group results by que_id ====
    results.sort(key=lambda x:x[5])
    for _que, _as in groupby(results, key=lambda x:x[5]):
        sp = _que.find(' ')
        que_id, question = _que[:sp], _que[sp+1:]
        answers = list(_as)
        for _query in answers:

            # question
            html = ['<html>\n', '<head>{}</head><body>'.format(tplt_style)]
            html.append('<h2>Question</h2>')
            html.append(tplt_tbl_question.format(que_id=que_id, question=question))

            # query answer
            query_id, pred, score, neighbors, confidence, sentence,  = _query[0], float(_query[1]), float(_query[2]), _query[8], float(_query[14]), _query[7]
            # query id is in format of a.b.c
            query_id = query_id.split(' ')[-1]
            # question id is a.b
            # que_id = '.'.join(query_id.split('.')[:-1])
            # query index is c
            query_idx = query_id.split('.')[-1]

            # read weights of words
            with open(RESULTS_PATH + "/word_weights/" + que_id, errors='ignore') as f_weights:
                line = f_weights.readlines()[int(query_idx) - 1]
                weight_dict = dict([(item.split(':')[0], float(item.split(':')[1])) for item in line.split(',')])

            html.append('<h2>Query answer</h2>')
            sentence = text_weight_color(sentence[sentence.find(' ')+1:], weight_dict)
            html.append(tplt_tbl_query.format(query_id=query_idx, truth=score, prediction=pred, confidence=confidence,
                                              query_ans=sentence))

            # training data
            html.append('<h2>Training answers</h2>')
            html.append('<table>')
            html.append(tplt_tr_title)
            training_ans = answers[:]
            # sort training answers by the difference to query answer
            training_ans.sort(key=lambda x:abs(float(x[2])-score))
            neighbors = neighbors.split(',')
            for _train in training_ans:
                a_id, pred, score, confidence, sentence, = _train[0], float(_train[1]), float(_train[2]), float(_train[14]), _train[7]
                a_id = a_id.split(' ')[-1]
                if a_id.strip() == query_id.strip():
                    continue
                sentence = text_weight_color(sentence[sentence.find(' ') + 1:], weight_dict)
                nei = '●' if a_id in neighbors else ''
                html.append(tplt_tr.format(ans_id=a_id, neighbor=nei, score='{:.1f}'.format(score), sentence=sentence))
            html.append('</table></body></html>')
            html_dir = result_dir_name + "/html"
            if not os.path.exists(html_dir):
                os.mkdir(html_dir)
            with open(html_dir + '/' + query_id + '.html', 'w') as f_html:
                # print('write to ' + html_dir + '/' + query_id + '.html')
                f_html.write('\n'.join(html))


    # read results
    # read questions
    #
    # with open(RAW_PATH + '/questions') as f_questions:
    #     for q in f_questions:
    #         q_id = q.split(' ')[0]
    #
    #         # read scores
    #         with open(DATA_PATH + '/scores/{}/ave'.format(q_id), 'r') as f_score:
    #             scores = f_score.readlines()
    #
    #         # with open(RAW_PATH_STU + '/' + q_id, 'r', errors='ignore') as f_answers:
    #         #     answers = f_answers.readlines()
    #         #     for i in range(len(answers)):
    #         #         # read word_weight_dict
    #         #         a_id = i+1
    #         #         with open(RESULTS_PATH + "/word_weights/" + q_id + "." + str(a_id)) as f_weights:
    #         #             line = f_weights.readline()
    #         #             weight_dict = dict([(item.split(':')[0], float(item.split(':')[1])) for item in line.split(',')])
    #         #
    #         #         # virtualize answers
    #         #         html = ['<html>\n', '<div>' + q + '</div><br>', ]
    #         #
    #         #         for i in range(len(answers)):
    #         #             score = scores[i].strip()
    #         #             ans = answers[i].replace('<br>', '')
    #         #             ans = "{}.{}({}) {}".format(q_id, i + 1, score, ans.strip()[ans.find(' ') + 1:])
    #         #             html.append(text_weight_color(ans, weight_dict) + '\n')
    #         #         html.append('</html>')
    #         #         path_html = RESULTS_PATH + "/html"
    #         #         if not os.path.exists(path_html):
    #         #             os.mkdir(path_html)
    #         #         with open("{}/{}.{}.html".format(path_html, q_id, a_id), 'w', encoding='utf-8',
    #         #                   errors="ignore") as f_html:
    #         #             # with open(path_html + '/' + q_id + '.html', 'w', encoding='utf-8', errors="ignore") as f_html:
    #         #             f_html.writelines(html)
    #
    #
    #         # read word-weight dict
    #         weight_dicts = []
    #         with open(RESULTS_PATH + "/word_weights/" + q_id, errors='ignore') as f_weights:
    #             for line in f_weights:
    #                 weights = dict([(item.split(':')[0], float(item.split(':')[1])) for item in line.split(',')])
    #                 weight_dicts.append(weights)
    #
    #
    #
    #
    #
    #         # read answers
    #         for a_idx in range(len(weight_dicts)):
    #         # for weight_dict in weight_dicts:
    #             weight_dict = weight_dicts[a_idx]
    #
    #             with open(RAW_PATH_STU + '/' + q_id, encoding='utf-8', errors="ignore") as f_answers:
    #                 answers = f_answers.readlines()
    #                 for i in range(len(answers)):
    #                     score = scores[i].strip()
    #                     ans = answers[i].replace('<br>', '')
    #                     ans = "{}.{}({}) {}".format(q_id, i+1, score, ans.strip()[ans.find(' ')+1:])
    #                     html.append(text_weight_color(ans, weight_dict)+'\n')
    #             html.append('</html>')
    #             path_html = RESULTS_PATH + "/html"
    #             if not os.path.exists(path_html):
    #                 os.mkdir(path_html)
    #             with open("{}/{}.{}.html".format(path_html, q_id, a_idx+1), 'w', encoding='utf-8', errors="ignore") as f_html:
    #             # with open(path_html + '/' + q_id + '.html', 'w', encoding='utf-8', errors="ignore") as f_html:
    #                 f_html.writelines(html)

if __name__ == '__main__':
    file_list = os.listdir(RESULTS_PATH + '/results')
    print(file_list)
    for f in file_list:
        if f.startswith('.'):
            continue
        if not os.path.exists(RESULTS_PATH + '/results/' + f + '/result_confidence.txt'):
            continue
        # try:
        print(f)
            # if f.startswith('gb.kmenas'):
            #     print(f)
        virtualize_results(RESULTS_PATH + '/results/' + f)
        # except:
        #     print("FAILED!")