from xml.etree.ElementTree import ElementTree, Element
import os
basic_path = '../data/sciEntsBank_2Ways'
question_paths = (
    # basic_path + '/test-unseen-domains/',
    # basic_path + '/test-unseen-questions/',
    # basic_path + '/test-unseen-answers/',
    # basic_path + '/train/'
    basic_path + '/all/',
)
LABELS_5WAYS = dict((
    ('non_domain', '0'),
    ('irrelevant', '1'),
    ('contradictory', '2'),
    ('partially_correct_incomplete', '3'),
    ('correct', '4')))

LABELS_3WAYS = dict((
    ('non_domain', '0'),
    ('irrelevant', '0'),
    ('contradictory', '0'),
    ('partially_correct_incomplete', '1'),
    ('correct', '2')))
LABELS_2WAYS = dict((
    ('non_domain', '0'),
    ('incorrect', '0'),
    ('irrelevant', '0'),
    ('contradictory', '0'),
    ('partially_correct_incomplete', '0'),
    ('correct', '1')))

Scores = LABELS_2WAYS

def read_xml(in_path):
    '''''读取并解析xml文件
       in_path: xml路径
       return: ElementTree'''
    tree = ElementTree()
    tree.parse(in_path)
    return tree


# question_path + '/Dependency/ReferenceAnswers/' + ref_id + '.txt.label'
def read_parses(path_to_file):
    #     print(path_to_file)
    edges = []
    graph_id = os.path.split(path_to_file)[1].split('.')[0:-2]
    graph_id = '_'.join(graph_id)

    with open(path_to_file, 'r') as f_parse:
        lines = f_parse.readlines()
        lines = [l for l in lines if l.strip()]
        for i in range(len(lines)):
            lines[i] = lines[i].split('\t')

        reformed_lines = []
        base_number = 0
        for line in lines:
            if not line[0].strip():
                base_number = int(reformed_lines[-1][0])
                continue
            line[0] = str(int(line[0]) + base_number)
            line[5] = str(int(line[5]) + base_number)
            reformed_lines.append(line)

        for line in reformed_lines:
            #             print('line1')
            if len(line) == 1 and not line[0].strip():
                continue
            if not line[5].isdigit():
                continue
            # print(line)
            dep = line[6].lower()
            word1 = line[2]
            seq1 = line[0]
            pos1 = line[3]
            if word1 in '()':
                continue

            to = int(line[5]) - 1
            if -1 == to:
                word2 = 'ROOT'
                pos2 = 'ROOT'
                seq2 = '0'
            else:
                line2 = lines[to]
                #                 print(line2)
                word2 = line2[2]
                seq2 = line2[0]
                pos2 = line2[3]
            if word2 in '()':
                continue
            edge = '{}({}:{}:{}, {}:{}:{})'.format(dep, word1, pos1, seq1, word2, pos2, seq2)
            edges.append(edge)
    return graph_id + ' ' + '|||'.join(edges) + '\n'


for question_path in question_paths:
    path_xml = question_path + '/Core/'
    path_raw = question_path + 'raw/'
    if not os.path.exists(path_raw): os.mkdir(path_raw)

    path_scores = question_path + 'scores/'
    if not os.path.exists(path_scores): os.mkdir(path_scores)

    path_parses = question_path + 'parses/'
    if not os.path.exists(path_parses): os.mkdir(path_parses)

    raw_questions, raw_answers = [], []
    parses_ref, parses_que = [], [],
    print(question_path)
    print(os.listdir(path_xml))
    for f_xml in os.listdir(path_xml):
        scores, parses_stu, raw_ans_stu = [], [], []
        doc = read_xml(path_xml + f_xml)
        root = doc.getroot()
        # question_id = root.get('id')
        question_id =  ''.join(f_xml.split('.')[:-1])
        question_text = root.find('questionText').text
        raw_questions.append(question_id + ' ' + question_text + '\n')
        path_to_file = question_path + 'Dependency/Questions/'
        # if not os.path.exists(path_to_file):
        #     os.mkdir(path_to_file)
        print('path_to_file', path_to_file)
        print('question_id:', question_id)
        # parses_que.append(read_parses(path_to_file + question_id + '.txt.label'))

        ref_ans = root.find('referenceAnswers/referenceAnswer')
        ref_id = ''.join(f_xml.split('.')[:-1])
        ref_id += '-' + ref_ans.get('id')
        raw_answers.append(ref_id + ' ' + ref_ans.text + '\n')
        # path_to_file = question_path + 'Dependency/ReferenceAnswers/'
        # if not os.path.exists(path_to_file):
        #     os.mkdir(path_to_file)
        # parses_ref.append(read_parses(path_to_file + ref_id + '.txt.label'))

        # path_to_file = question_path + 'Dependency/StudentAnswersRaw/'
        # if not os.path.exists(path_to_file):
        #     os.mkdir(path_to_file)
        for ans_stu in root.iterfind('studentAnswers/studentAnswer'):
            ans_stu_id = ans_stu.get('id')
            ans_stu_score = Scores[ans_stu.get('accuracy')]
            ans_stu_text = ans_stu.text

            raw_ans_stu.append(ans_stu_id + ' ' + ans_stu_text + '\n')
            scores.append(ans_stu_score + '\n')
            # parse_stu = read_parses(path_to_file + ans_stu_id + '.txt.label')
            # parses_stu.append(parse_stu)

        # write raw for student answers
        with open(path_raw + question_id, 'wt') as f_raw_stu:
            f_raw_stu.writelines(raw_ans_stu)

        # write scores
        with open(path_scores + question_id, 'wt') as f_scores:
            f_scores.writelines(scores)

        # write parses of student answers
        with open(path_parses + question_id, 'wt') as f_parses:
            f_parses.writelines(parses_stu)

    # write raw for questions and ref-answers
    with open(path_raw + 'questions', 'wt') as f_raw_q, open(path_raw + 'answers', 'wt') as f_raw_a:
        f_raw_q.writelines(raw_questions)
        f_raw_a.writelines(raw_answers)

    # write parses of questions and ref-answers
    with open(path_parses + 'questions', 'wt') as f_parses_q, open(path_parses + 'answers', 'wt') as f_parses_a:
        f_parses_q.writelines(parses_que)
        f_parses_a.writelines(parses_ref)

print('done')
