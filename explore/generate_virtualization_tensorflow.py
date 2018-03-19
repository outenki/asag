from config import *
from basic_util import check_dir
# generate vector data
# replace ',' to '\t'


def convert_comma2tab_tsv(file_input, file_output):
    with open(file_input, 'r') as f_input:
        lines = f_input.readlines()
        lines = map(lambda x: x.split(','), lines)
        lines = list(map(lambda x: '\t'.join(x), lines))
    with open(file_output, 'w') as f_output:
        f_output.writelines(lines)


def generate_label_raw_tsv(file_label, file_raw, titles, file_output):
    with open(file_label, 'r') as f_label:
        labels = f_label.readlines()
        labels = list(map(lambda x: x.strip(), labels))
    with open(file_raw, 'r', errors='ignore') as f_raw:
        raws = f_raw.readlines()
        raws = list(map(lambda x:x[x.find(' ')+1:], raws))
    assert len(labels) == len(raws)

    title = '\t'.join(titles) + '\n'
    label_raw_list = list(map(lambda x: '\t'.join(x), zip(labels, raws)))

    with open(file_output, 'w') as f_output:
        f_output.write(title)
        f_output.writelines(label_raw_list)


if __name__ == '__main__':
    path_tensorflow = '%s/virtualization_tensorflow' % DATA_PATH
    path_tsv_vector = '%s/vectors' % path_tensorflow
    path_tsv_raw = '%s/raw' % path_tensorflow
    check_dir(path_tensorflow)
    check_dir(path_tsv_vector)
    check_dir(path_tsv_raw)

    path_feature = '%s/features_sent2vec' % RESULTS_PATH
    path_label = '%s/scores' % DATA_PATH
    path_raw = RAW_PATH_STU

    que_ids = sorted(os.listdir(RAW_PATH_STU))
    for que_id in que_ids:
        print(que_id)
        fn_feature = '%s/%s' % (path_feature, que_id)
        fn_output = '%s/%s' % (path_tsv_vector, que_id)
        convert_comma2tab_tsv(fn_feature, fn_output)

        fn_label = '%s/%s/ave' % (path_label, que_id)
        fn_raw = '%s/%s' % (path_raw, que_id)
        fn_output = '%s/%s' % (path_tsv_raw, que_id)
        generate_label_raw_tsv(fn_label, fn_raw, ('Score', 'Answers'), fn_output)
