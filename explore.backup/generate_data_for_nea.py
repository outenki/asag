from config import *
import os
TSV_NAME = 'data.tsv'

title = 'essay_id\tessay_set\tessay\trater1_domain1\trater2_domain1\trater3_domain1\tdomain1_score\trater1_domain2\trater2_domain2\tdomain2_score\trater1_trait1\trater1_trait2\trater1_trait3\trater1_trait4\trater1_trait5\trater1_trait6\trater2_trait1\trater2_trait2\trater2_trait3\trater2_trait4\trater2_trait5\trater2_trait6\trater3_trait1\trater3_trait2\trater3_trait3\trater3_trait4\trater3_trait5\trater3_trait6\n'
data = []
# read file list of raw data
files = sorted(os.listdir(RAW_PATH_STU))
for i, fi in enumerate(files):
    with open('%s/%s' % (RAW_PATH_STU, fi), 'r', errors='ignore') as fn:
        raw_data = fn.readlines()
        raw_data = list(map(lambda x:x.strip(), raw_data))
    with open('%s/scores/%s/ave' % (DATA_PATH, fi), 'r') as fn:
        scores = fn.readlines()
        scores = list(map(lambda x:x.strip(), scores))
    prmpt = [i+1] * len(raw_data)
    raw_score = list(zip(raw_data, prmpt, scores))
    data.extend(raw_score)


print('%s/%s' % (DATA_PATH, TSV_NAME))
with open(TSV_NAME, 'w') as fn, open('idx_prmpt_id.txt', 'w') as fd:
    fn.write(title)
    ans_id = 0
    que_id_last = ''
    for i, item in enumerate(data):
        raw, prmpt, score = item
        raw = raw[raw.find(' ')+1:]
        raw = raw.replace('<br>', ' ')
        fn.write('{}\t{}\t{}\t{}\n'.format(i+1, prmpt, raw, '\t'.join([score] * 25)))

        que_id = files[prmpt-1]
        if que_id != que_id_last:
            ans_id = 0
        que_id_last = ans_id
        ans_id += 1
        fd.write('%d\t%s\t%d' % (i+1, que_id, ans_id))
