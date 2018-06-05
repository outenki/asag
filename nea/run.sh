# !/bin/bash

for i in $(ls $1)
do 
    MKL_THREADING_LAYER=GNU KERAS_BACKEND='theano' THEANO_FLAGS="device=cudaxx,floatX=float32" python train_nea.py -tr data/kaggle_ext/$i/train.tsv -tu data/kaggle_ext/$i/dev.tsv -ts data/kaggle_ext/$i/test.tsv -p $i -o data/kaggle_ext_result_cnn10_cpu/$i -e 300 -r 250 -t breg --skip-init-bias --aggregation attsum -c 10 -w 5 --emb glove.6B.300d.txt
done

