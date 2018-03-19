# !/bin/bash

for i in $(ls data/$1)
do 
    MKL_THREADING_LAYER=GNU KERAS_BACKEND='theano' THEANO_FLAGS="device=cuda1,floatX=float32" python train_nea.py -tr data/$1/$i/train.tsv -tu data/$1/$i/dev.tsv -ts data/$1/$i/test.tsv -p 0 -o $1/$i/
done

