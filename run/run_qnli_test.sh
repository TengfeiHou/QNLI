#!/bin/bash

read_model_path=$1
bert=bert-base-uncased # bert-base-uncased, bert-base-cased, bert-large-uncased, bert-large-cased
loss=ce
reduction=mean # mean, sum, none
label_smoothing=0.1
batch_size=10
drop=0.1
deviceId=0
seed=999

python scripts/qnli_classifier_bert.py --testing --read_model_path $read_model_path --bert $bert \
    --loss $loss --reduction $reduction --label_smoothing $label_smoothing \
    --batch_size $batch_size --dropout $drop --deviceId $deviceId --seed $seed