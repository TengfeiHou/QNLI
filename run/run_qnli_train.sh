#!/bin/bash

bert=bert-base-uncased # bert-base-uncased, bert-base-cased, bert-large-uncased, bert-large-cased
loss=ce
reduction=mean # mean, sum, none
label_smoothing=0.1
lr=5e-5
l2=0.01
batch_size=10
drop=0.1
optim=adam # bertadam
warmup=0.1
schedule=warmup_linear # warmup_linear, warmup_constant, warmup_cosine, none
init_weights=0.1
max_epoch=100
deviceId=0
seed=999

python scripts/qnli_classifier_bert.py --bert $bert --loss $loss --reduction $reduction --label_smoothing $label_smoothing \
    --lr $lr --weight_decay $l2 --batch_size $batch_size --dropout $drop \
    --optim $optim --warmup $warmup --schedule $schedule --init_weights $init_weights \
    --max_epoch $max_epoch --deviceId $deviceId --seed $seed