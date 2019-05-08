#!/bin/bash
if [ "$1" ] ; then
    echo "Set proxy to ${1}"
    PROXY=$1
else
    echo "Set default proxy to https://127.0.0.1:1080"
    PROXY=https://127.0.0.1:1080
fi
python data/download_pretrained_bert.py --proxy $PROXY