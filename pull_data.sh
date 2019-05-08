#!/bin/bash
if [ "$1" ] ; then
    echo "Set proxy to ${1}; Start download ..."
    export HTTP_PROXY=$1
    export HTTPS_PROXY=$1
else
    echo "Set default proxy to https://127.0.0.1:1080; Start download ..."
    export HTTP_PROXY="https://127.0.0.1:1080"
    export HTTPS_PROXY="https://127.0.0.1:1080"
fi
python data/download_glue_data.py --data_dir data --tasks QNLI