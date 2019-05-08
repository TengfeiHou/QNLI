#!/bin/bash
if [ -f "submission.zip" ] ; then
    rm submission.zip
fi
zip -rj submission.zip data/submission/*.tsv