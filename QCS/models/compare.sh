#!/bin/bash -l

FILES='CNN Decider Encoder Ensemble Guesser Oracle QGen QGenImgCap'

for fname in $FILES; do
    echo '--------------------'
    echo $fname
    diff $fname.py ~/gw-updated/models/$fname.py
done
