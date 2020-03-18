#!/usr/bin/env bash 
set -e

python preprocess.py -tra_src 'data/en2ch.train.src' -tra_tgt 'data/en2ch.train.tgt'\
                              -val_src 'data/en2ch.valid.src' -val_tgt 'data/en2ch.valid.tgt'\
                              -tst_src 'data/en2ch.dev.src' -tst_tgt 'data/en2ch.dev.tgt'\
                              -save 'vocabs/vocab.EnCh.json'
