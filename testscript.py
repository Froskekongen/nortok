#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import nortok
import argparse
import numpy as np


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Tool to test the tokenizer')
    parser.add_argument("--outfilepath",\
        dest='ofpath', help="File to write tokenizer")
    parser.add_argument("--infilepath",\
        dest='ifpath',help="File with texts, newline per doc")
    args=parser.parse_args()

    tok=nortok.WordTokenizer(preserve_case=False)
    with open(args.ifpath) as ff:
        txts=ff.readlines()

    tok.fit_tokenizer(txts,1000,max_words=1000)
    tok.save_tokenizer(args.ofpath)

    tok2=nortok.WordTokenizer.load_tokenizer(args.ofpath,nortok.WordTokenizer)

    seq1=tok.texts_to_sequences(txts,200)
    seq2=tok2.texts_to_sequences(txts,200)

    print('Sequences equal after loading tokenizer:',not np.all(seq1-seq2))

    tok.prune_vocab(max_words=900)

    seq3=tok.texts_to_sequences(txts,200)

    cols,rows=np.where((seq1-seq3)!=0)

    print('After pruning, {0} \
entries are different from original. Original vocab size: {1}, \
pruned vocab size: {2}'.format(len(cols),tok2.max_words,tok.max_words))

    bpe=nortok.BPE(n_symbols=4000)
    with open(args.ifpath) as ff:
        bpe.get_vocabulary(ff)
    bpe.train_bpe()

    bpe.fit_tokenizer(txts,1000,max_words=1000)
    bpeseq=bpe.texts_to_sequences(txts,200)
    bpe.save_tokenizer(args.ofpath+'.bpe')

    bpe2=nortok.BPE.load_tokenizer(args.ofpath+'.bpe',nortok.BPE)
