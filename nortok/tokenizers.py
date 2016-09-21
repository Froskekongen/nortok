#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from nltk.tokenize import TweetTokenizer
import numpy as np
from collections import defaultdict,Counter
import logging as logger

def _fit_tokenizer(texts,tokfunc,max_length=None):
    """
    Makes word dictionaries based on text. Zero (0) is reserved.
    """
    doc_count=0
    wordcount=Counter()
    ind2word={}
    word2ind=defaultdict(lambda:0)
    maxInd=0
    for text in texts:
        out=tokfunc(text,max_length=max_length)
        for c in out:
            wordcount[c]+=1
        os=set(out)
        for c in os:
            if c not in word2ind:
                maxInd+=1
                word2ind[c]=maxInd
                ind2word[maxInd]=c
    return word2ind,ind2word,wordcount

def _max_word_vocab(word2ind,ind2word,wordcount,max_words=None):
    if max_words is None:
        return word2ind,ind2word
    if len(word2ind)<max_words:
        logger.info('len(word2ind)<=max_words:{0}'.format(max_words))
        return word2ind,ind2word
    wcs=list(wordcount.items())
    wcs.sort(key=lambda x:x[1],reverse=True)
    w2i=defaultdict(lambda:0)
    i2w={}
    wordInd=1
    for w in wcs:
        word=w[0]
        if wordInd>=max_words:
            break
        w2i[word]=wordInd
        i2w[wordInd]=word
        wordInd+=1
    return w2i,i2w


def _texts_to_seqs(texts,tokfunc,word2ind,max_len,n_texts=None):
    if n_texts is None:
        n_texts=len(texts)
    seqs=np.zeros((n_texts,max_len),dtype='int32')
    for iii,txt in enumerate(texts):
        toks=tokfunc(txt,max_len)
        seqs[iii,:max_len]=[word2ind[tok] for tok in toks]
    return seqs

def texts_to_seqs_var(texts,tokfunc,word2ind,max_len):
    for txt in texts:
        toks=tokfunc(txt,max_len)
        yield [word2ind[tok] for tok in toks]


class WordTokenizer(TweetTokenizer):
    def __init__(self,word2ind=None,max_words=None,**kwargs):
        super(WordTokenizer, self).__init__(**kwargs)
        if word2ind is not None:
            self.document_count=1
            self.word2ind=defaultdict(lambda:0,word2ind)

    def tokenize(self,text,max_length=512):
        toks=super(WordTokenizer,self).tokenize(text)[:max_length]
        return toks

    def texts_to_sequences(texts,max_len,n_texts=None):
        seqs=_texts_to_seqs(texts,self.tokenize,self.word2ind,max_len,n_texts)
        return seqs

    def fit_tokenizer(self,texts,max_length,max_words=None):
        word2ind,ind2word,wordcount=_fit_tokenizer(texts,self.tokenize,max_length=max_length)
        self.word2ind,self.ind2word=_max_word_vocab(word2ind,ind2word,wordcount,max_words)
        self.max_words=max_words


class RawCharTokenizer(object):
    def __init__(self,word2ind=None,max_words=None):
        self.max_words=max_words
        if word2ind is not None:
            self.document_count=1
            self.word2ind=defaultdict(lambda:0,word2ind)
        #self.max_words=max_words


    def tokenize(self,text,max_length=2048):
        return list(text.lower())[:max_length]

    def texts_to_sequences(texts,max_len,n_texts=None):
        seqs=_texts_to_seqs(texts,RawCharTokenizer.tokenize,self.word2ind,max_len,n_texts)
        return seqs

    def fit_tokenizer(self,texts,max_length,max_words=None):
        word2ind,ind2word,wordcount=_fit_tokenizer(texts,self.tokenize,max_length=max_length)
        self.word2ind,self.ind2word=_max_word_vocab(word2ind,ind2word,wordcount,max_words)
        self.max_words=max_words




class HierarchicalTokenizer(object):
    def __init__(self,word2ind=None,max_words=None):
        if word2ind is not None:
            self.document_count=1
            self.word2ind=defaultdict(lambda:0,word2ind)
        self.max_words=max_words
        self.wordtok=WordTokenizer()
        self.chartok=RawCharTokenizer(max_words=max_words)

    def tokenize(self,text,max_len_words=512,max_len_chars=20):
        wds=self.wordtok.tokenize(text,max_length=max_len_words)
        toks=[self.chartok.tokenize(wd,max_length=max_len_chars)]
        return toks
