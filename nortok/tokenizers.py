#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from nltk.tokenize import TweetTokenizer
from nltk.stem.snowball import SnowballStemmer
import numpy as np
from collections import defaultdict,Counter
import logging as logger
from nortok.stopwords import get_norwegian_stopwords
import pickle
import gzip

def dd_def():
    return 0

def _fit_tokenizer(texts,tokfunc,max_length=None):
    """
    Makes word dictionaries based on text. Zero (0) is reserved.
    """

    doc_count=0
    wordcount=Counter()
    ind2word={}
    word2ind=defaultdict(dd_def)
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

def _max_word_vocab(word2ind,ind2word,wcs,max_words=None):
    if max_words is None:
        return word2ind,ind2word
    if len(word2ind)<max_words:
        logger.info('len(word2ind)<=max_words:{0}'.format(max_words))
        return word2ind,ind2word
    w2i=defaultdict(dd_def)
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
        ml=min(max_len,len(toks))
        seqs[iii,:ml]=[word2ind[tok] for tok in toks]
    return seqs

def texts_to_seqs_var(texts,tokfunc,word2ind,max_len=None):
    for txt in texts:
        toks=tokfunc(txt,max_length=max_len)
        yield [word2ind[tok] for tok in toks]

class BaseTokenizer(object):
    def __init__(self,word2ind=None,max_words=None,min_freq=2,**kwargs):
        if word2ind is not None:
            self.document_count=1
            self.word2ind=defaultdict(dd_def,word2ind)
        self.min_freq=min_freq

    def tokenize(self,text,max_length=None):
        toks=text.split()
        if max_length:
            toks=toks[:max_length]
        return toks

    def texts_to_sequences(self,texts,max_len,n_texts=None):
        seqs=_texts_to_seqs(texts,self.tokenize,self.word2ind,max_len,n_texts)
        return seqs

    def var_length_texts_to_sequences(self,texts):
        tt=texts_to_seqs_var(texts,self.tokenize,self.word2ind)
        for seq in tt:
            yield seq

    def fit_tokenizer(self,texts,max_length,max_words=None):
        word2ind,ind2word,wordcount=_fit_tokenizer(texts,self.tokenize,max_length=max_length)
        wordcount=dict((q,r) for q,r in wordcount.items() if r>=self.min_freq)

        def skey(x):
            return x[1]
        wcs=list(wordcount.items())
        wcs.sort(key=skey,reverse=True)

        self.wordcount=wcs
        self.word2ind,self.ind2word=_max_word_vocab(word2ind,ind2word,self.wordcount,max_words)
        self.max_words=max_words

    def prune_vocab(self,max_words=None):
        if max_words>=self.max_words:
            raise ValueError("Can't prune with larger vocabulary.")
        self.word2ind,self.ind2word=_max_word_vocab(self.word2ind,self.ind2word,self.wordcount,max_words)
        self.max_words=max_words

    def save_tokenizer(self,savepath,extraobjs=None):
        w2i=list(self.word2ind.items())
        i2w=list(self.ind2word.items())
        outdict={'word2ind':w2i,'ind2word':i2w,'max_words':self.max_words}
        if extraobjs:
            outdict.update(extraobjs)
        with gzip.open(savepath,'wb') as ff:
            pickle.dump(outdict,ff)

    @staticmethod
    def load_tokenizer(savepath):
        with gzip.open(savepath,'rb') as ff:
            indict=pickle.load(ff)
        indict['word2ind']=defaultdict(dd_def,indict['word2ind'])
        indict['ind2word']=dict(indict['ind2word'])
        tok=WordTokenizer(indict)
        for k,v in indict.items():
            setattr(tok,k,v)
        return tok

class WordTokenizer(BaseTokenizer):
    def __init__(self,word2ind=None,use_stopwords=False,use_stemmer=False,max_words=None,**kwargs):
        super(WordTokenizer, self).__init__(**kwargs)
        self.tweetok=TweetTokenizer(**kwargs)
        if use_stopwords:
            if isinstance(use_stopwords,set):
                self.stopwords=use_stopwords
            else:
                self.stopwords=get_norwegian_stopwords()
        else:
            self.stopwords=False

        self.use_stemmer=use_stemmer
        if use_stemmer==True:
            self.stemmer=SnowballStemmer('norwegian')

    def def_eobjs(self):
        return {'use_stemmer':self.use_stemmer,'stopwords':self.stopwords}

    def tokenize(self,text,max_length=512):
        toks=self.tweetok.tokenize(text)
        if max_length:
            toks=toks[:max_length]
        if self.stopwords and (not self.use_stemmer):
            toks=[t for t in toks if t not in self.stopwords]
        elif self.stopwords and self.use_stemmer:
            toks=[self.stemmer.stem(t) for t in toks if t not in self.stopwords]
        elif (not self.stopwords) and self.use_stemmer:
            toks=[self.stemmer.stem(t) for t in toks]
        return toks

    def save_tokenizer(self,savepath):
        eobjs=self.def_eobjs()
        super(WordTokenizer, self).save_tokenizer(savepath=savepath,extraobjs=eobjs)


class RawCharTokenizer(BaseTokenizer):
    def __init__(self,word2ind=None,max_words=None):
        self.max_words=max_words

    def tokenize(self,text,max_length=None):
        toks=list(text.lower())
        if max_length:
            toks=toks[:max_length]
        return toks


class HierarchicalTokenizer(BaseTokenizer):
    def __init__(self,word2ind=None,max_words=None):
        self.max_words=max_words
        self.wordtok=WordTokenizer()
        self.chartok=RawCharTokenizer(max_words=max_words)

    def tokenize(self,text,max_len_words=512,max_len_chars=20):
        wds=self.wordtok.tokenize(text,max_length=max_len_words)
        toks=[self.chartok.tokenize(wd,max_length=max_len_chars)]
        return toks
