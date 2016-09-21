#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from unittest import TestCase
import nortok

class TestWordTok(TestCase):

    def test_tokenize(self):
        tok=nortok.WordTokenizer()
        txt='Jeg gikk en tur i skogen, og hørte skogens ro!'
        tokens=tok.tokenize(txt,max_length=512)
        tokens_true=['Jeg','gikk','en','tur','i','skogen',',','og','hørte','skogens','ro','!']
        trues=[]
        for tok1,tok2 in zip(tokens,tokens_true):
            trues.append(tok1==tok2)
        self.assertTrue(all(trues))

    def test_maxwords(self):
        mw=4
        tok=nortok.WordTokenizer(max_words=mw)
        alltoks=set()
        txt1=['aa']*50+['bb']*30+['cc']*10+ ['dd']
        alltoks.update(set(txt1))
        txt1=' '.join(txt1)
        txt2=['cc']*50+['dd']*10
        alltoks.update(set(txt2))
        txt2=' '.join(txt2)
        txts=[txt1,txt2]
        tok.fit_tokenizer(txts,2048,max_words=tok.max_words)
        trues=[]
        for ttok in ['aa','bb','cc']:
            trues.append(ttok in tok.word2ind)
        trues.append(len(tok.word2ind)==(mw-1))

        tok2=nortok.WordTokenizer()
        tok2.fit_tokenizer(txts,2048,max_words=tok2.max_words)
        for ttok in alltoks:
            trues.append(ttok in tok2.word2ind)
        trues.append(len(tok2.word2ind)==len(alltoks))
        self.assertTrue(all(trues))
