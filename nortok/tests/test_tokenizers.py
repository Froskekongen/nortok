#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from unittest import TestCase
import nortok

class TestWordTok(TestCase):

    def test_tokenize(self):
        tok=nortok.WordTokenizer()
        txt='Jeg gikk en tur i skogen, og hørte skogens ro!'
        tokens=tok.tokenize(txt,max_length=512)
        print(tokens)
        tokens_true=['Jeg','gikk','en','tur','i','skogen',',','og','hørte','skogens','ro','!']
        trues=[]
        for tok1,tok2 in zip(tokens,tokens_true):
            trues.append(tok1==tok2)
        self.assertTrue(all(trues))
