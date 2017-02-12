#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import pkg_resources



def get_norwegian_stopwords():
    DATA_PATH = pkg_resources.resource_filename('nortok', 'data/stopwords.txt')
    sw=set()
    with open(DATA_PATH) as ff:
        for line in ff:
            swtok=line.split('|')[0].strip()
            sw.add(swtok)
    sw.update({':',';','.',',','!','?','-','\\','/','f.eks',"–",')'\
        ,'(',"”","får","000",'to','sier','fikk',"«","»",'les',\
        'blant','annet','the','a','of','mye','jo','få','får','må',\
        'litt','andre','mer','kommer','går','veldig','les',\
        'gjør','under','meir','fekk','vert','berre','frå',\
        "\'",'"',"'","\x96",'kl','igjen','hos','alt','hatt',\
        'tatt','tok','rundt','tre','fire',"\xad",'ta'})
    temp={str(iii) for iii in range(10)}
    sw.update(temp)
    temp1={str(iii) for iii in range(1880,2030)}
    sw.update(temp1)
    return sw
