#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from lxml import etree
from io import BytesIO
import tarfile
import pickle


class ParseText(object):
    def __init__(self,tokenizer):
        """
        Tokenizer must be explicitly fit on a relevant corpus. Should be
        able to match indices to words, or characters.
        """
        if tokenizer.document_count>0:
            self.tok = tokenizer #tokenizer must have method texts_to_sequences
        else:
            raise ValueError("Initiating parser with a tokenizer that has no docs.")
        self.word2ind = tokenizer.word2ind


class NTBParse(object):

    @staticmethod
    def parseCategories(tree,outD):
        aa=tree.findall('.//tobject')
        if not aa:
            outD['overkategori']=None
            outD['underkategori']=None
            outD['overkategoriid']=None
            outD['underkategoriid']=None
            outD['kategoriid']=None
            outD['innenriks/utenriks']=None
            return None
        aa=aa[0]
        vals=aa.values()
        if vals:
            outD['innenriks/utenriks']=vals[0]
        outD['overkategori']=[]
        outD['underkategori']=[]
        outD['overkategoriid']={}
        outD['underkategoriid']={}
        for chld in aa.iterchildren():
            if chld.tag=='tobject.property':
                if chld.values():
                    outD['nyhetstype']=chld.values()[0]
            else:
                vals=chld.values()
                if vals:
                    kdict=dict(chld.items())
                    if 'tobject.subject.matter' in kdict:
                        outD['underkategoriid'][kdict['tobject.subject.refnum']]=kdict['tobject.subject.matter']
                        outD['underkategori'].append(kdict['tobject.subject.matter'])
                    if 'tobject.subject.type' in kdict:
                        outD['overkategoriid'][kdict['tobject.subject.refnum']]=kdict['tobject.subject.type']
                        outD['overkategori'].append(kdict['tobject.subject.type'])
                # if vals:
                #     outD['overkategori'].append(vals[0])
                #     outD['underkategori'].append(vals[2])
        outD['overkategori']=list(set(outD['overkategori']))
        outD['underkategori']=list(set(outD['underkategori']))


    @staticmethod
    def parseNTBText(tree,outD):
        aa=tree.findall('.//hl1')[0]
        if aa.text:
            outD['title']=aa.text
        aa=tree.findall('.//body.content')[0]

        txt=''
        for chld in aa.iterchildren():
            if chld.text:
                txt+=chld.text+' \n'
        outD['text']=txt

    @staticmethod
    def parseNTBMessage(msg):
        tree=etree.parse(BytesIO(msg))
        outD={}
        timestamp=tree.findall('.//meta[@name="timestamp"]')
        if timestamp:
            outD[timestamp[0].values()[0]]=timestamp[0].values()[1]
        NTBParse.parseCategories(tree,outD)
        NTBParse.parseNTBText(tree,outD)
        return outD

    @staticmethod
    def parseFromTGZ(maxkat=8,targzfiles=None,jsonfilepath=None,picklefilepath=None):
        def checkLength(dd,kkey,maxkat=maxkat):
            ret=False
            if kkey in dd:
                if dd[kkey]:
                    nkat=len(dd[kkey])
                    if nkat<=maxkat and nkat>0:
                        ret=True
            return ret


        with open(jsonfilepath,'w') as jsonFile:
            iii=0
            odList=[]
            for tgz in targzfiles:
                with tarfile.open(tgz,'r:gz') as ff:
                    for tarf in ff:
                        #print(tf)
                        exfile=ff.extractfile(tarf)
                        if exfile:
                            msg=exfile.read()
                        else:
                            continue
                        try:
                            outD=self.parseNTBMessage(msg)
                        except Exception as exc:
                            logger.info('Message not parseable: {0}, {1}'.format(msg,iii))
                        if checkLength(outD,'underkategori') and checkLength(outD,'overkategori') and (len(outD['text'].split())>11):
                            jsonFile.write(json.dumps(outD)+'\n')
                            odList.append(outD)
                            iii+=1
                            if (iii%5000)==0:
                                print(iii,outD['underkategoriid'],outD['overkategoriid'])
        with open(picklefilepath,'wb') as pfile:
            pickle.dump(odList,pfile,protocol=4)
