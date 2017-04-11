#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# 
# The BPE encoding provided herein is adapted from
# https://github.com/rsennrich/subword-nmt


from collections import defaultdict, Counter
import copy
from nortok.stopwords import get_example_data
from nltk.tokenize import TweetTokenizer
from nortok.tokenizers import _texts_to_seqs,_fit_tokenizer,_max_word_vocab
import re
import sys


class BPE(object):
    def __init__(self,n_symbols=32000,min_frequency=2,tokenizer=None,separator='@@'):
        if not tokenizer:
            self.tokenizer=TweetTokenizer(preserve_case=False)
            self.btok=self.tokenizer.tokenize
        else:
            self.tok=tokenizer
        self.n_symbols=n_symbols
        self.min_frequency=2
        self.separator=separator
        self.number_replace=re.compile('\d')

    def tok(self,txt):
        txt=self.number_replace.sub('¤',txt)
        return self.btok(txt)

    def _get_vocab_from_file(self,fobj):
        """Read text and return dictionary that encodes vocabulary
        """
        vocab = Counter()
        for line in fobj:
            if isinstance(fobj,dict):
                word, count = line.strip().split()
                vocab[word] = int(count)
            else:
                toks=self.tok(line.strip())
                for word in toks:
                    vocab[word] += 1
        return vocab

    def get_vocabulary(self,ff):
        vocab=self._get_vocab_from_file(ff)
        vocab = dict([(tuple(x)+('</w>',) ,y) for (x,y) in vocab.items()])
        self.sorted_vocab = sorted(vocab.items(), key=lambda x: x[1], reverse=True)

    @staticmethod
    def update_pair_statistics(pair, changed, stats, indices):
        """Minimally update the indices and frequency of symbol pairs
        if we merge a pair of symbols, only pairs that overlap with occurrences
        of this pair are affected, and need to be updated.
        """
        stats[pair] = 0
        indices[pair] = defaultdict(int)
        first, second = pair
        new_pair = first+second
        for j, word, old_word, freq in changed:

            # find all instances of pair, and update frequency/indices around it
            i = 0
            while True:
                try:
                    i = old_word.index(first, i)
                except ValueError:
                    break
                if i < len(old_word)-1 and old_word[i+1] == second:
                    if i:
                        prev = old_word[i-1:i+1]
                        stats[prev] -= freq
                        indices[prev][j] -= 1
                    if i < len(old_word)-2:
                        # don't double-count consecutive pairs
                        if old_word[i+2] != first or i >= len(old_word)-3 or old_word[i+3] != second:
                            nex = old_word[i+1:i+3]
                            stats[nex] -= freq
                            indices[nex][j] -= 1
                    i += 2
                else:
                    i += 1

            i = 0
            while True:
                try:
                    i = word.index(new_pair, i)
                except ValueError:
                    break
                if i:
                    prev = word[i-1:i+1]
                    stats[prev] += freq
                    indices[prev][j] += 1
                # don't double-count consecutive pairs
                if i < len(word)-1 and word[i+1] != new_pair:
                    nex = word[i:i+2]
                    stats[nex] += freq
                    indices[nex][j] += 1
                i += 1

    @staticmethod
    def get_pair_statistics(vocab):
        """Count frequency of all symbol pairs, and create index"""

        # data structure of pair frequencies
        stats = defaultdict(int)

        #index from pairs to words
        indices = defaultdict(lambda: defaultdict(int))

        for i, (word, freq) in enumerate(vocab):
            prev_char = word[0]
            for char in word[1:]:
                stats[prev_char, char] += freq
                indices[prev_char, char][i] += 1
                prev_char = char

        return stats, indices

    @staticmethod
    def replace_pair(pair, vocab, indices):
        """Replace all occurrences of a symbol pair ('A', 'B') with a new symbol 'AB'"""
        first, second = pair
        pair_str = ''.join(pair)
        pair_str = pair_str.replace('\\','\\\\')
        changes = []
        pattern = re.compile(r'(?<!\S)' + re.escape(first + ' ' + second) + r'(?!\S)')
        if sys.version_info < (3, 0):
            iterator = indices[pair].iteritems()
        else:
            iterator = indices[pair].items()
        for j, freq in iterator:
            if freq < 1:
                continue
            word, freq = vocab[j]
            new_word = ' '.join(word)
            new_word = pattern.sub(pair_str, new_word)
            new_word = tuple(new_word.split())

            vocab[j] = (new_word, freq)
            changes.append((j, new_word, word, freq))

        return changes

    @staticmethod
    def prune_stats(stats, big_stats, threshold):
        """Prune statistics dict for efficiency of max()
        The frequency of a symbol pair never increases, so pruning is generally safe
        (until we the most frequent pair is less frequent than a pair we previously pruned)
        big_stats keeps full statistics for when we need to access pruned items
        """
        for item,freq in list(stats.items()):
            if freq < threshold:
                del stats[item]
                if freq < 0:
                    big_stats[item] += freq
                else:
                    big_stats[item] = freq

    def train_bpe(self,verbose=1):
        self.mf=[]
        stats, indices = BPE.get_pair_statistics(self.sorted_vocab)
        big_stats = copy.deepcopy(stats)
        # threshold is inspired by Zipfian assumption, but should only affect speed
        threshold = max(stats.values()) / 10
        for i in range(self.n_symbols):
            if stats:
                most_frequent = max(stats, key=lambda x: (stats[x], x))

            # we probably missed the best pair because of pruning; go back to full statistics
            if not stats or (i and stats[most_frequent] < threshold):
                BPE.prune_stats(stats, big_stats, threshold)
                stats = copy.deepcopy(big_stats)
                most_frequent = max(stats, key=lambda x: (stats[x], x))
                # threshold is inspired by Zipfian assumption, but should only affect speed
                threshold = stats[most_frequent] * i/(i+10000.0)
                BPE.prune_stats(stats, big_stats, threshold)

            if stats[most_frequent] < self.min_frequency:
                sys.stderr.write('no pair has frequency >= {0}. Stopping\n'.format(self.min_frequency))
                break

            if verbose:
                sys.stderr.write('pair {0}: {1} {2} -> {1}{2} (frequency {3})\n'.format(i, most_frequent[0], most_frequent[1], stats[most_frequent]))
            self.mf.append('{0} {1}'.format(*most_frequent))
            #args.output.write('{0} {1}\n'.format(*most_frequent))
            changes = BPE.replace_pair(most_frequent, self.sorted_vocab, indices)
            BPE.update_pair_statistics(most_frequent, changes, stats, indices)
            stats[most_frequent] = 0
            if not i % 100:
                BPE.prune_stats(stats, big_stats, threshold)
        temp_codes=[tuple(item.split()) for item in self.mf]
        self.bpe_codes=dict([(code,i) for (i,code) in reversed(list(enumerate(temp_codes)))])

    @staticmethod
    def get_pairs(word):
        """Return set of symbol pairs in a word.
        word is represented as tuple of symbols (symbols being variable-length strings)
        """
        pairs = set()
        prev_char = word[0]
        for char in word[1:]:
            pairs.add((prev_char, char))
            prev_char = char
        return pairs

    @staticmethod
    def encode(orig, bpe_codes, cache={}):
        """Encode word based on list of BPE merge operations, which are applied consecutively
        """

        if orig in cache:
            return cache[orig]

        word = tuple(orig) + ('</w>',)
        pairs = BPE.get_pairs(word)

        while True:
            bigram = min(pairs, key = lambda pair: bpe_codes.get(pair, float('inf')))
            if bigram not in bpe_codes:
                break
            first, second = bigram
            new_word = []
            i = 0
            while i < len(word):
                try:
                    j = word.index(first, i)
                    new_word.extend(word[i:j])
                    i = j
                except:
                    new_word.extend(word[i:])
                    break

                if word[i] == first and i < len(word)-1 and word[i+1] == second:
                    new_word.append(first+second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_word = tuple(new_word)
            word = new_word
            if len(word) == 1:
                break
            else:
                pairs = BPE.get_pairs(word)

        # don't print end-of-word symbols
        if word[-1] == '</w>':
            word = word[:-1]
        elif word[-1].endswith('</w>'):
            word = word[:-1] + (word[-1].replace('</w>',''),)

        cache[orig] = word
        return word

    def tokenize(self, sentence,max_length=512):
        """segment single sentence (whitespace-tokenized string) with BPE encoding"""

        output = []
        toks1=self.tok(sentence)
        self.cache={}
        for word in toks1:
            new_word = BPE.encode(word, self.bpe_codes,self.cache)

            for item in new_word[:-1]:
                output.append(item + self.separator)
            output.append(new_word[-1])

        return output[:max_length]

    def texts_to_sequences(self,texts,max_len,n_texts=None):
        seqs=_texts_to_seqs(texts,self.tokenize,self.word2ind,max_len,n_texts)
        return seqs

    def fit_tokenizer(self,texts,max_length,max_words=None):
        word2ind,ind2word,wordcount=_fit_tokenizer(texts,self.tokenize,max_length=max_length)
        self.word2ind,self.ind2word=_max_word_vocab(word2ind,ind2word,wordcount,max_words)
        self.max_words=max_words



if __name__ == "__main__":
    datapath=get_example_data()
    bpe=BPE(n_symbols=4000)
    with open(datapath) as ff:
        bpe.get_vocabulary(ff)
    bpe.train_bpe()
    txts=[]
    with open(datapath) as ff:
        bpe.fit_tokenizer(ff,512,max_words=4200)
