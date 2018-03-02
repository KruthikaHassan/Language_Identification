
import sys
import time
import csv
import re
import numpy as np

class VocabVector(object):
    def __init__(self, vocab, embeddings):
        self._vocab      = vocab
        self._embeddings = embeddings
        self._dimension  = len(embeddings[0])
        
    @property
    def vocab(self):
        return self._vocab
    
    @property
    def embeddings(self):
        return np.array(self._embeddings, dtype=np.float32)
    
    @property
    def dimension(self):
        return self._dimension
    
#########################################################################

class LoadVectors(VocabVector):

    def __init__(self, vocab_file_path):

        start_time = time.time()
        print("Loading File:", vocab_file_path)
        vocab, embeddings = self.__load_glove_vectors(vocab_file_path)
        time_taken = time.time() - start_time
        print("%s Loaded: %.3f secs!" % (vocab_file_path, time_taken))

        super().__init__(vocab, embeddings)
    
    @property
    def embeddings(self):
        return self._embeddings

    def __load_glove_vectors(self, filename):
        vocab = []
        embd = []
        with open(filename, 'r') as file:
            for line in file:
                row = line.strip().split(' ')
                dim = len(row[1:])
                isexpected_dim = (dim == 25) or (dim == 50) or (dim == 100) or (dim == 200)
                if not isexpected_dim:
                    continue
                vocab.append(row[0])
                embd.append([ float(s) for s in row[1:] ])
        return vocab, embd

#########################################################################################################

def build_lang_vocab(langs):
    start_time = time.time()
    print("Building lang vocab")

     # binary embeddings
    num_langs = len(langs)
    max_bits  = len(list(bin(num_langs))[2:])
    vocab, embds = [], []
    for lang in langs:
        bin_embd  = [0 for i in range(max_bits)]
        
        bits = list(bin(num_langs))[2:]
        bits_len = len(bits)
        bin_embd[-bits_len:] = [int(i) for i in bits]

        vocab.append(lang)
        embds.append(bin_embd)
        num_langs -= 1
   
    time_taken = time.time() - start_time
    print("Vocabulary built: %.3f secs!" % (time_taken))

    #print("Total Langs:", len(vocab))
    #print("Lang names:", vocab)
    #print("Lang embeddings:", embds)

    return VocabVector(vocab, embds)


def build_text_vocab(text):

    start_time = time.time()
    print("Building vocab")

    all_vocab = {}
    for line in text:
        for words in line.split():
            chars = list(words)
            for char in chars:
                if char not in all_vocab:
                    all_vocab[char] = 1
                else:
                    all_vocab[char] += 1

    # add unknown, start, end to required list
    required_vocab = {'<UNK>' : 0, '<S>': 10000, '</S>': 10000}
    for char in all_vocab:
        val = all_vocab[char]
        if val >= 10:
            required_vocab[char] = val
        else:
            required_vocab['<UNK>'] += 1
    
    # Sort according to highest 
    sorted_chars = sorted(required_vocab, key=lambda k: required_vocab[k], reverse=True)

    # binary embeddings
    num_chars = len(sorted_chars)
    max_bits  = len(list(bin(num_chars))[2:])
    vocab, embds = [], []
    for char in sorted_chars:
        bin_embd  = [0 for i in range(max_bits)]
        
        bits = list(bin(num_chars))[2:]
        bits_len = len(bits)
        bin_embd[-bits_len:] = [int(i) for i in bits]

        vocab.append(char)
        embds.append(bin_embd)
        num_chars -= 1
    
    time_taken = time.time() - start_time
    print("Vocabulary built: %.3f secs!" % (time_taken))

    print("Total chars:", len(vocab)-3)
    print("Chars not included:", required_vocab['<UNK>'])

    return VocabVector(vocab, embds)