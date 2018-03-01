import csv
import numpy as np 
import re
import time

class Vocab(object):

    def __init__(self, text):
        start_time = time.time()
        print("Building Vocabulary:")
        self._vocab_size, self._percent_oov = self.vocab_building(text)
        time_taken = time.time() - start_time
        print("Built Vocab in: %.3f secs!" % (time_taken))
        print("Vocabulary Size: %d !" % (self._vocab_size))
        print("Percentage of Out of Vocabulary tokens: %.3f!" % (self._percent_oov))

    @property
    def vocab_dict(self):
        return self._vocab_dict

    def percent_oov(self):
        return self._percent_oov

    def vocab_size(self):
        return self._vocab_size

    def vocab_building(self,text):
        token_list = []
        vocab_dict = {}
        outofvocab = []
        characterlist = []
        counter = 0

        for row in range(0,len(text)):
            row_split = text[row][0].split()            #split into row
            for index in range(0,len(row_split)):       #split into characters
               c = list(row_split[index])
               characterlist.extend(c)

        for char in characterlist:
            for index in range(0,len(characterlist)):
                if char == characterlist[index]:
                    counter += 1
            if (counter >= 10):        
                vocab_dict[char] = counter
                token_list.extend(char)
            else:
                outofvocab.append('<oov>')

        token_list.extend(['<S>','</S>'])
        
        vocabsize = len(token_list)
        percentoov = len(outofvocab)/vocabsize

        return vocabsize, percentoov

