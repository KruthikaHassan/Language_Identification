import numpy as np 
import csv
import re
import time
import sys

class DataSet(object):

    def __init__(self, data_file_path):

        start_time = time.time()
        print("Loading File:", data_file_path)
        self._text, self._langid = self.load_file(data_file_path)
        time_taken = time.time() - start_time
        print("%s Loaded: %.3f secs!" % (data_file_path, time_taken))

    @property
    def text(self):
        return self._text

    @property
    def langid(self):
        return self._langid
    
    def load_file(self, filename):
        file = open(filename, encoding = 'utf-8')
        data  = [row for row in csv.reader(file, delimiter = '\t')]
        text, lang_id = [], []
        for row in range(0,len(data)):
            text.append(data[row][1:])
            lang_id.append(self.get_lang_label(data[row][0]))
        return text, lang_id

    def get_lang_label(self, langid_text):

        if langid_text == 'es':
            return [1, 0, 0, 0, 0, 0, 0, 0, 0]
        elif langid_text == 'en':
            return [0, 1, 0, 0, 0, 0, 0, 0, 0]
        elif langid_text == 'pt':
            return [0, 0, 1, 0, 0, 0, 0, 0, 0]
        elif langid_text == 'fr':
            return [0, 0, 0, 1, 0, 0, 0, 0, 0]
        elif langid_text == 'de':
            return [0, 0, 0, 0, 1, 0, 0, 0, 0]
        elif langid_text == 'gl':
            return [0, 0, 0, 0, 0, 1, 0, 0, 0]
        elif langid_text == 'eu':
            return [0, 0, 0, 0, 0, 0, 1, 0, 0]
        elif langid_text == 'it':
            return [0, 0, 0, 0, 0, 0, 0, 1, 0]
        else:
            return [0, 0, 0, 0, 0, 0, 0, 0, 1]
        