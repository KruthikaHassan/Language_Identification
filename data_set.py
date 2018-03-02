import sys
import time
import csv
import re
import numpy as np
import random
from vocab_vectors import VocabVector

# -*- coding: unicode -*-

class DataSet(object):

    def __init__(self, text, labels, isVectorized=False):
        
        # General Init
        self._text         = text
        self._labels       = labels
        self._isVectorized = isVectorized

        # For batchwise data retrival
        self._record_indecies  =   [i for i in range(self.num_records)]
        

    #####################  Properties #################################

    @property
    def text(self):
        return self._text 
    
    @property
    def labels(self):
        return self._labels
    
    @property
    def vec_labels(self):
        if not self._vec_labels:
            self.__vectorize_labels()
        return self._vec_labels

    @property
    def labels_list(self):
        return self._labels_list

    @property
    def num_records(self):
        return len(self._labels)
    
    @property
    def num_classes(self):
        return len(self.labels_list)
    
    @property 
    def max_text_length(self):
        return len(self._text[0])
    
    @property
    def isVectorized(self):
        return self._isVectorized
    
    @property
    def epoch_completed(self):
        return self._epoch_completed
    
    @property
    def records_used(self):
        return self._records_used

    ######################### Methods ##################################

    def reset_epoch(self, shuffle=False):
        if shuffle:
            random.shuffle(self._record_indecies)
        else:
            self._record_indecies = [i for i in range(self.num_records)]
        self._epoch_completed  = False
        self._records_used     = 0
    
    def get_next_batch(self, batch_size=None):
        ''' Gives data in batches '''
        
        if not batch_size:
            return self._text, self._vec_labels
        
        text = []
        vec_labels = []
        records_retrived = 0
        for index in self._record_indecies[self._records_used:]:
            text.append(self.text[index])
            vec_labels.append(self.vec_labels[index])
            records_retrived += 1
            if records_retrived >= batch_size:
                break
        
        self._records_used += records_retrived

        if records_retrived < batch_size:
            self._records_used       = 0
            self._epoch_completed    = True
            extra_records_needed     = batch_size-records_retrived
            extra_text, extra_labels = self.get_next_batch(extra_records_needed)
            text                    += extra_text
            vec_labels              += extra_labels

        return text, vec_labels

    def vectorize_text(self, vocab, max_chars_limit=100):
        ''' vectorize the cleaned up text '''
        
        # Already vectorized, no need to continue further
        if self.isVectorized:
            return True

        # Now lets indexise the words
        print("Vectorizing text: %d lines" % (len(self._text)))
        start_time = time.time()
        
        raw_text           = self._text
        text_vec           = []
        max_chars_inline   = 0
        line_num           = 0
        unknown_char_index = vocab.index('<UNK>')
        start_char_index   = vocab.index('<S>')
        end_char_index     = vocab.index('</S>')
        for line in raw_text:
            char_indicies = []
            char_indicies.append(start_char_index)
            for word in line.split():
                chars = list(word)
                for char in chars:
                    try:
                        char_indx = vocab.index(char)
                    except ValueError:
                        char_indx = unknown_char_index
                    char_indicies.append(char_indx)
            char_indicies.append(end_char_index)
            
            text_vec.append(char_indicies)
            # Calculating max word in a line
            if len(char_indicies) > max_chars_inline:
                max_chars_inline = len(char_indicies)

            # Print a dot every 100 lines
            line_num += 1
            if line_num % 100 == 0:
                print('.', end='', flush=True)

        if max_chars_limit < max_chars_inline:
            print("\nWarning: max_chars_limit (%d) is less than max chars (%d) found in a line" % (max_chars_limit, max_chars_inline))
            exit()
        
        # Make all lines same length as max + buffer ( append zeroes to the end )
        total_lines     = line_num
        self._text      = np.zeros((total_lines, max_chars_limit), dtype='int32')
        for line_num in range(total_lines):
            lineLen = len(text_vec[line_num])
            self._text[line_num, 0:lineLen-1] = text_vec[line_num]
            self._text[line_num, lineLen:] = end_char_index
        
        # hot vector for labels
        self.__vectorize_labels()

        time_taken = time.time() - start_time
        print("\n %d lines of text vectorized in %.3f secs!" % (len(self._text), time_taken))

        # Set vectorized as true
        self._isVectorized = True
    
    def __vectorize_labels(self):
        labels_map = {}
        labels_vector = []
        for label in self._labels:
            label_vec = []
            for ref in self.labels_list:
                if ref == label:
                    label_vec.append(1)
                else:
                    label_vec.append(0)
            
            if label not in labels_map:
                labels_map[label] = label_vec
            
            labels_vector.append(label_vec)
        
        self._vec_labels = labels_vector
        return True
        
#################################################### Child class to laod data from file #######################################################
    
class LoadTSV(DataSet):

    def __init__ (self, data_file_path, labels_list=False, test=False):
        self._text_token_flags = re.MULTILINE | re.DOTALL

        start_time = time.time()
        print("Loading File:", data_file_path)
        text, labels = self.__load_tsv_file(data_file_path, test)
        time_taken = time.time() - start_time
        print("%s Loaded: %.3f secs!" % (data_file_path, time_taken))
        
        if labels_list:
            self._labels_list  = labels_list
        else:
            self._labels_list  = list(set(labels))

        super().__init__(text, labels)

    def __load_tsv_file(self, filename, test):
        file = open(filename)
        read_text, read_labels = [], []

        max_char_len = 0
        for line in file:

            if test:
                c_row = self.__cleanup(line.strip())
            else:
                row = line.strip().split('	')
                c_row = self.__cleanup(row[1])
                read_labels.append(row[0])
            read_text.append(c_row)
            
        return read_text, read_labels

    def __cleanup(self, string):
        ''' Cleans up the given tweet '''
        
        #convert to unicode
       # string = unicode(string, 'utf-8')

        #encode it with string escape
       # string = string.encode('unicode_escape')

        # function so code less repetitive
        def re_sub(pattern, repl):
            try:
                return re.sub(pattern, repl, string)
            except ValueError:
                return string

        # Different regex parts for smiley faces
        eyes = r"[8:=;]"
        nose = r"['`\-]?"

        string = re_sub(r"https?:\/\/\S+\b|www\.(\w+\.)+\S*", "")  # urls
        string = re_sub(r"@\w+", " ")  # @user
        string = re_sub(r"#\S+", " ")  # #hashtag
        string = re_sub(r"/"," / ")   # 
        string = re_sub(r"{}{}[)dD]+|[)dD]+{}{}".format(eyes, nose, nose, eyes), " ")  # smile
        string = re_sub(r"{}{}p+".format(eyes, nose), " ") #lol face
        string = re_sub(r"{}{}\(+|\)+{}{}".format(eyes, nose, nose, eyes), " ") # sad face
        string = re_sub(r"{}{}[\/|l*]".format(eyes, nose), " ") # neutralface
        string = re_sub(r"<3"," ")  # heart
        string = re_sub(r"[-+]?[.\d]*[\d]+[:,.\d]*", " ") # numbers

        
        return string