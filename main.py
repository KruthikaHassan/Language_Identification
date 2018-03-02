"""
Main function module
"""
#!/usr/bin/env python3

__author__ = "Kruthika H A"
__email__ = "kruthika@uw.edu"

import sys
import numpy as np
import random
import time
from vocab_vectors import VocabVector
from vocab_vectors import build_lang_vocab
from vocab_vectors import build_text_vocab
from data_set import DataSet
from data_set import LoadTSV
from lang_classifier import LangClassifier

class Configuration:
    def print(self):
        attrs = vars(self)
        print("Configuration:")
        for item in attrs:
            print("%s : %s" % (item, attrs[item]))

def main(train_file_path, val_file_path=None, test_file_path=None):
    """ Main function """

    # 131 is max in train data, making it 150 to have some buffer
    max_chars_limit = 150   

    # Load datasets
    train_data_set = LoadTSV(train_file_path)  

    # vocabulary
    text_vocab_vector = build_text_vocab(train_data_set.text)
    lang_vocab_vector = build_lang_vocab(train_data_set.labels_list)

    # vectorize train set
    train_data_set.vectorize_text(text_vocab_vector.vocab, max_chars_limit)

    # Set some config params for this dataset
    config = Configuration()
    config.batchSize     =  4000
    config.lstmUnits     =  12
    config.epochs        =  1
    config.numClasses    =  train_data_set.num_classes
    config.maxSeqLength  =  train_data_set.max_text_length
    config.numDimensions =  text_vocab_vector.dimension
    config.print()

    # Init classifier
    classifier = LangClassifier(config, text_vocab_vector.embeddings, lang_vocab_vector.embeddings)

    if val_file_path:
        val_data_set  = LoadTSV(val_file_path, labels_list=train_data_set.labels_list)
        val_data_set.vectorize_text(text_vocab_vector.vocab, max_chars_limit)

    start_time = time.time()
    print("Starting training, Epochs = ", config.epochs)
    for epoch_num in range(config.epochs):
        classifier.fit_epoch(train_data_set, epoch_num)
        print('.', end='', flush=True)
    run_time = time.time() - start_time
    print("Training completed in %.2f secs!" % (run_time))

    val_preds = classifier.predict(val_data_set)

    print(val_preds[0:10])
    exit()
    
    if val_data_set:
        val_accuracy = classifier.accuracy(val_data_set) * 100
    else:
        val_accuracy = 0
    
    train_accuracy = classifier.accuracy(train_data_set) * 100
    print("%d:%.2f:%.2f" % (epoch_num, train_accuracy, val_accuracy))

    print("")
    exit()
    
    if test_file_path:
        test_data_set  = LoadTSV(test_file_path, labels_list=train_data_set.labels_list, test=True)
        test_data_set.vectorize_text(text_vocab_vector.vocab, max_chars_limit)
    
    
if __name__ == "__main__":
    ''' Start the program here '''
    
    train_file_path = None
    val_file_path   = None
    test_file_path  = None
    
    if len(sys.argv) > 1:
        train_file_path = sys.argv[1]
    if len(sys.argv) > 2:
        val_file_path = sys.argv[2]
    if len(sys.argv) > 3:
        test_file_path = sys.argv[2]
    
    if len(sys.argv) < 2 or len(sys.argv) > 4:
        print("Usage: $python3 main.py train.tsv val.tsv test.tsv")
        exit()

    # Run the program!
    main(train_file_path, val_file_path, test_file_path)