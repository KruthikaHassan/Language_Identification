from data_preprocess import DataSet
from build_vocab import Vocab 

def main(data_file_path):

    data_set = DataSet(data_file_path)

    text_subset = data_set.text[0:100]

    voc = Vocab(text_subset)

    return voc.vocab_size, voc.percent_oov

size, oov = main('train.tsv')

