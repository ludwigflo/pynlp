from nlp.data.corpus import Corpus
import os


data_path = 'test_corpus/'
yaml_file = 'test.yaml'
#
# corpus = Corpus(corpus_name='test corpus', param_file=yaml_file)


corpus = Corpus(corpus_name='test corpus', param_file=yaml_file)


for document in corpus.corpus:
    for sentence in document:
        print(sentence)
    print()

print(corpus.voc.word2index)