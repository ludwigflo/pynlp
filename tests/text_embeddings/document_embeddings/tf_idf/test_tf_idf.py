from nlp.text_embeddings.document_embeddings import tf_idf
from nlp.data.corpus import Corpus
import numpy as np
import unittest


class TestTfIdf(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """
        Defines a toy corpus, which is used as a base for our BagOfWords class in all of our tests.
        """

        # define a parameter file
        param_file = 'test.yaml'

        # create a toy corpus
        corpus_list = [[['first', 'sentence', 'and', 'first', 'document'],
                        ['second', 'sentence', 'in', 'first', 'document']],

                       [['first', 'sentence', 'and', 'second', 'document'],
                        ['second', 'sentence', 'and', 'second', 'document']],
                       [['a', 'third', 'document']]]

        # store the toy corpus
        cls.corpus = Corpus(corpus_name = 'Toy Corpus', param_file = param_file, corpus_list = corpus_list)


    def test_compute_tf_values(self):
        self.emb = tf_idf.TfIdf(self.corpus)

        pre0 = self.emb.compute_tf_values(self.corpus.corpus[0])
        tar0 = {'first': 3./10, 'sentence': 2./10, 'and': 1./10, 'document': 2./10, 'second': 1./10, 'in': 1./10}
        self.assertEqual(pre0, tar0)

        pre1 = self.emb.compute_tf_values(self.corpus.corpus[1])
        tar1 = {'first': 1./10, 'sentence': 2./10, 'and': 2./10, 'document': 2./10, 'second': 3./10}
        self.assertEqual(pre1, tar1)

        pre2 = self.emb.compute_tf_values(self.corpus.corpus[2])
        tar2 = {'a': 1./3, 'third': 1./3, 'document': 1./3}
        self.assertEqual(pre2, tar2)

    def test_compute_idf_values(self):
        self.emb = tf_idf.TfIdf(self.corpus)
        tar = {'first': np.log(3./2), 'sentence': np.log(3./2), 'and': np.log(3./2), 'document': np.log(1),
               'second': np.log(3./2), 'in': np.log(3./1), 'a': np.log(3./1), 'third': np.log(3./1)}
        idf_dict = self.emb.compute_idf_values()
        self.assertEqual(tar, idf_dict)

    def test_embedding(self):
        idf_tar = {'first': np.log(3./2), 'sentence': np.log(3./2), 'and': np.log(3./2), 'document': np.log(1),
                   'second': np.log(3./2), 'in': np.log(3./1), 'a': np.log(3./1), 'third': np.log(3./1)}

        tf0_tar = {'first': 3./10, 'sentence': 2./10, 'and': 1./10, 'document': 2./10, 'second': 1./10, 'in': 1./10}
        tf1_tar = {'first': 1./10, 'sentence': 2./10, 'and': 2./10, 'document': 2./10, 'second': 3./10}
        tf2_tar = {'a': 1./3, 'third': 1./3, 'document': 1./3}

        self.emb = tf_idf.TfIdf(self.corpus)
        tf_idf_array = self.emb.compute_embedding()

        tf_idf_tar0 = np.asarray([0, 0, 0,
                                  tf0_tar['first']*idf_tar['first'],
                                  tf0_tar['sentence']*idf_tar['sentence'],
                                  tf0_tar['and'] * idf_tar['and'],
                                  tf0_tar['document'] * idf_tar['document'],
                                  tf0_tar['second'] * idf_tar['second'],
                                  tf0_tar['in'] * idf_tar['in'],
                                  0, 0])

        np.testing.assert_array_equal(tf_idf_array[0, ...], tf_idf_tar0)

        tf_idf_tar1 = np.asarray([0, 0, 0,
                                  tf1_tar['first']*idf_tar['first'],
                                  tf1_tar['sentence']*idf_tar['sentence'],
                                  tf1_tar['and'] * idf_tar['and'],
                                  tf1_tar['document'] * idf_tar['document'],
                                  tf1_tar['second'] * idf_tar['second'],
                                  0, 0, 0])
        np.testing.assert_array_equal(tf_idf_array[1, ...], tf_idf_tar1)

        tf_idf_tar2 = np.asarray([0, 0, 0, 0, 0, 0,
                                  tf2_tar['document'] * idf_tar['document'],
                                  0, 0,
                                  tf2_tar['a'] * idf_tar['a'],
                                  tf2_tar['third'] * idf_tar['third']])
        np.testing.assert_array_equal(tf_idf_array[2, ...], tf_idf_tar2)



if __name__ == '__main__':
    unittest.main()
