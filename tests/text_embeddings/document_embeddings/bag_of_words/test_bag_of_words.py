from nlp.text_embeddings.document_embeddings import bag_of_words
from nlp.text_embeddings.embedding import EmbeddingAlgorithm
from nlp.data.corpus import Corpus
import torch_testing as tt
import torch.nn as nn
import unittest
import torch
import os

class TestBagOfWords(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """
        Defines a toy corpus, which is used as a base for our BagOfWords class in all of our tests.
        """

        # define a parameter file
        param_file = 'test.yaml'

        # create a toy corpus
        corpus_list = [[['Das', 'ist', 'der', 'erste', 'Satz', 'im', 'ersten', 'Dokument'],
                        ['Das', 'ist', 'der', 'zweite', 'Satz', 'im', 'ersten', 'Dokument']],
                       [['Das', 'ist', 'der', 'erste', 'Satz', 'im', 'zweiten', 'Dokument'],
                        ['Das', 'ist', 'der', 'zweite', 'Satz', 'im', 'zweiten', 'Dokument']]]

        # store the toy corpus
        cls.corpus = Corpus(corpus_name = 'Toy Corpus', param_file = param_file, corpus_list = corpus_list)
        cls.bow = bag_of_words.BagOfWords(cls.corpus)
        print()
        print()


    @classmethod
    def tearDownClass(cls):
        """
        """
        pass

    def setUp(self):
        """
        """
        pass

    def tearDown(self):
        """
        """
        pass

    def test_constructor_property_types(self):
        """
        Test, if the property types after object instanciation are correct.
        """
        print("test_constructor_property_types")
        self.assertIs(type(self.bow.embedding), nn.Embedding)
        self.assertIs(type(self.bow.corpus), Corpus)

    def test_constructor_corpus_property(self):
        """
        Test, if the corpus, which was constructed, is stored correctly.
        """
        print("test_constructor_corpus_property")
        self.assertEqual(len(self.bow.corpus), 2)
        self.assertEqual(len(self.bow.corpus.voc.word2index), 13)
        self.assertEqual(len(self.bow.corpus.voc.index2word), 13)
        self.assertEqual(len(self.bow.corpus.voc), 13)

    def test_constructor_embedding_property(self):
        """
        Test, if the computed embeddings are correct.
        """

        print("test_constructor_embedding_property")
        embeddings = self.bow.embedding
        self.assertEqual(embeddings.weight.data.size()[0], 2)
        self.assertEqual(embeddings.weight.data.size()[1], 13)

        target_tensor0 = torch.Tensor([[0, 0, 0, 2, 2, 2, 1, 2, 2, 2, 2, 1, 0]])
        target_tensor1 = torch.Tensor([[0, 0, 0, 2, 2, 2, 1, 2, 2, 0, 2, 1, 2]])

        tt.assert_equal(embeddings(torch.LongTensor([0])), target_tensor0)
        tt.assert_equal(embeddings(torch.LongTensor([1])), target_tensor1)

    def test_compute_similarity(self):
        """
        Tests the similarity function of the BagOfWord class.
        """
        print("test_compute_similarity")
        sim_target1 = 26./30.
        doc1 = 0
        doc2 = 1
        self.assertEqual(self.bow.compute_similarity(doc1, doc2), sim_target1)

        doc1 = 'Das ist der erste Satz im ersten Dokument. Das ist der zweite Satz im ersten Dokument.'
        doc2 = 'Das ist der erste Satz im zweiten Dokument. Das ist der zweite Satz im zweiten Dokument.'
        self.assertEqual(self.bow.compute_similarity(doc1, doc2), sim_target1)

        doc1 = [0]
        doc2 = [1]
        self.assertEqual(self.bow.compute_similarity(doc1, doc2), sim_target1)

        doc1 = ['Das ist der erste Satz im ersten Dokument.', 'Das ist der zweite Satz im ersten Dokument.']
        doc2 = ['Das ist der erste Satz im zweiten Dokument.', 'Das ist der zweite Satz im zweiten Dokument.']
        self.assertEqual(self.bow.compute_similarity(doc1, doc2), sim_target1)

        sim_target2 = 1.
        doc1 = 0
        doc2 = 0
        self.assertEqual(self.bow.compute_similarity(doc1, doc2), sim_target2)

        doc1 = 1
        doc2 = 1
        self.assertEqual(self.bow.compute_similarity(doc1, doc2), sim_target2)

    def test_saving_and_loading(self):
        """
        Tests the saving and loading capabilities of the corpus object.
        """
        print('Test Saving and Loading')
        file_name = 'test_bow.pickl'

        # if the file exeists in the current directory, then remove it
        file_list = os.listdir('./')
        if file_name in file_list:
            os.remove(file_name)
        # save a the bow instance
        self.bow.save_object(file_name=file_name)
        file_list = os.listdir('./')

        # check if the object was saved in the output directory
        self.assertIn(file_name, file_list)

        # load the bow object again
        bow2 = bag_of_words.BagOfWords.load_object(file_name)
        embeddings = bow2.embedding

        # check if the embeddings are equal to the embeddings of the object, which has previously been stored.
        target_tensor0 = torch.Tensor([[0, 0, 0, 2, 2, 2, 1, 2, 2, 2, 2, 1, 0]])
        target_tensor1 = torch.Tensor([[0, 0, 0, 2, 2, 2, 1, 2, 2, 0, 2, 1, 2]])

        tt.assert_equal(embeddings(torch.LongTensor([0])), target_tensor0)
        tt.assert_equal(embeddings(torch.LongTensor([1])), target_tensor1)

        # load the object by calling the function from its parents class
        bow3 = EmbeddingAlgorithm.load_object(file_name)
        embeddings = bow2.embedding
        tt.assert_equal(embeddings(torch.LongTensor([0])), target_tensor0)
        tt.assert_equal(embeddings(torch.LongTensor([1])), target_tensor1)

if __name__ == '__main__':
    unittest.main()
