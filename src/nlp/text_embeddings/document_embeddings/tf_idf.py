from nlp.text_embeddings.embedding import EmbeddingAlgorithm
from nlp.utils import cosine_similarity
from torch import LongTensor, Tensor
from collections import defaultdict
from nlp.data.corpus import Corpus
from typing import Union, List
import numpy as np
import sys


class TfIdf(EmbeddingAlgorithm):

    def __init__(self, corpus: Corpus, log_norm: bool = False):
        """

        Parameters
        ----------
        corpus
        log_norm
        """

        super(TfIdf, self).__init__(corpus, None)
        self.log_norm = log_norm

        # compute the idf values for all word terms
        print('Compute idf values...')
        self.idf_dict = self.compute_idf_values()
        print('Done!\n')

    def compute_tf_values(self, document: List[List[str]]):
        """

        Parameters
        ----------
        document: List representation of a document.

        Returns
        -------
        tf_dict: Dictionary containing the term frequencies.
        """

        tf_dict = defaultdict(lambda: 0)

        # iterate through the sentences in the document
        total_num_words = 0
        for sentence in document:

            # compute the total number of words in the document
            total_num_words += len(sentence)

            # compute the word count of the current sentence
            for word in sentence:
                tf_dict[word] += 1

        # normalize the frequencies
        for word, value in tf_dict.items():
            if self.log_norm:
                tf_dict[word] = np.log(1 + value)
            else:
                tf_dict[word] = float(value) / total_num_words

        return tf_dict

    def compute_idf_values(self) -> dict:
        """
        Computes the inverse document frequency for all words in the corpus.

        Returns
        -------
        idf_dict: Dictionary containing the idf values of each term in the corpus.
        """

        num_docs = len(self.corpus)
        idf_dict = defaultdict(lambda : 0)

        # iterate through the corpus and check, which words occur in the current document.
        for document in self.corpus.corpus:

            # initialize a set, in which the words of the current documents are stored
            doc_words = set()
            for sentence in document:
                for word in sentence:
                    doc_words.add(word)

            # increase the document term count by 1 for each word in the doc_words set
            for word in doc_words:
                idf_dict[word] += 1

        # compute the idf values
        for word, value in idf_dict.items():
            idf_dict[word] = np.log(num_docs/float(value))
        return idf_dict

    def document2embedding(self, document: Union[str, List[Union[str, List[str]]]],
                           preprocessing: bool = False) -> np.ndarray:
        """
        Computes a tf-idf document embedding based on a word count dictionary.

        Parameters
        ----------
        document: Representation of a document.
        preprocessing: Indicates, whether we need to preprocess the provided document or not.

        Returns
        -------

        embedding: Embedding Tensor, in which the word counts, provided by the count dict, are included.
        """

        embedding = np.zeros((1, len(self.corpus.voc)))

        if preprocessing:
            document = self.corpus.preprocessor.preprocess_corpus([document])[0]

        # get the term frequencies for the current document
        tf_dict = self.compute_tf_values(document)

        # store the tf-idf values into the embedding matrix
        for word, tf in tf_dict.items():

            # compute the tf-idf values
            tf_idf_value = tf * self.idf_dict[word]

            # get the index of the current word and store the value in the embedding matrix
            word_index = self.corpus.voc.word2index(word)
            embedding[0, word_index] = tf_idf_value
        return embedding


    def compute_embedding(self, *args, **kwargs) -> np.ndarray:
        """
        Computes the tf-idf representation of the current corpus.

        Returns
        -------
        embeddings: Document embeddings for all documents in the provided corpus. The shape of these embeddings is:
                    number of documents x number of words.
        """

        # get the dimension of the document embedding matrix and initialize the embeddings
        num_documents = len(self.corpus)
        num_words = len(self.corpus.voc)
        embeddings = np.zeros((num_documents, num_words))

        # compute the embedding based on the word dict and add it to its position in the embeddings array
        print('Computing TF-IDF document embeddings...')
        for i, document in enumerate(self.corpus.corpus):

            # print the current status
            i = float(i)
            out_string = "\rDocuments: {0}%".format(int((i+1) / len(self.corpus.corpus) * 100))
            sys.stdout.write(out_string)
            sys.stdout.flush()
            i = int(i)

            # compute the current embedding
            embeddings[i:i+1, ...] = self.document2embedding(document)

        print('\n')
        return embeddings

    # noinspection PyCallingNonCallable
    def compute_similarity(self, doc1: Union[list, str, int], doc2: Union[list, str, int]) -> float:
        """
        Computes the similarity between two documents, which are either provided by their string value or
        by their index (int) in the vocabulary. It is also possible to provide a list of string or integer values for
        both of these documents

        Parameters
        ----------
        doc1: First text representation (or list of text representations), which should be compared to the other
              text representation. In case of a list representation, the corresponding embeddings are averaged.
        doc2: Second text representation (or list of text representations), which should be compared to the other
              text representation. In case of a list representation, the corresponding embeddings are averaged.

        Returns
        -------
        similarity: A float value representing the Cosine Similarity between the words
        """

        # compute the average embeddings of document 1 and document 2
        if type(doc1) != list:
            doc1 = [doc1]
        doc1 = [self.document2embedding(x, True) if type(x) == str else self.embedding(LongTensor([x])) for x in doc1]
        doc1 = np.concatenate([x.detach().cpu().numpy() if type(x) == Tensor else x for x in doc1], axis=0)
        doc1 = np.mean(doc1, axis=0)

        if type(doc2) != list:
            doc2 = [doc2]
        doc2 = [self.document2embedding(x, True) if type(x) == str else self.embedding(LongTensor([x])) for x in doc2]
        doc2 = np.concatenate([x.detach().cpu().numpy() if type(x) == Tensor else x for x in doc2], axis=0)
        doc2 = np.mean(doc2, axis=0)

        # compute the cosine similarity between these average vectors
        similarity = cosine_similarity(doc1, doc2)
        return similarity
