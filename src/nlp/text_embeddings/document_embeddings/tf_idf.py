from nlp.text_embeddings.embedding import EmbeddingAlgorithm
from torch import LongTensor, Tensor
from collections import defaultdict
from nlp.data.corpus import Corpus
from typing import Union, List
import numpy as np


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




    def compute_embedding(self, *args, **kwargs) -> np.ndarray:
        """
        Computes the tf-idf representation of the current corpus.

        Returns
        -------
        embeddings: Document embeddings for all documents in the provided corpus. The shape of these embeddings is:
                    number of documents x number of words.
        """
