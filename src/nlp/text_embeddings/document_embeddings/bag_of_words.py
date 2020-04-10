from nlp.utils import compute_document_word_count, cosine_similarity
from nlp.text_embeddings.embedding import EmbeddingAlgorithm
from torch import LongTensor, Tensor
from nlp.data.corpus import Corpus
from typing import Union, List
import numpy as np
import sys


class BagOfWords(EmbeddingAlgorithm):
    def __init__(self, corpus: Corpus):
        """
        This class computes the bag of words embedding representation of documents for a given corpus.

        Parameters
        ----------
        corpus: Corpus, for which the bag of words representation should be computed.
        """

        # call the constructor of the super class in order to initialize the BagOfWords class as EmbeddingAlgorithm
        super(BagOfWords, self).__init__(corpus, None)

        # compute the bag of words embeddings
        self.embedding = self.compute_embedding()

    def document2embedding(self, document: Union[str, List[Union[str, List[str]]]], preprocessing: bool = False,
                                 embedding: Union[None, np.ndarray] = None, ) -> np.ndarray:
        """
        Computes a bag of words document embedding based on a word count dictionary.

        Parameters
        ----------
        document:
        embedding: (Optional), which should be updated. If no embedding tensor is provided, a new Tensor is created.
        preprocessing: Indicates, whether we need to preprocess the provided document or not.

        Returns
        -------

        embedding: Embedding Tensor, in which the word counts, provided by the count dict, are included.
        """

        if preprocessing:
            document = self.corpus.preprocessor.preprocess_corpus([document])[0]

        # compute the word counts for the current document
        word_count_dict = compute_document_word_count(document)

        # create a new embedding, if no embedding was provided.
        if embedding is None:
            embedding = np.zeros((1, len(self.corpus.voc)))

        # iterate through the count dict and increase the count at the corresponding index in the embedding tensor
        for word, word_count in word_count_dict.items():
            index = self.corpus.voc.word2index[word]
            embedding[0, index] += word_count
        return embedding


    def compute_embedding(self) -> np.ndarray:
        """
        Computes the Bag of words representation of the current corpus.

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
        print('Computing Bag of Words document embeddings...')
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
        similarity: A float value representing the similarity between the words
                    (common choice: Cosine Similarity).
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
