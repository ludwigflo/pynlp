from typing import Union, Tuple, List
from abc import ABC, abstractmethod
from nlp.data.corpus import Corpus
from torch.nn import Embedding
from torch import Tensor
import numpy as np
import pickle
import torch


class EmbeddingAlgorithm(ABC):

    def __init__(self, corpus: Corpus, embedding: Union[None, np.ndarray, Tensor, Embedding] = None) -> None:
        """
        Abstract class, which represents an interface for different kinds of embedding algorithms like character -,
        word - or document embeddings. The embedding algorithms combine test representation (in form of a corpus Object)
        with a numerical representation (in form of an embedding layer). It defines abstract methods for training the
        embedding layer with respect to a provided corpus and for computing similarities between embeddings.

        Parameters
        ----------
            corpus: A Corpus object, which is used to represent the corpus in text form and provides the vocabulary.
            embedding: Embedding layer (optional), which is used to represent the vocabulary in numeric form.
        """

    # ------------------------------- Attributes -------------------------------
        # attributes
        self.dim = None
        self.num_embedding_units = None

        # private attributes
        self._corpus = None
        self._embedding = None

        # properties attributes
        self.corpus = corpus
        self.embedding = embedding

    # -------------------------- Property definitions --------------------------
    @property
    def corpus(self) -> Corpus:
        """
        Getter method for the Corpus object.

        Returns
        -------
            self.corpus: Corpus object, representing the corpus (together with its vocabulary) in text form.
        """
        return self._corpus

    @corpus.setter
    def corpus(self, corpus: Corpus) -> None:
        """
        Setter Function for te Corpus object. Embeddings try to embed parts of the corpus in a numeric form. The corpus,
        together with its corresponding embedding, represent the complete data, which can be further used by other
        algorithms to process the text.

        Parameters
        ----------
            corpus: Corpus object, representing the corpus (together with its vocabulary in text form).
        """
        self._corpus = corpus

    @property
    def embedding(self) -> Embedding:
        """
        Getter method for the embedding layer.

        Returns
        -------
            self._embedding: Embedding layer, which defines the numeric representation of the vocabulary.
        """
        return self._embedding

    @embedding.setter
    def embedding(self, embedding: Union[None, np.ndarray, Tensor, Embedding]) -> None:
        """
        Setter Function for te embedding layer. The embedding layer embeds parts of the corpus to a numeric
        representation.

        Parameters
        ----------
            embedding: Representation of an embedding (or None), which should be used to initialize the embedding layer.
        """

        # initialize the embedding either with a provided matrix or with None.
        if isinstance(embedding, np.ndarray):
            embedding = torch.from_numpy(embedding)
        if isinstance(embedding, Tensor):
            assert len(embedding.size()) == 2, "You must provide a 2 dimensional Tensor as embedding!"
            n_vocab, embedding_dim = embedding.size()
            self._embedding = Embedding(n_vocab, embedding_dim)
            self._embedding.weight.data = embedding
            num_embedding_units, dim = self._embedding.weight.data.size()
        elif isinstance(embedding, Embedding):
            self._embedding = embedding
            num_embedding_units, dim = self._embedding.weight.data.size()
        else:
            # if the embedding is provided as None, initalize everything as None
            self._embedding, num_embedding_units, dim = None, None, None

        # store the dimension and the number of embedding units as attributes of the class
        self.num_embedding_units = num_embedding_units
        self.dim = dim

    # ---------------------------- Abstract methods ----------------------------
    @abstractmethod
    def compute_embedding(self, *args, **kwargs) -> Union[np.ndarray, Tensor, Embedding]:
        """
        Trains / computes and returns the word embedding on the provided corpus, if the corpus is not empty.
        """
        raise NotImplementedError

    @abstractmethod
    def compute_similarity(self, text1: Union[list, str, int], text2: Union[list, str, int]) -> float:
        """
        Computes the similarity between two words, which are either provided by their string value or
        by their index in the vocabulary.

        Parameters
        ----------
            text1: First text representation (or list of text representations), which should be compared to the other
                   text representation. In case of a list representation, the corresponding embeddings are averaged.
            text2: Second text representation (or list of text representations), which should be compared to the other
                   text representation. In case of a list representation, the corresponding embeddings are averaged.

        Returns
        -------
            similarity: A float value representing the similarity between the words
                        (common choice: Cosine Similarity).
        """
        raise NotImplementedError

    # -------------------------------- methods ---------------------------------
    def set_embedding_gradient(self, requires_grad: bool = False) -> None:
        """
        Allows the embedding layer to receive gradients from the optimization procedure.

        Parameters
        ----------
            requires_grad: Variable, indicating whether to allow gradients for the embedding layer or not.
        """
        self._embedding.weight.requires_grad = requires_grad

    def compute_index_list(self, text_list: list) -> list:
        """
        Computes a list of indices from a list of text representations.

        Parameters
        ----------
            text_list: List of Text, which should be represented as a list of indices.

        Returns
        -------
            text_list: List containing the corresponding indices of the input text.
        """

        # convert the positive and negative words into a list of indices (integer representation)

        text_list = [self.corpus.voc.word2index(x) if type(x) is str else x for x in text_list]
        return text_list

    def compute_nearest_words(self, positive: Union[None, list, str, int], negative: Union[None, list, str, int],
                              n: int = 1) -> List[Tuple[str, int, float]]:
        """
        Searches for the n nearest words given a positive text representation (or a list of positive text
        representations) and a negative text representation (or a list of negative text representations). 'Nearest'
        is defined by the similarity method of the class (it is common to  use the cosine similarity). The similarity
        method computes its similarity based on the vector representation the embeddings, which correspond to the
        provided text representation.

        For the typical example like "king" - "man" + "women" = x ? (where the word queen should be the correct asnwere)
        we can provide as positive list ["king", "woman"] and as negative list ["man"].

        Parameters
        ----------
        positive: List of words, which should be considered as positve in the computation of the most similar words.
        negative: List of words, which should be considered as negative in the computation of the most similar words.
        n: Number of most similar words, which should be computed.

        Returns
        -------
            similarity_list: List of tuples, representing the most similar words. The tuple contains the index of the
                             word in the vocabulary dictionary, the word itself as string representation as well as the
                             similarity value computed by the similarity function for comparing word similarities.
        """

        # convert the positive and negative text representations into their indices
        positive, negative = self.compute_index_list(positive), self.compute_index_list(negative)

        # compute the similarity for the input for each word, which is not in the positive, negative or token_list list
        similarity_list = []
        tokens_list = [self.corpus.voc.pad_token, self.corpus.voc.sos_token, self.corpus.voc.eos_token]

        # iterate through the vocabulary indices
        for index in self.corpus.voc.index2word:

            # if the current word index is not a sos, eos or pad token, and not one of the requested words
            if (index not in tokens_list) and (index not in positive) and (index not in negative):

                # Distance between [sum(positive words) - sum(negative words)] and [index] (index of our current word)
                # is equal to the distance between [sum(positive words)] and [sum((index + negative words)]
                # therefore we combine the current word with the negative list in order to compute the distance
                neg = [].extend(negative).append(index)
                similarity_score = self.compute_similarity(positive, neg)
                similarity_list.append((index, self.corpus.voc.index2word[index], similarity_score))

        similarity_list = sorted(similarity_list, key=lambda x: x[2], reverse=True)[:n]
        return similarity_list

    def save_object(self, file_name: str) -> None:
        """
        Saves the current EmbeddingAlgorithm object to a pickle file.

        Parameters
        ----------
            file_name: File name, in which the EmbeddingAlgorithm object should be saved.
        """

        with open(file_name, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load_object(file_name) -> 'EmbeddingAlgorithm':
        """
        Loads a EmbeddingAlgorithm object.

        Parameters
        ----------
            file_name: File name, under which the EmbeddingAlgorithm object is saved.
        """

        with open(file_name, 'rb') as f:
            # load the provided pickle file
            embedding_object = pickle.load(f)
        return embedding_object
