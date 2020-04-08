from nlp.data.preprocessing import tokenize_words
from typing import Union
import numpy as np
import scipy as sc


def sent2bag_of_words(word2index: dict, sent: Union[str, list], max_count: int=1, normalize: bool=False) -> np.ndarray:
    """
    Parameters
    ----------
        word2index: Dictionary, which maps words to their corresponding indices.
        sent: Word representation of a sentence.
        max_count: Maximal count, which a certain vocabulary can have.
        normalize: Variable, which indicates whether to normalize the vector or not.
    Returns
    -------
        bag_of_words: Bag of words representation of the sentence.
    """

    # convert the sentence into its list representation, if it is represented as string
    if type(sent) is str:
        sent = tokenize_words(sent)

    total_num_words = len(word2index)
    bag_of_words = np.zeros((1, total_num_words))

    # for each word in the current sentence, get its index and increase the counter on its position in the vector
    for word in sent:
        if word in word2index:
            index = word2index[word]
            bag_of_words[0, index] +=1

    # clip the vector, if required
    if max_count > 0:
        bag_of_words = np.clip(bag_of_words, a_min=0, a_max=max_count)

    # normalize the vector, if required
    if normalize:
        bag_of_words -= bag_of_words.min()
        bag_of_words = bag_of_words / bag_of_words.max()
    return bag_of_words


def sent2cooc_matrix(word2index: dict, sent: Union[str, list], window_size: int=5, weighted: bool=True) -> np.ndarray:
    """
    """

    vocab_size = len(word2index)
    cooc_matrix = np.zeros((vocab_size, vocab_size))

    # convert the sentence into its list representation, if it is represented as string
    if type(sent) is str:
        sent = tokenize_words(sent)

    # iterate through the words in the sentence
    for word_in_sentence_index, word in enumerate(sent):
        if word in word2index:
            row = word2index[word]

            # get the indices of the neighbours within the window size
            min_neighbour_index = word_in_sentence_index - window_size
            max_neighbour_index = word_in_sentence_index + window_size + 1
            if min_neighbour_index < 0:
                min_neighbour_index = 0
            if max_neighbour_index > len(sent) - 1:
                max_neighbour_index = len(sent)

            # iterate through the neighbours of the current word
            for neighbour_index in range(min_neighbour_index, max_neighbour_index):
                if neighbour_index != word_in_sentence_index:
                    neighbour_word = sent[neighbour_index]

                    if neighbour_word in word2index:
                        neighbour_word_index = word2index[neighbour_word]
                        if weighted:
                            weight = 1./abs(neighbour_index - word_in_sentence_index)
                            cooc_matrix[row, neighbour_word_index] += weight
                        else:
                            cooc_matrix[row, neighbour_word_index] += 1

    return cooc_matrix


def doc2cooc_matrix(word2index: dict, document: list, window_size: int=5, weighted: bool=False):
    """
    """

    vocab_size = len(word2index)
    cooc_matrix = np.zeros((vocab_size, vocab_size))
    for sentence in document:
        sentence_cooc_matrix = sent2cooc_matrix(word2index, sentence, window_size, weighted)
        cooc_matrix += sentence_cooc_matrix
    return cooc_matrix


def corpus2cooc_matrix(word2index: dict, corpus: list, window_size: int=5, weighted: bool=False):
    """
    """

    vocab_size = len(word2index)
    cooc_matrix = np.zeros((vocab_size, vocab_size))

    for document in corpus:
        document_cooc_matrix = doc2cooc_matrix(word2index, document, window_size, weighted)
        cooc_matrix += document_cooc_matrix
    return cooc_matrix


def reduce_dimensionality(cooc_matrix: np.ndarray, k: int):
    """
    """

    u, s, v = sc.sparse.linalg.svds(cooc_matrix, k=k)
    embeddings = u # np.matmul(u, s)
    # TODO: Test
    return embeddings
