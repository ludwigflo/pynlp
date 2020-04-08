from collections import defaultdict
from typing import List, Dict
import numpy as np
import yaml



def cosine_similarity(vector1: np.ndarray, vector2: np.ndarray) -> float:
    """
    Computes the cosine similarity between two vectors.

    Parameters
    ----------
    vector1: Vector, which is the first candidate for computing the cosine similarity.
    vector2: Vector, which is the second candidate for computing the cosine similarity.

    Returns
    -------
        sim: Cosine similarity between both provided vectors.
    """

    # reshape the vectors into the correct shapes and compute their norms
    vector1, vector2 = vector1.reshape(-1), vector2.reshape(-1)
    norm1, norm2 = np.linalg.norm(vector1), np.linalg.norm(vector2)

    # compute the cosine similarity between these vectors
    sim = np.dot(vector1, vector2)/(norm1*norm2)
    return sim


def merge_dict(dict1: dict, dict2: dict) -> dict:
    """
    Merges two dictionaries into one by adding the values of their common keys.

    Parameters
    ----------
    dict1: First dictionary which should be merged.
    dict2: First dictionary which should be merged.

    Returns
    -------
    out_dict: Merged dictionary, which added common keys.
    """

    # merge all keys into one single dictionary
    out_dict = {**dict1, **dict2}

    # check for keys which occur in both dictionaries and store the some of their corresponding values.
    for key, value in out_dict.items():
        if key in dict1 and key in dict2:
            out_dict[key] = dict1[key] + dict2[key]
    return out_dict


def read_parameter_file(parameter_file_path: str) -> dict:
    """
    Reads the parameters from a yaml file into a dictionary.

    Parameters
    ----------
    parameter_file_path: path to a parameter file, which is stored as a yaml file.

    Returns
    -------
    params: Dictionary containing the parameters defined in the provided yam file
    """

    with open(parameter_file_path, 'r') as f:
        params = yaml.safe_load(f)
    return params


def compute_sentence_word_counts(sentence: List[str]) -> Dict[str, int]:
    """
    Computes the word counts of a sentence.

    Parameters
    ----------
    sentence: Input sentence, from which the single words should be counted.

    Returns
    -------
    count_dict: dictionary, which maps words (keys) to their corresponding counts (values).
    """

    # initialize a default dict, which initializes unknown words with a count of 1
    init_fun = lambda : 0
    count_dict = defaultdict(init_fun)

    # count the words in the current sentence
    for word in sentence:
        count_dict[word] += 1

    return count_dict


def compute_document_word_count(document: List[List[str]]) -> Dict[str, int]:
    """
    Computes the word counts for a complete document.

    Parameters
    ----------
    document: List representation of a document.

    Returns
    -------
    count_dict: dictionary, which maps words (keys) to their corresponding counts (values).
    """

    # initialize the count dict and merge all count dicts for each sentence in the corpus
    count_dict = dict()
    for sentence in document:
        count_dict = merge_dict(count_dict, compute_sentence_word_counts(sentence))
    return count_dict
