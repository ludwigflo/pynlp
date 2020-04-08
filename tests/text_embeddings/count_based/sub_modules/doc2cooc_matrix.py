import numpy as np
from copy import deepcopy


def get_test_cases():
    """
    """

    word2index = {'Das': 1, 'der': 2, 'erste': 3, 'Test': 4, 'Satz': 5, 'zweiter': 6}
    for key in word2index:
        word2index[key] -= 1

    documents = [
                 [['Das', 'ist', 'der', 'erste', 'Test', 'Satz'],
                  ['Das', 'ist', 'der', 'erste', 'Test', 'Satz']],
                 ['Das ist der erste Test Satz',
                  'Das ist der erste Test Satz']
                ]
    window_sizes = [1, 2]
    weighted_list = [False, True]

    output_sent1 = [[[0, 0, 0, 0, 0, 0],
                     [0, 0, 2, 0, 0, 0],
                     [0, 2, 0, 2, 0, 0],
                     [0, 0, 2, 0, 2, 0],
                     [0, 0, 0, 2, 0, 0],
                     [0, 0, 0, 0, 0, 0]],

                    [[0, 2, 0, 0, 0, 0],
                     [2, 0, 2, 2, 0, 0],
                     [0, 2, 0, 2, 2, 0],
                     [0, 2, 2, 0, 2, 0],
                     [0, 0, 2, 2, 0, 0],
                     [0, 0, 0, 0, 0, 0]],

                    [[0, 0, 0, 0, 0, 0],
                     [0, 0, 2, 0, 0, 0],
                     [0, 2, 0, 2, 0, 0],
                     [0, 0, 2, 0, 2, 0],
                     [0, 0, 0, 2, 0, 0],
                     [0, 0, 0, 0, 0, 0]],

                    [[0, 1, 0, 0, 0, 0],
                     [1, 0, 2, 1, 0, 0],
                     [0, 2, 0, 2, 1, 0],
                     [0, 1, 2, 0, 2, 0],
                     [0, 0, 1, 2, 0, 0],
                     [0, 0, 0, 0, 0, 0]]]
    output_sent1 = [np.asarray(x) for x in output_sent1]

    params = []
    output_list = []
    for sentence in documents:
        for i, weighted in enumerate(weighted_list):
            for j, window_size in enumerate(window_sizes):
                params.append((word2index, sentence, window_size, weighted))
                output_list.append(output_sent1[2*i+j])
    return params, output_list

if __name__ == '__main__':
    get_test_cases()