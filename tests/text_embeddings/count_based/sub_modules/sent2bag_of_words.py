import numpy as np
from copy import deepcopy


def get_test_cases():
    """
    """

    word2index1 = {'Das': 1, 'ist': 2, 'der': 3, 'erste': 4, 'Test': 5, 'Satz': 6, 'ein': 7, 'zweiter': 8, 'in': 9,
                   'dem': 10, 'Wörter': 11, 'wie': 12, 'oder': 13, 'mehrfach': 14, 'vorkommen': 15}
    word2index2 = {'Das': 1, 'der': 2, 'erste': 3, 'Test': 4, 'Satz': 5, 'zweiter': 6, 'in': 7,
                   'dem': 8, 'Wörter': 9, 'wie': 10, 'oder': 11, 'mehrfach': 12, 'vorkommen': 13}
    for key in word2index1:
        word2index1[key] -= 1

    for key in word2index2:
        word2index2[key] -= 1

    sentence1 = ['Das', 'ist', 'der', 'erste', 'Test', 'Satz']
    sentence2 = 'Das ist ein zweiter Test Satz in dem Wörter wie ein oder Das mehrfach vorkommen.'

    output_base1 = [[1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [2, 1, 0, 0, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1]]

    output_base2 = [[1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                    [2, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]

    max_count1 = 1
    max_count2 = -1

    normalize1 = True
    normalize2 = False

    word2index_list = [word2index1, word2index2]
    sentence_list = [sentence1, sentence2]
    max_count_list = [max_count1, max_count2]
    normalize_list = [normalize1, normalize2]

    train_param_list = []
    output_list = []
    for i, word_to_index in enumerate(word2index_list):
        for j, sentence in enumerate(sentence_list):
            for max_count in max_count_list:
                for normalize in normalize_list:
                    train_param_list.append((word_to_index, sentence, max_count, normalize))
                    if i == 0:
                        output_act = deepcopy(output_base1[j])
                    else:
                        output_act = deepcopy(output_base2[j])
                    if max_count > 0:
                        output_act = [min([x, max_count]) for x in output_act]
                    if normalize:
                        min_val = min(output_act)
                        output_act = [x - min_val for x in output_act]
                        max_val = max(output_act)
                        output_act = [float(x) / max_val for x in output_act]
                    output_list.append(np.asarray([output_act]))
    return train_param_list, output_list

if __name__ == '__main__':
    get_test_cases()