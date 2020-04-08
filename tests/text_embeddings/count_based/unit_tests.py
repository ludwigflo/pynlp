from nlp.word_embeddings import count_based
import numpy as np
import importlib
import os


def test_sub_module(sub_module_name: str) -> tuple:
    """
    Parameters
    ----------
        sub_module_name: name of the modules, which should be tested.

    Returns
    -------
        output: Number of positive and number of negative tests.
    """

    # load the submodule which should be tested as well as its corresponding test cases
    sub_module = importlib.import_module('sub_modules.' + sub_module_name)
    (test_cases, expected_outcomes) = sub_module.get_test_cases()
    sub_module_function = getattr(count_based, sub_module_name)

    print()
    print('================================================================================')
    print('Testing function ' + sub_module_name + ' from module preprocessing')
    print('--------------------------------------------------------------------------------')

    # iterate through the test cases, check if the tests are passed, store, report and return the results
    num_positive_tests = 0
    num_negative_tests = 0
    for i, test_case in enumerate(test_cases):
        out_come = sub_module_function(*test_case)

        if np.array_equal(out_come, expected_outcomes[i]):
            num_positive_tests += 1
            print('-------------------------')
            print('Test ' + str(i) + ' Passed')
            print('-------------------------')
        else:
            num_negative_tests += 1
            print()
            print('-------------------------')
            print('Test ' + str(i) + ' Failed')
            print('Target: ')
            print(expected_outcomes[i])
            print('Computed: ')
            print(out_come)
            print('-------------------------')
            print()
    print('-------------------------------------------------')
    print('Summary -   Function Name: ' + sub_module_name + ', ' + str(num_positive_tests) +
          ' Tests passed, ' + str(num_negative_tests) + ' Tests failed')
    print('================================================================================')

    output = (num_positive_tests, num_negative_tests)
    return output


def sub_modules_unit_tests():
    """
    """

    exclude_files = ['__init__.py', '__pycache__']
    path = os.path.realpath(__file__)
    path = path.replace('unit_tests.py', '')
    sub_module_path = path + 'sub_modules/'
    sub_module_names = [x for x in os.listdir(sub_module_path) if x not in exclude_files]

    num_positive_tests, num_negative_tests = 0, 0
    num_positive_modules, num_negative_modules = 0, 0

    for sub_module_name in sub_module_names:
        positive_tests, negative_tests = test_sub_module(sub_module_name.replace('.py', ''))
        num_positive_tests += positive_tests
        num_negative_tests += negative_tests
        if negative_tests == 0:
            num_positive_modules += 1
        else:
            num_negative_modules += 1
    print()
    print()
    print('================================================================================')
    print('Unit Tests of Module Word Embeddings.Count Based - Summary')
    print('--------------------------------------------')
    print('    Total Number of Modules: ' + str(len(sub_module_names)))
    print('    Total Number of Unit Tests: ' + str(num_positive_tests + num_negative_tests))
    print('    Number of modules, which passed the Unit test: ' + str(num_positive_modules))
    print('    Number of modules, which failed the Unit test: ' + str(num_negative_modules))
    print('    Number of unit tests, which passed : ' + str(num_positive_tests))
    print('    Number of unit tests, which failed: ' + str(num_negative_tests))
    print('================================================================================')


if __name__ == '__main__':
    sub_modules_unit_tests()
