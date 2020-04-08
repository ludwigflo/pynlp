def get_test_cases() -> tuple:
    """
    Returns
    -------
        output:
    """

    test_strings = ['Here is a large string. We want to tokenize it into a list of sentences.',
                      'We also try to use different characters like , ..., and others.']
    target = [['Here is a large string.', 'We want to tokenize it into a list of sentences.'],
                        ['We also try to use different characters like , ..., and others.']]
    output = (test_strings, target)
    return output
