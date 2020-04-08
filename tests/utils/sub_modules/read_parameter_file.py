def get_test_cases() -> tuple:
    """
    Returns
    -------
        output:
    """

    file_path = 'sub_modules/dummy_parameters.yaml'
    target = {'parameter1': 10, 'parameter2': 3.74, 'parameter3': 'Hello World', 'parameter4': [1, 2, 3, 'test string'],
              'parameter_group': {'subgroup1': 10, 'subgroup2': ['another', 'list']}}

    output = ([file_path], [target])
    return output