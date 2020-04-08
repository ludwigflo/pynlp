def get_test_cases() -> tuple:
    """
    Returns
    -------
        output:
    """

    test_strings = ['Das ist ein ganz normaler test satz', 'Das ist ein schwerer test satz, datei.txt.']
    target = [['Das', 'ist', 'ein', 'ganz', 'normaler', 'test', 'satz'],
              ['Das', 'ist', 'ein', 'schwerer', 'test', 'satz', ',', 'datei.txt', '.']]

    output = (test_strings, target)
    return output
