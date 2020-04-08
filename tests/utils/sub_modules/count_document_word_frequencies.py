from collections import OrderedDict


def get_test_cases() -> tuple:
    """
    Returns
    -------
        output:
    """

    document = [['Das', 'ist', 'der', 'erste', 'Satz'],
                ['Hier', 'ist', 'ein', 'zweiter', 'Satz'],
                ['Schließlich', 'noch', 'ein', 'dritter', 'Satz']]

    target = {'Das': 1, 'ist': 2, 'der': 1, 'erste': 1, 'Satz': 3, 'Hier': 1,
              'ein': 2, 'zweiter': 1, 'Schließlich': 1, 'noch': 1, 'dritter': 1}
    target = OrderedDict(sorted(target.items(), key=lambda x: x[1], reverse=True))
    output = ([document], [target])
    return output
