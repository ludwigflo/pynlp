from collections import OrderedDict


def get_test_cases() -> tuple:
    """
    Returns
    -------
        output:
    """

    Corpus = [[['Das', 'ist', 'der', 'erste', 'Satz', 'im', 'ersten', 'Dokument'],
                ['Hier', 'ist', 'ein', 'zweiter', 'Satz', 'im', 'ersten', 'Dokument'],
                ['Schließlich', 'noch', 'ein', 'dritter', 'Satz', 'im', 'ersten', 'Dokument']],
              [['Das', 'ist', 'der', 'erste', 'Satz', 'im', 'zweiten', 'Dokument'],
               ['Hier', 'ist', 'ein', 'zweiter', 'Satz', 'im', 'zweiten', 'Dokument'],
               ['Schließlich', 'noch', 'ein', 'dritter', 'Satz', 'im', 'zweiten', 'Dokument']]
              ]

    target = {'Das': 2, 'ist': 4, 'der': 2, 'erste': 2, 'Satz': 6, 'Hier': 2,
              'ein': 4, 'zweiter': 2, 'Schließlich': 2, 'noch': 2, 'dritter': 2,
              'im': 6, 'ersten': 3, 'zweiten': 3, 'Dokument': 6}
    target = OrderedDict(sorted(target.items(), key=lambda x: x[1], reverse=True))
    output = ([Corpus], [target])
    return output
