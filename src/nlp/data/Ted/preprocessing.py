import re


def remove_speaker_names(line: str):
    """
    Estimates speaker's names by some heuristics (not necessary optimal).

    Parameters
    ----------
        line: String representing one line of the transcribed TED talk data.
    """

    m = re.match(r'^(?:(?P<precolon>[^:]{,20}):)?(?P<postcolon>.*)$', line)
    # output = m.groupdict()['postcolon']
    output = [sent for sent in m.groupdict()['postcolon'].split('.') if sent]
    return output
