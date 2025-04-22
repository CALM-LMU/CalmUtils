import re


def split_str_digit(s):
    """
    split s into numeric (integer) and non-numeric parts
    return split as a tuple of ints and strings
    """
    res = []
    for m in re.finditer('(\d*)(\D*)', s):
        for g in m.groups():
            if g != '':
                try:
                    res.append(int(g))
                except ValueError:
                    res.append(g)
    return tuple(res)