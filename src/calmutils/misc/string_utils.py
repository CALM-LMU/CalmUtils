def pad(s, length=3, pad_char='0'):
    """
    left-pad a string to desired length with given padding char
    """
    s = str(s)
    return pad_char * (length - len(s)) + s