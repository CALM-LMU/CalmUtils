import os
import json
import re
from collections.abc import Sequence, Mapping


# pattern for query splitting
_p_query = re.compile('(.*)\[([0-9]+)\]')


def _split_query(q):
    """
    split query into leading string key and trailing int indices
    Parameters
    ----------
    q : query of the form key[idx_1][idx_2] .. [idx_n]

    Returns
    -------
    split query : list[str, int, int, ..]

    """

    res = []
    done = False
    while not done:
        if _p_query.match(q):
            q, idx = _p_query.match(q).groups()
            res.append(int(idx))
        else:
            res.append(q)
            done = True
    return list(reversed(res))


def _query_single_key(d, query):

    query = _split_query(query)
    try:
        for q in query:
            d = d[q]

    except (KeyError, IndexError, TypeError):
        return None

    return d


def recursive_dict_query(d, query, sep='/'):
    """
    XPath-like querying of nested dicts
    Supports nested keys: /foo/bar
    and index-based access for nested sequences /foo/bar[0]/baz

    Parameters
    ----------
    d : dict to query
    query : query string
    sep : separator in query string, default: "/"

    Returns
    -------
    queried element or None if it could not be found
    """

    # ignore leading/trailing seps
    query = query.strip(sep)
    # split on sep
    query = query.split(sep)

    try:
        while len(query) >= 1:

            # we have query left, but no more dict
            if not isinstance(d, Mapping):
                return None

            # last element
            if len(query) == 1:
                return _query_single_key(d,query[0])

            else:
                d = _query_single_key(d,query[0])

                # continue with rest of query
                query = query[1:]

    # could not find element -> return None
    except (KeyError, IndexError):
        return None


def query_json(filepath, query, sep='/'):
    """
    wrapper for recursive_dict_query that reads a json file and queries its contents

    Parameters
    ----------
    filepath : JSON-file to query
    query : query string
    sep : separator char, default: "/"

    Returns
    -------
    query result or None if not found

    """
    if not os.path.exists(filepath):
        return None
    with open(filepath, 'r') as fd:
        d = json.load(fd)
    return recursive_dict_query(d, query, sep)


if __name__ == '__main__':

    # some simple tests

    d = {
        'a' : [
                1,
                2,
                {'c' : 3, 'd' : [4, 5, 6]},
                [{'e': 3, 'f': [4, 5, 6]}, 23]
            ],
        'b' : 12
    }

    print(recursive_dict_query(d, 'b'))
    print(recursive_dict_query(d, 'a[0]'))
    print(recursive_dict_query(d, 'a[2]/c'))
    print(recursive_dict_query(d, 'a[2]/d[2]'))
    print(recursive_dict_query(d, 'a[3][0]/f[2]'))