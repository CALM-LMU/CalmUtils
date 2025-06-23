import os
import numpy as np

def get_common_subpath(path1, path2):

    # hopefully platform-agnostic splitting into folders
    # TODO: check on Windows
    path1 = os.path.normpath(path1.replace("\\", "/")).split(os.sep)
    path2 = os.path.normpath(path2.replace("\\", "/")).split(os.sep)

    # longest subsequence following the dynamic programming approach from
    # https://en.wikipedia.org/wiki/Longest_common_substring#Dynamic_programming 
    common_prefix_len = np.zeros((len(path1), len(path2)), dtype=int)
    for i in range(len(path1)):
        for j in range(len(path2)):
            if path1[i] == path2[j]:
                if i == 0 or j == 0:
                    common_prefix_len[i, j] = 1
                else:
                    common_prefix_len[i, j] = common_prefix_len[i-1, j-1] + 1

    max_len = common_prefix_len.max()
    maxi, maxj = np.unravel_index(common_prefix_len.argmax(), common_prefix_len.shape)

    # if we found no common part, return original paths as "prefix"
    if max_len == 0:
        return "", (os.path.join(*path1), os.path.join(*path2)), ("", "")

    # cut common path, prefixes, suffixes and return
    longest_common_path = os.path.join(*path1[maxi - max_len + 1 : maxi + 1])
    prefixes = (
        os.path.join(*path1[: maxi - max_len + 1 ]) if maxi - max_len + 1 > 0 else "", 
        os.path.join(*path2[: maxj - max_len + 1 ]) if maxj - max_len + 1 > 0 else ""
    )
    suffixes = (
        os.path.join(*path1[maxi + 1 :]) if maxi < len(path1) - 1 else "", 
        os.path.join(*path2[maxj + 1 :]) if maxj < len(path2) - 1 else ""
    )

    return longest_common_path, prefixes, suffixes