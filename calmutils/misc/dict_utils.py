from functools import reduce
from operator import add, or_


def combine_dicts_along_keys(*dicts, combine_function=add, error_missing_key=False):

    """
    Combine the values for each key in multiple dicts via reduction with a user-specified function.
    By default, combines all present values for a key, skipping missing,
    but can also be set to raise error if a key is not present in all dicts.
    """

    # TODO: general-purpose function, move to CalmUtils?

    # get all keys present in any dict
    all_keys = reduce(or_, [d.keys() for d in dicts])
    combined_dicts = {}

    for k in all_keys:
        # get present values for key
        present_values = [d[k] for d in dicts if k in d]

        # number of present values does not match number of dicts -> raise error if desired
        if error_missing_key and (len(present_values) != len(dicts)):
            raise ValueError(f"key '{k}' not found in all dicts.")

        # combine via reduction with combine_function
        combined_value = reduce(combine_function, present_values)
        combined_dicts[k] = combined_value

    return combined_dicts