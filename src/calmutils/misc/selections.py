import numpy as np

def minmax_to_slices(mins, maxs):
    return tuple((slice(int(mi), int(ma)) for mi, ma in zip(mins, maxs)))

def safe_rectangular_selection(arr: np.array, loc, size, return_offsets=False):
    
    if arr.ndim != len(loc):
        raise ValueError('dimensionality of array (is {}) and location (is {}) have to match.'.format(arr.ndim, len(loc)))
    
    # allow scalar size -> will be the same in each dimension
    if np.isscalar(size):
        size = [size] * len(loc)

    # TODO: warn and explain implicit behaviour
    # * rounding of loc
    # * in case of even sized cut: loc will be right of center

    slices = []
    min_offs = []
    max_offs = []
    for l, s, c in zip(loc, arr.shape, size):
        
        even = c % 2 == 0
        min_ = np.round(l) - c // 2
        # add 1 for odd: compensate for floor division
        max_ = np.round(l) + c // 2 + (0 if even else 1)
        min_off = -min_ if min_ < 0 else 0
        max_off = s - max_ if max_ > s else c
        min_ = np.clip(min_, 0, s)
        max_ = np.clip(max_, 0, s)
        slices.append(slice(int(min_), int(max_)))
        min_offs.append(min_off)
        max_offs.append(max_off)

    if return_offsets:
        return tuple(slices), min_offs, max_offs
    else:
        return tuple(slices)
    
