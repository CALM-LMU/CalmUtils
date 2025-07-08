import numpy as np
from functools import lru_cache


def get_sinusoidal_fadeout_profile_1d(size, fadeout_width=10):

    """
    Get a 1-dimensional profile of length size that fades out on the ends following a sinusoidal profile.
    The profile does not fall all the way to zero but rather that elements -1 and size would be zero.
    The fadeout will happen over a width of fadeout_width on both sizes, elements in the center will be 1.
    If size is smaller than 2*fadeout_width, the profile will be scaled,
    with 1 (odd) or 2 (even size) elements of value 1 in the middle.
    """

    if fadeout_width * 2 > size:
        # NOTE: subtract 1 from size so we get 1/2 elements of weight 1 in the middle
        fadeout_width = (size - 1) // 2

    # get sinusoidal ramp from 0 to 1, exclude first and last (exactly 0 and 1)
    ramp = np.sin(np.linspace(0, np.pi/2, fadeout_width+2))[1:-1]

    # concat ramps with constant center
    profile = np.concatenate([ramp, np.ones(size - 2 * fadeout_width), ramp[::-1]])

    return profile


@lru_cache()
def get_sinusoidal_fadeout(shape, fadeout_width=10):

    """
    Get array that is 1 in the center and fades out at the borders following a sinusiodal profile.
    N-dimensional fadeout is implemented as outer product of 1D profiles. 
    The function is cached as it is intended to create weight map for stitching and images often have the same shape.
    """

    LETTERS = "abcdefghijklmnopqrstuvwxyz"

    # catch 1D case:
    if np.isscalar(shape):
        shape = (shape, )

    profiles_1d = [get_sinusoidal_fadeout_profile_1d(s, fadeout_width) for s in shape]

    # outer product of 1D profiles via einsum (signature: "a,b,... -> ab...")
    # we construct signature on-the-fly to support dimensions up to len(LETTERS)
    einsum_signature = f"{','.join(LETTERS[:len(shape)])} -> {LETTERS[:len(shape)]}"
    fadeout_field = np.einsum(einsum_signature, *profiles_1d)

    return fadeout_field
