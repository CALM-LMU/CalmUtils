import warnings

def scale_intensities(arr, in_range, out_range=(0, 1), clip=False):
    """
    Non-clipping (optional) intensity rescaling.
    Implemented using basic array calculations & selections, should work with both NumPy and PyTorch arrays.
    """

    in_low, in_high = in_range

    if in_high == in_low:
        warnings.warn(f"Input range low and high are equal, will result in division-by-zero.")

    # scale to 0-1
    arr = (arr - in_low) / (in_high - in_low)

    # scale to output range if it is not (0-1)
    if out_range != (0, 1):
        out_low, out_high = out_range
        arr = arr * (out_high - out_low) + out_low

    if clip:
        arr[arr < out_low] = out_low
        arr[arr > out_high] = out_high

    return arr