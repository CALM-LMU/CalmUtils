import numpy as np


def _line_coords(start, alpha, shape):
    """
    Get coordinates of a Bresenham line of given angle

    Parameters
    ----------
    start: array-like (1d)
        Starting point of the line
    alpha: float \in (0,2\pi)
        Angle to 0th axis (radians)
    shape: array-like
        Shape of image in which we are creating line (necessary as upper bound for coordinates)

    Returns
    -------
    coords: list of 1d-array
        (y,x) coordinates of all points on the line
    """

    _upper_bound = np.array(shape, dtype=np.int32)
    _lower_bound = np.zeros_like(start, dtype=np.int32)

    # directions we are going in for both axes
    first_axis_sign = np.sign(np.cos(alpha))
    second_axis_sign = np.sign(np.sin(alpha))

    # do we have an angle in the lower half-quadrant?
    # adjust steps acordingly
    step = np.abs(np.tan(alpha))
    high = step > 1.0
    if high:
        step = step ** -1

    position = np.array(start)
    error = 0.0
    res = []

    # high angle == low angle to x-axis --> iterate x
    if high:
        while np.all(position >= _lower_bound) and np.all(position < _upper_bound):
            res.append(np.array(position))
            error += step
            position[1] += 1 * second_axis_sign
            if error >= 0.5:
                position[0] += 1 * first_axis_sign
                error -= 1.0

    # low angle --> iterate y
    else:
        while np.all(position >= _lower_bound) and np.all(position < _upper_bound):
            res.append(np.array(position))
            error += step
            position[0] += 1 * first_axis_sign
            if error >= 0.5:
                position[1] += 1 * second_axis_sign
                error -= 1.0
    return res


def directed_dist(mask, angle, output=None):
    """
    Get the (approximate) distance to the next object border along a line of a given angle
    NB: this is rather slow at the moment, consider parallelizing when calculating along multiple angles!

    Parameters
    ----------
    mask: 2d-array
        Labelled image, background should be 0
    angle: float
        Angle at which to calculate distance, in radians
        NB: is measured from the 0th axis (y)
    output: 2d-array
        Output array of the same shape as mask to stor results, will be modified.
        Optional, if not provided, a new array will be created

    Returns
    -------
    output: 2d-array
        Newly created output array or reference to the provided destination array.

    Example
    -------
    # image with two box objects
    m = np.zeros((800,800))
    m[10:190, 10:70] = 1
    m[110:190, 110:199] = 1

    alpha = 6 * np.pi / 8
    d = directed_dist(m, alpha)
    """

    # new output array if not provided
    if output is None:
        output = - np.ones_like(mask, dtype=np.float32)

    # get step length
    step = np.abs(np.tan(angle))
    if step > 1.0:
        step = step ** -1
    step = np.sqrt(1 + step ** 2)

    # in which direction are we going?
    first_axis_sign = np.sign(np.cos(angle))
    second_axis_sign = np.sign(np.sin(angle))
    up = first_axis_sign < 0
    left = second_axis_sign < 0

    # iterate along y-axis
    a_imo = None
    for y in reversed(range(0, mask.shape[0])) if up else range(0, mask.shape[0]):
        for a in _line_coords([y, 0 if not left else mask.shape[1] - 1], angle, mask.shape):

            # was not set before
            if mask[tuple(a)] != 0 and output[tuple(a)] == -1:
                output[tuple(a)] = 0

            # we found a new object
            if a_imo is not None and mask[tuple(a_imo)] == 0 and mask[tuple(a)] != 0:
                output[tuple(a)] = 0

            # continue along line, only update dist if we found a shorter line
            if a_imo is not None and mask[tuple(a_imo)] != 0 and mask[tuple(a)] != 0:
                do_min = output[tuple(a)] > 0
                old = output[tuple(a)]
                if do_min:
                    output[tuple(a)] = np.min([output[tuple(a_imo)] + step, old])
                else:
                    output[tuple(a)] = output[tuple(a_imo)] + step

            a_imo = a

    # iterate along x-axis
    a_imo = None
    for x in reversed(range(0, mask.shape[1])) if left else range(0, mask.shape[1]):
        for a in _line_coords([0 if not up else mask.shape[0] - 1, x], angle, mask.shape):

            # was not set before
            if mask[tuple(a)] != 0 and output[tuple(a)] == -1:
                output[tuple(a)] = 0

            # we found a new object
            if a_imo is not None and mask[tuple(a_imo)] == 0 and mask[tuple(a)] != 0:
                output[tuple(a)] = 0

            # continue along line, only update dist if we found a shorter line
            if a_imo is not None and mask[tuple(a_imo)] != 0 and mask[tuple(a)] != 0:
                do_min = output[tuple(a)] > 0
                old = output[tuple(a)]
                if do_min:
                    output[tuple(a)] = np.min([output[tuple(a_imo)] + step, old])
                else:
                    output[tuple(a)] = output[tuple(a_imo)] + step

            a_imo = a

    # pixels we did not modify -> background
    output[output == -1] = 0
    return output
