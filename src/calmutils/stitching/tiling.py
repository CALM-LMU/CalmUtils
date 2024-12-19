from itertools import repeat

import numpy as np


def centered_tiles_1d(center_position, fov_size, n_tiles, overlap):

    # integer steps
    x = np.arange(np.ceil(-n_tiles / 2), np.ceil(n_tiles / 2))
    # times distance considering overlap
    x *= fov_size * (1 - overlap)

    # offset by half a tile of even number of tiles
    if not (n_tiles % 2):
        x += fov_size * (1 - overlap) / 2

    x += center_position
    return x


def minmax_tiles_1d(min_position, max_position, fov_size, overlap):

    # swap min/max (so it works even if we swap inputs, e.g. flipped stage directions)
    min_position, max_position = min(min_position, max_position), max(min_position, max_position)

    # center of range
    center_position = (min_position + max_position) / 2

    # necessary number of tiles
    total_length = max_position - min_position
    n_tiles = np.ceil(total_length / (fov_size * (1 - overlap))) + 1

    return centered_tiles_1d(center_position, fov_size, n_tiles, overlap)


def alternate_axes(arr, alternate_along_axis, axes_to_alternate, offset=1):

    # wrap single axis to alternate
    if np.isscalar(axes_to_alternate):
        axes_to_alternate = (axes_to_alternate, )

    # allow axis specification from the back (negative indices)
    if alternate_along_axis < 0:
        alternate_along_axis = arr.ndim + alternate_along_axis
    axes_to_alternate = tuple(ax if ax>=0 else arr.ndim + ax for ax in axes_to_alternate)

    # copy arr as to not modify input
    arr=arr.copy()

    # select every second position along the axis to alternate on
    selection = tuple(slice(offset,s,2) if i==alternate_along_axis else slice(s) for i,s in enumerate(arr.shape))

    # perform inversion of the axes to alternate
    # note the clumsy slice(s, -(s+1), -1) seems to be necessary to represent [::-1] indexing
    alternation_selection = tuple(slice(s, -(s+1), -1) if i in axes_to_alternate else slice(s) for i,s in enumerate(arr[selection].shape))

    # perform actual alternation
    arr[selection] = arr[selection][alternation_selection]

    return arr


def centered_tiles(center_position, fov_size, n_tiles, overlap=0, snake_rows=True, flip_axes=()):

    # repeat length, n_tiles, overlap if only scalar value is provided
    if np.isscalar(fov_size):
        fov_size = repeat(fov_size)
    if np.isscalar(n_tiles):
        n_tiles = repeat(n_tiles)
    if np.isscalar(overlap):
        overlap = repeat(overlap)

    grid = np.meshgrid(*(centered_tiles_1d(x, l, n, o) for x, l, n, o in zip(center_position, fov_size, n_tiles, overlap)), indexing='ij')
    grid = np.stack(grid, -1)

    if snake_rows :
        # alternate along rows (last two dimensions of regular grid assuming numpy zyx convention)
        # as the last dimension are the coordinates themselves, we alternate dim -2 (columns) along -3 (rows)
        grid = alternate_axes(grid, -3, -2)

    # flip specified axes
    flip_selection = tuple(slice(s, -(s+1), -1) if i in flip_axes else slice(s) for i,s in enumerate(grid.shape[:-1]))
    grid = grid[flip_selection]

    return grid.reshape((-1, len(center_position)))


def minmax_tiles(min_position, max_position, fov_size, overlap=0, snake_rows=True, flip_axes=()):

    # repeat length, overlap if only scalar value is provided
    if np.isscalar(fov_size):
        fov_size = repeat(fov_size)
    if np.isscalar(overlap):
        overlap = repeat(overlap)

    grid = np.meshgrid(*(minmax_tiles_1d(mi, ma, l, o) for mi, ma, l, o in zip(min_position, max_position, fov_size, overlap)), indexing='ij')
    grid = np.stack(grid, -1)

    if snake_rows :
        # alternate along rows (last two dimensions of regular grid assuming numpy zyx convention)
        # as the last dimension (-1) are the coordinates themselves, we alternate dim -2 (columns) along -3 (rows)
        grid = alternate_axes(grid, -3, -2)

    # flip specified axes
    flip_selection = tuple(slice(s, -(s+1), -1) if i in flip_axes else slice(s) for i,s in enumerate(grid.shape[:-1]))
    grid = grid[flip_selection]

    return grid.reshape((-1, len(min_position)))