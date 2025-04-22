from operator import add
from functools import reduce

import numpy as np
from matplotlib.colors import to_rgb
from matplotlib.colors import get_named_colors_mapping


# use the same colors as ImageJ Merge Channels by default
DEFAULT_COLOR_NAMES = ('red', 'green', 'blue', 'gray', 'cyan', 'magenta', 'yellow')


def grayscale_image_to_named_color(img_grayscale, color_name):

    # get normalized RGB values of named color
    color = np.array(to_rgb(get_named_colors_mapping()[color_name]))
    color /= np.max(color)

    # normalize image
    img_grayscale = img_grayscale / np.max(img_grayscale)

    # to (... original shape ..., 3)-shape RGB via outer product
    img_color = np.outer(img_grayscale, color).reshape(img_grayscale.shape + (-1,))
    return img_color


def gray_images_to_rgb_composite(images, color_names=DEFAULT_COLOR_NAMES):

    # check that images have the same shape
    shape = None
    for image in images:
        if shape is not None and image.shape != shape:
            raise ValueError('Images must have the same shape')
        shape = image.shape

    # make channels, compose via add, clip overflowing values
    composite_channels = [grayscale_image_to_named_color(image, color_name) for image, color_name in zip(images, color_names)]
    composite = np.clip(reduce(add, composite_channels), 0, 1)

    return composite