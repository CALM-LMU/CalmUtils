import numpy as np
from PIL import Image


def read_tiff_stack(path, correct16bit, get_info=False):
    """
    :param path: path to .tif file
    :param correct16bit: whether to subtract 2^16-1 from pixel values (hack for images converted from Imspector .msr images)
    :param get_info whether to return infos
    :return: stack of layers or tuple (stack, info) if get_info
    """
    image_file = Image.open(path)
    info = image_file.info
    image_list = list()
    n = 0
    while True:
        (w, h) = image_file.size
        image_list.append(np.array(image_file.getdata()).reshape(h, w))
        n += 1
        try:
            image_file.seek(n)
        except:
            break

    if len(image_list) == 1:
        res = image_list[0]
    else:
        res = np.dstack(image_list)

    if correct16bit:
        res = correct_16bit(res)

    image_file.close()

    if get_info:
        return res, info
    else:
        return res


def correct_16bit(img):
    '''
    subtract 2^16-1 from pixel values (hack for images converted from Imspector .msr images)
    :param img: image with wrong values
    :return: corrected image
    '''
    return img - np.iinfo(np.int16).max - 1