import numpy as np

def translation_matrix(translation):
    """
    Construct (augmented) Affine Matrix representing a translation by translation vector translation
    """
    mat = np.eye(len(translation) + 1)
    mat[0:-1, -1] = translation
    return mat


def scale_matrix(scale):
    """
    Construct (augmented) Affine Matrix representing scaling by scale vector scale
    """
    return np.diag(list(scale) + [1])