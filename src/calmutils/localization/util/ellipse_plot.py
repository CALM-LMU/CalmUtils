import numpy as np

def get_ellipse_params(cov, q=0.5):
    """
    get ellipse parameters for matplotlib for an ellipse representing the
    full-width-at-quantile of a 2d-Gaussian

    Parameters
    ----------
    cov: np-array
        2x2 covaraince matrix
    p: float
        quantile at which to draw the ellipse
        default: 0.5 -> FWHM

    Returns
    -------
    a: float
        horizontal axis length
    b: float
        vertical axis length
    alpha: float \in (-180, 180)
        counterclockwise rotation of the ellipse in degrees
    """
    w, v = np.linalg.eig(cov)
    lens = [sig_to_full_width_at_quantile(np.sqrt(wi), q) for wi in w]
    a = _deg_angle(v[:, 0] * lens[0])
    return (lens[0], lens[1], a)


def _deg_angle(a):
    """
    angle between vector a and x-axis, in degrees
    """
    return 180 * np.arctan2(a[1], a[0]) / np.pi


def sig_to_full_width_at_quantile(sig, q=0.5):
    """
    get Gaussian Full-width-at-quantile given std.dev.
    """
    return sig * 2 * np.sqrt(2 * np.log(q ** -1))