
from collections import namedtuple

import numpy as np
from skimage.measure import ransac

from calmutils.misc.selections import safe_rectangular_selection
from calmutils.morphology.structuring_elements import hypersphere_centered


RANSACRadialSymmetryResult = namedtuple('RANSACRadialSymmetryResult', ['center_estimate', 'candidates', 'inliers'])


def get_coords_and_gradients(img, loc, radius):

    # get slices to cut area around loc from original image
    slices, min_off, max_off = safe_rectangular_selection(img, loc, 2*radius+1, return_offsets=True)
    # and relative slices to cut from hypershpere structuring element in case of loc on edges
    slices_rel = tuple((slice(mi, ma) for mi, ma in zip(min_off, max_off)))

    # hypershpere structuring element to select only pixels in radius from rectangular slices
    selem = hypersphere_centered(img.ndim, radius)[slices_rel].astype(bool)

    # calculate gradients, select with hypershere
    gradients = np.stack(np.gradient(img[slices]), axis=-1)
    gradients = gradients[selem]

    # coordinates
    # 1. make full coordinate array
    coords = np.stack(np.meshgrid(*(np.arange(l-radius, l+radius+1, dtype=float) for l in loc), indexing='ij'), axis=-1)
    # 2. rectangular slice
    coords = coords[slices_rel]
    # 3. select hypersphere
    coords = coords[selem]

    return coords, gradients


def get_normal_distances(center, coords, gradients):
    # vectors from coordinates to (estimated) center 
    vc = (center - coords)

    # vector projection of vc center vector onto gradient vector -> closest point along gradient direction
    vc_proj_grad = ( gradients.T * np.sum(gradients * vc, axis=1) / np.sum(gradients**2, axis=1) ).T

    # magnitude of difference of center vector and projected -> distance / residual
    return np.linalg.norm(vc - vc_proj_grad, axis=1)


class RadialSymmetryLocalization():

    """
    Radial Symmetry Localization following the derivations in:
    Bahry et al., 2022 (RS-FISH)
    Liu et al., 2011
    implemented as skimage.measure.ransac-compatible class
    """

    def __init__(self, weighted=False):
        self.weighted = weighted

    def estimate(self, coords, gradients):

        # coords (and gradients) are NxD arrays
        ndim = coords.shape[1]

        # square gradient magnitude
        m2 = (gradients**2).sum(axis=1)

        if self.weighted:
            # weight: m2 divided by distance to centroid
            centroid = coords.mean(axis=0)
            centroid_distance = np.linalg.norm(coords - centroid)
            # clip centroid distances to avoid zero division in edge cases
            weights = m2 / np.clip(centroid_distance, 0.001, None)
        else:
            weights = np.ones_like(m2)

        # clip gradient magnitide to be !=0 and avoid zero division nans
        # NOTE: terms with zero gradient magnitude do not contribute anyway,
        # as they always have a gradient compontent in numerator
        m2 = np.clip(m2, 0.00001, None)

        # get lhs matrix delta and rhs vector theta to solve equation:
        # delta * center = theta
        delta = np.zeros((ndim, ndim))
        for i in range(ndim):
            for j in range(i, ndim):
                delta[i,j] = - (weights * gradients[:,i] * gradients[:,j] / m2).sum()
                delta[j,i] = delta[i,j]
        delta += np.diag([weights.sum()] * ndim)

        theta = np.zeros(ndim)
        for i in range(ndim):
            for j in range(ndim):
                theta[i] -= (weights * gradients[:,i] * gradients[:, j] * coords[:,j] / m2).sum()
        theta += (coords * weights[:,np.newaxis]).sum(axis=0)

        self.center_estimate = np.linalg.inv(delta) @ theta

    def residuals(self, coords, gradients):
        return get_normal_distances(self.center_estimate, coords, gradients)


def refine_ransac_radial_symmetry_coords(coords, gradients, max_error=0.5, min_inliers=10, max_rounds_multiconsensus=1,
                                         center_weighted_gradients=True, max_trials=1_000):
    res = []
    for _ in range(max_rounds_multiconsensus):
        mr, inliers = ransac((coords, gradients), lambda: RadialSymmetryLocalization(center_weighted_gradients), coords.shape[1], max_error, max_trials=max_trials)

        n_inliers = inliers.sum()
        n_candidates = len(coords)

        if n_inliers < min_inliers:
            break

        res.append(RANSACRadialSymmetryResult(mr.center_estimate, n_candidates, n_inliers))

        sel_outliers = ~inliers
        coords = coords[sel_outliers]
        gradients = gradients[sel_outliers]
    return res


def refine_ransac_radial_symmetry(image, candidate_coordinates, gradient_radius=2, anisotropy=None, **kwargs):

    # check if we have anisotropy and that it is a vector matching image dimensionality
    if anisotropy is not None:
        anisotropy = np.array(anisotropy)
        if len(anisotropy) != image.ndim:
            raise ValueError("anisotropy / pixel size vector length needs to match image dimensionality")

    res = []
    for candidate_coord in candidate_coordinates:
        coords, grads = get_coords_and_gradients(image, candidate_coord, gradient_radius)

        # scale coordinates / gradients to anisotropy weighted coords
        if anisotropy is not None:
            coords *= anisotropy
            grads *= anisotropy

        res.extend(refine_ransac_radial_symmetry_coords(coords, grads, **kwargs))

    # scale results back to pixel coords
    if anisotropy is not None:
        res = [RANSACRadialSymmetryResult(r.center_estimate / anisotropy, r.candidates, r.inliers) for r in res]

    return res
