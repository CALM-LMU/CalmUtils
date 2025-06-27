import numpy as np
from random import shuffle
import networkx as nx
from skimage.transform import AffineTransform


# first try, 2d affine test

def make_coeff_affine2d(x1, x2, group_idx1=0, group_idx2=1, ngroups_nonfixed=2):
    res = np.zeros((2, 6*(ngroups_nonfixed)))
    
    res[0,group_idx1*6:group_idx1*6+2] = x1
    res[0,group_idx1*6+2] = 1
    res[0,group_idx2*6:group_idx2*6+2] = -x2
    res[0,group_idx2*6+2] = -1
    
    res[1,3+group_idx1*6:3+group_idx1*6+2] = x1
    res[1,3+group_idx1*6+2] = 1
    res[1,3+group_idx2*6:3+group_idx2*6+2] = -x2
    res[1,3+group_idx2*6+2] = -1
    
    return res, np.zeros(2)

def make_coeff_affine2d_fixed(xf, x2, group_idx2=0, ngroups_nonfixed=1):
    res = np.zeros((2, 6*(ngroups_nonfixed)))
    
    res[0,group_idx2*6:group_idx2*6+2] = x2
    res[0,group_idx2*6+2] = 1
    
    res[1,3+group_idx2*6:3+group_idx2*6+2] = x2
    res[1,3+group_idx2*6+2] = 1
    
    return res, xf

# n-dimensional coeffs for affine and translation

def make_coeffs_affine_nd(x1, x2, n_dim=2, group_idx1=0, group_idx2=1, ngroups_nonfixed=2):
    mat_len = (n_dim+1)*n_dim
    res = np.zeros((n_dim, mat_len*(ngroups_nonfixed)))
    for n in range(n_dim):
        res[n, n*(n_dim+1)+group_idx1*mat_len:n*(n_dim+1)+group_idx1*mat_len+n_dim] = x1
        res[n, n*(n_dim+1)+group_idx1*mat_len+n_dim] = 1
        res[n, n*(n_dim+1)+group_idx2*mat_len:n*(n_dim+1)+group_idx2*mat_len+n_dim] = -x2
        res[n, n*(n_dim+1)+group_idx2*mat_len+n_dim] = -1
    return res, np.zeros(n_dim)

def make_coeffs_affine_nd_fixed(xf, x2, n_dim=2, group_idx2=0, ngroups_nonfixed=1):
    mat_len = (n_dim+1)*n_dim
    res = np.zeros((n_dim, mat_len*(ngroups_nonfixed)))
    for n in range(n_dim):
        res[n,n*(n_dim+1)+group_idx2*mat_len:n*(n_dim+1)+group_idx2*mat_len+n_dim] = x2
        res[n,n*(n_dim+1)+group_idx2*mat_len+n_dim] = 1
    return res, xf

def make_coeffs_translation_nd(x1, x2, n_dim=2, group_idx1=0, group_idx2=1, ngroups_nonfixed=2):
    mat_len = n_dim
    res = np.zeros((n_dim, mat_len*(ngroups_nonfixed)))
    for n in range(n_dim):
        res[n, n+group_idx1*mat_len] = 1
        res[n, n+group_idx2*mat_len] = -1
    return res, x2-x1

def make_coeffs_translation_nd_fixed(xf, x2, n_dim=2, group_idx2=0, ngroups_nonfixed=1):
    mat_len = n_dim
    res = np.zeros((n_dim, mat_len*(ngroups_nonfixed)))
    for n in range(n_dim):
        res[n, n+group_idx2*mat_len] = 1
    return res, xf - x2


import itertools
def register_affine(matched_points, fixed_indices):
    model_idx = itertools.count()
    idx_to_model_idx = {}
    
    ndim = None;
    coeffs = []
    ys = []
    
    for ((i1,i2),(points1, points2)) in matched_points.items():
        
        if ndim is None:
            point = next(iter(points1))
            ndim = len(point)
        
        if i1 not in fixed_indices:
            if i1 not in idx_to_model_idx:
                idx_to_model_idx[i1] = next(model_idx)

        if i2 not in fixed_indices:
            if i2 not in idx_to_model_idx:
                idx_to_model_idx[i2] = next(model_idx)

    for ((i1,i2),(points1, points2)) in matched_points.items():
        if i1 not in fixed_indices:
            m_idx1 = idx_to_model_idx[i1]
        if i2 not in fixed_indices:
            m_idx2 = idx_to_model_idx[i2]
            
        if i1 not in fixed_indices and i2 not in fixed_indices:
            for x1, x2 in zip(points1, points2):
                c, y = make_coeffs_affine_nd(x1, x2, ndim, m_idx1, m_idx2, len(idx_to_model_idx))
                coeffs.append(c)
                ys.append(y)
        
        elif i1 in fixed_indices and i2 not in fixed_indices:
            for x1, x2 in zip(points1, points2):
                c, y = make_coeffs_affine_nd_fixed(x1, x2, ndim, m_idx2, len(idx_to_model_idx))
                coeffs.append(c)
                ys.append(y)
        
        elif i1 not in fixed_indices and i2 in fixed_indices:
            for x1, x2 in zip(points1, points2):
                c, y = make_coeffs_affine_nd_fixed(x2, x1, ndim, m_idx1, len(idx_to_model_idx))
                coeffs.append(c)
                ys.append(y)
        
    coeffs = np.vstack(coeffs)
    ys = np.concatenate(ys)
    r, _, _, _ = np.linalg.lstsq(coeffs, ys, rcond=None)
    
    mat_len = (ndim+1)*ndim
    
    res = {}
    for idx, midx in idx_to_model_idx.items():
        res[idx] = aug_mat(r[mat_len*midx:mat_len*midx+mat_len].reshape((ndim, ndim+1)))
        
    for idx in fixed_indices:
        res[idx] = np.eye(ndim+1)
        
    return res

def register_translations(matched_points, fixed_indices):
    model_idx = itertools.count()
    idx_to_model_idx = {}
    
    ndim = None;
    coeffs = []
    ys = []
    
    for ((i1,i2),(points1, points2)) in matched_points.items():
        
        if ndim is None:
            point = next(iter(points1))
            ndim = len(point)
        
        if i1 not in fixed_indices:
            if i1 not in idx_to_model_idx:
                idx_to_model_idx[i1] = next(model_idx)

        if i2 not in fixed_indices:
            if i2 not in idx_to_model_idx:
                idx_to_model_idx[i2] = next(model_idx)

    for ((i1,i2),(points1, points2)) in matched_points.items():
        if i1 not in fixed_indices:
            m_idx1 = idx_to_model_idx[i1]
        if i2 not in fixed_indices:
            m_idx2 = idx_to_model_idx[i2]
            
        if i1 not in fixed_indices and i2 not in fixed_indices:
            for x1, x2 in zip(points1, points2):
                c, y = make_coeffs_translation_nd(x1, x2, ndim, m_idx1, m_idx2, len(idx_to_model_idx))
                coeffs.append(c)
                ys.append(y)
        
        elif i1 in fixed_indices and i2 not in fixed_indices:
            for x1, x2 in zip(points1, points2):
                c, y = make_coeffs_translation_nd_fixed(x1, x2, ndim, m_idx2, len(idx_to_model_idx))
                coeffs.append(c)
                ys.append(y)
        
        elif i1 not in fixed_indices and i2 in fixed_indices:
            for x1, x2 in zip(points1, points2):
                c, y = make_coeffs_translation_nd_fixed(x2, x1, ndim, m_idx1, len(idx_to_model_idx))
                coeffs.append(c)
                ys.append(y)
        
    coeffs = np.vstack(coeffs)
    ys = np.concatenate(ys)
    r, _, _, _ = np.linalg.lstsq(coeffs, ys, rcond=None)
    
    mat_len = ndim
    
    res = {}
    for idx, midx in idx_to_model_idx.items():
        res[idx] = aug_mat(np.hstack((np.eye(ndim), r[mat_len*midx:mat_len*midx+mat_len].reshape((ndim, 1)))))
        
    for idx in fixed_indices:
        res[idx] = np.eye(ndim+1)
        
    return res


def aug_mat(arr):
    """
    Augment a $ndim/times(ndim+1)$ affine+translation matrix with an additional 0, 0, ..., 1 row
    to get an $(ndim+1)/times(ndim+1)$ matrix that can be applied to an augmented vector
    """
    aug = np.zeros(arr.shape[1])
    aug[-1] = 1
    return np.vstack((arr, aug))


def aug_vec(v):
    """
    append a 1 to a vector, so that an $(ndim+1)/times(ndim+1)$ augmented affine
    matrix can be applied to it
    """
    a = np.ones((len(v)+1,))
    a[:len(v)] = v
    return a


def register_iterative(matched_points, fixed_indices=(), transform_type=AffineTransform, max_iterations=200):

    """
    Iterative global optimization of point matches.
    Following Saalfeld et al., Bioinformatics, 2010 (https://doi.org/10.1093/bioinformatics/btq219)
    """

    # get any set of matched coords to determine dimensionality
    coords1, coords2 = next(iter(matched_points.values()))
    ndim = coords1.shape[1]

    # get all indices
    all_indices = set(sum(matched_points.keys(), ()))

    # list of non-fixed := free indices
    free_indices = [idx for idx in all_indices if idx not in fixed_indices]

    # init all transforms with identity
    transforms = {idx: transform_type(dimensionality=ndim) for idx in all_indices}

    # make graph of connected indices
    G = nx.Graph()
    G.add_edges_from(matched_points.keys())

    # iterate connected components
    for cc_nodes in nx.connected_components(G):
        Gi = G.subgraph(cc_nodes)

        # start for bfs: any fixed node or any node if none are fixed
        bfs_start = next((n for n in Gi.nodes if n in fixed_indices), None) or next(iter(Gi.nodes))

        for idx1, idx2 in nx.bfs_edges(Gi, bfs_start):

            # we want to update idx2, if it is fixed, we skip it
            if idx2 in fixed_indices:
                continue

            # handle two possible orders in matched point dict
            if (idx1, idx2) in matched_points:
                coords1, coords2 = matched_points[(idx1, idx2)]
            else:
                coords2, coords1 = matched_points[(idx2, idx1)]

            # apply existing transform to idx1 (idx2 is still identity)
            coords1 = transforms[idx1](coords1)

            # estimate additional transform for idx2 and concatenate
            tr = transform_type(dimensionality=ndim)
            tr.estimate(coords2, coords1)
            transforms[idx2] = transforms[idx2] + tr


    # iteratively refine
    for _ in range(max_iterations):

        # shuffle indices
        shuffle(free_indices)

        for idx in free_indices:

            # collect coordinates of idx and matched of others with current transforms applied
            coords_self = []
            coords_other = []

            for (idx1, idx2), (coords1, coords2) in matched_points.items():
                if idx1 == idx:
                    coords_self.append(transforms[idx](coords1))
                    coords_other.append(transforms[idx2](coords2))
                elif idx2 == idx:
                    coords_self.append(transforms[idx](coords2))
                    coords_other.append(transforms[idx1](coords1))

            coords_self = np.vstack(coords_self)
            coords_other = np.vstack(coords_other)

            # estimate new transform
            tr = transform_type(dimensionality=ndim)
            tr.estimate(coords_self, coords_other)

            # append new transform to existing
            transforms[idx] = transforms[idx] + tr

            # TODO: early stop if converged

    return transforms