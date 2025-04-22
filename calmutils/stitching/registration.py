import numpy as np

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