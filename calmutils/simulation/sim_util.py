import numpy as np


def runit_vec(n=1, d=3):
    '''
    random unit vectors/random hypersphere point picking
    see http://mathworld.wolfram.com/HyperspherePointPicking.html
    :param n: shape of first output dimensions
    :param d: size of last dimension (dimensionality of unit vecs)
    :return: (n_0, ..., n_n, d) array of unit vectors along last dimension
    '''

    if np.isscalar(n):
        n = [n]

    vec = np.random.normal(size=(np.prod(n), d))

    vec = (vec.T / np.linalg.norm(vec, axis=1)).T
    return vec.reshape(tuple(list(n) + [d]))

if __name__ == '__main__':
    print((runit_vec((2, 2), 1).transpose() * np.array([[[1, 0], [0, 1]]])).transpose())