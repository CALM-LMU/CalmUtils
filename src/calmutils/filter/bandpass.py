import numpy as np

def soft_fourier_ring_filter(img, radius=1/4, sigma=1/8, amplitude=0.5, background=1.0, return_mask=False):
    """
    Soft "bandpass" filter that modulates a Gaussian ring of an images's FFT
    
    Parameters
    ----------
    img: np-array
        the image to filter
    radius: float
        radius of the ring, in fraction of image size
    sigma: float
        stddev of ring, in fraction of image size
    amplitude: float
        by how much parts of the FFT in the ring are modulated (should probably be > -background)
    background: float
        by how much parts of the FFT outside of the ring are modulated
    return_mask: boolean
        flag to return the ring mask in addition to filtered image
        
    Returns
    -------
    result: np-array
        filtered image
    mask: np-array (optional)
        mask with wich the image's FFT was multiplied
    """    
    
    shape = img.shape
    mid = (np.array(shape)//2 + 1)

    # coords shape // 2 + 1 along last axis, for rfft
    coords = np.stack(np.meshgrid(*([np.arange(s) for s in shape[:-1]] + [np.arange(shape[-1]//2+1)]), indexing='ij'), -1)
    
    # coords in FFT space, so we do not have to do fftshift
    for i in range(len(shape)):
        coords_i = coords[tuple([slice(s) for s in shape] + [i])]
        coords_i[coords_i >= mid[i]] -= shape[i]
    
    dists = np.linalg.norm(coords, axis=-1)  

    # radius & sigma are in units of fractions of image size
    _rad = np.mean(shape) * radius
    _sig = np.mean(shape) * sigma

    # make Gaussian ring mask
    mask = np.exp(-1/2 * (dists - _rad)**2 / _sig**2)
    mask = background + mask * amplitude


        
    # multiply FFT of image by mask
    result = np.fft.irfftn(np.fft.rfftn(img) * mask)
    
    
    # TODO: make sure result intenisty range is similar to input
    # NB: e.g. this (insert above) does not work
    #mask *= img.size / np.sum(mask)
    
    return result if not return_mask else (result, np.fft.fftshift(mask, axes=tuple(range(len(shape)-1))).real)