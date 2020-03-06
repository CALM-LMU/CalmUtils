from itertools import product
import warnings

import numpy as np


def get_axes_aligned_overlap(shape1, shape2, transform1=None, transform2=None):

    if transform1 is None:
        transform1 = np.eye(len(shape1) + 1)
    if transform2 is None:
        transform2 = np.eye(len(shape2) + 1)

    corners1 = np.array(list(product(*(list(zip([0] * len(shape1), shape1)) + [[1]]))))
    corners1 = corners1 @ transform1.transpose()
    corners2 = np.array(list(product(*(list(zip([0] * len(shape2), shape2)) + [[1]]))))
    corners2 = corners2 @ transform2.transpose()

    mins = np.max([np.min(corners1, axis=0), np.min(corners2, axis=0)], axis=0)
    maxs = np.min([np.max(corners1, axis=0), np.max(corners2, axis=0)], axis=0)

    return mins[:-1], maxs[:-1]

def get_axes_aligned_bbox(shapes, transforms):
    mins = []
    maxs = []
    for shape, transform in zip(shapes, transforms):
        corners = np.array(list(product(*(list(zip([0] * len(shape), shape)) + [[1]]))))
        corners = corners @ transform.transpose()
        min_i = np.min(corners, axis=0)
        max_i = np.max(corners, axis=0)
        mins.append(min_i)
        maxs.append(max_i)
    mins = np.min(mins, axis=0)
    maxs = np.max(maxs, axis=0)
    return mins[:-1], maxs[:-1]

def get_transform_from_shift(shift):
    tr = np.eye(len(shift) + 1)
    tr[:-1,-1] = shift
    return tr

def phasecorr_align(img1, img2):

    freq1 = np.fft.rfftn(img1)
    freq2 = np.fft.rfftn(img2).conj()

    fccor = freq1*freq2
    fccor1 = fccor / (np.abs(fccor))
    fccor1[np.abs(fccor) < 1e-12] = 0

    pcm = np.fft.irfftn(fccor1, img1.shape)

    # get ndim^2 largest values
    # TODO: subpixel localization!
    idxs = np.argpartition(-pcm.ravel(), int(img1.ndim**2))[:int(img1.ndim**2)]

    shift_max = None
    ccor = None
    for idx in idxs:

        shifts = np.unravel_index(idx, pcm.shape)
        shifts = np.array(shifts, dtype=np.float64)

        for off_i in list(product(*zip([0] * len(pcm.shape), pcm.shape))):

            shift_i = shifts - np.array(off_i)
            min_1, max_1 = get_axes_aligned_overlap(img1.shape, img2.shape, transform2=get_transform_from_shift(shift_i))
            min_2, max_2 = get_axes_aligned_overlap(img1.shape, img2.shape, transform2=get_transform_from_shift(-shift_i))
            min_1, min_2, max_1, max_2 = (arg.astype(np.int) for arg in (min_1, min_2, max_1, max_2))

            patch1 = img1[tuple((slice(mi, ma) for mi, ma in zip(min_1, max_1)))]
            patch2 = img2[tuple((slice(mi, ma) for mi, ma in zip(min_2, max_2)))]

            # skip zero volumes
            if np.prod(patch1.shape) < 1:
                continue

            # normalized ccor
            patch1 = patch1 - np.mean(patch1)
            patch2 = patch2 - np.mean(patch2)
            ccor_i = np.sum(patch1 * patch2) / np.sqrt(np.sum((patch1**2))) / np.sqrt(np.sum((patch2**2)))

            #print(shift_i, ccor_i, np.prod(patch1.shape))
            if ccor is None or ccor_i > ccor:
                ccor = ccor_i
                shift_max = shift_i

    # TODO: return PCM?
    return np.array(shift_max), ccor #, pcm

try:
    import torch

    def phasecorr_align_torch(img1, img2, device=None):

        with torch.no_grad():
            img1_ = torch.from_numpy(img1).type(torch.FloatTensor)
            img2_ = torch.from_numpy(img2).type(torch.FloatTensor)
            if device is not None:
                img1_ = img1_.to(device)
                img2_ = img2_.to(device)

            freq1 = torch.rfft(img1_, len(img1_.shape))
            freq2 = torch.rfft(img2_, len(img2_.shape))

            # complex conjugate
            freq2[tuple(slice(s) for s in freq2.shape[:-1]) + (1,)] *= -1

            fccor = freq1*freq2
            fccor1 = fccor / (torch.abs(fccor))
            fccor1[torch.abs(fccor) < 1e-12] = 0

            pcm = torch.irfft(fccor1, len(img1_.shape), signal_sizes=img1_.shape)

            # get ndim^2 largest values
            # TODO: subpixels
            _, idxs = torch.topk(pcm.flatten(), int(img1.ndim**2))

            shift_max = None
            ccor = None
            for idx in idxs:

                shifts = np.unravel_index(idx if device is None else idx.cpu(), pcm.shape)
                shifts = np.array(shifts, dtype=np.float64)

                # check the different possible positive and negative shifts
                for off_i in list(product(*zip([0] * len(pcm.shape), pcm.shape))):

                    shift_i = shifts - np.array(off_i)
                    min_1, max_1 = get_axes_aligned_overlap(img1.shape, img2.shape, transform2=get_transform_from_shift(shift_i))
                    min_2, max_2 = get_axes_aligned_overlap(img1.shape, img2.shape, transform2=get_transform_from_shift(-shift_i))
                    min_1, min_2, max_1, max_2 = (arg.astype(np.int) for arg in (min_1, min_2, max_1, max_2))

                    patch1 = img1_[tuple((slice(mi, ma) for mi, ma in zip(min_1, max_1)))]
                    patch2 = img2_[tuple((slice(mi, ma) for mi, ma in zip(min_2, max_2)))]

                    # skip zero volumes
                    if np.prod(patch1.shape) < 1:
                        continue

                    # normalized ccor
                    patch1 = patch1 - torch.mean(patch1)
                    patch2 = patch2 - torch.mean(patch2)
                    ccor_i = torch.sum(patch1 * patch2) / torch.sqrt(torch.sum((patch1**2))) / torch.sqrt(torch.sum((patch2**2)))

                    #print(shift_i, ccor_i)
                    if ccor is None or ccor_i > ccor:
                        ccor = ccor_i
                        shift_max = shift_i

        return np.array(shift_max), ccor.numpy()  if device is None else ccor.cpu().numpy()#, pcm

except ImportError as e:
    warnings.warn('PyTorch not available. Install it for accelerated versions of phase correlation')