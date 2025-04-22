import numpy as np
from scipy.optimize import linear_sum_assignment
from skimage.measure import regionprops
from skimage.segmentation import relabel_sequential
import fastremap


def _correct_next_plane(img1, img2, iou_threshold=0.0, max_label=None):

    # no maximum label given, start after max of img1
    if max_label is None:
        max_label = img1.max()

    # relabel img2 so everything has a higher label than img1 (or previous highest label)
    img2_relab, _, _ = relabel_sequential(img2, max_label+1)

    # get regionprops
    rprops1 = regionprops(img1)
    rprops2 = regionprops(img2_relab)

    # one of the images is empty -> nothing to do
    if len(rprops1) == 0 or len(rprops2) == 0:
        return img2_relab

    # maximum number of objects in one of the two images
    n_labs = np.max([len(rprops1), len(rprops2)])

    # init IOU matrix
    ious = np.zeros((n_labs, n_labs))

    # fill IOU matrix (rows: objects in img1, cols: objects in img2)
    for i, rprop1 in enumerate(rprops1):
        for j, rprop2 in enumerate(rprops2):

            # area within prop1 mask that has prop2 label in img2
            intersection = np.sum(img2_relab[rprop1.slice][rprop1.image] == rprop2.label)
            # union: area sum - intersection
            union = rprop1.area + rprop2.area - intersection
            iou = intersection / union

            ious[i, j] = iou

    # find maximum IOU matching
    row_idx, col_idx = linear_sum_assignment(ious, True)

    # init remap table to keep everything
    labs2 = fastremap.unique(img2_relab)
    remap_table = dict(zip(labs2, labs2))

    for ri, ci in zip(row_idx, col_idx):
        # if we have a valid match
        # (object exists in img1 and img2)
        valid_match = ri < len(rprops1) and ci < len(rprops2)
        # and the IOU of the matched objects is bigger than threshold
        if valid_match and ious[ri,ci] > iou_threshold:
            # add to remap table
            remap_table[rprops2[ci].label] = rprops1[ri].label

    return fastremap.remap(img2_relab, remap_table)


def merge_label_slices(imgs, iou_threshold = 0.0):

    """
    Merge label maps along the first dimension (via LAP matching based on overlap).
    Can be used to merge 2D segmentations into 3D, but also e.g. for matching GT to mask predictions for evaluation.

    Example call:
    planes = [prediction plane1, prediction plane2, ...]
    labels_3d = merge_label_slices(planes)
    """

    # relabel first plane -> first "corrected" plane of result
    res = [relabel_sequential(imgs[0])[0]]
    max_label = res[0].max()
    
    for img in imgs[1:]:
        
        # correct each other plane in comparison to last (corrected) plane
        res.append(_correct_next_plane(res[-1], img, iou_threshold, max_label))

        # new maximum label: maximum of previous max and max in corrected plane
        max_label = max(max_label, res[-1].max())
    
    # relabel everything, as we might have skipped some numbers
    return relabel_sequential(np.stack(res))[0]
