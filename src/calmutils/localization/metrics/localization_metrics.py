import numpy as np
from scipy.optimize import linear_sum_assignment

def get_coord_distance_matrix(coords_1, coords_2):
    # get MxN matrix of Euclidean distances between each pair of D-dimensional spots
    # in two MxD and NxD arrays of coordinates coords_1, coords_2
    # NOTE: the einsum calls repeat coordinates the two coordinate sets as rows/columns to common MxNxD shape
    d_vec = np.einsum('ik,jk->ijk', coords_1, np.ones_like(coords_2)) - np.einsum('ik,jk->ijk', np.ones_like(coords_1), coords_2)
    # Norm along last dim: magnitude of distance vectors
    return np.linalg.norm(d_vec, axis=-1)

def get_precision_recall_coordinate_sets(gt_coords, pred_coords, max_dist):

    # get pairwise distance matrix
    # set distances of infeasible matches to large number
    # -> matches that are slightly over max_dist get penalized in matching
    d = get_coord_distance_matrix(gt_coords, pred_coords)
    d[d>max_dist] = max_dist * 9000

    # get optimal matching
    ci, ri = linear_sum_assignment(d)

    # get Precision, Recall
    # precision: # true positives / # predicted positives
    # recall: # true positives / # ground truth positives
    n_true_positives = np.sum(d[ci, ri] < max_dist)
    recall = n_true_positives / len(gt_coords)
    precision = n_true_positives / len(pred_coords)

    return precision, recall

def get_pr_curve_coordinates(gt_coords, predicted_coords, max_dist, prediction_scores=None):
    # get precision-recall curve
    # predicted_coords should be sorted according to some score or will be sorted according to the prediction_scores vector
    # e.g. when doing peak_local_max on LoG response, the highest peaks will come first
    n_candidates = len(predicted_coords)

    # re-order predicted coordinates to be in descending score order
    if prediction_scores is not None:
        predicted_coords = predicted_coords[np.argsort(-prediction_scores)]

    # get prec., rec. for growing subset of predicted peaks
    pr_curve = [get_precision_recall_coordinate_sets(gt_coords, predicted_coords[:t], max_dist) for t in np.arange(1, n_candidates + 1)]

    precision_values, recall_values = np.array(pr_curve).T.astype(float)
    return precision_values, recall_values