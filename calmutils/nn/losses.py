"""Loss functions not covered by torch/torchvision built-ins."""

import torch
import torch.nn.functional as F

__all__ = ["focal_loss", "cosine_distance_loss", "negative_cosine_similarity_loss"]


def focal_loss(logits_predicted, target, gamma=1, class_weights=None):
    """
    Multi-class focal loss (Lin et al.): CE * (1 - p_t)^gamma, with (1 - p_t) = exp(-CE).
    class_weights play the role of the alpha balancing term.
    (torchvision.ops.sigmoid_focal_loss covers only the binary case.)
    """
    ce_loss = F.cross_entropy(logits_predicted, target, class_weights, reduction='none')
    return (ce_loss * (1 - torch.exp(-ce_loss)) ** gamma).mean()


def cosine_distance_loss(vecs_pred, vecs_true, mask=None):
    """
    1 - mean cosine similarity between two spatial vector maps (B, C, ...),
    optionally restricted to masked positions (mask broadcastable to the similarity map).
    Returns 0.0 for an empty mask (constant, zero gradient).
    """
    sim_cosine = F.cosine_similarity(vecs_pred, vecs_true)
    if mask is not None:
        if mask.sum() > 0:
            sim_cosine = sim_cosine[mask.bool().squeeze()]
        else:
            return torch.zeros((), device=vecs_pred.device, dtype=vecs_pred.dtype)
    return 1 - sim_cosine.mean()


# legacy name kept for backward compatibility
negative_cosine_similarity_loss = cosine_distance_loss
