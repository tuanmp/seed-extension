import torch


def info_nce_loss(scores, targets, temperature=0.1):
    """Multi-positive InfoNCE loss.

    Args:
        scores: (B, N_s, N_h) similarity scores between seeds and hits
        targets: (B, N_s, N_h) binary matrix, 1 if seed_i matches hit_j
        temperature: softmax temperature

    Returns:
        Scalar loss, averaged over batch and seeds that have positives.
    """
    if scores.numel() == 0:
        return scores.sum() * 0.0
    if scores.shape[-1] == 0:
        return scores.sum() * 0.0
    scaled = scores / temperature
    scaled = scaled - scaled.max(dim=-1, keepdim=True).values
    exp_scores = torch.exp(scaled)
    numerator = (exp_scores * targets).sum(dim=-1)
    denominator = exp_scores.sum(dim=-1)
    has_positives = numerator > 0
    if not has_positives.any():
        return scores.sum() * 0.0
    loss_per_seed = -torch.log(numerator[has_positives] / denominator[has_positives])
    return loss_per_seed.mean()
