import torch
import torch.nn.functional as F


def info_nce_loss(scores, targets, temperature=1.0):
    """Multi-positive InfoNCE loss (L_out / sum-outside-log variant).

    For each seed i with positives P(i):
        L_i = -(1/|P(i)|) * sum_{p in P(i)} log(exp(s_ip/T) / sum_a exp(s_ia/T))

    Args:
        scores: (B, N_s, N_h) similarity scores between seeds and hits
        targets: (B, N_s, N_h) binary matrix, 1 if seed_i matches hit_j
        temperature: softmax temperature (already applied in model forward
                     by convention, keep at 1.0)

    Returns:
        Scalar loss, averaged over batch and seeds that have positives.
    """
    if scores.numel() == 0 or scores.shape[-1] == 0:
        return scores.sum() * 0.0

    scaled = scores / temperature
    scaled = scaled - scaled.max(dim=-1, keepdim=True).values
    log_denom = torch.logsumexp(scaled, dim=-1, keepdim=True)
    log_prob = scaled - log_denom

    pos_counts = targets.sum(dim=-1)
    has_positives = pos_counts > 0
    if not has_positives.any():
        return scores.sum() * 0.0

    sum_log_prob_pos = (log_prob * targets).sum(dim=-1)
    loss_per_seed = -sum_log_prob_pos[has_positives] / pos_counts[has_positives]
    return loss_per_seed.mean()


def hit_side_ce_loss(scores, targets, temperature=1.0):
    """Column-wise cross-entropy: for each hit, classify which seed owns it.

    Each hit has exactly one true seed (or none — masked out).  This
    complements the seed-side InfoNCE by providing gradients in the
    hit-to-seed direction.

    Args:
        scores: (B, N_s, N_h) similarity scores between seeds and hits
        targets: (B, N_s, N_h) binary matrix, 1 if seed_i matches hit_j
        temperature: softmax temperature (already applied in model forward
                     by convention, keep at 1.0)

    Returns:
        Scalar loss, averaged over hits that have a true seed.
    """
    if scores.numel() == 0 or scores.shape[-1] == 0:
        return scores.sum() * 0.0

    scores_t = scores.transpose(1, 2) / temperature       # (B, N_h, N_s)
    targets_t = targets.transpose(1, 2)                    # (B, N_h, N_s)

    true_seed = targets_t.argmax(dim=-1)                   # (B, N_h)
    has_seed = targets_t.sum(dim=-1) > 0                   # (B, N_h)

    if not has_seed.any():
        return scores.sum() * 0.0

    B, N_h, N_s = scores_t.shape
    log_p = F.log_softmax(scores_t, dim=-1)                # (B, N_h, N_s)
    nll = -log_p.reshape(-1, N_s)[torch.arange(B * N_h), true_seed.reshape(-1)]
    nll = nll.reshape(B, N_h)                              # (B, N_h)

    loss = nll[has_seed].mean()
    return loss


def joint_loss(scores, targets, temperature=1.0, lambda_seed=1.0, lambda_hit=1.0):
    """Combined seed-side InfoNCE and hit-side cross-entropy.

    Args:
        scores: (B, N_s, N_h) similarity scores
        targets: (B, N_s, N_h) binary target matrix
        temperature: passed through to both sub-losses (default 1.0)
        lambda_seed: weight for seed-side InfoNCE
        lambda_hit: weight for hit-side cross-entropy

    Returns:
        total: scalar joint loss
        L_seed: unscaled seed-side component (for logging)
        L_hit: unscaled hit-side component (for logging)
    """
    L_seed = info_nce_loss(scores, targets, temperature)
    L_hit = hit_side_ce_loss(scores, targets, temperature)
    total = lambda_seed * L_seed + lambda_hit * L_hit
    return total, L_seed, L_hit
