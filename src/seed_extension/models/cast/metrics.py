import torch


def _bin_values(values, bin_edges):
    """Return bin index (0..n_bins-1) for each value. -1 means out of range."""
    bin_idx = torch.bucketize(values, bin_edges, right=False) - 1
    return bin_idx.clamp(-1, len(bin_edges) - 2)


def compute_efficiency(preds, targets):
    """Per-seed efficiency: fraction of true hits correctly predicted.

    Args:
        preds: (N_s, N_h) binary predictions
        targets: (N_s, N_h) binary ground truth
    Returns:
        Mean efficiency across seeds with >=1 positive hit.
    """
    n_true = targets.sum(dim=-1)
    n_correct = (preds * targets).sum(dim=-1)
    has_pos = n_true > 0
    if has_pos.sum() == 0:
        return 0.0
    eff = n_correct[has_pos].float() / n_true[has_pos].float()
    return eff.mean().item()


def compute_purity(preds, targets):
    """Per-seed purity: fraction of predicted hits that are true positives.

    Args:
        preds: (N_s, N_h) binary predictions
        targets: (N_s, N_h) binary ground truth
    Returns:
        Mean purity across seeds with >=1 prediction.
    """
    n_pred = preds.sum(dim=-1)
    n_correct = (preds * targets).sum(dim=-1)
    has_pred = n_pred > 0
    if has_pred.sum() == 0:
        return 0.0
    pur = n_correct[has_pred].float() / n_pred[has_pred].float()
    return pur.mean().item()


def compute_binned_metrics(preds, targets, kinematics):
    """Compute metrics binned by eta, pT, d0, z0.

    Args:
        preds: (N_s, N_h) binary predictions
        targets: (N_s, N_h) binary ground truth
        kinematics: (N_s, 6) [eta, pt, d0, z0, theta, phi]

    Returns:
        Dict with keys like 'eff_mean', 'pur_mean', 'eff_eta_bin0', etc.
    """
    metrics = {}
    metrics["eff_mean"] = compute_efficiency(preds, targets)
    metrics["pur_mean"] = compute_purity(preds, targets)

    bin_configs = {
        "eta": (0, torch.tensor([-2.5, -2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 2.5])),
        "pt": (1, torch.tensor([0.0, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0])),
        "d0": (2, torch.tensor([0.0, 0.1, 0.5, 1.0, 5.0, 10.0])),
        "z0": (3, torch.tensor([0.0, 0.5, 1.0, 5.0, 10.0, 20.0])),
    }

    for name, (col_idx, bin_edges) in bin_configs.items():
        vals = kinematics[:, col_idx]
        bin_idx = _bin_values(vals, bin_edges)
        for b in range(len(bin_edges) - 1):
            mask = bin_idx == b
            if mask.sum() == 0:
                continue
            bin_eff = compute_efficiency(preds[mask], targets[mask])
            metrics[f"eff_{name}_bin{b}"] = bin_eff

    return metrics
