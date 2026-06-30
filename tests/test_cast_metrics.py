import torch
from seed_extension.models.cast.metrics import compute_efficiency, compute_purity, compute_binned_metrics


class TestComputeEfficiency:
    def test_perfect(self):
        targets = torch.tensor([[1, 1, 0, 0], [0, 0, 1, 1]])
        preds = torch.tensor([[1, 1, 0, 0], [0, 0, 1, 1]])
        assert compute_efficiency(preds, targets) == 1.0

    def test_half(self):
        targets = torch.tensor([[1, 1, 0, 0]])
        preds = torch.tensor([[1, 0, 0, 0]])
        assert compute_efficiency(preds, targets) == 0.5

    def test_empty_seed(self):
        targets = torch.tensor([[0, 0, 0, 0]])
        preds = torch.tensor([[0, 0, 0, 0]])
        assert compute_efficiency(preds, targets) == 0.0


class TestComputePurity:
    def test_perfect(self):
        targets = torch.tensor([[1, 1, 0, 0]])
        preds = torch.tensor([[1, 1, 0, 0]])
        assert compute_purity(preds, targets) == 1.0

    def test_false_positive(self):
        targets = torch.tensor([[1, 1, 0, 0]])
        preds = torch.tensor([[1, 1, 1, 0]])
        assert abs(compute_purity(preds, targets) - 2.0 / 3.0) < 1e-6


class TestBinnedMetrics:
    def test_output_structure(self):
        targets = torch.zeros(3, 100)
        preds = torch.zeros(3, 100)
        kinematics = torch.zeros(3, 6)
        kinematics[:, 0] = torch.tensor([-2.3, -1.7, 0.5])
        for i in range(3):
            targets[i, i * 10 : (i + 1) * 10] = 1
            preds[i, i * 10 : (i + 1) * 10] = 1
        result = compute_binned_metrics(preds, targets, kinematics)
        assert "eff_mean" in result
        assert "pur_mean" in result
        assert "eff_eta_bin0" in result
