import pytest
import torch
from seed_extension.models.cast.loss import info_nce_loss


class TestInfoNCELoss:
    def test_perfect_prediction_zero_loss(self):
        scores = torch.zeros(4, 3, 5)
        targets = torch.zeros(4, 3, 5)
        for b in range(4):
            for i in range(3):
                scores[b, i, i] = 10.0
                targets[b, i, i] = 1.0
        loss = info_nce_loss(scores, targets, temperature=1.0)
        assert loss.item() < 0.1

    def test_random_prediction_high_loss(self):
        scores = torch.randn(4, 10, 100)
        targets = torch.zeros(4, 10, 100)
        for b in range(4):
            for i in range(10):
                matches = torch.randperm(100)[:5]
                targets[b, i, matches] = 1.0
                scores[b, i, matches] += 2.0
        loss = info_nce_loss(scores, targets, temperature=0.1)
        assert loss.item() > 0.5

    def test_no_positives_returns_zero(self):
        scores = torch.randn(1, 1, 10)
        targets = torch.zeros(1, 1, 10)
        loss = info_nce_loss(scores, targets, temperature=1.0)
        assert torch.isfinite(loss)
        assert loss.item() >= 0.0

    def test_multiple_positives(self):
        scores = torch.randn(2, 3, 10)
        targets = torch.zeros(2, 3, 10)
        for b in range(2):
            for i in range(3):
                pos_indices = [i, i + 3, i + 6]
                targets[b, i, pos_indices] = 1.0
                scores[b, i, pos_indices] = 5.0 - i
        loss = info_nce_loss(scores, targets, temperature=1.0)
        assert loss.item() >= 0.0

    def test_temperature_effect(self):
        scores = torch.tensor([[[0.0, 1.0, 0.0]]])
        targets = torch.tensor([[[0.0, 1.0, 0.0]]])
        loss_high_t = info_nce_loss(scores, targets, temperature=10.0)
        loss_low_t = info_nce_loss(scores, targets, temperature=0.1)
        assert loss_low_t < loss_high_t

    def test_batch_independence(self):
        scores1 = torch.randn(2, 5, 20)
        targets1 = torch.zeros(2, 5, 20)
        for b in range(2):
            for i in range(5):
                targets1[b, i, i] = 1.0
        loss_batch = info_nce_loss(scores1, targets1)
        loss0 = info_nce_loss(scores1[0:1], targets1[0:1])
        loss1 = info_nce_loss(scores1[1:2], targets1[1:2])
        loss_manual = (loss0 + loss1) / 2.0
        assert abs(loss_batch.item() - loss_manual.item()) < 1e-4
