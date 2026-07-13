import pytest
import torch
from seed_extension.models.cast.loss import hit_side_ce_loss, info_nce_loss, joint_loss


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

    def test_l_out_matches_l_in_with_single_positive(self):
        """With exactly 1 positive, L_out = L_in, so results are unchanged."""
        scores = torch.randn(1, 5, 20)
        targets = torch.zeros(1, 5, 20)
        for i in range(5):
            targets[0, i, i] = 1.0
            scores[0, i, i] += 3.0
        loss = info_nce_loss(scores, targets, temperature=1.0)
        assert loss.item() > 0.0

    def test_l_out_averages_over_positives(self):
        """Seeds with multiple positives average individual log-probabilities."""
        scores = torch.zeros(1, 1, 4)
        scores[0, 0, 0] = 5.0
        scores[0, 0, 1] = 2.0
        scores[0, 0, 2] = 0.0
        scores[0, 0, 3] = 0.0
        targets = torch.tensor([[[1.0, 1.0, 0.0, 0.0]]])
        loss = info_nce_loss(scores, targets, temperature=1.0)
        assert loss.item() >= 0.0
        assert torch.isfinite(loss)


class TestHitSideCELoss:
    def test_perfect_prediction_zero_loss(self):
        scores = torch.zeros(2, 4, 10)
        targets = torch.zeros(2, 4, 10)
        for b in range(2):
            for j in range(10):
                seed_idx = j % 4
                scores[b, seed_idx, j] = 10.0
                targets[b, seed_idx, j] = 1.0
        loss = hit_side_ce_loss(scores, targets, temperature=1.0)
        assert loss.item() < 0.1

    def test_random_prediction_high_loss(self):
        scores = torch.randn(1, 5, 100)
        targets = torch.zeros(1, 5, 100)
        for j in range(100):
            seed_idx = j % 5
            targets[0, seed_idx, j] = 1.0
        loss = hit_side_ce_loss(scores, targets, temperature=1.0)
        assert loss.item() > 0.5

    def test_orphan_hits_excluded(self):
        """Hits with no true seed (all-zero column) contribute zero gradient."""
        scores = torch.randn(1, 3, 5)
        targets = torch.zeros(1, 3, 5)
        for j in range(3):
            targets[0, j % 3, j] = 1.0
        loss_with_orphans = hit_side_ce_loss(scores, targets, temperature=1.0)

        scores_no_orphans = scores[:, :, :3]
        targets_no_orphans = targets[:, :, :3]
        loss_no_orphans = hit_side_ce_loss(scores_no_orphans, targets_no_orphans, temperature=1.0)

        assert abs(loss_with_orphans.item() - loss_no_orphans.item()) < 1e-4

    def test_all_orphans_returns_zero(self):
        scores = torch.randn(1, 2, 4)
        targets = torch.zeros(1, 2, 4)
        loss = hit_side_ce_loss(scores, targets, temperature=1.0)
        assert torch.isfinite(loss)
        assert loss.item() == 0.0

    def test_empty_input_returns_zero(self):
        scores = torch.randn(0, 3, 5)
        targets = torch.zeros(0, 3, 5)
        loss = hit_side_ce_loss(scores, targets)
        assert torch.isfinite(loss)


class TestJointLoss:
    def test_combines_both_terms(self):
        scores = torch.randn(1, 5, 20)
        targets = torch.zeros(1, 5, 20)
        for i in range(5):
            for j in range(i * 4, (i + 1) * 4):
                targets[0, i, j] = 1.0
                scores[0, i, j] += 3.0

        total, L_seed, L_hit = joint_loss(scores, targets, lambda_seed=1.0, lambda_hit=1.0)
        assert total.item() == pytest.approx(L_seed.item() + L_hit.item())

    def test_lambda_zero_suppresses_term(self):
        scores = torch.randn(1, 5, 20)
        targets = torch.zeros(1, 5, 20)
        for i in range(5):
            targets[0, i, i] = 1.0

        total_seed_only, L_seed, L_hit = joint_loss(
            scores, targets, lambda_seed=1.0, lambda_hit=0.0,
        )
        assert L_hit.item() >= 0.0
        assert total_seed_only.item() == pytest.approx(L_seed.item())

        total_hit_only, L_seed2, L_hit2 = joint_loss(
            scores, targets, lambda_seed=0.0, lambda_hit=1.0,
        )
        assert L_seed2.item() >= 0.0
        assert total_hit_only.item() == pytest.approx(L_hit2.item())

    def test_perfect_prediction_zero_total(self):
        scores = torch.zeros(2, 3, 6)
        targets = torch.zeros(2, 3, 6)
        for b in range(2):
            for i in range(3):
                for j in range(i * 2, (i + 1) * 2):
                    scores[b, i, j] = 10.0
                    targets[b, i, j] = 1.0
        total, L_seed, L_hit = joint_loss(scores, targets, lambda_seed=1.0, lambda_hit=1.0)
        assert L_hit.item() < 0.1
        assert L_seed.item() == pytest.approx(0.693, abs=1e-2)

    def test_gradients_flow_both_terms(self):
        scores = torch.randn(1, 5, 10, requires_grad=True)
        targets = torch.zeros(1, 5, 10)
        for i in range(5):
            targets[0, i, i] = 1.0

        total, L_seed, L_hit = joint_loss(scores, targets, lambda_seed=1.0, lambda_hit=1.0)
        total.backward()
        assert scores.grad is not None
        assert scores.grad.abs().sum() > 0
