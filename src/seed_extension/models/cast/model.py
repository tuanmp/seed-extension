import lightning as L
import torch
import torch.nn as nn

from .attention import CrossAttentionDecoder, SeedSelfAttention
from .embedders import HitEmbedder, SeedEmbedder
from .encoders import IdentityEncoder
from .loss import joint_loss
from .metrics import compute_binned_metrics, compute_efficiency, compute_purity


class CASTModel(L.LightningModule):
    def __init__(
        self,
        d_model=256,
        n_cross_attn_layers=3,
        n_heads=8,
        ff_dim=1024,
        dropout=0.1,
        temperature=0.1,
        fourier_L=8,
        use_cylindrical_pe=True,
        use_seed_self_attn=True,
        hit_encoder="identity",
        learning_rate=1e-4,
        weight_decay=1e-4,
        lr_scheduler="cosine",
        warmup_steps=1000,
        lambda_seed=1.0,
        lambda_hit=1.0,
        use_null_seed=False,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.hit_embedder = HitEmbedder(
            d_model=d_model,
            fourier_L=fourier_L,
            use_detector_features=True,
            use_cylindrical_pe=use_cylindrical_pe,
            dropout=dropout,
            dataset="colliderml"
        )
        self.seed_embedder = SeedEmbedder(d_model=d_model, dropout=dropout)
        self.hit_encoder = self._build_hit_encoder(hit_encoder)
        self.cross_attn = CrossAttentionDecoder(
            d_model=d_model,
            n_heads=n_heads,
            ff_dim=ff_dim,
            n_layers=n_cross_attn_layers,
            dropout=dropout,
        )
        self.seed_self_attn = (
            SeedSelfAttention(d_model=d_model, n_heads=n_heads, dropout=dropout)
            if use_seed_self_attn
            else None
        )
        self.temperature = temperature

        self.null_seed = (
            nn.Parameter(torch.zeros(d_model))
            if use_null_seed
            else None
        )

    def _build_hit_encoder(self, encoder_type: str):
        if encoder_type == "identity":
            return IdentityEncoder()
        elif encoder_type == "binned_self_attn":
            from .encoders import BinnedSelfAttentionEncoder
            return BinnedSelfAttentionEncoder()
        else:
            raise ValueError(f"Unknown hit_encoder: {encoder_type}")

    def forward(self, hits: torch.Tensor, detector_info: torch.Tensor, seed_hit_idx: torch.Tensor) -> torch.Tensor:
        h_emb = self.hit_embedder(hits, detector_info)
        s_emb = self.seed_embedder(h_emb, seed_hit_idx)
        h_emb = self.hit_encoder(h_emb)
        s_emb = self.cross_attn(s_emb, h_emb)
        if self.seed_self_attn is not None:
            s_emb = self.seed_self_attn(s_emb)
        scores = torch.bmm(s_emb, h_emb.transpose(1, 2)) / self.temperature

        if self.null_seed is not None:
            null_scores = torch.matmul(h_emb, self.null_seed) / self.temperature  # (B, N_h)
            scores = torch.cat([scores, null_scores.unsqueeze(1)], dim=1)

        return scores
    
    def get_input_tensors(self, batch):
        hit_coords = batch["hits"]
        hit_detector_info = batch["hit_detector_info"]
        seed_hit_idx = batch["seeds"]
        return hit_coords, hit_detector_info, seed_hit_idx

    @staticmethod
    def _compute_preds(scores_2d, targets_2d, null_seed_active):
        """Build a binary prediction matrix from scores.

        Args:
            scores_2d: (N, N_h) where N = N_s + (1 if null_seed_active else 0)
            targets_2d: (N_s, N_h) binary ground truth
            null_seed_active: if True, the last row is the null seed and hits
                              assigned to it are excluded from predictions.

        Returns:
            preds: (N_s, N_h) binary predictions, null-assigned hits are all zeros.
        """
        best_seed = scores_2d.argmax(dim=0)             # (N_h,)
        N_s_real = targets_2d.shape[0]

        if null_seed_active:
            real_mask = best_seed < N_s_real
        else:
            real_mask = torch.ones_like(best_seed, dtype=torch.bool)

        preds = torch.zeros(N_s_real, scores_2d.shape[1], device=scores_2d.device)
        preds[best_seed[real_mask], torch.arange(scores_2d.shape[1], device=scores_2d.device)[real_mask]] = 1.0
        return preds

    def training_step(self, batch, batch_idx):

        hit_x, hit_detector_info, seed_hit_idx = self.get_input_tensors(batch)
        targets = batch["targets"]

        scores = self(hit_x, hit_detector_info, seed_hit_idx)
        loss, L_seed, L_hit = joint_loss(
            scores, targets, temperature=1.0,
            lambda_seed=self.hparams.lambda_seed,
            lambda_hit=self.hparams.lambda_hit,
            use_null_seed=self.null_seed is not None,
        )
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train_loss_seed", L_seed, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train_loss_hit", L_hit, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):

        hit_x, hit_detector_info, seed_hit_idx = self.get_input_tensors(batch)
        targets = batch["targets"]

        scores = self(hit_x, hit_detector_info, seed_hit_idx)
        loss, L_seed, L_hit = joint_loss(
            scores, targets, temperature=1.0,
            lambda_seed=self.hparams.lambda_seed,
            lambda_hit=self.hparams.lambda_hit,
            use_null_seed=self.null_seed is not None,
        )

        scores_2d = scores.squeeze(0)
        if scores_2d.size(0) == 0 or scores_2d.size(1) == 0:
            self.log("val_loss", loss, on_step=False, on_epoch=True)
            return loss

        targets_2d = targets.squeeze(0)
        preds = self._compute_preds(
            scores_2d, targets_2d, null_seed_active=self.null_seed is not None,
        )

        eff = compute_efficiency(preds, targets_2d)
        pur = compute_purity(preds, targets_2d)

        metrics = {"val_loss": loss, "val_eff": eff, "val_pur": pur}

        kinematics = batch.get("kinematics")
        if kinematics is not None and kinematics.numel() > 0:
            kin = kinematics.squeeze(0) if kinematics.dim() == 3 else kinematics
            binned = compute_binned_metrics(preds, targets_2d, kin)
            metrics.update({f"val_{k}": v for k, v in binned.items()})

        self.log_dict(metrics, on_step=False, on_epoch=True)
        return loss

    def test_step(self, batch, batch_idx):
        batch = batch[0] if isinstance(batch, list) else batch
        hit_x, hit_detector_info, seed_hit_idx = self.get_input_tensors(batch)
        targets = batch["targets"]

        scores = self(hit_x, hit_detector_info, seed_hit_idx)
        loss, L_seed, L_hit = joint_loss(
            scores, targets, temperature=1.0,
            lambda_seed=self.hparams.lambda_seed,
            lambda_hit=self.hparams.lambda_hit,
            use_null_seed=self.null_seed is not None,
        )

        scores_2d = scores.squeeze(0)
        self.log("test_loss", loss, on_step=False, on_epoch=True)

        if scores_2d.size(0) == 0 or scores_2d.size(1) == 0:
            return loss

        targets_2d = targets.squeeze(0)
        preds = self._compute_preds(
            scores_2d, targets_2d, null_seed_active=self.null_seed is not None,
        )

        eff = compute_efficiency(preds, targets_2d)
        pur = compute_purity(preds, targets_2d)
        metrics = {"test_loss": loss, "test_eff": eff, "test_pur": pur}

        kinematics = batch.get("kinematics")
        if kinematics is not None and kinematics.numel() > 0:
            kin = kinematics.squeeze(0) if kinematics.dim() == 3 else kinematics
            binned = compute_binned_metrics(preds, targets_2d, kin)
            metrics.update({f"test_{k}": v for k, v in binned.items()})

        self.log_dict(metrics, on_step=False, on_epoch=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,
        )
        try:
            max_epochs = self.trainer.max_epochs
        except RuntimeError:
            max_epochs = 50
        warmup = self.hparams.warmup_steps
        lr_scheduler_cfg = self.hparams.lr_scheduler

        if warmup > 0:
            warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
                optimizer, start_factor=0.01, total_iters=warmup,
            )
        else:
            warmup_scheduler = None

        if lr_scheduler_cfg == "cosine":
            main_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=max_epochs,
            )
        elif lr_scheduler_cfg == "plateau":
            main_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="min", patience=5,
            )
        else:
            raise ValueError(f"Unknown lr_scheduler: {lr_scheduler_cfg}")

        if warmup_scheduler is not None:
            scheduler = torch.optim.lr_scheduler.SequentialLR(
                optimizer,
                schedulers=[warmup_scheduler, main_scheduler],
                milestones=[warmup],
            )
        else:
            scheduler = main_scheduler

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
                "interval": "epoch",
            },
        }
