import lightning as L
import torch

from .attention import CrossAttentionDecoder, SeedSelfAttention
from .embedders import HitEmbedder, SeedEmbedder
from .encoders import IdentityEncoder
from .loss import info_nce_loss
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
    ):
        super().__init__()
        self.save_hyperparameters()

        self.hit_embedder = HitEmbedder(
            d_model=d_model,
            fourier_L=fourier_L,
            use_detector_features=True,
            use_cylindrical_pe=use_cylindrical_pe,
            dropout=dropout,
        )
        self.seed_embedder = SeedEmbedder(d_model=d_model, dropout=dropout)
        self.hit_encoder = IdentityEncoder()
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

    def forward(self, hits, seeds):
        h_emb = self.hit_embedder(hits)
        s_emb = self.seed_embedder(seeds)
        h_emb = self.hit_encoder(h_emb)
        s_emb = self.cross_attn(s_emb, h_emb)
        if self.seed_self_attn is not None:
            s_emb = self.seed_self_attn(s_emb)
        scores = torch.bmm(s_emb, h_emb.transpose(1, 2)) / self.temperature
        return scores

    def training_step(self, batch, batch_idx):
        if isinstance(batch, list):
            sample = batch[0]
        else:
            sample = batch
        hits = sample["hits"]
        seeds = sample["seeds"]
        targets = sample["targets"]
        if hits.dim() == 2:
            hits = hits.unsqueeze(0)
        if seeds.dim() == 2:
            seeds = seeds.unsqueeze(0)
        if targets.dim() == 2:
            targets = targets.unsqueeze(0)

        scores = self(hits, seeds)
        loss = info_nce_loss(scores, targets, temperature=1.0)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        if isinstance(batch, list):
            sample = batch[0]
        else:
            sample = batch
        hits = sample["hits"]
        seeds = sample["seeds"]
        targets = sample["targets"]
        if hits.dim() == 2:
            hits = hits.unsqueeze(0)
        if seeds.dim() == 2:
            seeds = seeds.unsqueeze(0)
        if targets.dim() == 2:
            targets = targets.unsqueeze(0)

        scores = self(hits, seeds)
        loss = info_nce_loss(scores, targets, temperature=1.0)

        scores_2d = scores.squeeze(0)
        best_seed = scores_2d.argmax(dim=0)
        N_s = scores_2d.shape[0]
        preds = torch.zeros(N_s, scores_2d.shape[1], device=scores.device)
        preds[best_seed, torch.arange(scores_2d.shape[1])] = 1.0

        targets_2d = targets.squeeze(0)

        eff = compute_efficiency(preds, targets_2d)
        pur = compute_purity(preds, targets_2d)

        metrics = {"val_loss": loss, "val_eff": eff, "val_pur": pur}

        kinematics = sample.get("kinematics")
        if kinematics is not None and kinematics.numel() > 0:
            kin = kinematics.squeeze(0) if kinematics.dim() == 3 else kinematics
            binned = compute_binned_metrics(preds, targets_2d, kin)
            metrics.update({f"val_{k}": v for k, v in binned.items()})

        self.log_dict(metrics, on_step=False, on_epoch=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,
        )
        max_epochs = self.trainer.max_epochs if self._trainer is not None else 50
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max_epochs
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
                "interval": "epoch",
            },
        }
