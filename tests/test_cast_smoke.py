"""Smoke test: 1-epoch CAST training on a few ColliderML events."""
import os
import pytest
import lightning as L

from seed_extension.data.dataset import SeedExtensionDataset
from seed_extension.models.cast.model import CASTModel
from colliderml_dataloader import ColliderMLDataModule


@pytest.mark.slow
def test_cast_smoke_train():
    datadir = os.environ.get(
        "COLLIDERML_DATA_DIR",
        "/pscratch/sd/p/pmtuan/.cache/colliderml",
    )

    dm = ColliderMLDataModule(
        data_dir=datadir,
        process="ttbar",
        pileup="pu200",
        max_train_events=4,
        max_val_events=2,
        max_test_events=0,
        batch_size=1,
        num_workers=0,
        dataset_cls=SeedExtensionDataset,
        min_track_hits=5,
        seed_strategy="fixed_innermost",
        target_vertices=1,
        primary_only=True,
        predict_seed_hits=False,
    )

    model = CASTModel(
        d_model=64,
        n_cross_attn_layers=1,
        n_heads=2,
        ff_dim=256,
        dropout=0.0,
        temperature=0.1,
        fourier_L=2,
        use_cylindrical_pe=False,
        use_seed_self_attn=False,
        hit_encoder="identity",
        learning_rate=1e-3,
    )

    trainer = L.Trainer(
        max_epochs=1,
        accelerator="cpu",
        devices=1,
        deterministic=True,
        enable_progress_bar=False,
        enable_model_summary=False,
        logger=False,
        limit_train_batches=2,
        limit_val_batches=1,
    )

    trainer.fit(model, dm)
    assert trainer.state.finished
