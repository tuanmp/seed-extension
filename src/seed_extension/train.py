"""CAST seed-extension training orchestration."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import lightning as L
import yaml
from lightning.pytorch.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from lightning.pytorch.loggers import CSVLogger

from ml_cookbook.utils.repro import seed_everything
from colliderml_dataloader import ColliderMLDataModule

import seed_extension.data  # noqa: F401 — monkey-patches feature lists
from seed_extension.data.dataset import SeedExtensionDataset
from seed_extension.models.cast.model import CASTModel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the CAST seed-extension transformer."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/cast_default.yaml"),
        help="Path to the YAML config file.",
    )
    return parser.parse_args()


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    seed = int(cfg.get("seed", 42))
    deterministic = bool(cfg.get("trainer", {}).get("deterministic", True))
    seed_everything(seed=seed, deterministic=deterministic)

    data_cfg = cfg["data"]
    model_cfg = cfg["model"]
    trainer_cfg = cfg["trainer"]

    default_datadir = os.environ.get(
        "COLLIDERML_DATA_DIR",
        "/pscratch/sd/p/pmtuan/.cache/colliderml",
    )

    datamodule = ColliderMLDataModule(
        data_dir=default_datadir,
        process=data_cfg["process"],
        pileup=data_cfg["pileup"],
        max_train_events=data_cfg["max_train_events"],
        max_val_events=data_cfg["max_val_events"],
        max_test_events=data_cfg["max_test_events"],
        batch_size=1,
        num_workers=data_cfg["num_workers"],
        dataset_cls=SeedExtensionDataset,
        min_track_hits=data_cfg.get("min_track_hits", 5),
        seed_strategy=data_cfg.get("seed_strategy", "random_consecutive"),
        target_vertices=data_cfg.get("target_vertices", 200),
        primary_only=data_cfg.get("primary_only", False),
        predict_seed_hits=data_cfg.get("predict_seed_hits", False),
    )

    model = CASTModel(
        d_model=int(model_cfg["d_model"]),
        n_cross_attn_layers=int(model_cfg["n_cross_attn_layers"]),
        n_heads=int(model_cfg["n_heads"]),
        ff_dim=int(model_cfg["ff_dim"]),
        dropout=float(model_cfg["dropout"]),
        temperature=float(model_cfg["temperature"]),
        fourier_L=int(model_cfg["fourier_L"]),
        use_cylindrical_pe=bool(model_cfg["use_cylindrical_pe"]),
        use_seed_self_attn=bool(model_cfg["use_seed_self_attn"]),
        hit_encoder=str(model_cfg["hit_encoder"]),
        learning_rate=float(model_cfg["learning_rate"]),
        weight_decay=float(model_cfg.get("weight_decay", 1e-4)),
        lr_scheduler=str(model_cfg.get("lr_scheduler", "cosine")),
        warmup_steps=int(model_cfg.get("warmup_steps", 1000)),
    )

    exp_name = cfg.get("experiment_name", "cast_baseline")
    logger = CSVLogger(save_dir="logs", name=exp_name)

    callbacks = [
        ModelCheckpoint(
            monitor="val_loss",
            mode="min",
            save_top_k=1,
            filename="best",
        ),
        EarlyStopping(monitor="val_loss", mode="min", patience=3),
        LearningRateMonitor(logging_interval="epoch"),
    ]

    trainer = L.Trainer(
        max_epochs=int(trainer_cfg["max_epochs"]),
        accelerator=trainer_cfg.get("accelerator", "auto"),
        devices=trainer_cfg.get("devices", 1),
        deterministic=bool(trainer_cfg.get("deterministic", True)),
        log_every_n_steps=int(trainer_cfg.get("log_every_n_steps", 1)),
        enable_progress_bar=bool(trainer_cfg.get("enable_progress_bar", True)),
        limit_train_batches=trainer_cfg.get("limit_train_batches", 1.0),
        limit_val_batches=trainer_cfg.get("limit_val_batches", 1.0),
        callbacks=callbacks,
        logger=logger,
    )

    trainer.fit(model=model, datamodule=datamodule)
    trainer.test(model=model, datamodule=datamodule, ckpt_path="best")


if __name__ == "__main__":
    main()
