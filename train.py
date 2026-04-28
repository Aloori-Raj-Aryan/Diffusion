import yaml
from pathlib import Path
from datetime import datetime
import argparse

import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from utils.dataloader import DiffusionDataModule
from utils.checkpoints import get_checkpoints
from utils.train_pipeline import DiffusionLightning

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    model = DiffusionLightning(cfg)
    data_module = DiffusionDataModule(cfg)

    # Use fast cuDNN kernels when image size is fixed and GPU is available
    torch.backends.cudnn.benchmark = True
    use_gpu = torch.cuda.is_available()

    # Set up TensorBoard logger
    logger = TensorBoardLogger(save_dir="outputs", name="diffusion")
    checkpoint_callbacks = get_checkpoints(cfg)
    trainer = pl.Trainer(
        max_epochs=cfg["training"]["epochs"],
        accelerator="gpu" if use_gpu else "cpu",
        devices=1,
        precision=16 if use_gpu else 32,
        callbacks=checkpoint_callbacks,
        logger=logger,
    )

    trainer.fit(model, datamodule=data_module)


if __name__ == "__main__":
    main()