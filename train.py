import yaml
from pathlib import Path
from datetime import datetime
import argparse

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
    
    # Set up TensorBoard logger
    logger = TensorBoardLogger(
        save_dir="outputs",
        name="diffusion",
    )
    
    # Get all checkpoint callbacks from config
    checkpoint_callbacks = get_checkpoints(cfg)

    trainer = pl.Trainer(
        max_epochs=cfg["training"]["epochs"],
        accelerator="auto",
        devices="auto",
        precision=16,  # mixed precision (optional but recommended)
        callbacks=checkpoint_callbacks,
        logger=logger,
        default_root_dir="outputs",
    )

    trainer.fit(model, datamodule=data_module)


if __name__ == "__main__":
    main()