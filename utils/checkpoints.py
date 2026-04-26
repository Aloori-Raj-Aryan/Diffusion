from pathlib import Path
from pytorch_lightning.callbacks import ModelCheckpoint


def get_checkpoints(cfg: dict) -> list:
   
    checkpoints_list = []
    checkpoint_dir = Path(cfg.get("paths", {}).get("checkpoint_dir", "runs/checkpoints"))
    
    for ckpt_name, ckpt_cfg in cfg.get("checkpoints", {}).items():
        # Set defaults
        save_top_k = ckpt_cfg.get("save_top_k", 1)
        filename = ckpt_cfg.get("filename", ckpt_name)
        
        # Create callback based on type
        if ckpt_name == "latest":
            # Save every N steps/epochs
            callback = ModelCheckpoint(
                dirpath=checkpoint_dir,
                save_top_k=save_top_k,  # Save all checkpoints
                every_n_epochs=ckpt_cfg.get("interval", 10),
                filename=filename,
            )
        elif ckpt_name == "best":
            # Monitor-based checkpoint (best model)
            callback = ModelCheckpoint(
                dirpath=checkpoint_dir,
                save_top_k=save_top_k,
                monitor=ckpt_cfg.get("monitor", "val_loss"),
                mode=ckpt_cfg.get("mode", "min"),
                filename=filename,
            )
        else:
            raise ValueError(f"Unknown checkpoint type: {ckpt_name}")
        
        checkpoints_list.append(callback)
    
    return checkpoints_list
