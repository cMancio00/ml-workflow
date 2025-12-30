import warnings

import torch
from lightning.pytorch import Trainer, seed_everything
import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig

from models.classifier import Classifier

def build_callbacks(cfg: DictConfig):
    if "callbacks" not in cfg:
        return None
    return [instantiate(cb) for cb in cfg.callbacks.values()]

@hydra.main(version_base=None, config_path="pkg://config", config_name="trainer")
def train(cfg: DictConfig):
    warnings.filterwarnings("ignore", category=UserWarning, module="lightning")

    seed_everything(cfg.seed, workers=True)

    torch.set_float32_matmul_precision(cfg.precision)

    model = Classifier(model=instantiate(cfg.model), optimizer=instantiate(cfg.optim))

    data = instantiate(cfg.data)
    data.prepare_data()
    data.setup("fit")
    train_dataloader = data.train_dataloader()
    val_dataloader = data.val_dataloader()

    callbacks = build_callbacks(cfg)


    trainer = Trainer(
        enable_model_summary=False,
        max_epochs=cfg.trainer.epochs,
        accelerator=cfg.trainer.accelerator,
        devices=cfg.trainer.devices,
        callbacks=callbacks,
    )

    trainer.fit(model, train_dataloader, val_dataloader)

    return trainer.callback_metrics[cfg.loss.monitor].item()
