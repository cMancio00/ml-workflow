from warnings import filterwarnings

from hydra import main
from hydra.utils import instantiate
from lightning import Callback
from lightning.pytorch import Trainer, seed_everything
from omegaconf import DictConfig


def build_callbacks(cfg: DictConfig) -> list[Callback] | None:
    if "callbacks" not in cfg:
        return None
    return [instantiate(cb) for cb in cfg.callbacks.values()]


@main(version_base=None, config_path="pkg://config", config_name="train")
def train(cfg: DictConfig):
    filterwarnings("ignore", category=UserWarning, module="lightning")

    seed_everything(cfg.seed, workers=True)

    model = instantiate(cfg.model)

    model = model(optim=instantiate(cfg.optim))

    data = instantiate(cfg.data)
    data.prepare_data()
    data.setup("fit")
    train_dataloader = data.train_dataloader()
    val_dataloader = data.val_dataloader()

    callbacks = build_callbacks(cfg)

    trainer: Trainer = instantiate(cfg.trainer, callbacks=callbacks)

    trainer.fit(model, train_dataloader, val_dataloader)

    return trainer.callback_metrics[cfg.loss.monitor].item()
