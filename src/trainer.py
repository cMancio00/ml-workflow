import warnings

import torch
from lightning.pytorch import Trainer, seed_everything
import hydra
from hydra.utils import instantiate, get_class
from omegaconf import DictConfig
from torch import Tensor

from models.classifier import Classifier
import optuna
from optuna import Trial
from optuna.integration.pytorch_lightning import PyTorchLightningPruningCallback
from utils.optim_search_space import optim_hpo


def build_callbacks(cfg: DictConfig):
    if "callbacks" not in cfg:
        return None
    return [instantiate(cb) for cb in cfg.callbacks.values()]


def objective(trial: Trial, cfg: DictConfig):
    return run_train(cfg, trial)


def run_train(cfg: DictConfig, trial: Trial | None = None):
    warnings.filterwarnings("ignore", category=UserWarning, module="lightning")

    seed_everything(cfg.seed, workers=True)

    torch.set_float32_matmul_precision(cfg.precision)

    if trial:
        opt = instantiate(cfg.optim)
        opt = opt(params=[Tensor([0])])
        cfg = optim_hpo(opt, trial=trial, cfg=cfg)
        del opt

    model = Classifier(model=instantiate(cfg.model), optimizer=instantiate(cfg.optim))

    if trial:
        data_cls = get_class(cfg.data.module._target_)
        cfg = data_cls.hpo(trial, cfg)
        del data_cls

    data = instantiate(cfg.data.module)
    data.prepare_data()
    data.setup("fit")
    train_dataloader = data.train_dataloader()
    val_dataloader = data.val_dataloader()

    callbacks = build_callbacks(cfg)

    if trial:
        callbacks.append(
            PyTorchLightningPruningCallback(trial=trial, monitor=cfg.loss.monitor)
        )

    trainer = Trainer(
        enable_model_summary=False,
        max_epochs=cfg.trainer.epochs,
        accelerator=cfg.trainer.accelerator,
        devices=cfg.trainer.devices,
        callbacks=callbacks,
    )

    trainer.fit(model, train_dataloader, val_dataloader)

    return trainer.callback_metrics[cfg.loss.monitor].item()


@hydra.main(version_base=None, config_path="pkg://config", config_name="trainer")
def train(cfg: DictConfig):
    run_train(cfg)


@hydra.main(version_base=None, config_path="pkg://config", config_name="trainer")
def hpo(cfg: DictConfig):
    study_name = cfg.optuna.name
    storage_name = "sqlite:///{}.db".format(study_name)
    study = optuna.create_study(
        study_name=study_name, storage=storage_name, load_if_exists=True
    )
    study.optimize(lambda trial: objective(trial, cfg), n_trials=cfg.optuna.trials, n_jobs=1)
    print(
        f"{study.best_trial.number}\n"
        f"{study.best_value}\n"
        f"{study.best_trial.params}\n"
    )
