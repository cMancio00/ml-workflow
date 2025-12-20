import warnings

import torch
from lightning.pytorch import Trainer, seed_everything
import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig

from models.classifier import Classifier
import optuna
from optuna import Trial
from optuna.integration.pytorch_lightning import PyTorchLightningPruningCallback


def build_callbacks(cfg: DictConfig):
    if "callbacks" not in cfg:
        return None
    return [instantiate(cb) for cb in cfg.callbacks.values()]

def objective(trial: Trial, cfg: DictConfig):

    cfg.optim.lr = trial.suggest_float(
        name='lr',
        low=1e-4,
        high=1e-1,
        log=True
    )


    cfg.data.batch_size = trial.suggest_categorical('batch_size', [32, 64, 128])

    return run_train(cfg, trial)


def run_train(cfg: DictConfig, trial: Trial | None = None):
    warnings.filterwarnings(
        "ignore",
        category=UserWarning,
        module="lightning"
    )

    seed_everything(42, workers=True)

    torch.set_float32_matmul_precision('high')

    model = Classifier(
        model=instantiate(cfg.model),
        lr=cfg.optim.lr
    )

    data = instantiate(
        cfg.data
    )
    data.prepare_data()
    data.setup("fit")
    train_dataloader = data.train_dataloader()
    val_dataloader = data.val_dataloader()

    callbacks = build_callbacks(cfg)

    if trial:
        callbacks.append(
            PyTorchLightningPruningCallback(
                trial=trial,
                monitor=cfg.loss.monitor
            )
        )

    trainer = Trainer(
        enable_model_summary=False,
        max_epochs=cfg.trainer.epochs,
        accelerator='gpu',
        devices="1",
        callbacks=callbacks
    )

    trainer.fit(model, train_dataloader, val_dataloader)

    return trainer.callback_metrics[cfg.loss.monitor].item()

@hydra.main(version_base=None, config_path="pkg://config", config_name="trainer")
def train(cfg: DictConfig):
    run_train(cfg)

@hydra.main(version_base=None, config_path="pkg://config", config_name="trainer")
def hpo(cfg: DictConfig):
    torch.set_float32_matmul_precision('high')
    study_name = "MNIST"
    storage_name = "sqlite:///{}.db".format(study_name)
    study = optuna.create_study(study_name=study_name, storage=storage_name, load_if_exists=True)
    study.optimize(lambda trial: objective(trial, cfg), n_trials=50, n_jobs=1)
    print(f"{study.best_trial.number}"
          f"{study.best_value}\n"
          f"{study.best_trial.params}\n")
