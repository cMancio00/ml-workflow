from functools import singledispatch

from omegaconf import DictConfig
from optuna import Trial
import torch.optim as optim


@singledispatch
def optim_hpo(opt, trial: Trial, cfg: DictConfig) -> DictConfig:
    raise TypeError(f"Optimizer not supported: {type(opt)}.")


@optim_hpo.register(optim.Adam)
def _(opt: optim.Adam, trial: Trial, cfg: DictConfig) -> DictConfig:
    cfg.optim.lr = trial.suggest_float("lr", 1e-8, 1e-1, log=True)
    cfg.optim.eps = trial.suggest_float("eps", 1e-8, 1e-1, log=True)
    return cfg


@optim_hpo.register
def _(opt: optim.SGD, trial: Trial, cfg: DictConfig) -> DictConfig:
    cfg.optim.lr = trial.suggest_float("lr", 1e-8, 1e-1, log=True)
    cfg.optim.momentum = trial.suggest_float("momentum", 0.0, 0.99)
    return cfg
