import torch
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.callbacks import RichProgressBar, EarlyStopping
import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig

from models.classifier import Classifier
import optuna
from optuna import Trial
from optuna.integration.pytorch_lightning import PyTorchLightningPruningCallback

def objective(trial: Trial, cfg: DictConfig):
    seed_everything(42, workers=True)

    torch.set_float32_matmul_precision('high')
    lr = trial.suggest_float(
        name='lr',
        low=1e-4,
        high=1e-1,
        log=True
    )

    model = Classifier(
        model=instantiate(cfg.model),
        lr=lr
    )

    data = instantiate(
        cfg.data,
        batch_size = trial.suggest_categorical('batch_size', [32, 64, 128])
    )
    data.prepare_data()
    data.setup("fit")
    train_dataloader = data.train_dataloader()
    val_dataloader = data.val_dataloader()

    trainer = Trainer(
        enable_model_summary=False,
        max_epochs=cfg.trainer.epochs,
        accelerator='gpu',
        devices=[2],
        callbacks=[
            RichProgressBar(leave=False),
            PyTorchLightningPruningCallback(trial, monitor='val_loss'),
            EarlyStopping(
                'val_loss',
                patience=2
            )
        ]
    )

    trainer.fit(model, train_dataloader, val_dataloader)

    return trainer.callback_metrics['val_loss'].item()

@hydra.main(version_base=None, config_path="pkg://config", config_name="trainer")
def main(cfg: DictConfig):
    study_name = "MNIST"
    storage_name = "sqlite:///{}.db".format(study_name)
    study = optuna.create_study(study_name=study_name, storage=storage_name, load_if_exists=True)
    study.optimize(lambda trial: objective(trial, cfg), n_trials=50, n_jobs=1)
    print(f"{study.best_trial.number}"
          f"{study.best_value}\n"
          f"{study.best_trial.params}\n")

if __name__ == "__main__":
    main()