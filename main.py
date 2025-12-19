import torch
from data import MNISTDataModule
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.callbacks import RichProgressBar, EarlyStopping

from models.classifier import Classifier
from models.simple_cnn import CnnClassifier
import optuna
from optuna import Trial
from optuna.integration.pytorch_lightning import PyTorchLightningPruningCallback

def objective(trial: Trial):
    seed_everything(42, workers=True)

    torch.set_float32_matmul_precision('high')
    lr = trial.suggest_float(
        name='lr',
        low=1e-4,
        high=1e-1,
        log=True
    )
    batch_size = trial.suggest_categorical('batch_size', [32, 64, 128])


    model = Classifier(
        model=CnnClassifier(),
        lr=lr
    )

    data = MNISTDataModule(batch_size=batch_size)
    data.prepare_data()
    data.setup("fit")
    train_dataloader = data.train_dataloader()
    val_dataloader = data.val_dataloader()

    trainer = Trainer(
        enable_model_summary=False,
        max_epochs=50,
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


def main():
    study_name = "MNIST"
    storage_name = "sqlite:///{}.db".format(study_name)
    study = optuna.create_study(study_name=study_name, storage=storage_name, load_if_exists=True)
    study.optimize(objective, n_trials=50, n_jobs=1)
    print(f"{study.best_trial.number}"
          f"{study.best_value}\n"
          f"{study.best_trial.params}\n")

if __name__ == "__main__":
    main()