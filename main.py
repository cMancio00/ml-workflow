import torch
from data import MNISTDataModule
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.callbacks import RichProgressBar

from models.classifier import Classifier
from models.simple_cnn import CnnClassifier


def main():
    seed_everything(42, workers=True)

    torch.set_float32_matmul_precision('high')

    model = Classifier(
        model=CnnClassifier()
    )

    data = MNISTDataModule()
    data.prepare_data()
    data.setup("fit")
    train_dataloader = data.train_dataloader()
    val_dataloader = data.val_dataloader()


    trainer = Trainer(
        max_epochs=50,
        accelerator='gpu',
        devices=[2],
        callbacks=[
            RichProgressBar(leave=False)
        ]
    )

    trainer.fit(model, train_dataloader, val_dataloader)

if __name__ == "__main__":
    main()