import lightning as l
import torch.nn.functional as F
import torch.optim as optim
from lightning.pytorch.utilities.types import STEP_OUTPUT
from torch import Tensor, nn


class Classifier(l.LightningModule):
    def __init__(self, net: nn.Module, optimizer: optim.Optimizer):
        super().__init__()
        self.net = net
        self.save_hyperparameters(ignore=["net"])

    def configure_optimizers(self):
        optimizer = self.hparams.optimizer(params=self.parameters())
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.97)
        return [optimizer], [scheduler]

    def training_step(self, batch: Tensor) -> STEP_OUTPUT:
        x, labels = batch

        y = self.net(x)
        loss = F.cross_entropy(y, labels)

        return loss

    def validation_step(self, batch: Tensor) -> STEP_OUTPUT:
        x, labels = batch

        y = self.net(x)
        loss = F.cross_entropy(y, labels)
        self.log("val_loss", loss, prog_bar=True)

        return loss
