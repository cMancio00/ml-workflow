from lightning.pytorch.utilities.types import STEP_OUTPUT
import lightning as l
import torch.optim as optim
from torch import nn
from torch import Tensor
import torch.nn.functional as F



class Classifier(l.LightningModule):

    def __init__(self, model: nn.Module, optimizer: optim.Optimizer):
        super().__init__()
        self.model = model
        self.save_hyperparameters(ignore=['model'])

    def configure_optimizers(self):
        optimizer = self.hparams.optimizer(params=self.parameters())
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.97)
        return [optimizer] , [scheduler]

    def training_step(self, batch: Tensor) -> STEP_OUTPUT:
        x, labels = batch

        y = self.model(x)
        loss = F.cross_entropy(y, labels)

        return loss

    def validation_step(self, batch: Tensor) -> STEP_OUTPUT:
        x, labels = batch

        y = self.model(x)
        loss = F.cross_entropy(y, labels)
        self.log('val_loss', loss, prog_bar=True)

        return loss



