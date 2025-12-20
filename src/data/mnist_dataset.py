from torchvision import transforms
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import MNIST
import lightning as l

class MNISTDataModule(l.LightningDataModule):

    def __init__(self, data_dir: str = "./datasets", batch_size: int = 32, num_workers: int = 4):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        self.data_train = None
        self.data_val = None
        self.data_test = None
        self.num_classes = 10
        self.shape = (1, 28, 28)

    def prepare_data(self):
        MNIST(self.data_dir, train=True, download=True)
        MNIST(self.data_dir, train=False, download=True)

    def setup(self, stage: str):
        if stage == "fit":
            mnist_full = MNIST(self.data_dir, train=True, download=True, transform=self.transform)
            mnist_full.data = mnist_full.data[:500]
            mnist_full.targets = mnist_full.targets[:500]
            self.data_train, self.data_val = random_split(
                mnist_full, [11/12, 1/12]
            )
        if stage == "test":
            self.data_test = MNIST(self.data_dir, train=False, download=True, transform=self.transform)

    def train_dataloader(self):
        return DataLoader(self.data_train, batch_size=self.batch_size, drop_last=True, shuffle=True, pin_memory=True,
                          num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.data_val, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.data_test, batch_size=self.batch_size, num_workers=self.num_workers)