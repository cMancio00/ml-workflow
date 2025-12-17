from omegaconf import DictConfig, OmegaConf
import torch
from torch import nn
from torch import optim
import hydra
from hydra.utils import instantiate

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}\n")


@hydra.main(version_base=None, config_path="pkg://config", config_name="trainer")
def train(cfg: DictConfig):


    epochs = cfg.trainer.epochs

    data = instantiate(cfg.data)
    data.prepare_data()
    data.setup("fit")
    train_loader = data.train_dataloader()

    model = instantiate(cfg.model).to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    criterion = nn.CrossEntropyLoss()

    train_losses = []
    for epoch in range(epochs):
        running_loss = 0.0

        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if (i + 1) % 100 == 0:
                print(
                    f'Epoch [{epoch+1}/{epochs}], '
                    f'Step [{i+1}/{len(train_loader)}], '
                    f'Loss: {running_loss/(i+1):.4f}',
                    flush=True, end='\r'
                )
            train_losses.append(running_loss/100)

    print('Training finished!')

if __name__ == "__main__":
    train()