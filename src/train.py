import torch
from models.simple_cnn import CnnClassifier
from data.mnist_dataset import MNISTDataModule
from torch import nn
from torch import optim

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}\n")

data = MNISTDataModule()
data.prepare_data()
data.setup("fit")
train_loader = data.train_dataloader()

model = CnnClassifier().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)


num_epochs = 5
train_losses = []

def main():
    for epoch in range(num_epochs):
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
                    f'Epoch [{epoch+1}/{num_epochs}], '
                    f'Step [{i+1}/{len(train_loader)}], '
                    f'Loss: {running_loss/(i+1):.4f}',
                    flush=True, end='\r'
                )
            train_losses.append(running_loss/100)

    print('Training finished!')