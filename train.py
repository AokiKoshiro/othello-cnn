import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

from config import hyperparameters


class BoardTransform:
    def __init__(self):
        pass

    def __call__(self, sample, num):
        board, move = self._reshape(sample)
        board, move = self._apply_transformations(board, move, num)
        return board, move

    def _reshape(self, sample):
        board = sample[:-1].reshape(2, 8, 8).astype(np.float32)
        move = sample[-1].reshape(8, 8).astype(np.float32)
        return torch.from_numpy(board), torch.from_numpy(move)

    def _apply_transformations(self, board, move, num):
        board, move = self._rotate(board, move, num)
        if num >= 4:
            board, move = self._transpose(board, move)
        return board, move.reshape(64)

    def _rotate(self, board, move, num):
        rotate = num % 4
        return (
            torch.rot90(board, rotate, (1, 2)),
            torch.rot90(move, rotate, (0, 1)),
        )

    def _transpose(self, board, move):
        return (
            torch.transpose(board, 1, 2),
            torch.transpose(move, 0, 1),
        )


class OthelloDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, multiple=8, transform=BoardTransform()):
        self.multiple = multiple
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset) * self.multiple

    def __getitem__(self, idx):
        sample = self.dataset[idx // self.multiple]
        num = idx % self.multiple
        return self.transform(sample, num) if self.transform else sample


class SqueezeExcitationBlock(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(SqueezeExcitationBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.linear1 = nn.Linear(in_channels, in_channels // reduction)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(in_channels // reduction, in_channels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.relu(self.linear1(y))
        y = self.sigmoid(self.linear2(y)).view(b, c, 1, 1)
        return x * y


class ConvolutionBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(ConvolutionBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.se = SqueezeExcitationBlock(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.se(x)
        return x


class Model(nn.Module):
    def __init__(self, hidden_size, num_block, dropout):
        super(Model, self).__init__()
        self.convs = self._create_convs(hidden_size, num_block)
        self.flatten = nn.Flatten()
        self.bn = nn.BatchNorm1d(hidden_size * 8 * 8)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size * 8 * 8, 64)
        self.relu = nn.ReLU()

    def _create_convs(self, hidden_size, num_block):
        convs = nn.ModuleList()
        for i in range(num_block):
            in_channels = 2 if i == 0 else hidden_size
            convs.append(ConvolutionBlock(in_channels, hidden_size, 3, 1, 1))
        return convs

    def forward(self, x):
        for conv in self.convs:
            x = conv(x)
        x = self.flatten(x)
        x = self.bn(x)
        x = self.dropout(x)
        x = self.fc(x)
        x = self.relu(x)
        return x


def train(model, device, train_loader, criterion, optimizer, epoch, log_batch_interval):
    model.train()
    train_loss = 0
    for batch_idx, (board, move) in enumerate(train_loader):
        board, move = board.to(device), move.to(device)
        optimizer.zero_grad()
        output = model(board)
        loss = criterion(output, torch.argmax(move, dim=1))
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

        if batch_idx % log_batch_interval == 0:
            print(
                f"Train Epoch: {epoch} | Progress: {batch_idx * len(board)}/{len(train_loader.dataset)} "
                f"({100. * batch_idx / len(train_loader):.0f}%) | Loss: {loss.item():.6f}"
            )

    train_loss /= len(train_loader)

    return train_loss


def validate(model, device, val_loader, criterion):
    model.eval()
    val_loss = 0
    correct = 0
    with torch.no_grad():
        for board, move in val_loader:
            board, move = board.to(device), move.to(device)
            output = model(board)
            val_loss += criterion(output, torch.argmax(move, dim=1)).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(torch.argmax(move, dim=1).view_as(pred)).sum().item()

    val_loss /= len(val_loader)
    val_accuracy = correct / len(val_loader.dataset)

    return val_loss, val_accuracy


def main():
    # hyperparameters
    batch_size = hyperparameters["batch_size"]
    hidden_size = hyperparameters["hidden_size"]
    num_block = hyperparameters["num_block"]
    dropout = hyperparameters["dropout"]
    learning_rate = hyperparameters["learning_rate"]
    scheduler_step_size = hyperparameters["scheduler_step_size"]
    scheduler_gamma = hyperparameters["scheduler_gamma"]
    num_epochs = hyperparameters["num_epochs"]
    log_interval = hyperparameters["log_interval"]
    log_batch_interval = hyperparameters["log_batch_interval"]

    # dataset
    np_train_dataset = np.load("./data/train_winner_othello_dataset.npy")
    train_dataset = OthelloDataset(np_train_dataset, multiple=8, transform=BoardTransform())

    np_val_dataset = np.load("./data/val_winner_othello_dataset.npy")
    val_dataset = OthelloDataset(np_val_dataset, multiple=8, transform=BoardTransform())

    # dataloader
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    # Set up device, model, criterion, optimizer, and scheduler
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Model(hidden_size, num_block, dropout).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_step_size, gamma=scheduler_gamma)

    # Train and validate the model for num_epochs
    train_loss_list = []
    val_loss_list = []
    val_accuracy_list = []
    for epoch in range(1, num_epochs + 1):
        train_loss = train(model, device, train_loader, criterion, optimizer, epoch, log_batch_interval)
        val_loss, val_accuracy = validate(model, device, val_loader, criterion)

        train_loss_list.append(train_loss)
        val_loss_list.append(val_loss)
        val_accuracy_list.append(val_accuracy)

        scheduler.step()

        if epoch % log_interval == 0:
            print(
                f"Epoch: {epoch}/{num_epochs} | Train Loss: {train_loss:.6f} | Validation Loss: {val_loss:.6f} | "
                f"Validation Accuracy: {100. * val_accuracy:.2f}%"
            )

        # Save the trained model for each epoch
        if epoch == num_epochs:
            torch.save(model.state_dict(), "./weights/model.pth")
        else:
            torch.save(model.state_dict(), f"./weights/model_{epoch}.pth")

        # Plot loss and accuracy curves after each epoch
        fig, ax = plt.subplots(1, 2, figsize=(15, 5))
        ax[0].plot(range(1, epoch + 1), train_loss_list, label="train_loss")
        ax[0].plot(range(1, epoch + 1), val_loss_list, label="val_loss")
        ax[0].set_xlabel("Epoch")
        ax[0].set_ylabel("Loss")
        ax[0].set_xlim([1, num_epochs])
        ax[0].legend()
        ax[1].plot(range(1, epoch + 1), val_accuracy_list)
        ax[1].set_xlabel("Epoch")
        ax[1].set_ylabel("Accuracy (%)")
        ax[1].set_xlim([1, num_epochs])
        plt.show()


if __name__ == "__main__":
    main()
