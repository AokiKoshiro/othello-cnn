import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt


class Transform:
    def __init__(self):
        pass

    def __call__(self, sample, num):
        board = sample[:-1]
        move = sample[-1]
        board = board.reshape(2, 8, 8).astype(np.float32)
        move = move.reshape(8, 8).astype(np.float32)
        board = torch.from_numpy(board)
        move = torch.from_numpy(move)

        # rotate 0, 90, 180, 270
        rotate = num % 4
        board = torch.rot90(board, rotate, (1, 2))
        move = torch.rot90(move, rotate, (0, 1))

        # transpose
        if num >= 4:
            board = torch.transpose(board, 1, 2)
            move = torch.transpose(move, 0, 1)

        move = move.reshape(64)
        return board, move


class Dataset(torch.utils.data.Dataset):
    def __init__(self, dataset, multiple=8, transform=Transform()):
        self.multiple = multiple
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset) * self.multiple

    def __getitem__(self, idx):
        sample = self.dataset[idx // self.multiple]
        num = idx % self.multiple
        if self.transform:
            return self.transform(sample, num)
        return sample


class SEBlock(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.linear1 = nn.Linear(in_channels, in_channels // reduction)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(in_channels // reduction, in_channels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.linear1(y)
        y = self.relu(y)
        y = self.linear2(y)
        y = self.sigmoid(y).view(b, c, 1, 1)
        return x * y


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.se = SEBlock(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.se(x)
        return x


class Model(nn.Module):
    def __init__(self, hidden_size, num_block, dropout):
        super(Model, self).__init__()
        self.convs = nn.ModuleList()
        for i in range(num_block):
            if i == 0:
                self.convs.append(ConvBlock(2, hidden_size, 3, 1, 1))
            else:
                self.convs.append(ConvBlock(hidden_size, hidden_size, 3, 1, 1))
        self.flatten = nn.Flatten()
        self.bn = nn.BatchNorm1d(hidden_size * 8 * 8)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size * 8 * 8, 64)
        self.relu = nn.ReLU()

    def forward(self, x):
        for conv in self.convs:
            x = conv(x)
        x = self.flatten(x)
        x = self.bn(x)
        x = self.dropout(x)
        x = self.fc(x)
        x = self.relu(x)
        return x


# hyperparameters
batch_size = 2**10
hidden_size = 128
num_block = 5
dropout = 0.6
learning_rate = 5e-4
scheduler_step_size = 1
scheduler_gamma = 1
num_epochs = 20
log_interval = 1

if __name__ == "__main__":
    # dataset
    np_train_dataset = np.load("./train_wthor_winner_dataset.npy")
    train_dataset = Dataset(np_train_dataset, multiple=8, transform=Transform())

    np_val_dataset = np.load("./val_wthor_winner_dataset.npy")
    val_dataset = Dataset(np_val_dataset, multiple=8, transform=Transform())

    # dataloader
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # model, criterion, optimizer, scheduler
    model = Model(hidden_size, num_block, dropout)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_step_size, gamma=scheduler_gamma)

    # train and calc loss & accuracy for 10 epochs
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    train_loss_list = []
    val_loss_list = []
    val_accuracy_list = []
    for epoch in range(num_epochs):
        # train
        model.train()
        for batch, (board, move) in enumerate(train_loader):
            board = board.to(device)
            move = move.to(device)
            optimizer.zero_grad()
            outputs = model(board)
            loss = criterion(outputs, torch.argmax(move, dim=1))
            loss.backward()
            optimizer.step()
        train_loss = loss.item()
        train_loss_list.append(train_loss)
        scheduler.step()

        # val
        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch, (board, move) in enumerate(val_loader):
                board = board.to(device)
                move = move.to(device)
                outputs = model(board)
                loss = criterion(outputs, torch.argmax(move, dim=1))
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += move.size(0)
                correct += (predicted == torch.argmax(move, dim=1)).sum().item()
        val_loss /= batch + 1
        val_accuracy = correct / total
        val_loss_list.append(val_loss)
        val_accuracy_list.append(val_accuracy)
        if (epoch + 1) % log_interval == 0:
            print(f"-----Epoch {epoch + 1}/{num_epochs} -----")
            print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Accuracy: {val_accuracy:.4f}")
            print()

    # plot 2 graphs (loss & accuracy)
    fig, ax = plt.subplots(1, 2, figsize=(15, 5))
    ax[0].plot(train_loss_list, label="train_loss")
    ax[0].plot(val_loss_list, label="val_loss")
    ax[0].set_xlabel("epoch")
    ax[0].set_ylabel("loss")
    ax[0].legend()
    ax[1].plot(val_accuracy_list)
    ax[1].set_xlabel("epoch")
    ax[1].set_ylabel("accuracy")
    plt.show()

    # save model
    torch.save(model.state_dict(), "./model.pth")
