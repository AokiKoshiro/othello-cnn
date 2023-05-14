import torch
import torch.nn as nn
import numpy as np

from config import hyperparameters
from train import BoardTransform, OthelloDataset
from model import Model


def test(model, device, test_loader, criterion):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for board, move in test_loader:
            board, move = board.to(device), move.to(device)
            output = model(board)
            test_loss += criterion(output, torch.argmax(move, dim=1)).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(torch.argmax(move, dim=1).view_as(pred)).sum().item()

    test_loss /= len(test_loader)
    test_accuracy = correct / len(test_loader.dataset)

    return test_loss, test_accuracy


def main():
    # hyperparameters
    batch_size = hyperparameters["batch_size"]
    hidden_size = hyperparameters["hidden_size"]
    num_block = hyperparameters["num_block"]
    dropout = hyperparameters["dropout"]

    # dataset
    np_test_dataset = np.load("./data/test_winner_othello_dataset.npy")
    test_dataset = OthelloDataset(np_test_dataset, multiple=8, transform=BoardTransform())

    # dataloader
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    # Set up device, model, and criterion
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Model(hidden_size, num_block, dropout).to(device)
    model.load_state_dict(torch.load("./weights/model.pth", map_location=device))
    criterion = nn.CrossEntropyLoss()

    # Test the model
    test_loss, test_accuracy = test(model, device, test_loader, criterion)

    print(f"Test Loss: {test_loss:.6f} | Test Accuracy: {100. * test_accuracy:.2f}%")


if __name__ == "__main__":
    main()
