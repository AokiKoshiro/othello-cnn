import torch
import numpy as np


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
