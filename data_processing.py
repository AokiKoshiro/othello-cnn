import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from utils import get_winner_indices, moves2boards, moves2matrices, transcript2moves

def prepare_othello_dataset():
    df_data = pd.read_csv(f"./data/original_data/othello_dataset.csv")
    df_transcript = df_data["game_moves"]
    df_winner = df_data["winner"]
    df_move = df_transcript.apply(transcript2moves)
    df_move_and_winner = pd.concat([df_move, df_winner], axis=1)
    df_winner_index = df_move_and_winner.apply(lambda x: get_winner_indices(x[0], x[1]), axis=1)
    df_move_matrix = df_move.apply(moves2matrices)
    df_board = df_move.apply(moves2boards).apply(lambda x: x[:-1])
    np_winner_move_matrix = create_winner_move_matrix(df_move_matrix, df_winner_index)
    np_winner_board = create_winner_board(df_board, df_winner_index)
    np_dataset = np.concatenate([np_winner_board, np_winner_move_matrix], axis=1)

    return np_dataset

def create_winner_move_matrix(df_move_matrix, df_winner_index):
    winner_move_matrix = [df_move_matrix.values[i][j] for i, winner_index in enumerate(df_winner_index.values) for j in winner_index]
    np_winner_move_matrix = np.array(winner_move_matrix)
    np_winner_move_matrix = np_winner_move_matrix.reshape(-1, 1, 8, 8)

    return np_winner_move_matrix

def create_winner_board(df_board, df_winner_index):
    winner_board = [df_board.values[i][j] for i, winner_index in enumerate(df_winner_index.values) for j in winner_index]
    np_winner_board = np.array(winner_board)

    return np_winner_board

def split_dataset(np_dataset, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2):
    assert train_ratio + val_ratio + test_ratio == 1, "Ratios must add up to 1"

    train_val, test = train_test_split(np_dataset, test_size=test_ratio, random_state=42)
    train, val = train_test_split(train_val, test_size=val_ratio / (train_ratio + val_ratio), random_state=42)

    return train, val, test

def save_datasets(train, val, test):
    np.save("./data/train_winner_othello_dataset.npy", train)
    np.save("./data/val_winner_othello_dataset.npy", val)
    np.save("./data/test_winner_othello_dataset.npy", test)

if __name__ == "__main__":
    np_dataset = prepare_othello_dataset()
    train, val, test = split_dataset(np_dataset)
    save_datasets(train, val, test)
