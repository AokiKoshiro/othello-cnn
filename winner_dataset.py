import numpy as np
import pandas as pd

from utils import get_winner_index, moves2boards, moves2matrices, transcript2moves

phase = "val"

df_data = pd.read_csv(f"./data/original_data/{phase}_wthor.csv")
df_transcript = df_data["transcript"]
df_black_score = df_data["blackScore"]

df_winner = df_black_score.apply(lambda x: 1 if x > 32 else -1)
df_move = df_transcript.apply(transcript2moves)
df_move_and_winner = pd.concat([df_move, df_winner], axis=1)

df_winner_index = df_move_and_winner.apply(lambda x: get_winner_index(x[0], x[1]), axis=1)

df_move_matrix = df_move.apply(moves2matrices)
df_board = df_move.apply(moves2boards).apply(lambda x: x[:-1])

np_winner_move_matrix = np.array(
    [df_move_matrix.values[i][j] for i, winner_index in enumerate(df_winner_index.values) for j in winner_index]
)  # (?, 8, 8)
np_winner_move_matrix = np_winner_move_matrix.reshape(-1, 1, 8, 8)  # (?, 1, 8, 8)

np_winner_board = np.array(
    [df_board.values[i][j] for i, winner_index in enumerate(df_winner_index.values) for j in winner_index]
)  # (?, 2, 8, 8)

np_dataset = np.concatenate([np_winner_board, np_winner_move_matrix], axis=1)  # (?, 3, 8, 8)

np.save(f"./data/{phase}_wthor_winner_dataset.npy", np_dataset)
