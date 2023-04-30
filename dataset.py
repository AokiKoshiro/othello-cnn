import pandas as pd
import numpy as np
from utils import transcript2moves, moves2matrices, moves2boards


df_data = pd.read_csv("./data/wthor.csv")
df_transcript = df_data["transcript"]
df_move = df_transcript.apply(transcript2moves)
df_move_matrix = df_move.apply(moves2matrices)
df_board = df_move.apply(moves2boards).apply(lambda x: x[:-1])

df_move_matrix = df_move_matrix.apply(pd.Series).stack().reset_index(drop=True)
df_board = df_board.apply(pd.Series).stack().reset_index(drop=True)

np_move_matrix = np.array([i for i in df_move_matrix.values])  # (?, 8, 8)
np_move_matrix = np_move_matrix.reshape(-1, 1, 8, 8)  # (?, 1, 8, 8)
np_board = np.array([i for i in df_board.values])  # (?, 2, 8, 8)

np_dataset = np.concatenate([np_board, np_move_matrix], axis=1)  # (?, 3, 8, 8)

np.save("./data/wthor_dataset.npy", np_dataset)
