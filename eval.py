import numpy as np
import torch

from config import hyperparameters
from train import Model
from utils import get_legal_moves, init_board, reverse_disks

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

hidden_size = hyperparameters['hidden_size']
num_block = hyperparameters['num_block']
dropout = hyperparameters['dropout']

model = Model(hidden_size, num_block, dropout)
model.load_state_dict(torch.load("./weights/model.pth", map_location=device))
model.eval()


def get_legal_output(board, legal_moves):
    with torch.no_grad():
        np_board = np.array(board).astype(np.float32)
        tensor_board = torch.from_numpy(np_board)
        output = model(tensor_board.unsqueeze(0))
        output = output.reshape(8, 8)
        legal_output = np.array([output[move] for move in legal_moves])
    return legal_output


def strongest_ai_move(board, legal_moves):
    legal_output = get_legal_output(board, legal_moves)
    move = legal_moves[np.argmax(legal_output)]
    return reverse_disks(board, move)


def strong_ai_move(board, legal_moves):
    legal_output = get_legal_output(board, legal_moves)
    if len(legal_moves) > 1:
        # The best 3 moves except the best move are selected at random.
        limit = min(len(legal_moves), 4)
        move = legal_moves[np.random.choice(np.argsort(legal_output)[-limit:-1])]
    else:
        move = legal_moves[0]
    return reverse_disks(board, move)


def weak_ai_move(board, legal_moves):
    legal_output = get_legal_output(board, legal_moves)
    # The worst 3 moves are selected at random.
    limit = min(len(legal_moves) - 1, 3) if len(legal_moves) > 1 else 1
    move = legal_moves[np.random.choice(np.argsort(legal_output)[:limit])]
    return reverse_disks(board, move)


def random_move(board, legal_moves):
    move = legal_moves[np.random.randint(len(legal_moves))]
    return reverse_disks(board, move)


def switch_turn(turn, board):
    turn *= -1
    board = board[::-1]
    return turn, board


def play_game(ai_color):
    # Initialize the board.
    board = init_board()

    # Play the game.
    turn = ai_color
    consecutive_paths = 0

    while True:
        # If the game is over, break.
        if np.sum(board) == 64 or consecutive_paths == 2:
            break

        # Get legal moves.
        legal_moves = get_legal_moves(board)

        # If there is no legal move, pass.
        if len(legal_moves) == 0:
            turn, board = switch_turn(turn, board)
            consecutive_paths += 1
            continue
        consecutive_paths = 0

        # If there is a legal move, ask the next move.
        if turn == 1:
            board = strongest_ai_move(board, legal_moves)
            # board = random_move(board, legal_moves)
        else:
            board = strong_ai_move(board, legal_moves)
            # board = weak_ai_move(board, legal_moves)
            # board = random_move(board, legal_moves)

        last_board = board[:: ai_color * turn]

        # Switch the turn.
        turn, board = switch_turn(turn, board)

    return last_board


if __name__ == "__main__":
    # number of games
    num_games = 100

    for ai_color in [1, -1]:
        win_count = 0
        lose_count = 0
        draw_count = 0
        for i in range(num_games):
            # Play the game.
            last_board = play_game(ai_color)

            # Print the final score.
            black_score = np.sum(last_board[0])
            white_score = np.sum(last_board[1])

            if black_score > white_score:
                if ai_color == 1:
                    win_count += 1
                else:
                    lose_count += 1
            elif black_score < white_score:
                if ai_color == 1:
                    lose_count += 1
                else:
                    win_count += 1
            else:
                draw_count += 1

        if ai_color == 1:
            print("black AI")
        else:
            print("white AI")

        print(f"    Win: {win_count}, Lose: {lose_count}, Draw: {draw_count}")
