import numpy as np
import torch

from train import Model
from utils import get_legal_moves, init_board, print_board, reverse_disks

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = Model()
model.load_state_dict(torch.load("./weights/model.pth", map_location=device))
model.eval()


def ask_color():
    # Ask if you are black or white.
    color = input("Choose black(●) or white(○) (b/w): ")
    if color == "b":
        return 1
    if color == "w":
        return -1
    print("Invalid color.")
    return ask_color()


def random_move(board, legal_moves):
    # Get the next move.
    move = legal_moves[np.random.randint(len(legal_moves))]
    print(f"Random move: {chr(ord('a') + move[1]) + str(move[0] + 1)}")
    return reverse_disks(board, move)


def human_move(board, legal_moves, ai_board_history, human_color):
    # If there is a legal move, ask the next move.
    move_index = input("Your move: ")
    if len(move_index) == 2:
        if move_index[0].isalpha() and move_index[1].isdigit():
            move = (int(move_index[1]) - 1, ord(move_index[0]) - ord("a"))
            if move in legal_moves:
                return reverse_disks(board, move), ai_board_history
    if move_index == "undo":
        ai_board_history.pop()
        board = ai_board_history[-1]
        legal_moves = get_legal_moves(board)
        print_board(board[::human_color])
        return human_move(board, legal_moves, ai_board_history, human_color)
    if move_index == "legal":
        for move in legal_moves:
            print(f"{chr(ord('a') + move[1]) + str(move[0] + 1)}", end=" ")
        print()
        return human_move(board, legal_moves, ai_board_history, human_color)
    if move_index == "exit":
        exit()
    print("Invalid move.")
    return human_move(board, legal_moves, ai_board_history, human_color)


def ai_move(board, legal_moves, ai_board_history):
    # Get the next move.
    with torch.no_grad():
        np_board = np.array(board).astype(np.float32)
        tensor_board = torch.from_numpy(np_board)
        output = model(tensor_board.unsqueeze(0))
        output = output.reshape(8, 8)
        legal_output = np.array([output[move] for move in legal_moves])
        move = legal_moves[np.argmax(legal_output)]

    print(f"AI move: {chr(ord('a') + move[1]) + str(move[0] + 1)}")

    # Update the board.
    board = reverse_disks(board, move)
    ai_board_history.append(np.copy(board[::-1]))
    return board, ai_board_history


def switch_turn(turn, board):
    turn *= -1
    board = board[::-1]
    return turn, board


# Initialize the board.
board = init_board()  # board.shape = (2, 8, 8) , board = (my_board, enemy_board)
ai_board_history = []

# Play the game.
human_color = ask_color()
turn = human_color
consecutive_paths = 0
print_board(board)
if human_color == 1:
    ai_board_history.append(np.copy(board))

while True:
    # If the game is over, break.
    if np.sum(board) == 64 or consecutive_paths == 2:
        print("Game over.\n")
        break

    # Get legal moves.
    legal_moves = get_legal_moves(board)

    # If there is no legal move, pass.
    if len(legal_moves) == 0:
        print("Pass\n")
        turn, board = switch_turn(turn, board)
        consecutive_paths += 1
        continue
    consecutive_paths = 0

    # If there is a legal move, ask the next move.
    if turn == 1:
        board = random_move(board, legal_moves)
        # board, ai_board_history = human_move(board, legal_moves, ai_board_history, human_color)
        # board, ai_board_history = ai_move(board, legal_moves, ai_board_history)
    else:
        board, ai_board_history = ai_move(board, legal_moves, ai_board_history)

    last_board = board[:: human_color * turn]
    print_board(last_board)

    # Switch the turn.
    turn, board = switch_turn(turn, board)

# Print the final score.
black_score = np.sum(last_board[0])
white_score = np.sum(last_board[1])
print(f"Black score: {black_score}")
print(f"White score: {white_score}")

if black_score > white_score:
    if human_color == 1:
        print("You win.")
    else:
        print("You lose.")
elif black_score < white_score:
    if human_color == 1:
        print("You lose.")
    else:
        print("You win.")
else:
    print("Draw.")
