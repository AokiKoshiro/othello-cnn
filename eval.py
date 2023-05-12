import numpy as np
import torch

from config import hyperparameters
from train import Model
from utils import get_legal_moves, init_board, reverse_disks

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Othello:
    def __init__(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.load_model(device)
        self.board = init_board()

    @staticmethod
    def load_model(device):
        hidden_size = hyperparameters['hidden_size']
        num_block = hyperparameters['num_block']
        dropout = hyperparameters['dropout']
        model = Model(hidden_size, num_block, dropout)
        model.load_state_dict(torch.load("./weights/model.pth", map_location=device))
        model.eval()
        return model

    def get_legal_output(self, board, legal_moves):
        with torch.no_grad():
            np_board = np.array(board).astype(np.float32)
            tensor_board = torch.from_numpy(np_board)
            output = self.model(tensor_board.unsqueeze(0))
            output = output.reshape(8, 8)
            legal_output = np.array([output[move] for move in legal_moves])
        return legal_output

    def ai_move(self, board, legal_moves, strength):
        legal_output = self.get_legal_output(board, legal_moves)
        if strength == "strongest":
            move = legal_moves[np.argmax(legal_output)]
        elif strength == "strong":
            limit = min(len(legal_moves), 4)
            move = legal_moves[np.random.choice(np.argsort(legal_output)[-limit:-1])]
        elif strength == "weak":
            limit = min(len(legal_moves) - 1, 3) if len(legal_moves) > 1 else 1
            move = legal_moves[np.random.choice(np.argsort(legal_output)[:limit])]
        elif strength == "random":
            move = legal_moves[np.random.randint(len(legal_moves))]
        else:
            raise ValueError("Invalid strength!")
        return reverse_disks(board, move)

    def switch_turn(self, turn, board):
        turn *= -1
        board = board[::-1]
        return turn, board

    def play_game(self, ai_color, strength):
        board = init_board()
        turn = ai_color
        consecutive_paths = 0

        while True:
            if np.sum(board) == 64 or consecutive_paths == 2:
                break

            legal_moves = get_legal_moves(board)

            if len(legal_moves) == 0:
                turn, board = self.switch_turn(turn, board)
                consecutive_paths += 1
                continue
            consecutive_paths = 0

            if turn == 1:
                board = self.ai_move(board, legal_moves, strength)
            else:
                board = self.ai_move(board, legal_moves, "random")

            last_board = board[:: ai_color * turn]
            turn, board = self.switch_turn(turn, board)

        return last_board

def compute_results(ai_color, num_games, strength):
    game = Othello()
    win_count = 0
    lose_count = 0
    draw_count = 0
    for i in range(num_games):
        last_board = game.play_game(ai_color, strength)
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

    return win_count, lose_count, draw_count


if __name__ == "__main__":
    num_games = 100
    strength = "strongest"

    for ai_color in [1, -1]:
        if ai_color == 1:
            print("black AI")
        else:
            print("white AI")

        win_count, lose_count, draw_count = compute_results(ai_color, num_games, strength)
        print(f"    Win: {win_count}, Lose: {lose_count}, Draw: {draw_count}")
