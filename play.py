import numpy as np
import torch

from config import hyperparameters
from model import Model
from utils import get_legal_moves, init_board, print_board, reverse_disks


class Othello:
    def __init__(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.load_model(device)
        self.board = init_board()
        self.ai_board_history = []
        self.human_color = self.ask_color()
        self.turn = self.human_color
        self.consecutive_paths = 0

    @staticmethod
    def load_model(device):
        hidden_size = hyperparameters["hidden_size"]
        num_block = hyperparameters["num_block"]
        dropout = hyperparameters["dropout"]
        model = Model(hidden_size, num_block, dropout)
        model.load_state_dict(torch.load("./weights/model.pth", map_location=device))
        model.eval()
        return model

    @staticmethod
    def ask_color():
        color = input("Choose black(●) or white(○) (b/w): ")
        return 1 if color == "b" else -1 if color == "w" else Othello.ask_color()

    def random_move(self, legal_moves):
        move = legal_moves[np.random.randint(len(legal_moves))]
        print(f"Random move: {chr(ord('a') + move[1]) + str(move[0] + 1)}")
        return reverse_disks(self.board, move)

    def human_move(self, legal_moves):
        while True:
            move_index = input("Your move: ")

            if move_index == "undo":
                self.ai_board_history.pop()
                self.board = self.ai_board_history[-1]
                legal_moves = get_legal_moves(self.board)
                print_board(self.board[:: self.human_color])
                continue

            if move_index == "legal":
                for move in legal_moves:
                    print(f"{chr(ord('a') + move[1]) + str(move[0] + 1)}", end=" ")
                print()
                continue

            if move_index == "exit":
                exit()

            if len(move_index) == 2:
                if move_index[0].isalpha() and move_index[1].isdigit():
                    move = (int(move_index[1]) - 1, ord(move_index[0]) - ord("a"))
                    if move in legal_moves:
                        return reverse_disks(self.board, move)

            print("Invalid move.")

    def ai_move(self, legal_moves):
        with torch.no_grad():
            np_board = np.array(self.board).astype(np.float32)
            tensor_board = torch.from_numpy(np_board)
            output = self.model(tensor_board.unsqueeze(0))
            output = output.reshape(8, 8)
            legal_output = np.array([output[move] for move in legal_moves])
            move = legal_moves[np.argmax(legal_output)]

        print(f"AI move: {chr(ord('a') + move[1]) + str(move[0] + 1)}")
        self.board = reverse_disks(self.board, move)
        self.ai_board_history.append(np.copy(self.board[::-1]))
        return self.board, self.ai_board_history

    def switch_turn(self):
        self.turn *= -1
        self.board = self.board[::-1]

    def print_final_score(self, last_board):
        black_score = np.sum(last_board[0])
        white_score = np.sum(last_board[1])
        print(f"Black score: {black_score}")
        print(f"White score: {white_score}")

        if black_score > white_score:
            if self.human_color == 1:
                print("You win.")
            else:
                print("You lose.")
        elif black_score < white_score:
            if self.human_color == 1:
                print("You lose.")
            else:
                print("You win.")
        else:
            print("Draw.")

    def paly_game(self):
        print_board(self.board)
        if self.human_color == 1:
            self.ai_board_history.append(np.copy(self.board))

        while True:
            if np.sum(self.board) == 64 or self.consecutive_paths == 2:
                print("Game over.\n")
                break

            legal_moves = get_legal_moves(self.board)

            if len(legal_moves) == 0:
                print("Pass\n")
                self.switch_turn()
                self.consecutive_paths += 1
                continue
            self.consecutive_paths = 0

            if self.turn == 1:
                self.board = self.random_move(legal_moves)
                # self.board = self.human_move(legal_moves)
                # self.board, self.ai_board_history = self.ai_move(legal_moves)
            else:
                self.board, self.ai_board_history = self.ai_move(legal_moves)

            last_board = self.board[:: self.human_color * self.turn]
            print_board(last_board)
            self.switch_turn()

        self.print_final_score(last_board)


if __name__ == "__main__":
    game = Othello()
    game.paly_game()
