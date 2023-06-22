import numpy as np
import itertools

DIRECTIONS = list(itertools.product([-1, 0, 1], repeat=2))
DIRECTIONS.remove((0, 0))


def init_board():
    """
    Initialize the game board.
    """
    black_board = np.zeros((8, 8), dtype=np.int8)
    white_board = np.zeros((8, 8), dtype=np.int8)
    black_board[3, 4] = black_board[4, 3] = 1
    white_board[3, 3] = white_board[4, 4] = 1
    return black_board, white_board


def is_legal_direction(my_board, enemy_board, move, direction):
    """
    Check if it's possible to sandwich the opponent's discs in the specified direction.
    """
    for i in range(1, 8):
        next_coord = tuple(np.add(move, i * np.array(direction)))
        if not all(0 <= c <= 7 for c in next_coord):
            return False
        if enemy_board[next_coord] == 1:
            continue
        elif my_board[next_coord] == 1:
            return i > 1
        else:
            return False
    return False


def get_legal_directions(board, move):
    """
    Return the directions in which it is possible to flip the opponent's discs.
    """
    my_board, enemy_board = board
    return [direction for direction in DIRECTIONS if is_legal_direction(my_board, enemy_board, move, direction)]


def is_legal_move(board, move):
    """
    Check if the specified move can flip the opponent's discs.
    """
    return any(is_legal_direction(board[0], board[1], move, direction) for direction in DIRECTIONS)


def get_legal_moves(board):
    """
    Return possible moves.
    """
    legal_moves = []
    for move in itertools.product(range(8), repeat=2):
        if is_legal_move(board, move) and all(b[move] == 0 for b in board):
            legal_moves.append(move)
    return legal_moves


def reverse_disks(board, move):
    """
    Flip the opponent's discs sandwiched by the specified move.
    """
    my_board, enemy_board = board
    my_board[move] = 1
    for direction in get_legal_directions(board, move):
        for i in range(1, 8):
            next_coord = tuple(np.add(move, i * np.array(direction)))
            if enemy_board[next_coord] == 1:
                enemy_board[next_coord] = 0
                my_board[next_coord] = 1
            else:
                break
    return my_board, enemy_board


def transcript2moves(transcript):
    """
    Convert string transcript to moves.
    """
    moves = []
    for i in range(0, len(transcript), 2):
        move = (int(transcript[i + 1]) - 1, ord(transcript[i]) - ord("a"))
        moves.append(move)
    return moves


def moves2matrices(moves):
    """
    Convert moves to matrices.
    """
    matrices = []
    for move in moves:
        matrix = np.zeros((8, 8), dtype=np.int8)
        matrix[move] = 1
        matrices.append(matrix)
    return matrices


def get_boards_winner_indices(moves, winner):
    """
    Return the boards and the indices of the winner's moves.
    """
    boards = [init_board()]
    turns = [1]
    turn = 1
    for move in moves:
        if not is_legal_move(boards[-1], move):
            turn *= -1
            continue
        new_board = np.copy(boards[-1])
        turn *= -1
        boards.append(reverse_disks(new_board, move)[::-1])
        turns.append(turn)
    winner_indices = [i - 1 for i, t in enumerate(turns) if t == winner]
    return boards, winner_indices


def print_board(boards):
    """
    Display the board(s).
    """
    boards = [boards] if type(boards) != list else boards
    for my_board, enemy_board in boards:
        print("  a b c d e f g h")
        for i in range(8):
            print(i + 1, end=" ")
            for j in range(8):
                if my_board[i, j] == 1:
                    print("●", end=" ")
                elif enemy_board[i, j] == 1:
                    print("○", end=" ")
                else:
                    print("・", end="")
            print(i + 1)
        print("  a b c d e f g h")
