import itertools

import numpy as np


def init_board():
    """
    盤を初期化する。
    """
    black_board = np.zeros((8, 8), dtype=np.int8)
    white_board = np.zeros((8, 8), dtype=np.int8)
    black_board[3, 4] = 1
    black_board[4, 3] = 1
    white_board[3, 3] = 1
    white_board[4, 4] = 1
    return black_board, white_board


def is_legal_direction(board, move, direction):
    """
    指定した方向に同じ色で挟めるかどうか判定する。
    """
    my_board, enemy_board = board
    for i in range(1, 8):
        next_coord = tuple(np.array(move) + i * np.array(direction))
        if all(0 <= c <= 7 for c in next_coord):
            if enemy_board[next_coord] == 1:
                continue
            elif my_board[next_coord] == 1:
                return i > 1
            return False
        return False
    return False


def get_legal_directions(board, move):
    """
    指定した指し手で同じ色に挟まれた石をひっくり返せる方向を返す。
    """
    legal_directions = []
    for direction in itertools.product([-1, 0, 1], [-1, 0, 1]):
        if direction == (0, 0):
            continue
        if is_legal_direction(board, move, direction):
            legal_directions.append(direction)
    return legal_directions


def is_legal_move(board, move):
    """
    指定した指し手で同じ色に挟まれた石をひっくり返せるかどうか判定する。
    """
    for direction in itertools.product([-1, 0, 1], [-1, 0, 1]):
        if direction == (0, 0):
            continue
        if is_legal_direction(board, move, direction):
            return True
    return False


def get_legal_moves(board):
    """
    指し手の候補を返す。
    """
    legal_moves = []
    for move in itertools.product(range(8), range(8)):
        if is_legal_move(board, move) and board[0][move] == 0 and board[1][move] == 0:
            legal_moves.append(move)
    return legal_moves


def reverse_disks(board, move):
    """
    指定した指し手で同じ色に挟まれた石をひっくり返す。
    """
    my_board, enemy_board = board
    my_board[move] = 1
    for direction in get_legal_directions(board, move):
        for i in range(1, 8):
            next_coord = tuple(np.array(move) + i * np.array(direction))
            if enemy_board[next_coord] == 1:
                enemy_board[next_coord] = 0
                my_board[next_coord] = 1
            else:
                break
    return my_board, enemy_board


def transcript2moves(transcript):
    """
    文字列のトランスクリプトを指し手に変換する。
    """
    moves = []
    for i in range(0, len(transcript), 2):
        moves.append((int(transcript[i + 1]) - 1, ord(transcript[i]) - ord("a")))
    return moves


def moves2matrices(moves):
    """
    指し手を行列に変換する。
    """
    matrices = []
    for move in moves:
        matrix = np.zeros((8, 8), dtype=np.int8)
        matrix[move] = 1
        matrices.append(matrix)
    return matrices


def moves2boards(moves):
    """
    指し手を盤面に変換する。
    """
    boards = [init_board()]
    turn = 1
    for move in moves:
        if not is_legal_move(boards[-1][::turn], move):
            turn *= -1
        board = np.copy(boards[-1][::turn])
        board_reversed = reverse_disks(board, move)
        boards.append(board_reversed[::turn])
        turn *= -1
    return boards


def get_winner_index(moves, winner):
    """
    勝者の指し手のインデックスを返す。
    """
    boards = [init_board()]
    winner_index = []
    index = 0
    turn = 1
    for move in moves:
        if not is_legal_move(boards[-1][::turn], move):
            turn *= -1
        board = np.copy(boards[-1][::turn])
        board_reversed = reverse_disks(board, move)
        boards.append(board_reversed[::turn])
        if turn == winner:
            winner_index.append(index)
        index += 1
        turn *= -1
    return winner_index


def print_board(boards):
    """
    盤面を表示する。
    """
    boards = [boards] if type(boards) != list else boards
    for board in boards:
        my_board, enemy_board = board
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
