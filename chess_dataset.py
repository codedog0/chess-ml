import chess.pgn

def load_games(pgn_path, max_games=1000):
    X, y = [], []
    with open(pgn_path) as f:
        for i in range(max_games):
            game = chess.pgn.read_game(f)
            if game is None:
                break
            board = game.board()
            for move in game.mainline_moves():
                X.append(board.fen())
                y.append(move.uci())
                board.push(move)
    return X, y
