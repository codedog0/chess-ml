import torch
import chess
from encoder import board_to_tensor, index_to_move
from chess_model import ChessNet
import random

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ChessNet().to(device)
model.load_state_dict(torch.load("chess_model.pt", map_location=device))
model.eval()

board = chess.Board()
print(board)

while not board.is_game_over():
    if board.turn == chess.WHITE:
        move_uci = input("Your move (e.g., e2e4): ")
        try:
            move = chess.Move.from_uci(move_uci)
            if move in board.legal_moves:
                board.push(move)
            else:
                print("Illegal move.")
                continue
        except:
            print("Invalid input.")
            continue
    else:
        input_tensor = (
            torch.tensor(board_to_tensor(board.fen())).unsqueeze(0).to(device)
        )
        with torch.no_grad():
            logits = model(input_tensor)
            move_idx = torch.argmax(logits).item()
        move_uci = index_to_move(move_idx)
        move = chess.Move.from_uci(move_uci)
        if move not in board.legal_moves:
            move = random.choice(list(board.legal_moves))
        board.push(move)
        print(f"Bot plays: {move.uci()}")
    print(board)
