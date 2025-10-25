import chess
import chess.engine
import torch
from chess_model import ChessNet
from encoder import board_to_tensor, index_to_move
import random

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ChessNet().to(device)
model.load_state_dict(torch.load("chess_model_250k.pt", map_location=device))
model.eval()

engine = chess.engine.SimpleEngine.popen_uci(
    "stockfish-windows-x86-64-avx2\stockfish\stockfish-windows-x86-64-avx2.exe"
)  

def ml_bot_move(board):
    input_tensor = torch.tensor(board_to_tensor(board.fen())).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(input_tensor)
        move_idx = torch.argmax(logits).item()
    move_uci = index_to_move(move_idx)
    move = chess.Move.from_uci(move_uci)

    if move not in board.legal_moves:
        move = random.choice(list(board.legal_moves))
    return move


def play_game():
    board = chess.Board()
    while not board.is_game_over():
        if board.turn == chess.WHITE:
            move = ml_bot_move(board)
        else:
            result = engine.play(board, chess.engine.Limit(time=0.1))
            move = result.move
        board.push(move)
    result = board.result()
    return result


# Elo variables
bot_rating = 1200
stockfish_rating = 1500
K = 32
num_games = 50

for i in range(num_games):
    result = play_game()
    if result == "1-0":  # bot won
        Sa = 1
    elif result == "0-1":  # bot lost
        Sa = 0
    else:
        Sa = 0.5
    Ea = 1 / (1 + 10 ** ((stockfish_rating - bot_rating) / 400))
    bot_rating += K * (Sa - Ea)
    print(f"Game {i + 1}: {result}, Bot Elo ≈ {bot_rating:.1f}")

engine.quit()
print(f"\n🏁 Final estimated Elo of bot: {bot_rating:.1f}")


# RESULT

# Game 1: 0-1, Bot Elo ≈ 1195.2
# Game 2: 0-1, Bot Elo ≈ 1190.5
# Game 3: 0-1, Bot Elo ≈ 1185.8
# Game 4: 0-1, Bot Elo ≈ 1181.3
# Game 5: 0-1, Bot Elo ≈ 1176.9
# Game 6: 0-1, Bot Elo ≈ 1172.6
# Game 7: 0-1, Bot Elo ≈ 1168.4
# Game 8: 0-1, Bot Elo ≈ 1164.3
# Game 9: 0-1, Bot Elo ≈ 1160.2
# Game 10: 0-1, Bot Elo ≈ 1156.3
# Game 11: 0-1, Bot Elo ≈ 1152.4
# Game 12: 0-1, Bot Elo ≈ 1148.6
# Game 13: 0-1, Bot Elo ≈ 1144.8
# Game 14: 0-1, Bot Elo ≈ 1141.2
# Game 15: 0-1, Bot Elo ≈ 1137.6
# Game 16: 0-1, Bot Elo ≈ 1134.0
# Game 17: 0-1, Bot Elo ≈ 1130.5
# Game 18: 0-1, Bot Elo ≈ 1127.1
# Game 19: 0-1, Bot Elo ≈ 1123.8
# Game 20: 0-1, Bot Elo ≈ 1120.5
# Game 21: 0-1, Bot Elo ≈ 1117.3
# Game 22: 0-1, Bot Elo ≈ 1114.1
# Game 23: 0-1, Bot Elo ≈ 1110.9
# Game 24: 0-1, Bot Elo ≈ 1107.9
# Game 25: 0-1, Bot Elo ≈ 1104.8
# Game 26: 0-1, Bot Elo ≈ 1101.9
# Game 27: 0-1, Bot Elo ≈ 1098.9
# Game 28: 0-1, Bot Elo ≈ 1096.0
# Game 29: 0-1, Bot Elo ≈ 1093.2
# Game 30: 0-1, Bot Elo ≈ 1090.4
# Game 31: 0-1, Bot Elo ≈ 1087.6
# Game 32: 0-1, Bot Elo ≈ 1084.9
# Game 33: 0-1, Bot Elo ≈ 1082.2
# Game 34: 0-1, Bot Elo ≈ 1079.5
# Game 35: 0-1, Bot Elo ≈ 1076.9
# Game 36: 0-1, Bot Elo ≈ 1074.4
# Game 37: 0-1, Bot Elo ≈ 1071.8
# Game 38: 0-1, Bot Elo ≈ 1069.3
# Game 39: 0-1, Bot Elo ≈ 1066.8
# Game 40: 0-1, Bot Elo ≈ 1064.4
# Game 41: 0-1, Bot Elo ≈ 1062.0
# Game 42: 0-1, Bot Elo ≈ 1059.6
# Game 43: 0-1, Bot Elo ≈ 1057.2
# Game 44: 0-1, Bot Elo ≈ 1054.9
# Game 45: 0-1, Bot Elo ≈ 1052.6
# Game 46: 0-1, Bot Elo ≈ 1050.4
# Game 47: 0-1, Bot Elo ≈ 1048.1
# Game 48: 0-1, Bot Elo ≈ 1045.9
# Game 49: 0-1, Bot Elo ≈ 1043.7
# Game 50: 0-1, Bot Elo ≈ 1041.6

# 🏁 Final estimated Elo of bot: 1041.6