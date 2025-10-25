import torch
import numpy as np
from tqdm import tqdm
import chess
from encoder import board_to_tensor,move_to_index
from chess_dataset import load_games
# (Include your board_to_tensor, move_to_index, and load_games functions here)



print("Loading raw game data...")
X_raw, y_raw = load_games(
    "lichess_db_standard_rated_2013-07.pgn", max_games=5000
)

print("Pre-processing data... This will take a while.")
all_board_tensors = []
all_move_indices = []
all_legal_masks = []

for fen, move_uci in tqdm(zip(X_raw, y_raw), total=len(X_raw)):
    # 1. Get board tensor
    board_tensor = board_to_tensor(fen)

    # 2. Get target move index
    move_index = move_to_index(move_uci)

    # 3. Get legal move mask
    mask = np.full((4096,), -np.inf, dtype=np.float32)  # Use numpy
    try:
        board = chess.Board(fen)
        for move in board.legal_moves:
            mask[move_to_index(move.uci())] = 0.0
    except Exception as e:
        print(f"Skipping bad FEN: {fen}, Error: {e}")
        continue

    all_board_tensors.append(board_tensor)
    all_move_indices.append(move_index)
    all_legal_masks.append(mask)

# Convert to tensors and save
print("Saving pre-processed tensors...")
torch.save(
    {
        "boards": torch.tensor(np.array(all_board_tensors), dtype=torch.float32),
        "moves": torch.tensor(all_move_indices, dtype=torch.long),
        "masks": torch.tensor(np.array(all_legal_masks), dtype=torch.float32),
    },
    "preprocessed_chess_data.pt",

)

print("✅ Pre-processing complete.")
