import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import numpy as np
import chess
import torch.nn as nn
import torch.nn.functional as F

# HELPER FUNCTIONS
PIECE_IDX = {
    "P": 0,
    "N": 1,
    "B": 2,
    "R": 3,
    "Q": 4,
    "K": 5,
    "p": 6,
    "n": 7,
    "b": 8,
    "r": 9,
    "q": 10,
    "k": 11,
}


def board_to_tensor(fen):
    board = chess.Board(fen)
    tensor = np.zeros((12, 8, 8), dtype=np.float32)
    for square, piece in board.piece_map().items():
        plane = PIECE_IDX[piece.symbol()]
        row, col = divmod(square, 8)
        tensor[plane, 7 - row, col] = 1.0
    return tensor


def move_to_index(move_uci):
    # Convert move like "e2e4" to index 0–4095
    from_sq = chess.parse_square(move_uci[:2])
    to_sq = chess.parse_square(move_uci[2:4])
    return from_sq * 64 + to_sq


def index_to_move(index):
    from_sq = index // 64
    to_sq = index % 64
    return chess.Move(from_sq, to_sq).uci()

# CHESSBUT ?
class ChessNet(nn.Module):
    def __init__(self):
        super(ChessNet, self).__init__()
        self.conv1 = nn.Conv2d(12, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(1024, 1024)  # adjust based on actual shape
        self.fc2 = nn.Linear(1024, 4672)  # number of possible chess moves

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        # print(x.shape)  # <-- print once to confirm
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# 

def load_games(pgn_path, max_games=1000):
    X, y = [], []

    with open(pgn_path) as f:
        for i in tqdm(range(max_games), desc="Loading games"):
            game = chess.pgn.read_game(f)
            if game is None:
                break
            board = game.board()
            for move in game.mainline_moves():
                X.append(board.fen())
                y.append(move.uci())
                board.push(move)
    return X, y


# Load dataset (small sample first)
X_raw, y_raw = load_games("lichess_db_standard_rated_2013-07.pgn", max_games=500)

# Encode boards and moves
X = np.array([board_to_tensor(fen) for fen in X_raw])
y = np.array([move_to_index(m) for m in y_raw])

# Split into train/test
split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# Torch datasets
train_ds = TensorDataset(torch.tensor(X_train), torch.tensor(y_train))
test_ds = TensorDataset(torch.tensor(X_test), torch.tensor(y_test))
train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)

# Model, optimizer, loss
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ChessNet().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
loss_fn = torch.nn.CrossEntropyLoss()

print("Training started...")
for epoch in range(1):  # small epochs to test
    total_loss = 0
    for xb, yb in tqdm(train_loader):
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        preds = model(xb)
        loss = loss_fn(preds, yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch + 1}, Loss = {total_loss / len(train_loader):.4f}")

torch.save(model.state_dict(), "chess_model.pt")
print("✅ Training complete. Model saved as chess_model.pt")
