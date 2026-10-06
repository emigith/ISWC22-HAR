from pathlib import Path
import json
import csv
import copy
import random

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score

import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from models.TinyHAR import TinyHAR_Model


SEED = 1
BATCH_SIZE = 256
EPOCHS = 10
LEARNING_RATE = 0.001

DATA_PATH = (
    REPO
    / "data"
    / "SIDDHA"
    / "processed"
    / "siddha_phone_accel_20hz_5s_stride50.npz"
)

SPLIT_PATH = (
    REPO
    / "data"
    / "SIDDHA"
    / "processed"
    / "siddha_subject_split_seed1.json"
)

STATS_PATH = (
    REPO
    / "data"
    / "SIDDHA"
    / "processed"
    / "siddha_train_normalization_seed1.npz"
)

OUTPUT_DIR = (
    REPO
    / "results"
    / "assessment2_siddha"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# =========================================================
# Reproducibility
# =========================================================
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# =========================================================
# Load processed SIDDHA
# =========================================================
data = np.load(DATA_PATH)

X = data["X"]
y = data["y"]
subjects = data["subjects"]

with SPLIT_PATH.open() as f:
    split = json.load(f)

stats = np.load(STATS_PATH)

mean = stats["mean"]
std = stats["std"]

train_subjects = np.asarray(
    split["train_subjects"]
)

val_subjects = np.asarray(
    split["validation_subjects"]
)

test_subjects = np.asarray(
    split["test_subjects"]
)

train_mask = np.isin(
    subjects,
    train_subjects,
)

val_mask = np.isin(
    subjects,
    val_subjects,
)

test_mask = np.isin(
    subjects,
    test_subjects,
)


# =========================================================
# Dataset
# =========================================================
class SIDDHADataset(Dataset):

    def __init__(self, X, y, mask, mean, std):

        self.X = X[mask]
        self.y = y[mask]

        self.mean = mean.astype(
            np.float32
        )

        self.std = std.astype(
            np.float32
        )

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):

        x = self.X[idx].astype(
            np.float32
        )

        # Training-derived normalization only
        x = (
            x - self.mean
        ) / self.std

        # TinyHAR expects:
        # [filter_channel, time, sensor_channel]
        x = np.expand_dims(
            x,
            axis=0,
        )

        return (
            torch.from_numpy(x).double(),
            torch.tensor(
                self.y[idx],
                dtype=torch.long,
            ),
        )


train_dataset = SIDDHADataset(
    X,
    y,
    train_mask,
    mean,
    std,
)

val_dataset = SIDDHADataset(
    X,
    y,
    val_mask,
    mean,
    std,
)

test_dataset = SIDDHADataset(
    X,
    y,
    test_mask,
    mean,
    std,
)


train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
)


# =========================================================
# Model
# =========================================================
device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print("========================================")
print(" BUILD TINYHAR FOR SIDDHA")
print("========================================")

print("Device:", device)

model = TinyHAR_Model(
    (1, 1, 100, 3),
    18,
    filter_num=16,
    cross_channel_interaction_type="attn",
    cross_channel_aggregation_type="FC",
    temporal_info_interaction_type="lstm",
    temporal_info_aggregation_type="tnaive",
).double().to(device)

parameter_count = sum(
    p.numel()
    for p in model.parameters()
    if p.requires_grad
)

print("Parameters:", parameter_count)

print("\nTrain windows:", len(train_dataset))
print("Validation windows:", len(val_dataset))
print("Test windows:", len(test_dataset))

print("Train subjects:", len(train_subjects))
print("Validation subjects:", len(val_subjects))
print("Test subjects:", len(test_subjects))

print("Classes: 18")


# =========================================================
# Evaluation
# =========================================================
criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE,
)


def evaluate(loader):

    model.eval()

    losses = []
    predictions = []
    truths = []

    with torch.no_grad():

        for batch_x, batch_y in loader:

            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            outputs = model(batch_x)

            loss = criterion(
                outputs,
                batch_y,
            )

            losses.append(
                loss.item()
            )

            pred = torch.argmax(
                outputs,
                dim=1,
            )

            predictions.extend(
                pred.cpu().numpy()
            )

            truths.extend(
                batch_y.cpu().numpy()
            )

    accuracy = accuracy_score(
        truths,
        predictions,
    )

    macro_f1 = f1_score(
        truths,
        predictions,
        average="macro",
        zero_division=0,
    )

    weighted_f1 = f1_score(
        truths,
        predictions,
        average="weighted",
        zero_division=0,
    )

    return (
        float(np.mean(losses)),
        accuracy,
        macro_f1,
        weighted_f1,
    )


# =========================================================
# Training
# =========================================================
print("\n========================================")
print(" TRAIN SIDDHA")
print("========================================")

metrics_path = (
    OUTPUT_DIR
    / "metrics.csv"
)

checkpoint_path = (
    OUTPUT_DIR
    / "best_model.pth"
)

best_val_loss = float("inf")
best_state = None
best_epoch = None

with metrics_path.open(
    "w",
    newline="",
) as f:

    writer = csv.writer(f)

    writer.writerow([
        "epoch",
        "train_loss",
        "val_loss",
        "val_accuracy",
        "val_macro_f1",
        "val_weighted_f1",
    ])

    for epoch in range(EPOCHS):

        model.train()

        train_losses = []

        for batch_x, batch_y in train_loader:

            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()

            outputs = model(
                batch_x
            )

            loss = criterion(
                outputs,
                batch_y,
            )

            loss.backward()

            optimizer.step()

            train_losses.append(
                loss.item()
            )

        train_loss = float(
            np.mean(train_losses)
        )

        (
            val_loss,
            val_accuracy,
            val_macro_f1,
            val_weighted_f1,
        ) = evaluate(
            val_loader
        )

        print(
            f"Epoch {epoch + 1:02d} | "
            f"Train Loss {train_loss:.4f} | "
            f"Val Loss {val_loss:.4f} | "
            f"Val Acc {val_accuracy:.4f} | "
            f"Val Macro-F1 {val_macro_f1:.4f}"
        )

        writer.writerow([
            epoch + 1,
            train_loss,
            val_loss,
            val_accuracy,
            val_macro_f1,
            val_weighted_f1,
        ])

        f.flush()

        if val_loss < best_val_loss:

            best_val_loss = val_loss
            best_epoch = epoch + 1

            best_state = copy.deepcopy(
                model.state_dict()
            )


# =========================================================
# Final held-out test evaluation
# =========================================================
model.load_state_dict(
    best_state
)

torch.save(
    best_state,
    checkpoint_path,
)

(
    test_loss,
    test_accuracy,
    test_macro_f1,
    test_weighted_f1,
) = evaluate(
    test_loader
)


print("\n========================================")
print(" SIDDHA FINAL RESULTS")
print("========================================")

print("Best epoch:", best_epoch)
print(
    f"Best validation loss: "
    f"{best_val_loss:.4f}"
)

print(
    f"Test loss: "
    f"{test_loss:.4f}"
)

print(
    f"Test accuracy: "
    f"{test_accuracy:.4f}"
)

print(
    f"Test Macro-F1: "
    f"{test_macro_f1:.4f}"
)

print(
    f"Test weighted F1: "
    f"{test_weighted_f1:.4f}"
)

print()
print("Parameters:", parameter_count)

print(
    "Metrics saved to:",
    metrics_path
)

print(
    "Checkpoint saved to:",
    checkpoint_path
)

print(
    "\n=== SIDDHA EXPERIMENT COMPLETE ==="
)
