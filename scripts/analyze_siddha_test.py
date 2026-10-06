from pathlib import Path
import json
import sys

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    accuracy_score,
)

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from models.TinyHAR import TinyHAR_Model


DATA_PATH = (
    REPO / "data" / "SIDDHA" / "processed"
    / "siddha_phone_accel_20hz_5s_stride50.npz"
)

SPLIT_PATH = (
    REPO / "data" / "SIDDHA" / "processed"
    / "siddha_subject_split_seed1.json"
)

STATS_PATH = (
    REPO / "data" / "SIDDHA" / "processed"
    / "siddha_train_normalization_seed1.npz"
)

CHECKPOINT = (
    REPO / "results" / "assessment2_siddha"
    / "best_model.pth"
)

OUTPUT_DIR = (
    REPO / "results" / "assessment2_siddha"
)

ACTIVITY_NAMES = [
    "Walking",
    "Jogging",
    "Stairs",
    "Sitting",
    "Standing",
    "Typing",
    "Brushing Teeth",
    "Eating Soup",
    "Eating Chips",
    "Eating Pasta",
    "Drinking",
    "Eating Sandwich",
    "Kicking",
    "Playing Catch",
    "Dribbling",
    "Writing",
    "Clapping",
    "Folding Clothes",
]

ACTIVITY_CODES = [
    "A", "B", "C", "D", "E", "F",
    "G", "H", "I", "J", "K", "L",
    "M", "O", "P", "Q", "R", "S",
]


# ---------------------------------------------------------
# Load dataset and split
# ---------------------------------------------------------
data = np.load(DATA_PATH)

X = data["X"]
y = data["y"]
subjects = data["subjects"]

with SPLIT_PATH.open() as f:
    split = json.load(f)

stats = np.load(STATS_PATH)

mean = stats["mean"].astype(np.float32)
std = stats["std"].astype(np.float32)

test_subjects = np.asarray(
    split["test_subjects"]
)

test_mask = np.isin(
    subjects,
    test_subjects,
)


class TestDataset(Dataset):

    def __init__(self):
        self.X = X[test_mask]
        self.y = y[test_mask]

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):

        x = self.X[idx].astype(np.float32)

        # Same training-derived normalization
        x = (x - mean) / std

        # TinyHAR shape: [F, T, C]
        x = np.expand_dims(x, axis=0)

        return (
            torch.from_numpy(x).double(),
            torch.tensor(
                self.y[idx],
                dtype=torch.long,
            ),
        )


loader = DataLoader(
    TestDataset(),
    batch_size=256,
    shuffle=False,
)


# ---------------------------------------------------------
# Rebuild TinyHAR
# ---------------------------------------------------------
device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

model = TinyHAR_Model(
    (1, 1, 100, 3),
    18,
    filter_num=16,
    cross_channel_interaction_type="attn",
    cross_channel_aggregation_type="FC",
    temporal_info_interaction_type="lstm",
    temporal_info_aggregation_type="tnaive",
).double().to(device)

model.load_state_dict(
    torch.load(
        CHECKPOINT,
        map_location=device,
    )
)

model.eval()


# ---------------------------------------------------------
# Predict held-out subjects
# ---------------------------------------------------------
predictions = []
truths = []

with torch.no_grad():

    for batch_x, batch_y in loader:

        batch_x = batch_x.to(device)

        outputs = model(batch_x)

        pred = torch.argmax(
            outputs,
            dim=1,
        )

        predictions.extend(
            pred.cpu().numpy()
        )

        truths.extend(
            batch_y.numpy()
        )

truths = np.asarray(truths)
predictions = np.asarray(predictions)


# ---------------------------------------------------------
# Metrics
# ---------------------------------------------------------
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

per_class_f1 = f1_score(
    truths,
    predictions,
    average=None,
    labels=np.arange(18),
    zero_division=0,
)

print("========================================")
print(" SIDDHA TEST ANALYSIS")
print("========================================")

print("Test subjects:", test_subjects.tolist())
print("Test windows:", len(truths))
print(f"Accuracy: {accuracy:.4f}")
print(f"Macro-F1: {macro_f1:.4f}")

print("\nPer-class F1:")

rows = []

for i in range(18):

    print(
        f"{ACTIVITY_CODES[i]} | "
        f"{ACTIVITY_NAMES[i]:18s} | "
        f"{per_class_f1[i]:.4f}"
    )

    rows.append({
        "class_id": i,
        "code": ACTIVITY_CODES[i],
        "activity": ACTIVITY_NAMES[i],
        "f1": per_class_f1[i],
    })


# ---------------------------------------------------------
# Save per-class F1
# ---------------------------------------------------------
f1_path = (
    OUTPUT_DIR
    / "siddha_per_class_f1.csv"
)

pd.DataFrame(rows).to_csv(
    f1_path,
    index=False,
)


# ---------------------------------------------------------
# Normalized confusion matrix
# ---------------------------------------------------------
cm = confusion_matrix(
    truths,
    predictions,
    labels=np.arange(18),
    normalize="true",
)

fig, ax = plt.subplots(
    figsize=(13, 11)
)

image = ax.imshow(cm)

fig.colorbar(
    image,
    ax=ax,
    label="Proportion of true class",
)

labels = [
    f"{code}: {name}"
    for code, name in zip(
        ACTIVITY_CODES,
        ACTIVITY_NAMES,
    )
]

ax.set_xticks(
    np.arange(18)
)

ax.set_yticks(
    np.arange(18)
)

ax.set_xticklabels(
    labels,
    rotation=70,
    ha="right",
    fontsize=8,
)

ax.set_yticklabels(
    labels,
    fontsize=8,
)

ax.set_xlabel("Predicted activity")
ax.set_ylabel("True activity")

ax.set_title(
    "TinyHAR on SIDDHA — Held-Out Subject Confusion Matrix"
)

plt.tight_layout()

figure_path = (
    OUTPUT_DIR
    / "siddha_confusion_matrix.png"
)

plt.savefig(
    figure_path,
    dpi=200,
    bbox_inches="tight",
)

print("\nSaved F1:", f1_path)
print("Saved matrix:", figure_path)

print("\n=== SIDDHA ANALYSIS COMPLETE ===")
