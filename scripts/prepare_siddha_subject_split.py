from pathlib import Path
import json
import numpy as np

REPO = Path(__file__).resolve().parents[1]

DATA = (
    REPO
    / "data"
    / "SIDDHA"
    / "processed"
    / "siddha_phone_accel_20hz_5s_stride50.npz"
)

OUT = (
    REPO
    / "data"
    / "SIDDHA"
    / "processed"
)

SEED = 1

data = np.load(DATA)

X = data["X"]
y = data["y"]
subjects = data["subjects"]

unique_subjects = np.unique(subjects)

rng = np.random.default_rng(SEED)
shuffled = rng.permutation(unique_subjects)

test_subjects = shuffled[:5]
val_subjects = shuffled[5:10]
train_subjects = shuffled[10:]

train_mask = np.isin(subjects, train_subjects)
val_mask = np.isin(subjects, val_subjects)
test_mask = np.isin(subjects, test_subjects)

# ---------------------------------------------------------
# Leakage checks
# ---------------------------------------------------------
assert set(train_subjects).isdisjoint(val_subjects)
assert set(train_subjects).isdisjoint(test_subjects)
assert set(val_subjects).isdisjoint(test_subjects)

assert (
    train_mask.sum()
    + val_mask.sum()
    + test_mask.sum()
    == len(X)
)

# ---------------------------------------------------------
# Normalization statistics from TRAINING DATA ONLY
# ---------------------------------------------------------
X_train = X[train_mask]

channel_mean = X_train.mean(
    axis=(0, 1)
)

channel_std = X_train.std(
    axis=(0, 1)
)

assert np.all(channel_std > 0)

# ---------------------------------------------------------
# Class coverage
# ---------------------------------------------------------
def class_counts(mask):
    labels, counts = np.unique(
        y[mask],
        return_counts=True
    )
    return {
        int(label): int(count)
        for label, count in zip(labels, counts)
    }

print("========================================")
print(" SIDDHA SUBJECT-INDEPENDENT SPLIT")
print("========================================")

print("\nTrain subjects:", sorted(train_subjects.tolist()))
print("Validation subjects:", sorted(val_subjects.tolist()))
print("Test subjects:", sorted(test_subjects.tolist()))

print("\nNumber of subjects:")
print("Train:", len(train_subjects))
print("Validation:", len(val_subjects))
print("Test:", len(test_subjects))

print("\nNumber of windows:")
print("Train:", int(train_mask.sum()))
print("Validation:", int(val_mask.sum()))
print("Test:", int(test_mask.sum()))

print("\nActivities represented:")
print("Train:", len(np.unique(y[train_mask])))
print("Validation:", len(np.unique(y[val_mask])))
print("Test:", len(np.unique(y[test_mask])))

print("\nTraining-only normalization:")
print("Mean XYZ:", channel_mean)
print("Std XYZ:", channel_std)

print("\nSubject overlap:")
print(
    "Train/Val:",
    set(train_subjects) & set(val_subjects)
)
print(
    "Train/Test:",
    set(train_subjects) & set(test_subjects)
)
print(
    "Val/Test:",
    set(val_subjects) & set(test_subjects)
)

# ---------------------------------------------------------
# Save reproducible split information
# ---------------------------------------------------------
split_info = {
    "seed": SEED,
    "strategy": "subject-independent 41/5/5 split",
    "train_subjects": [
        int(x) for x in train_subjects
    ],
    "validation_subjects": [
        int(x) for x in val_subjects
    ],
    "test_subjects": [
        int(x) for x in test_subjects
    ],
    "train_windows": int(train_mask.sum()),
    "validation_windows": int(val_mask.sum()),
    "test_windows": int(test_mask.sum()),
    "train_class_counts": class_counts(train_mask),
    "validation_class_counts": class_counts(val_mask),
    "test_class_counts": class_counts(test_mask),
}

split_path = (
    OUT
    / "siddha_subject_split_seed1.json"
)

with split_path.open("w") as f:
    json.dump(
        split_info,
        f,
        indent=2,
    )

stats_path = (
    OUT
    / "siddha_train_normalization_seed1.npz"
)

np.savez(
    stats_path,
    mean=channel_mean,
    std=channel_std,
)

print("\nSaved split:", split_path)
print("Saved normalization:", stats_path)

print("\n=== NO SUBJECT LEAKAGE ===")
