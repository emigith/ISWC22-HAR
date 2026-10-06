from pathlib import Path
from collections import Counter
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

REPO = Path(__file__).resolve().parents[1]
PARQUET = REPO / "data" / "SIDDHA" / "datasets" / "data.parquet"

pf = pq.ParquetFile(PARQUET)

subjects = Counter()
activities = Counter()
devices = Counter()
subject_activity_device = Counter()

missing = Counter()

valid_time_diffs = []

columns = [
    "device",
    "activity",
    "id",
    "gyro_x",
    "gyro_y",
    "gyro_z",
    "acc_x",
    "acc_y",
    "acc_z",
    "timestamp",
]

print("========================================")
print(" SIDDHA DATASET AUDIT")
print("========================================")
print("Rows:", pf.metadata.num_rows)
print("Columns:", pf.metadata.num_columns)
print("Row groups:", pf.metadata.num_row_groups)

for rg in range(pf.metadata.num_row_groups):

    print(f"\nProcessing row group {rg + 1}/{pf.metadata.num_row_groups}")

    df = pf.read_row_group(
        rg,
        columns=columns
    ).to_pandas()

    # -----------------------------------------------------
    # Basic counts
    # -----------------------------------------------------
    subjects.update(df["id"].dropna().astype(str))
    activities.update(df["activity"].dropna().astype(str))
    devices.update(df["device"].dropna().astype(str))

    combos = (
        df[["id", "activity", "device"]]
        .dropna()
        .astype(str)
        .value_counts()
    )

    for key, value in combos.items():
        subject_activity_device[key] += int(value)

    # -----------------------------------------------------
    # Missing values
    # -----------------------------------------------------
    for col in columns:
        missing[col] += int(df[col].isna().sum())

    # -----------------------------------------------------
    # Sampling interval
    # Only compare consecutive rows belonging to the same
    # participant + activity + device.
    # -----------------------------------------------------
    same_sequence = (
        (df["id"] == df["id"].shift(1))
        & (df["activity"] == df["activity"].shift(1))
        & (df["device"] == df["device"].shift(1))
    )

    diffs = df["timestamp"].diff()

    diffs = diffs[
        same_sequence
        & (diffs > 0)
        & (diffs < 1.0)
    ]

    # Keep a sample so memory remains small
    if len(diffs) > 100000:
        diffs = diffs.sample(
            100000,
            random_state=1
        )

    valid_time_diffs.extend(
        diffs.to_numpy().tolist()
    )

    del df


# =========================================================
# Final summary
# =========================================================
print("\n========================================")
print(" DATASET SUMMARY")
print("========================================")

print("\nUnique subjects:", len(subjects))
print("Subject IDs:")
print(sorted(subjects.keys(), key=lambda x: int(x)))

print("\nUnique activities:", len(activities))
print("Activities:")
print(sorted(activities.keys()))

print("\nUnique devices:", len(devices))
print("Devices:")
print(sorted(devices.keys()))

print("\nRows per device:")
for device, count in devices.most_common():
    print(f"{device:10s}: {count:,}")

print("\nRows per activity:")
for activity, count in sorted(activities.items()):
    print(f"{activity:5s}: {count:,}")

print("\nMissing values:")
for col in columns:
    print(f"{col:12s}: {missing[col]:,}")

print(
    "\nObserved subject/activity/device combinations:",
    len(subject_activity_device)
)

# ---------------------------------------------------------
# Sampling frequency estimate
# ---------------------------------------------------------
diffs = np.asarray(valid_time_diffs)

if len(diffs) > 0:

    median_dt = float(np.median(diffs))
    mean_dt = float(np.mean(diffs))

    print("\nSampling interval:")
    print(f"Median dt: {median_dt:.6f} s")
    print(f"Mean dt:   {mean_dt:.6f} s")

    print("\nEstimated sampling frequency:")
    print(f"From median: {1 / median_dt:.3f} Hz")
    print(f"From mean:   {1 / mean_dt:.3f} Hz")

    unique, counts = np.unique(
        np.round(diffs, 4),
        return_counts=True
    )

    order = np.argsort(counts)[::-1]

    print("\nMost common timestamp differences:")
    for idx in order[:10]:
        print(
            f"{unique[idx]:.4f} s : "
            f"{counts[idx]:,}"
        )

print("\n=== AUDIT COMPLETE ===")
