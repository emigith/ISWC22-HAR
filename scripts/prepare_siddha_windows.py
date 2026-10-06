from pathlib import Path
import json

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

REPO = Path(__file__).resolve().parents[1]

SOURCE = (
    REPO
    / "data"
    / "SIDDHA"
    / "datasets"
    / "data.parquet"
)

OUTPUT_DIR = (
    REPO
    / "data"
    / "SIDDHA"
    / "processed"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

DEVICE = "phone"

SENSOR_COLUMNS = [
    "acc_x",
    "acc_y",
    "acc_z",
]

WINDOW_SIZE = 100       # 5 seconds at 20 Hz
STRIDE = 50             # 50% overlap

ACTIVITIES = [
    "A", "B", "C", "D", "E", "F",
    "G", "H", "I", "J", "K", "L",
    "M", "O", "P", "Q", "R", "S",
]

ACTIVITY_TO_ID = {
    activity: idx
    for idx, activity in enumerate(ACTIVITIES)
}


print("========================================")
print(" PREPARE SIDDHA WINDOWS")
print("========================================")

print("Device:", DEVICE)
print("Sensors:", SENSOR_COLUMNS)
print("Window size:", WINDOW_SIZE)
print("Stride:", STRIDE)
print("Activities:", len(ACTIVITIES))


# =========================================================
# Load only columns needed for window construction
# =========================================================
table = pq.read_table(
    SOURCE,
    columns=[
        "device",
        "activity",
        "id",
        "timestamp",
        *SENSOR_COLUMNS,
    ],
)

df = table.to_pandas()

print("\nRaw rows loaded:", len(df))


# =========================================================
# Select phone accelerometer
# =========================================================
df = df[
    df["device"] == DEVICE
].copy()

print("Phone rows:", len(df))


# =========================================================
# Sort explicitly within recordings
# =========================================================
df["id"] = df["id"].astype(int)

df = df.sort_values(
    ["id", "activity", "timestamp"]
).reset_index(drop=True)


# =========================================================
# Construct windows WITHOUT crossing recordings
# =========================================================
X_windows = []
y_windows = []
subject_windows = []
activity_windows = []

session_rows = []

grouped = df.groupby(
    ["id", "activity"],
    sort=True,
)

print("\nConstructing windows...")

for (subject, activity), session in grouped:

    session = session.sort_values(
        "timestamp"
    )

    signal = session[
        SENSOR_COLUMNS
    ].to_numpy(dtype=np.float32)

    n_samples = len(signal)

    n_windows = 0

    for start in range(
        0,
        n_samples - WINDOW_SIZE + 1,
        STRIDE,
    ):

        stop = start + WINDOW_SIZE

        window = signal[start:stop]

        # Safety check
        if window.shape != (WINDOW_SIZE, 3):
            continue

        X_windows.append(window)

        y_windows.append(
            ACTIVITY_TO_ID[activity]
        )

        subject_windows.append(
            int(subject)
        )

        activity_windows.append(
            activity
        )

        n_windows += 1

    session_rows.append({
        "subject": int(subject),
        "activity": activity,
        "samples": n_samples,
        "windows": n_windows,
    })


# =========================================================
# Convert to arrays
# =========================================================
X = np.stack(
    X_windows
).astype(np.float32)

y = np.asarray(
    y_windows,
    dtype=np.int64,
)

subjects = np.asarray(
    subject_windows,
    dtype=np.int64,
)

activity_codes = np.asarray(
    activity_windows,
)


# =========================================================
# Validation
# =========================================================
assert X.ndim == 3
assert X.shape[1:] == (100, 3)

assert len(X) == len(y)
assert len(X) == len(subjects)
assert len(X) == len(activity_codes)

assert not np.isnan(X).any()

print("\n========================================")
print(" WINDOW DATASET SUMMARY")
print("========================================")

print("Windows:", len(X))
print("X shape:", X.shape)
print("y shape:", y.shape)

print("Subjects:", len(np.unique(subjects)))
print("Activities:", len(np.unique(y)))

print("NaN values:", int(np.isnan(X).sum()))

print(
    "Approximate data size:",
    round(X.nbytes / 1024**2, 2),
    "MB"
)


# =========================================================
# Class distribution
# =========================================================
print("\nWindows per activity:")

for activity in ACTIVITIES:

    label = ACTIVITY_TO_ID[activity]

    count = int(
        np.sum(y == label)
    )

    print(
        f"{activity:3s} -> "
        f"{label:2d}: "
        f"{count:,}"
    )


# =========================================================
# Save processed dataset
# =========================================================
npz_path = (
    OUTPUT_DIR
    / "siddha_phone_accel_20hz_5s_stride50.npz"
)

np.savez_compressed(
    npz_path,
    X=X,
    y=y,
    subjects=subjects,
    activity_codes=activity_codes,
)


# =========================================================
# Save session manifest
# =========================================================
manifest = pd.DataFrame(
    session_rows
)

manifest_path = (
    OUTPUT_DIR
    / "siddha_phone_session_manifest.csv"
)

manifest.to_csv(
    manifest_path,
    index=False,
)


# =========================================================
# Save metadata / label mapping
# =========================================================
metadata = {
    "source": "SIDDHA",
    "device": DEVICE,
    "sensor_columns": SENSOR_COLUMNS,
    "sampling_frequency_hz": 20,
    "window_size_samples": WINDOW_SIZE,
    "window_duration_seconds": 5,
    "stride_samples": STRIDE,
    "overlap_percent": 50,
    "number_of_subjects": int(
        len(np.unique(subjects))
    ),
    "number_of_activities": int(
        len(np.unique(y))
    ),
    "number_of_windows": int(
        len(X)
    ),
    "activity_to_id": ACTIVITY_TO_ID,
    "normalization": (
        "none during window construction; "
        "must be fitted on training subjects only"
    ),
}

metadata_path = (
    OUTPUT_DIR
    / "siddha_phone_metadata.json"
)

with metadata_path.open(
    "w"
) as f:
    json.dump(
        metadata,
        f,
        indent=2,
    )


print("\n========================================")
print(" SIDDHA WINDOW CONSTRUCTION COMPLETE")
print("========================================")

print("Dataset:", npz_path)
print("Manifest:", manifest_path)
print("Metadata:", metadata_path)
