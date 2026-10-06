from pathlib import Path
from collections import Counter
import itertools

import numpy as np
import pyarrow.parquet as pq

REPO = Path(__file__).resolve().parents[1]
PARQUET = REPO / "data" / "SIDDHA" / "datasets" / "data.parquet"

pf = pq.ParquetFile(PARQUET)

counts = Counter()

for rg in range(pf.metadata.num_row_groups):
    print(f"Processing row group {rg + 1}/{pf.metadata.num_row_groups}")

    df = pf.read_row_group(
        rg,
        columns=["id", "activity", "device"]
    ).to_pandas()

    group_counts = (
        df.groupby(["id", "activity", "device"])
          .size()
    )

    for key, value in group_counts.items():
        key = tuple(str(x) for x in key)
        counts[key] += int(value)


subjects = [str(i) for i in range(51)]

activities = [
    "A", "B", "C", "D", "E", "F",
    "G", "H", "I", "J", "K", "L",
    "M", "O", "P", "Q", "R", "S",
]

devices = ["phone", "watch"]

expected = set(
    itertools.product(subjects, activities, devices)
)

observed = set(counts.keys())

missing = sorted(
    expected - observed,
    key=lambda x: (int(x[0]), x[1], x[2])
)

lengths = np.asarray(list(counts.values()))

print("\n========================================")
print(" SIDDHA SESSION COMPLETENESS")
print("========================================")

print("Expected combinations:", len(expected))
print("Observed combinations:", len(observed))
print("Missing combinations:", len(missing))

print("\nMissing subject/activity/device sessions:")
for subject, activity, device in missing:
    print(
        f"subject={subject:>2s} | "
        f"activity={activity} | "
        f"device={device}"
    )

print("\nSamples per existing session:")
print("Minimum:", int(lengths.min()))
print("Median: ", int(np.median(lengths)))
print("Maximum:", int(lengths.max()))
print("Mean:   ", round(float(lengths.mean()), 2))

print("\nEquivalent duration at 20 Hz:")
print(
    "Minimum:",
    round(float(lengths.min()) / 20, 2),
    "seconds"
)
print(
    "Median:",
    round(float(np.median(lengths)) / 20, 2),
    "seconds"
)
print(
    "Maximum:",
    round(float(lengths.max()) / 20, 2),
    "seconds"
)

print("\nSessions shorter than 100 samples (5 sec):")
short = [
    (key, value)
    for key, value in counts.items()
    if value < 100
]

print(len(short))

for key, value in sorted(
    short,
    key=lambda x: x[1]
)[:20]:
    print(key, value)

print("\n=== COMPLETENESS CHECK COMPLETE ===")
