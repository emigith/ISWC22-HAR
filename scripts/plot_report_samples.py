from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[1]
OUTPUT = REPO / "report_figures"
OUTPUT.mkdir(parents=True, exist_ok=True)


def plot_wisdm_sample():
    source = REPO / "data" / "WISDM_Dataset" / "WISDM_ar_v1.1_raw.txt"

    rows = []
    with source.open() as f:
        for line in f:
            record = line.strip().rstrip(";")
            if not record:
                continue

            fields = record.split(",")
            if len(fields) != 6:
                continue

            subject, activity, timestamp, x, y, z = fields
            rows.append({
                "subject": int(subject),
                "activity": activity,
                "timestamp": int(timestamp),
                "x": float(x),
                "y": float(y),
                "z": float(z),
            })

    df = pd.DataFrame(rows)

    sample = None
    sample_subject = None
    sample_activity = None

    for (subject, activity), group in df.groupby(
        ["subject", "activity"],
        sort=False,
    ):
        if len(group) >= 100:
            sample = group.iloc[:100]
            sample_subject = subject
            sample_activity = activity
            break

    if sample is None:
        raise RuntimeError("No WISDM recording with at least 100 samples was found.")

    time = np.arange(100) / 20.0

    plt.figure(figsize=(7, 3.5))
    plt.plot(time, sample["x"], label="X")
    plt.plot(time, sample["y"], label="Y")
    plt.plot(time, sample["z"], label="Z")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Acceleration")
    plt.title(f"WISDM: {sample_activity}, Subject {sample_subject}")
    plt.legend()
    plt.tight_layout()

    out = OUTPUT / "wisdm_sample.png"
    plt.savefig(out, dpi=200)
    plt.close()
    return out


def plot_siddha_sample():
    source = (
        REPO
        / "data"
        / "SIDDHA"
        / "processed"
        / "siddha_phone_accel_20hz_5s_stride50.npz"
    )

    data = np.load(source)

    X = data["X"]
    activity_codes = data["activity_codes"]
    subjects = data["subjects"]

    sample = X[0]
    time = np.arange(100) / 20.0

    plt.figure(figsize=(7, 3.5))
    plt.plot(time, sample[:, 0], label="X")
    plt.plot(time, sample[:, 1], label="Y")
    plt.plot(time, sample[:, 2], label="Z")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Acceleration")
    plt.title(
        f"SIDDHA: Activity {activity_codes[0]}, Subject {subjects[0]}"
    )
    plt.legend()
    plt.tight_layout()

    out = OUTPUT / "siddha_sample.png"
    plt.savefig(out, dpi=200)
    plt.close()
    return out


if __name__ == "__main__":
    wisdm_out = plot_wisdm_sample()
    siddha_out = plot_siddha_sample()

    print("Created:")
    print(wisdm_out)
    print(siddha_out)
