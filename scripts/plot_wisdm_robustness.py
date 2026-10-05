from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[1]

input_file = (
    REPO
    / "results"
    / "constructed_wisdm_perturbations"
    / "summary.csv"
)

output_dir = (
    REPO
    / "results"
    / "constructed_wisdm_perturbations"
)

df = pd.read_csv(input_file)

labels = {
    "clean": "Clean",
    "gaussian_005": "Noise 0.05",
    "gaussian_010": "Noise 0.10",
    "gaussian_020": "Noise 0.20",
    "missing_010": "10% Missing",
    "missing_030": "30% Missing",
    "drop_x": "Drop X",
    "drop_y": "Drop Y",
    "drop_z": "Drop Z",
    "sampling_10hz": "10 Hz",
    "sampling_5hz": "5 Hz",
}

df["label"] = df["condition"].map(labels)
df["macro_f1_percent"] = df["macro_f1"] * 100

plt.figure(figsize=(11, 6))

bars = plt.bar(
    df["label"],
    df["macro_f1_percent"],
)

plt.axhline(
    y=df.loc[df["condition"] == "clean", "macro_f1_percent"].iloc[0],
    linestyle="--",
    linewidth=1,
    label="Clean baseline",
)

plt.ylabel("Macro-F1 (%)")
plt.xlabel("Test condition")
plt.title("TinyHAR Robustness on Constructed WISDM Test Conditions")

plt.xticks(rotation=40, ha="right")
plt.ylim(0, 100)

for bar, value in zip(bars, df["macro_f1_percent"]):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 1,
        f"{value:.1f}",
        ha="center",
        va="bottom",
        fontsize=8,
    )

plt.legend()
plt.tight_layout()

output = output_dir / "robustness_macro_f1.png"

plt.savefig(
    output,
    dpi=200,
    bbox_inches="tight",
)

print("Saved:", output)
