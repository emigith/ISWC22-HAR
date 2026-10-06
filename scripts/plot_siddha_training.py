from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[1]

metrics_path = (
    REPO
    / "results"
    / "assessment2_siddha"
    / "metrics.csv"
)

output_path = (
    REPO
    / "results"
    / "assessment2_siddha"
    / "siddha_training_curve.png"
)

df = pd.read_csv(metrics_path)

plt.figure(figsize=(8, 5))

plt.plot(
    df["epoch"],
    df["val_macro_f1"] * 100,
    marker="o",
)

best_idx = df["val_loss"].idxmin()
best_epoch = int(df.loc[best_idx, "epoch"])
best_f1 = df.loc[best_idx, "val_macro_f1"] * 100

plt.scatter(
    [best_epoch],
    [best_f1],
    s=80,
    label=f"Selected model: epoch {best_epoch}",
)

plt.xlabel("Epoch")
plt.ylabel("Validation Macro-F1 (%)")
plt.title("TinyHAR Training on SIDDHA")
plt.xticks(df["epoch"])
plt.ylim(0, 50)
plt.grid(alpha=0.25)
plt.legend()

plt.tight_layout()
plt.savefig(output_path, dpi=200)

print("Saved:", output_path)
