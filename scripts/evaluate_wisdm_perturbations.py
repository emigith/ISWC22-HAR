from pathlib import Path
import os
import sys
import csv
import random

import numpy as np
import torch
import torch.nn.functional as F
import yaml

from sklearn.metrics import accuracy_score, f1_score

# =========================================================
# Repository setup
# =========================================================
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

# Released repository assumes execution from notebooks/model
os.chdir(REPO / "notebooks" / "model")

from experiment import Exp
from dataloaders import data_dict
from models.TinyHAR import TinyHAR_Model


class dotdict(dict):
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


class PaperAlignedExp(Exp):
    def build_model(self):
        model = TinyHAR_Model(
            (
                1,
                self.args.f_in,
                self.args.input_length,
                self.args.c_in,
            ),
            self.args.num_classes,
            filter_num=16,
            cross_channel_interaction_type=
                self.args.cross_channel_interaction_type,
            cross_channel_aggregation_type=
                self.args.cross_channel_aggregation_type,
            temporal_info_interaction_type=
                self.args.temporal_info_interaction_type,
            temporal_info_aggregation_type=
                self.args.temporal_info_aggregation_type,
        )

        print("Build paper-aligned TinyHAR model!")
        return model.double()


# =========================================================
# Configuration
# =========================================================
args = dotdict()

args.seed = 1
args.data_name = "wisdm"

args.to_save_path = str(REPO / "run_logs")
args.freq_save_path = str(REPO / "data" / "Freq_data")
args.window_save_path = str(REPO / "data" / "Sliding_window")
args.root_path = str(REPO / "data")

args.drop_transition = False
args.datanorm_type = "standardization"

args.batch_size = 256
args.shuffle = True
args.drop_last = False
args.train_vali_quote = 0.90

args.difference = False
args.filtering = False
args.magnitude = False
args.weighted_sampler = False

args.pos_select = None
args.sensor_select = None

args.representation_type = "time"
args.exp_mode = "LOCV"

args.load_all = False
args.wavelet_function = None

args.use_gpu = torch.cuda.is_available()
args.gpu = 0
args.use_multi_gpu = False

args.output_attention = False
args.mixup = False

args.wavelet_filtering = False
args.wavelet_filtering_regularization = False
args.wavelet_filtering_finetuning = False
args.wavelet_filtering_finetuning_percent = 0.5
args.wavelet_filtering_learnable = False
args.wavelet_filtering_layernorm = False

args.regulatization_tradeoff = 0
args.number_wavelet_filtering = 12

args.model_type = "tinyhar"
args.cross_channel_interaction_type = "attn"
args.cross_channel_aggregation_type = "FC"
args.temporal_info_interaction_type = "lstm"
args.temporal_info_aggregation_type = "tnaive"

args.filter_scaling_factor = 1

# Values required by Exp
args.optimizer = "Adam"
args.criterion = "CrossEntropy"
args.learning_rate = 0.001
args.learning_rate_patience = 7
args.learning_rate_factor = 0.1
args.early_stop_patience = 15
args.train_epochs = 10


# =========================================================
# WISDM metadata
# =========================================================
with open(REPO / "configs" / "data.yaml", "r") as f:
    data_config = yaml.load(f, Loader=yaml.FullLoader)

config = data_config["wisdm"]

args.root_path = str(
    Path(args.root_path) / config["filename"]
)

args.sampling_freq = config["sampling_freq"]
args.num_classes = config["num_classes"]

window_seconds = config["window_seconds"]

args.windowsize = int(
    window_seconds * args.sampling_freq
)

args.input_length = args.windowsize
args.c_in = config["num_channels"]
args.f_in = 1


# =========================================================
# Reproducibility
# =========================================================
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)


# =========================================================
# Build model and load Fold 1 checkpoint
# =========================================================
print("\n========================================")
print(" LOAD PAPER-ALIGNED TINYHAR")
print("========================================")

exp = PaperAlignedExp(args)

checkpoint_path = (
    REPO
    / "results"
    / "assessment2_wisdm_fold1"
    / "best_model.pth"
)

if not checkpoint_path.exists():
    raise FileNotFoundError(
        f"Checkpoint not found: {checkpoint_path}"
    )

state = torch.load(
    checkpoint_path,
    map_location=exp.device,
)

exp.model.load_state_dict(state)
exp.model.eval()

print("Parameters:", exp.model_size)
print("Checkpoint:", checkpoint_path)


# =========================================================
# Reconstruct exactly the same Fold 1 test set
# =========================================================
print("\n========================================")
print(" LOAD HELD-OUT FOLD 1")
print("========================================")

dataset = data_dict["wisdm"](args)

# First LOCV split = subjects [1,2,3,4]
dataset.update_train_val_test_keys()

test_loader = exp._get_data(
    dataset,
    flag="test",
    weighted_sampler=False,
)

all_x = []
all_y = []

for batch_x1, batch_x2, batch_y in test_loader:
    all_x.append(batch_x1.double().cpu())
    all_y.append(batch_y.long().cpu())

X = torch.cat(all_x, dim=0)
y = torch.cat(all_y, dim=0)

print("Test subjects:", dataset.test_keys)
print("Test windows:", len(y))
print("Input shape:", tuple(X.shape))
print("Labels preserved:", len(y))


# =========================================================
# Perturbation functions
# =========================================================

def gaussian_noise(x, sigma, seed):
    """
    Add zero-mean Gaussian noise.

    WISDM is already standardized, so sigma is measured
    in standard-deviation units.
    """
    torch.manual_seed(seed)

    noise = torch.randn(
        x.shape,
        dtype=x.dtype,
    )

    return x + sigma * noise


def missing_packets(x, proportion, seed):
    """
    Simulate lost sensor packets.

    A proportion of timestamps is removed simultaneously
    from X, Y and Z and replaced by zero.

    Since data are standardized, zero corresponds to the
    training-data mean.
    """
    torch.manual_seed(seed)

    n, f, t, c = x.shape

    missing = (
        torch.rand(
            (n, 1, t, 1),
            dtype=x.dtype,
        )
        < proportion
    )

    return x.masked_fill(missing, 0.0)


def channel_dropout(x, channel):
    """
    Simulate complete failure of one accelerometer axis.
    """
    out = x.clone()

    out[:, :, :, channel] = 0.0

    return out


def lower_sampling_rate(x, factor):
    """
    Simulate a lower sampling frequency.

    Example:
      factor=2: 20 Hz -> 10 Hz
      factor=4: 20 Hz -> 5 Hz

    The reduced signal is interpolated back to 100 samples
    because TinyHAR expects a fixed 5-second input window.
    """
    n, f, t, c = x.shape

    # B,F,T,C -> B,F,C,T
    series = x.permute(0, 1, 3, 2)

    series = series.reshape(
        n * f * c,
        1,
        t,
    )

    reduced = series[:, :, ::factor]

    restored = F.interpolate(
        reduced,
        size=t,
        mode="linear",
        align_corners=False,
    )

    restored = restored.reshape(
        n,
        f,
        c,
        t,
    )

    return restored.permute(0, 1, 3, 2)


# =========================================================
# Evaluation
# =========================================================
label_names = [
    "Walking",
    "Jogging",
    "Sitting",
    "Standing",
    "Upstairs",
    "Downstairs",
]


def evaluate(condition_x):
    predictions = []

    exp.model.eval()

    with torch.no_grad():

        for start in range(
            0,
            len(condition_x),
            args.batch_size,
        ):
            stop = start + args.batch_size

            batch = (
                condition_x[start:stop]
                .double()
                .to(exp.device)
            )

            outputs = exp.model(batch)

            pred = torch.argmax(
                outputs,
                dim=1,
            )

            predictions.extend(
                pred.cpu().numpy().tolist()
            )

    true = y.numpy()
    pred = np.asarray(predictions)

    accuracy = accuracy_score(
        true,
        pred,
    )

    macro_f1 = f1_score(
        true,
        pred,
        average="macro",
    )

    weighted_f1 = f1_score(
        true,
        pred,
        average="weighted",
    )

    per_class = f1_score(
        true,
        pred,
        average=None,
        labels=list(range(6)),
        zero_division=0,
    )

    return (
        accuracy,
        macro_f1,
        weighted_f1,
        per_class,
    )


# =========================================================
# Construct experimental datasets
# =========================================================
conditions = [
    {
        "name": "clean",
        "family": "clean",
        "severity": "none",
        "seed": 1,
        "function": lambda x: x.clone(),
        "description": "Original held-out WISDM windows",
    },

    {
        "name": "gaussian_005",
        "family": "gaussian_noise",
        "severity": "sigma=0.05",
        "seed": 101,
        "function": lambda x: gaussian_noise(x, 0.05, 101),
        "description": "Gaussian noise, 0.05 standardized units",
    },
    {
        "name": "gaussian_010",
        "family": "gaussian_noise",
        "severity": "sigma=0.10",
        "seed": 102,
        "function": lambda x: gaussian_noise(x, 0.10, 102),
        "description": "Gaussian noise, 0.10 standardized units",
    },
    {
        "name": "gaussian_020",
        "family": "gaussian_noise",
        "severity": "sigma=0.20",
        "seed": 103,
        "function": lambda x: gaussian_noise(x, 0.20, 103),
        "description": "Gaussian noise, 0.20 standardized units",
    },

    {
        "name": "missing_010",
        "family": "missing_packets",
        "severity": "10%",
        "seed": 201,
        "function": lambda x: missing_packets(x, 0.10, 201),
        "description": "10% timestamps missing across all axes",
    },
    {
        "name": "missing_030",
        "family": "missing_packets",
        "severity": "30%",
        "seed": 202,
        "function": lambda x: missing_packets(x, 0.30, 202),
        "description": "30% timestamps missing across all axes",
    },

    {
        "name": "drop_x",
        "family": "channel_dropout",
        "severity": "X axis",
        "seed": 1,
        "function": lambda x: channel_dropout(x, 0),
        "description": "Complete loss of accelerometer X axis",
    },
    {
        "name": "drop_y",
        "family": "channel_dropout",
        "severity": "Y axis",
        "seed": 1,
        "function": lambda x: channel_dropout(x, 1),
        "description": "Complete loss of accelerometer Y axis",
    },
    {
        "name": "drop_z",
        "family": "channel_dropout",
        "severity": "Z axis",
        "seed": 1,
        "function": lambda x: channel_dropout(x, 2),
        "description": "Complete loss of accelerometer Z axis",
    },

    {
        "name": "sampling_10hz",
        "family": "lower_sampling_rate",
        "severity": "10 Hz",
        "seed": 1,
        "function": lambda x: lower_sampling_rate(x, 2),
        "description": "20 Hz reduced to 10 Hz then interpolated",
    },
    {
        "name": "sampling_5hz",
        "family": "lower_sampling_rate",
        "severity": "5 Hz",
        "seed": 1,
        "function": lambda x: lower_sampling_rate(x, 4),
        "description": "20 Hz reduced to 5 Hz then interpolated",
    },
]


# =========================================================
# Output paths
# =========================================================
output_dir = (
    REPO
    / "results"
    / "constructed_wisdm_perturbations"
)

output_dir.mkdir(
    parents=True,
    exist_ok=True,
)

summary_path = output_dir / "summary.csv"
manifest_path = output_dir / "manifest.csv"
class_path = output_dir / "per_class_f1.csv"


# =========================================================
# Run all conditions
# =========================================================
summary_rows = []
manifest_rows = []
class_rows = []

clean_macro_f1 = None

print("\n========================================")
print(" CONSTRUCT + EVALUATE DATASETS")
print("========================================")

for condition in conditions:

    print(
        "\nCondition:",
        condition["name"],
    )

    perturbed_x = condition["function"](X)

    # Labels are never transformed
    labels_changed = 0

    (
        accuracy,
        macro_f1,
        weighted_f1,
        per_class,
    ) = evaluate(perturbed_x)

    if condition["name"] == "clean":
        clean_macro_f1 = macro_f1

    delta_macro_f1 = (
        clean_macro_f1 - macro_f1
        if clean_macro_f1 is not None
        else 0.0
    )

    print(
        f"Accuracy: {accuracy:.4f} | "
        f"Macro-F1: {macro_f1:.4f} | "
        f"Delta: {delta_macro_f1:.4f}"
    )

    summary_rows.append([
        condition["name"],
        condition["family"],
        condition["severity"],
        accuracy,
        weighted_f1,
        macro_f1,
        delta_macro_f1,
    ])

    manifest_rows.append([
        condition["name"],
        condition["family"],
        condition["severity"],
        len(y),
        labels_changed,
        condition["seed"],
        condition["description"],
    ])

    for class_id, class_f1 in enumerate(per_class):
        class_rows.append([
            condition["name"],
            class_id,
            label_names[class_id],
            class_f1,
        ])

    del perturbed_x


# =========================================================
# Save experiment evidence
# =========================================================
with summary_path.open("w", newline="") as f:
    writer = csv.writer(f)

    writer.writerow([
        "condition",
        "family",
        "severity",
        "accuracy",
        "weighted_f1",
        "macro_f1",
        "delta_macro_f1",
    ])

    writer.writerows(summary_rows)


with manifest_path.open("w", newline="") as f:
    writer = csv.writer(f)

    writer.writerow([
        "condition",
        "family",
        "severity",
        "n_windows",
        "labels_changed",
        "seed",
        "description",
    ])

    writer.writerows(manifest_rows)


with class_path.open("w", newline="") as f:
    writer = csv.writer(f)

    writer.writerow([
        "condition",
        "class_id",
        "activity",
        "f1",
    ])

    writer.writerows(class_rows)


print("\n========================================")
print(" CONSTRUCTED DATA EXPERIMENT COMPLETE")
print("========================================")

print("Original test windows:", len(y))
print("Number of constructed conditions:", len(conditions) - 1)
print("Labels changed: 0")

print()
print("Summary:", summary_path)
print("Manifest:", manifest_path)
print("Per-class F1:", class_path)
