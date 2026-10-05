from pathlib import Path
import os
import sys
import csv
import copy
import random

import numpy as np
import torch
import torch.nn as nn
import yaml

# ---------------------------------------------------------
# Repository setup
# ---------------------------------------------------------
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

# Original code uses ../../configs/... relative to notebooks/model
os.chdir(REPO / "notebooks" / "model")

from experiment import Exp
from dataloaders import data_dict
from models.TinyHAR import TinyHAR_Model


class dotdict(dict):
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


class PaperAlignedExp(Exp):
    """
    Use the released experiment pipeline but construct TinyHAR
    with filter_num=16, which reproduces the 16,184 parameters
    reported for WISDM in the paper.
    """

    def build_model(self):
        model = TinyHAR_Model(
            (1, self.args.f_in,
             self.args.input_length,
             self.args.c_in),
            self.args.num_classes,
            filter_num=16,
            cross_channel_interaction_type=self.args.cross_channel_interaction_type,
            cross_channel_aggregation_type=self.args.cross_channel_aggregation_type,
            temporal_info_interaction_type=self.args.temporal_info_interaction_type,
            temporal_info_aggregation_type=self.args.temporal_info_aggregation_type,
        )

        print("Build paper-aligned TinyHAR model!")
        return model.double()


# ---------------------------------------------------------
# Experiment arguments
# ---------------------------------------------------------
args = dotdict()

args.seed = 1
args.data_name = "wisdm"

args.to_save_path = str(REPO / "run_logs")
args.freq_save_path = str(REPO / "data" / "Freq_data")
args.window_save_path = str(REPO / "data" / "Sliding_window")
args.root_path = str(REPO / "data")

for path in [
    args.to_save_path,
    args.freq_save_path,
    args.window_save_path,
]:
    Path(path).mkdir(parents=True, exist_ok=True)

# Data preprocessing
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

# Training
args.train_epochs = 10

args.learning_rate = 0.001
args.learning_rate_patience = 7
args.learning_rate_factor = 0.1
args.early_stop_patience = 15

args.use_gpu = torch.cuda.is_available()
args.gpu = 0
args.use_multi_gpu = False

args.optimizer = "Adam"
args.criterion = "CrossEntropy"

args.output_attention = False
args.mixup = False

# Wavelets disabled
args.wavelet_filtering = False
args.wavelet_filtering_regularization = False
args.wavelet_filtering_finetuning = False
args.wavelet_filtering_finetuning_percent = 0.5
args.wavelet_filtering_learnable = False
args.wavelet_filtering_layernorm = False

args.regulatization_tradeoff = 0
args.number_wavelet_filtering = 12

# TinyHAR architecture
args.model_type = "tinyhar"

args.cross_channel_interaction_type = "attn"
args.cross_channel_aggregation_type = "FC"
args.temporal_info_interaction_type = "lstm"
args.temporal_info_aggregation_type = "tnaive"

args.filter_scaling_factor = 1

# ---------------------------------------------------------
# WISDM configuration
# ---------------------------------------------------------
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


# ---------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(args.seed)


# ---------------------------------------------------------
# Build paper-aligned TinyHAR
# ---------------------------------------------------------
print("\n========================================")
print(" BUILD PAPER-ALIGNED TINYHAR")
print("========================================")

exp = PaperAlignedExp(args)

print("Expected paper parameters: 16184")
print("Actual parameters:", exp.model_size)

if exp.model_size != 16184:
    raise RuntimeError(
        f"Expected 16184 parameters, got {exp.model_size}"
    )


# ---------------------------------------------------------
# Load WISDM
# ---------------------------------------------------------
print("\n========================================")
print(" LOAD WISDM")
print("========================================")

dataset = data_dict["wisdm"](args)

# Calling this once selects CV fold 1
dataset.update_train_val_test_keys()

print("\nFold 1 test subjects:", dataset.test_keys)
print("Training subject groups:", len(dataset.train_keys))


# ---------------------------------------------------------
# Original loaders
# ---------------------------------------------------------
train_loader = exp._get_data(
    dataset,
    flag="train",
    weighted_sampler=False
)

val_loader = exp._get_data(
    dataset,
    flag="vali",
    weighted_sampler=False
)

test_loader = exp._get_data(
    dataset,
    flag="test",
    weighted_sampler=False
)

print("Train batches:", len(train_loader))
print("Validation batches:", len(val_loader))
print("Test batches:", len(test_loader))


# ---------------------------------------------------------
# Training setup
# ---------------------------------------------------------
criterion = nn.CrossEntropyLoss(
    reduction="mean"
).to(exp.device)

optimizer = torch.optim.Adam(
    exp.model.parameters(),
    lr=args.learning_rate
)

output_dir = REPO / "results" / "assessment2_wisdm_fold1"
output_dir.mkdir(parents=True, exist_ok=True)

csv_path = output_dir / "metrics.csv"
checkpoint_path = output_dir / "best_model.pth"

best_val_loss = float("inf")
best_epoch = None
best_state = None


# ---------------------------------------------------------
# Train fold 1
# ---------------------------------------------------------
print("\n========================================")
print(" TRAIN FOLD 1")
print("========================================")

with csv_path.open("w", newline="") as f:
    writer = csv.writer(f)

    writer.writerow([
        "epoch",
        "train_loss",
        "val_loss",
        "val_accuracy",
        "val_weighted_f1",
        "val_macro_f1",
        "val_micro_f1",
    ])

    for epoch in range(args.train_epochs):

        exp.model.train()
        losses = []

        for batch_x1, batch_x2, batch_y in train_loader:

            batch_x1 = batch_x1.double().to(exp.device)
            batch_y = batch_y.long().to(exp.device)

            optimizer.zero_grad()

            outputs = exp.model(batch_x1)

            loss = criterion(outputs, batch_y)

            loss.backward()
            optimizer.step()

            losses.append(loss.item())

        train_loss = float(np.mean(losses))

        (
            val_loss,
            val_acc,
            val_f_weighted,
            val_f_macro,
            val_f_micro,
        ) = exp.validation(
            exp.model,
            val_loader,
            criterion,
        )

        print(
            f"Epoch {epoch + 1:02d} | "
            f"Train Loss {train_loss:.4f} | "
            f"Val Loss {val_loss:.4f} | "
            f"Val Acc {val_acc:.4f} | "
            f"Val Macro-F1 {val_f_macro:.4f}"
        )

        writer.writerow([
            epoch + 1,
            train_loss,
            float(val_loss),
            val_acc,
            val_f_weighted,
            val_f_macro,
            val_f_micro,
        ])

        f.flush()

        if val_loss < best_val_loss:
            best_val_loss = float(val_loss)
            best_epoch = epoch + 1
            best_state = copy.deepcopy(
                exp.model.state_dict()
            )


# ---------------------------------------------------------
# Restore best validation model
# ---------------------------------------------------------
exp.model.load_state_dict(best_state)

torch.save(
    best_state,
    checkpoint_path,
)


# ---------------------------------------------------------
# Evaluate held-out test subjects
# ---------------------------------------------------------
(
    test_loss,
    test_acc,
    test_f_weighted,
    test_f_macro,
    test_f_micro,
) = exp.validation(
    exp.model,
    test_loader,
    criterion,
)


print("\n========================================")
print(" FOLD 1 FINAL RESULTS")
print("========================================")

print("Best epoch:", best_epoch)
print(f"Best validation loss: {best_val_loss:.4f}")

print(f"Test loss: {test_loss:.4f}")
print(f"Test accuracy: {test_acc:.4f}")
print(f"Test weighted F1: {test_f_weighted:.4f}")
print(f"Test Macro-F1: {test_f_macro:.4f}")
print(f"Test micro F1: {test_f_micro:.4f}")

print()
print("Parameters:", exp.model_size)
print("Metrics saved to:", csv_path)
print("Checkpoint saved to:", checkpoint_path)

print("\n=== FOLD 1 COMPLETE ===")
