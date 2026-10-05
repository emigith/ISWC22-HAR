from pathlib import Path
import os
import sys
import torch
import yaml

# ---------------------------------------------------------
# Locate the repository and make original imports available
# ---------------------------------------------------------
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

# The released code expects to run from notebooks/model
# because it uses paths such as ../../configs/model.yaml.
os.chdir(REPO / "notebooks" / "model")

from experiment import Exp
from dataloaders import data_dict


class dotdict(dict):
    """Allow dictionary values to be accessed as args.name."""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


args = dotdict()

# ---------------------------------------------------------
# Reproducible local paths
# ---------------------------------------------------------
args.to_save_path = str(REPO / "run_logs")
args.freq_save_path = str(REPO / "data" / "Freq_data")
args.window_save_path = str(REPO / "data" / "Sliding_window")
args.root_path = str(REPO / "data")

Path(args.to_save_path).mkdir(parents=True, exist_ok=True)
Path(args.freq_save_path).mkdir(parents=True, exist_ok=True)
Path(args.window_save_path).mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------
# Data settings from the authors' notebook
# ---------------------------------------------------------
args.seed = 1
args.data_name = "wisdm"

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

# Required by the base loader but irrelevant for time-only input
args.load_all = False
args.wavelet_function = None

# ---------------------------------------------------------
# Training settings from the authors' notebook
# ---------------------------------------------------------
args.train_epochs = 150
args.learning_rate = 0.001
args.learning_rate_patience = 7
args.learning_rate_factor = 0.1
args.early_stop_patience = 15

args.use_gpu = torch.cuda.is_available()
args.gpu = 0
args.use_multi_gpu = False

args.optimizer = "Adam"
args.criterion = "CrossEntropy"

# ---------------------------------------------------------
# Wavelet settings: disabled
# ---------------------------------------------------------
args.wavelet_filtering = False
args.wavelet_filtering_regularization = False
args.wavelet_filtering_finetuning = False
args.wavelet_filtering_finetuning_percent = 0.5
args.wavelet_filtering_learnable = False
args.wavelet_filtering_layernorm = False

args.regulatization_tradeoff = 0
args.number_wavelet_filtering = 12

# ---------------------------------------------------------
# Read WISDM configuration from the released repo
# ---------------------------------------------------------
with open(REPO / "configs" / "data.yaml", "r") as f:
    data_config = yaml.load(f, Loader=yaml.FullLoader)

config = data_config[args.data_name]

args.root_path = str(Path(args.root_path) / config["filename"])
args.sampling_freq = config["sampling_freq"]
args.num_classes = config["num_classes"]

window_seconds = config["window_seconds"]
args.windowsize = int(window_seconds * args.sampling_freq)
args.input_length = args.windowsize
args.c_in = config["num_channels"]
args.f_in = 1

# ---------------------------------------------------------
# TinyHAR configuration from Train model.ipynb
# ---------------------------------------------------------
args.filter_scaling_factor = 1
args.model_type = "tinyhar"

args.cross_channel_interaction_type = "attn"
args.cross_channel_aggregation_type = "FC"
args.temporal_info_interaction_type = "lstm"
args.temporal_info_aggregation_type = "tnaive"

# ---------------------------------------------------------
# Build model
# ---------------------------------------------------------
print("\n=== BUILDING TINYHAR ===")
exp = Exp(args)

# ---------------------------------------------------------
# Load WISDM using the released loader
# ---------------------------------------------------------
print("\n=== LOADING WISDM ===")
dataset = data_dict[args.data_name](args)

print("\n=== PIPELINE SUCCESS ===")
print("Dataset:", args.data_name)
print("Sampling frequency:", args.sampling_freq, "Hz")
print("Window size:", args.windowsize, "samples")
print("Window duration:", window_seconds, "seconds")
print("Sensor channels:", args.c_in)
print("Classes:", args.num_classes)
print("Raw valid samples:", len(dataset.data_x))
print("Training windows:", len(dataset.train_slidingwindows))
print("Test windows:", len(dataset.test_slidingwindows))
print("Cross-validation folds:", dataset.num_of_cv)
print("TinyHAR parameters:", exp.model_size)
