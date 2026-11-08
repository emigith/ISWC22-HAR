from ptflops import get_model_complexity_info

from models.TinyHAR import TinyHAR_Model


INPUT_SHAPE = (1, 1, 100, 3)
NUMBER_CLASSES = 6
FILTER_COUNTS = [16, 20]


def measure(filter_num):
    model = TinyHAR_Model(
        input_shape=INPUT_SHAPE,
        number_class=NUMBER_CLASSES,
        filter_num=filter_num,
        cross_channel_interaction_type="attn",
        cross_channel_aggregation_type="FC",
        temporal_info_interaction_type="lstm",
        temporal_info_aggregation_type="tnaive",
    )

    params_direct = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    macs, params_ptflops = get_model_complexity_info(
        model,
        (1, 100, 3),
        as_strings=False,
        print_per_layer_stat=False,
        verbose=True,
    )

    return {
        "filters": filter_num,
        "params_direct": int(params_direct),
        "params_ptflops": int(params_ptflops),
        "macs": int(macs),
        "mmacs": macs / 1_000_000,
    }


def main():
    print("========================================")
    print(" TINYHAR WISDM COMPLEXITY COMPARISON")
    print("========================================")
    print("Input shape:", INPUT_SHAPE)
    print("Classes:", NUMBER_CLASSES)
    print()

    for filter_num in FILTER_COUNTS:
        result = measure(filter_num)
        print("----------------------------------------")
        print("Filters:", result["filters"])
        print("Direct trainable parameters:", result["params_direct"])
        print("ptflops parameters:", result["params_ptflops"])
        print("MACs:", result["macs"])
        print("MMACs:", result["mmacs"])
        print()

    print("Paper-reported WISDM values:")
    print("Parameters: 16184")
    print("Reported FLOPs: 1466764")
    print()
    print(
        "Interpretation note: the authors' released notebooks use "
        "ptflops and print its raw 'macs' output as computational "
        "complexity. Therefore the raw MAC counts above should be "
        "treated as the primary reproduced measurements. A separate "
        "1 MAC ~= 2 FLOPs convention may be mentioned only as an "
        "alternative counting convention, not as an exact reproduction."
    )


if __name__ == "__main__":
    main()
