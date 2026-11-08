from ptflops import get_model_complexity_info

from models.TinyHAR import TinyHAR_Model


INPUT_SHAPE = (1, 1, 100, 3)
NUMBER_CLASSES = 6
FILTER_NUM = 16


def main():
    model = TinyHAR_Model(
        input_shape=INPUT_SHAPE,
        number_class=NUMBER_CLASSES,
        filter_num=FILTER_NUM,
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

    print("========================================")
    print(" PAPER-ALIGNED TINYHAR COMPLEXITY")
    print("========================================")
    print("Input shape:", INPUT_SHAPE)
    print("Classes:", NUMBER_CLASSES)
    print("Filters:", FILTER_NUM)
    print("Direct trainable parameters:", params_direct)
    print("ptflops parameters:", int(params_ptflops))
    print("MACs:", int(macs))
    print("MMACs:", macs / 1_000_000)
    print()
    print("Paper-reported WISDM values:")
    print("Parameters: 16184")
    print("FLOPs: 1466764")
    print()
    print(
        "Note: ptflops reports multiply-accumulate operations (MACs). "
        "The paper labels its complexity value as FLOPs, so the raw "
        "ptflops output should be reported separately before deciding "
        "whether the counting conventions are directly comparable."
    )


if __name__ == "__main__":
    main()
