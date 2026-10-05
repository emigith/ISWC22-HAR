from pathlib import Path

SOURCE = Path(
    "data/WISDM_Dataset/WISDM_ar_v1.1/WISDM_ar_v1.1_raw.txt"
)

OUTPUT = Path(
    "data/WISDM_Dataset/WISDM_ar_v1.1_raw.txt"
)

VALID_ACTIVITIES = {
    "Walking",
    "Jogging",
    "Sitting",
    "Standing",
    "Upstairs",
    "Downstairs",
}


def valid_record(fields):
    if len(fields) != 6:
        return False

    subject, activity, timestamp, x, y, z = fields

    if activity not in VALID_ACTIVITIES:
        return False

    try:
        int(subject)
        int(timestamp)
        float(x)
        float(y)
        float(z)
    except ValueError:
        return False

    return True


def main():
    if not SOURCE.exists():
        raise FileNotFoundError(f"Source file not found: {SOURCE}")

    raw_text = SOURCE.read_text(errors="replace")

    valid_records = []
    trailing_commas_fixed = 0
    malformed_records = 0

    # First split using the dataset's normal ";" record separator.
    for chunk in raw_text.split(";"):

        # Some corrupted chunks contain more than one physical line.
        # Treat each non-empty line as its own candidate record.
        for line in chunk.splitlines():

            record = line.strip()

            if not record:
                continue

            if record.endswith(","):
                record = record[:-1]
                trailing_commas_fixed += 1

            fields = [field.strip() for field in record.split(",")]

            if valid_record(fields):
                valid_records.append(",".join(fields) + ";")
            else:
                malformed_records += 1

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT.open("w") as f:
        for record in valid_records:
            f.write(record + "\n")

    print("=== WISDM PREPARATION COMPLETE ===")
    print(f"Source: {SOURCE}")
    print(f"Output: {OUTPUT}")
    print(f"Valid records: {len(valid_records)}")
    print(f"Trailing commas fixed: {trailing_commas_fixed}")
    print(f"Malformed records dropped: {malformed_records}")


if __name__ == "__main__":
    main()
