from pathlib import Path
import pandas as pd


RAW_DIR = Path("dataset/raw")


def load_raw_files(raw_dir: Path) -> pd.DataFrame:
    csv_files = sorted(raw_dir.glob("*.csv"))

    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {raw_dir}")

    print("Raw files found:")
    for file in csv_files:
        print(f" - {file.name}")

    dataframes = []
    for file in csv_files:
        print(f"\nLoading {file.name}...")
        df = pd.read_csv(file)
        print(f"Shape of {file.name}: {df.shape}")
        dataframes.append(df)

    merged_df = pd.concat(dataframes, ignore_index=True)
    return merged_df


def inspect_basic_info(df: pd.DataFrame) -> None:
    print("\n===== BASIC DATASET INFO =====")
    print(f"Total shape: {df.shape}")

    print("\nColumns:")
    print(list(df.columns))

    if "Label" not in df.columns:
        raise ValueError("Target column 'Label' was not found in the dataset.")

    print("\nLabel distribution:")
    print(df["Label"].value_counts(dropna=False))

    print("\nLabel percentages:")
    print(df["Label"].value_counts(normalize=True, dropna=False) * 100)


def main() -> None:
    df = load_raw_files(RAW_DIR)
    inspect_basic_info(df)


if __name__ == "__main__":
    main()