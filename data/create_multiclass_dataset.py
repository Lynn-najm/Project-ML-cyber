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

GROUPED_LABEL_MAP = {
    "BENIGN": "BENIGN",

    "DDOS-ICMP_FLOOD": "DDoS",
    "DDOS-UDP_FLOOD": "DDoS",
    "DDOS-TCP_FLOOD": "DDoS",
    "DDOS-PSHACK_FLOOD": "DDoS",
    "DDOS-SYN_FLOOD": "DDoS",
    "DDOS-RSTFINFLOOD": "DDoS",
    "DDOS-SYNONYMOUSIP_FLOOD": "DDoS",
    "DDOS-ICMP_FRAGMENTATION": "DDoS",
    "DDOS-ACK_FRAGMENTATION": "DDoS",
    "DDOS-UDP_FRAGMENTATION": "DDoS",
    "DDOS-HTTP_FLOOD": "DDoS",
    "DDOS-SLOWLORIS": "DDoS",

    "DOS-UDP_FLOOD": "DoS",
    "DOS-TCP_FLOOD": "DoS",
    "DOS-SYN_FLOOD": "DoS",
    "DOS-HTTP_FLOOD": "DoS",

    "MIRAI-GREETH_FLOOD": "Mirai",
    "MIRAI-UDPPLAIN": "Mirai",
    "MIRAI-GREIP_FLOOD": "Mirai",

    "VULNERABILITYSCAN": "Recon",
    "RECON-HOSTDISCOVERY": "Recon",
    "RECON-OSSCAN": "Recon",
    "RECON-PORTSCAN": "Recon",
    "RECON-PINGSWEEP": "Recon",

    "MITM-ARPSPOOFING": "Spoofing",
    "DNS_SPOOFING": "Spoofing",

    "DICTIONARYBRUTEFORCE": "BruteForce",

    "BROWSERHIJACKING": "Web",
    "COMMANDINJECTION": "Web",
    "SQLINJECTION": "Web",
    "XSS": "Web",
    "BACKDOOR_MALWARE": "Web",
    "UPLOADING_ATTACK": "Web",
}

def map_to_grouped_label(label: str) -> str:
    if label not in GROUPED_LABEL_MAP:
        raise ValueError(f"Unmapped label found: {label}")
    return GROUPED_LABEL_MAP[label]

def main() -> None:
    df = load_raw_files(RAW_DIR)
    inspect_basic_info(df)

    df["Grouped_Label"] = df["Label"].apply(map_to_grouped_label)

    print("\n===== GROUPED LABEL DISTRIBUTION =====")
    print(df["Grouped_Label"].value_counts())

    print("\n===== GROUPED LABEL PERCENTAGES =====")
    print(df["Grouped_Label"].value_counts(normalize=True) * 100)

if __name__ == "__main__":
    main()


