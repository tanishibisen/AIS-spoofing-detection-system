import pandas as pd
from pathlib import Path

DATA_PATH = Path("data/ais_all_message.csv")

def load_raw(path: Path = DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(
        path,
        dtype={"mmsi": str},
        low_memory=False
    )
    df.columns = df.columns.str.strip().str.lower()
    df["mmsi"] = df["mmsi"].astype(str).str.zfill(9)
    print(f"Loaded {len(df):,} rows, {df.shape[1]} columns")
    print(f"Message types found: {sorted(df['msg_type'].unique())}")
    print(f"Columns: {df.columns.tolist()}")
    return df

if __name__ == "__main__":
    df = load_raw()
    print(df.head())