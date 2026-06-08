import pandas as pd
from pathlib import Path
from math import radians, sin, cos, sqrt, atan2
from loader import load_raw
from validator import validate_dataframe

OUTPUT_CLEAN = Path("data/ais_clean.parquet")
OUTPUT_REJECTED = Path("data/ais_rejected.csv")

def haversine_km(lat1, lon1, lat2, lon2) -> float:
    R = 6371.0
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat/2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon/2)**2
    return R * 2 * atan2(sqrt(a), sqrt(1 - a))

def enrich_tracks(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy().sort_values(["mmsi", "second"])
    df["prev_lat"] = df.groupby("mmsi")["lat"].shift(1)
    df["prev_lon"] = df.groupby("mmsi")["lon"].shift(1)
    mask = df["prev_lat"].notna()
    df.loc[mask, "step_km"] = df[mask].apply(
        lambda r: haversine_km(r.prev_lat, r.prev_lon, r.lat, r.lon), axis=1
    )
    df["flag_position_jump"] = df["step_km"] > 500
    df["flag_anchored_moving"] = df["status"].isin([1, 5]) & (df["speed"] > 0.5)
    df["flag_heading_cog_mismatch"] = (
        (df["heading"] != 511) &
        (abs(df["heading"] - df["course"]) > 45) &
        (abs(df["heading"] - df["course"]) < 315)
    )
    df["flag_aivdo_unknown"] = (
        (df["sentence_type"] == "AIVDO") &
        (df["country"] == "Unknown")
    )
    return df.drop(columns=["prev_lat", "prev_lon"])

def run_pipeline():
    print("=" * 50)
    print("AIS SPOOFING DETECTION — INGESTION PIPELINE")
    print("=" * 50)

    raw = load_raw()
    pos_df = raw[raw["msg_type"].isin([1, 3])].copy()
    voy_df = raw[raw["msg_type"] == 5].copy()
    print(f"\nPosition reports : {len(pos_df):,}")
    print(f"Voyage reports   : {len(voy_df):,}")

    validated = validate_dataframe(pos_df)
    valid = validated[validated["is_valid"]].copy()
    rejected = validated[~validated["is_valid"]].copy()

    valid = enrich_tracks(valid)

    print(f"\n--- Validation Results ---")
    print(f"Valid rows        : {len(valid):,}")
    print(f"Rejected rows     : {len(rejected):,}")

    print(f"\n--- Spoofing Flags ---")
    print(f"Fake MMSI         : {valid['flag_fake_mmsi'].sum():,}")
    print(f"Position jumps    : {valid['flag_position_jump'].sum():,}")
    print(f"Anchored+moving   : {valid['flag_anchored_moving'].sum():,}")
    print(f"Heading/COG mis.  : {valid['flag_heading_cog_mismatch'].sum():,}")
    print(f"AIVDO unknown     : {valid['flag_aivdo_unknown'].sum():,}")

    valid.to_parquet(OUTPUT_CLEAN, index=False)
    rejected.to_csv(OUTPUT_REJECTED, index=False)
    print(f"\nSaved clean data  : {OUTPUT_CLEAN}")
    print(f"Saved rejected    : {OUTPUT_REJECTED}")
    print("\nPipeline complete!")
    return valid, rejected, voy_df

if __name__ == "__main__":
    valid, rejected, voyage = run_pipeline()