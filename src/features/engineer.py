import pandas as pd
import numpy as np
from pathlib import Path

CLEAN_PATH = Path("data/ais_clean.parquet")

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # --- Binary spoofing flags as features ---
    df["feat_fake_mmsi"] = df["flag_fake_mmsi"].astype(int)
    df["feat_aivdo_unknown"] = df["flag_aivdo_unknown"].astype(int)
    df["feat_anchored_moving"] = df["flag_anchored_moving"].astype(int)
    df["feat_heading_cog_mismatch"] = df["flag_heading_cog_mismatch"].astype(int)
    df["feat_position_jump"] = df["flag_position_jump"].fillna(False).astype(int)

    # --- Kinematic features ---
    df["feat_speed"] = df["speed"].fillna(0)
    df["feat_course"] = df["course"].fillna(0)
    df["feat_heading_valid"] = (df["heading"] != 511).astype(int)
    df["feat_heading_cog_diff"] = abs(df["heading"] - df["course"])
    df["feat_heading_cog_diff"] = df["feat_heading_cog_diff"].apply(
        lambda x: min(x, 360 - x) if pd.notna(x) else 0
    )

    # --- MMSI features ---
    df["feat_mmsi_is_reserved"] = df["mmsi"].isin(
        {"987654321", "123456789", "000000000"}
    ).astype(int)
    df["feat_mmsi_prefix"] = df["mmsi"].str[:3].astype(int)

    # --- Sentence type feature ---
    df["feat_is_aivdo"] = (df["sentence_type"] == "AIVDO").astype(int)

    # --- Step distance ---
    df["feat_step_km"] = df["step_km"].fillna(0)

    # --- Composite spoofing score (sum of all flags) ---
    flag_cols = [c for c in df.columns if c.startswith("feat_") and
                 df[c].max() <= 1 and df[c].min() >= 0]
    df["spoofing_score"] = df[flag_cols].sum(axis=1)

    # --- Label: 1 = spoofed, 0 = normal ---
    df["label"] = (df["flag_fake_mmsi"] | df["flag_aivdo_unknown"]).astype(int)

    return df

if __name__ == "__main__":
    df = pd.read_parquet(CLEAN_PATH)
    featured = build_features(df)

    feat_cols = [c for c in featured.columns if c.startswith("feat_")]
    print(f"Features created : {len(feat_cols)}")
    print(f"Feature columns  : {feat_cols}")
    print(f"\nLabel distribution:")
    print(featured["label"].value_counts())
    print(f"\nSpoofing score distribution:")
    print(featured["spoofing_score"].value_counts().sort_index())

    featured.to_parquet("data/ais_featured.parquet", index=False)
    print("\nSaved: data/ais_featured.parquet")