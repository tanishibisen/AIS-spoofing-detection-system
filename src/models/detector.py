import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from xgboost import XGBClassifier
import joblib

FEATURED_PATH = Path("data/ais_featured.parquet")
MODELS_DIR = Path("models")
MODELS_DIR.mkdir(exist_ok=True)

FEATURE_COLS = [
    "feat_fake_mmsi", "feat_aivdo_unknown", "feat_anchored_moving",
    "feat_heading_cog_mismatch", "feat_position_jump", "feat_speed",
    "feat_course", "feat_heading_valid", "feat_heading_cog_diff",
    "feat_mmsi_is_reserved", "feat_mmsi_prefix", "feat_is_aivdo", "feat_step_km"
]

def train_isolation_forest(df: pd.DataFrame):
    print("\n--- Isolation Forest (Unsupervised) ---")
    X = df[FEATURE_COLS].fillna(0)
    iso = IsolationForest(contamination=0.15, random_state=42, n_estimators=100)
    df["iso_pred"] = iso.fit_predict(X)
    df["iso_anomaly"] = (df["iso_pred"] == -1).astype(int)
    detected = df[df["iso_anomaly"] == 1]
    print(f"Anomalies detected     : {len(detected):,}")
    print(f"True spoofed caught    : {detected['label'].sum():,}")
    print(f"Detection rate         : {detected['label'].sum()/df['label'].sum():.1%}")
    joblib.dump(iso, MODELS_DIR / "isolation_forest.pkl")
    print("Saved: models/isolation_forest.pkl")
    return iso

def train_xgboost(df: pd.DataFrame):
    print("\n--- XGBoost Classifier (Supervised) ---")
    X = df[FEATURE_COLS].fillna(0)
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scale = (y_train == 0).sum() / (y_train == 1).sum()
    xgb = XGBClassifier(
        n_estimators=100,
        max_depth=4,
        scale_pos_weight=scale,
        random_state=42,
        eval_metric="logloss",
        verbosity=0
    )
    xgb.fit(X_train, y_train)
    y_pred = xgb.predict(X_test)

    print(f"Test set size          : {len(X_test):,}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["Normal", "Spoofed"]))
    print("Confusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(f"  TN={cm[0,0]}  FP={cm[0,1]}")
    print(f"  FN={cm[1,0]}  TP={cm[1,1]}")

    # Feature importance
    importance = pd.Series(xgb.feature_importances_, index=FEATURE_COLS)
    print("\nTop 5 important features:")
    print(importance.sort_values(ascending=False).head(5).to_string())

    joblib.dump(xgb, MODELS_DIR / "xgboost.pkl")
    print("\nSaved: models/xgboost.pkl")
    return xgb

def train_random_forest(df: pd.DataFrame):
    print("\n--- Random Forest Classifier ---")
    X = df[FEATURE_COLS].fillna(0)
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    rf = RandomForestClassifier(
        n_estimators=100, class_weight="balanced", random_state=42
    )
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)

    print(f"Test set size          : {len(X_test):,}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["Normal", "Spoofed"]))

    joblib.dump(rf, MODELS_DIR / "random_forest.pkl")
    print("Saved: models/random_forest.pkl")
    return rf

def run_all():
    print("=" * 50)
    print("AIS SPOOFING DETECTION — ML MODELS")
    print("=" * 50)

    df = pd.read_parquet(FEATURED_PATH)
    print(f"Loaded {len(df):,} rows with {len(FEATURE_COLS)} features")
    print(f"Spoofed: {df['label'].sum():,} | Normal: {(df['label']==0).sum():,}")

    iso = train_isolation_forest(df)
    xgb = train_xgboost(df)
    rf  = train_random_forest(df)

    print("\n" + "=" * 50)
    print("All models trained and saved to models/")
    print("=" * 50)

if __name__ == "__main__":
    run_all()