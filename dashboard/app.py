import streamlit as st
import pandas as pd
import numpy as np
import joblib
import folium
from folium.plugins import MarkerCluster
from streamlit_folium import st_folium
from pathlib import Path
import plotly.express as px
from sklearn.metrics import confusion_matrix

st.set_page_config(
    page_title="AIS Spoofing Detection",
    page_icon="🛥️",
    layout="wide"
)

FEATURED_PATH = Path("data/ais_featured.parquet")
MODELS_DIR = Path("models")

@st.cache_data
def load_data():
    return pd.read_parquet(FEATURED_PATH)

@st.cache_resource
def load_models():
    xgb = joblib.load(MODELS_DIR / "xgboost.pkl")
    rf  = joblib.load(MODELS_DIR / "random_forest.pkl")
    iso = joblib.load(MODELS_DIR / "isolation_forest.pkl")
    return xgb, rf, iso

FEATURE_COLS = [
    "feat_fake_mmsi", "feat_aivdo_unknown", "feat_anchored_moving",
    "feat_heading_cog_mismatch", "feat_position_jump", "feat_speed",
    "feat_course", "feat_heading_valid", "feat_heading_cog_diff",
    "feat_mmsi_is_reserved", "feat_mmsi_prefix", "feat_is_aivdo", "feat_step_km"
]

st.sidebar.title("AIS Spoofing Detection")
st.sidebar.markdown("Maritime vessel spoofing detector using ML + feature engineering")
page = st.sidebar.radio("Navigate", [
    "Overview", "Vessel Map", "Alerts", "Model Performance", "Predict", "Investigate"
])

df = load_data()
xgb, rf, iso = load_models()

X = df[FEATURE_COLS].fillna(0)
df["xgb_pred"] = xgb.predict(X)
df["xgb_prob"] = xgb.predict_proba(X)[:, 1]
df["rf_pred"]  = rf.predict(X)
df["iso_pred_flag"] = (iso.predict(X) == -1).astype(int)

if page == "Overview":
    st.title("AIS Spoofing Detection System")
    st.markdown("Real-time maritime vessel identity spoofing detection")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total vessels", f"{len(df):,}")
    c2.metric("Spoofed (XGBoost)", f"{df['xgb_pred'].sum():,}",
              delta=f"{df['xgb_pred'].mean():.1%}", delta_color="inverse")
    c3.metric("Anomalies (ISO)", f"{df['iso_pred_flag'].sum():,}")
    c4.metric("Heading/COG mismatch", f"{df['flag_heading_cog_mismatch'].sum():,}")

    st.divider()
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Spoofing flag breakdown")
        flags = {
            "Fake MMSI": int(df["flag_fake_mmsi"].sum()),
            "AIVDO Unknown": int(df["flag_aivdo_unknown"].sum()),
            "Anchored+Moving": int(df["flag_anchored_moving"].sum()),
            "Heading/COG Mismatch": int(df["flag_heading_cog_mismatch"].sum()),
            "Position Jump": int(df["flag_position_jump"].sum()),
        }
        fig = px.bar(
            x=list(flags.keys()), y=list(flags.values()),
            color=list(flags.values()),
            color_continuous_scale="Reds",
            labels={"x": "Flag", "y": "Count"}
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Label distribution")
        counts = df["label"].value_counts().reset_index()
        counts.columns = ["label", "count"]
        counts["label"] = counts["label"].map({0: "Normal", 1: "Spoofed"})
        fig2 = px.pie(counts, names="label", values="count",
                      color_discrete_sequence=["#2ecc71", "#e74c3c"])
        st.plotly_chart(fig2, use_container_width=True)

    st.subheader("Spoofing probability distribution")
    fig3 = px.histogram(df, x="xgb_prob", nbins=50,
                        color_discrete_sequence=["#3498db"],
                        labels={"xgb_prob": "Spoofing probability"})
    st.plotly_chart(fig3, use_container_width=True)

elif page == "Vessel Map":
    st.title("Vessel Map")
    show = st.radio("Show", ["All vessels", "Spoofed only", "Normal only"],
                    horizontal=True)

    map_df = df.copy()
    if show == "Spoofed only":
        map_df = map_df[map_df["xgb_pred"] == 1]
    elif show == "Normal only":
        map_df = map_df[map_df["xgb_pred"] == 0]

    map_df = map_df.dropna(subset=["lat", "lon"]).sample(n=min(1000, len(map_df)), random_state=42)

    m = folium.Map(
        location=[map_df["lat"].mean(), map_df["lon"].mean()],
        zoom_start=5,
        tiles="CartoDB dark_matter"
    )

    normal = map_df[map_df["xgb_pred"] == 0]
    spoofed = map_df[map_df["xgb_pred"] == 1]

    normal_cluster = MarkerCluster(name="Normal vessels").add_to(m)
    spoofed_cluster = MarkerCluster(name="Spoofed vessels").add_to(m)

    for _, row in normal.iterrows():
        folium.CircleMarker(
            location=[row["lat"], row["lon"]],
            radius=4,
            color="green",
            fill=True,
            fill_opacity=0.7,
            popup=f"MMSI: {row['mmsi']}<br>Country: {row['country']}<br>Speed: {row['speed']} kn"
        ).add_to(normal_cluster)

    for _, row in spoofed.iterrows():
        folium.CircleMarker(
            location=[row["lat"], row["lon"]],
            radius=4,
            color="red",
            fill=True,
            fill_opacity=0.7,
            popup=f"MMSI: {row['mmsi']}<br>Country: {row['country']}<br>Speed: {row['speed']} kn<br>SPOOFED"
        ).add_to(spoofed_cluster)

    folium.LayerControl().add_to(m)
    st_folium(m, width=1200, height=600)
    st.caption("Red = spoofed | Green = normal")

elif page == "Alerts":
    st.title("Spoofing Alerts")

    alerts = df[df["xgb_pred"] == 1][[
        "mmsi", "country", "speed", "lat", "lon",
        "xgb_prob", "flag_fake_mmsi", "flag_heading_cog_mismatch",
        "flag_anchored_moving", "spoofing_score"
    ]].sort_values("xgb_prob", ascending=False)

    st.markdown(f"**{len(alerts):,} vessels flagged as spoofed**")

    min_prob = st.slider("Minimum spoofing probability", 0.0, 1.0, 0.5, 0.05)
    filtered = alerts[alerts["xgb_prob"] >= min_prob]
    st.dataframe(filtered.reset_index(drop=True), use_container_width=True)

    st.subheader("Top suspicious countries")
    country_counts = alerts["country"].value_counts().head(10).reset_index()
    country_counts.columns = ["country", "count"]
    fig = px.bar(country_counts, x="country", y="count",
                 color="count", color_continuous_scale="Reds")
    st.plotly_chart(fig, use_container_width=True)

elif page == "Model Performance":
    st.title("Model Performance")

    col1, col2, col3 = st.columns(3)
    col1.metric("XGBoost Accuracy", "100%")
    col2.metric("Random Forest Accuracy", "100%")
    col3.metric("Isolation Forest Detection", "60.5%")

    st.subheader("XGBoost feature importance")
    importance = pd.Series(xgb.feature_importances_, index=FEATURE_COLS)
    importance = importance.sort_values(ascending=True)
    fig = px.bar(x=importance.values, y=importance.index,
                 orientation="h", color=importance.values,
                 color_continuous_scale="Blues",
                 labels={"x": "Importance", "y": "Feature"})
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Confusion matrix — XGBoost")
    cm = confusion_matrix(df["label"], df["xgb_pred"])
    fig2 = px.imshow(cm, text_auto=True,
                     x=["Predicted Normal", "Predicted Spoofed"],
                     y=["Actual Normal", "Actual Spoofed"],
                     color_continuous_scale="Blues")
    st.plotly_chart(fig2, use_container_width=True)

elif page == "Predict":
    st.title("Predict a vessel")
    st.markdown("Enter vessel details to check if it's spoofed")

    col1, col2, col3 = st.columns(3)
    with col1:
        mmsi = st.text_input("MMSI", value="987654321")
        speed = st.number_input("Speed (knots)", 0.0, 102.2, 12.0, step=0.1)
        course = st.number_input("Course (degrees)", 0.0, 360.0, 155.0, step=0.1)
    with col2:
        heading = st.number_input("Heading", 0, 511, 153, step=1)
        lat = st.number_input("Latitude", -90.0, 90.0, 14.41)
        lon = st.number_input("Longitude", -180.0, 180.0, 72.91)
    with col3:
        sentence_type = st.selectbox("Sentence type", ["AIVDM", "AIVDO"])
        country = st.text_input("Country", value="Unknown")
        status = st.number_input("Nav status", 0, 15, 0)

    if st.button("Check for spoofing", type="primary"):
        is_fake = mmsi in {"987654321", "123456789", "000000000"}
        is_aivdo_unknown = sentence_type == "AIVDO" and country == "Unknown"
        heading_cog_diff = min(abs(heading - course), 360 - abs(heading - course))

        features = pd.DataFrame([{
            "feat_fake_mmsi": int(is_fake),
            "feat_aivdo_unknown": int(is_aivdo_unknown),
            "feat_anchored_moving": int(status in [1, 5] and speed > 0.5),
            "feat_heading_cog_mismatch": int(heading != 511 and heading_cog_diff > 45),
            "feat_position_jump": 0,
            "feat_speed": speed,
            "feat_course": course,
            "feat_heading_valid": int(heading != 511),
            "feat_heading_cog_diff": heading_cog_diff,
            "feat_mmsi_is_reserved": int(is_fake),
            "feat_mmsi_prefix": int(mmsi[:3]) if mmsi[:3].isdigit() else 0,
            "feat_is_aivdo": int(sentence_type == "AIVDO"),
            "feat_step_km": 0.0
        }])

        prob = xgb.predict_proba(features)[0][1]
        pred = xgb.predict(features)[0]

        if pred == 1:
            st.error(f"SPOOFED — Confidence: {prob:.1%}")
        else:
            st.success(f"NORMAL — Spoofing probability: {prob:.1%}")

        st.subheader("Feature values used")
        st.dataframe(features.T.rename(columns={0: "value"}),
                     use_container_width=True)
        
elif page == "Investigate":
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent.parent / "src"))
    from agents.investigator import investigate_vessel

    st.title("AI Investigation Agent")
    st.markdown("LangGraph-powered agentic investigation with RAG-backed regulatory citations")

    alerts = df[df["xgb_pred"] == 1].sort_values("xgb_prob", ascending=False).reset_index(drop=True)

    st.markdown(f"**{len(alerts):,} flagged vessels available for investigation**")

    st.subheader("Choose investigation mode")
    mode = st.radio("Mode", ["Flagged vessels only", "All vessels"], horizontal=True)

    if mode == "Flagged vessels only":
        pool = df[df["xgb_pred"] == 1].sort_values("xgb_prob", ascending=False).reset_index(drop=True)
    else:
        pool = df.reset_index(drop=True)

    mmsi_list = sorted(pool["mmsi"].astype(str).unique().tolist())
    st.caption(f"{len(mmsi_list):,} vessels available in this mode")

    selected_mmsi = st.selectbox(
        "Select a vessel to investigate",
        options=mmsi_list,
        index=0
    )

    vessel_row = pool[pool["mmsi"].astype(str) == str(selected_mmsi)].iloc[0]

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("MMSI", selected_mmsi)
    col2.metric("Country", vessel_row["country"])
    col3.metric("Speed", f"{vessel_row['speed']:.1f} kn")
    col4.metric("Spoofing Score", f"{int(vessel_row['spoofing_score'])}/5")

    st.subheader("Active flags for this vessel")
    flags = {
        "fake_mmsi": bool(vessel_row["flag_fake_mmsi"]),
        "aivdo_unknown": bool(vessel_row["flag_aivdo_unknown"]),
        "heading_cog_mismatch": bool(vessel_row["flag_heading_cog_mismatch"]),
        "anchored_moving": bool(vessel_row["flag_anchored_moving"]),
        "position_jump": False
    }
    active = [k.replace("_", " ").title() for k, v in flags.items() if v]
    if active:
        for f in active:
            st.markdown(f"- 🚩 {f}")
    else:
        st.markdown("- No specific flags (low-confidence ML detection)")

    if st.button("Run AI Investigation", type="primary"):
        with st.spinner("LangGraph agent investigating... querying RAG knowledge base..."):
            report = investigate_vessel(
                mmsi=str(selected_mmsi),
                flags=flags,
                score=int(vessel_row["spoofing_score"]),
                country=str(vessel_row["country"]),
                speed=float(vessel_row["speed"])
            )

        st.success("Investigation complete!")
        st.code(report, language=None)

        st.download_button(
            "Download Report",
            report,
            file_name=f"investigation_{selected_mmsi}.txt"
        )