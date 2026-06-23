# AIS Spoofing Detection System

An AI-powered system for detecting maritime vessel identity spoofing using ML models and a LangGraph-powered AI Investigation Agent.

## Core Features
- **ML Detection**: Uses XGBoost, Random Forest, and Isolation Forest to identify anomalies (e.g., fake MMSIs, position jumps).
- **Streamlit Dashboard**: Provides real-time stats, interactive vessel mapping, and automated alerts.
- **AI Investigator**: A LangGraph agent with RAG capabilities for automated investigation and regulatory cross-referencing.

## Quick Start

1. **Install dependencies:**
   ```bash
   pip install streamlit pandas numpy scikit-learn xgboost folium streamlit-folium plotly joblib langchain chromadb
   ```

2. **Train Models:**
   ```bash
   python src/models/detector.py
   ```

3. **Run Dashboard:**
   ```bash
   streamlit run dashboard/app.py
   ```
