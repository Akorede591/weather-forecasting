# app.py
import streamlit as st
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

# Import the backend pipelines we just tested from model.py
from model import prepare_dataset, run_baseline_pipeline, run_mi_pipeline, run_pso_pipeline

# Configure the Streamlit page layout
st.set_page_config(page_title="Weather Forecast Optimization", layout="wide")

st.title("☀️ Weather Forecast Performance: MI vs PSO on Random Forest")
st.write("Compare standard Random Forest regressions against Mutual Information and Particle Swarm Optimization feature selections.")

# Hardcoded absolute path to your verified dataset
DATASET_PATH = r"C:\Users\USER\Desktop\Weather forecasting2\weather_prediction_dataset.csv"
TARGET_COLUMN = "HEATHROW_temp_mean"

# Check if file exists safely
if not os.path.exists(DATASET_PATH):
    st.error(f"Could not locate the dataset at: {DATASET_PATH}")
else:
    # --- Data Initialization ---
    with st.spinner("Initializing and preprocessing weather data..."):
        X_train, X_test, y_train, y_test, feat_names = prepare_dataset(DATASET_PATH, TARGET_COLUMN)
    
    st.sidebar.success("✅ Dataset Linked & Cached Successfully!")
    st.sidebar.markdown(f"**Target Variable:** `{TARGET_COLUMN}`")
    st.sidebar.markdown(f"**Total Available Features:** {len(feat_names)}")
    st.sidebar.markdown(f"**Training Rows:** {X_train.shape[0]}")
    st.sidebar.markdown(f"**Testing Rows:** {X_test.shape[0]}")

    # Create the 3 tabs exactly as requested
    tab1, tab2, tab3 = st.tabs([
        "📊 Baseline UI (All Features)", 
        "🧮 MI + RF UI", 
        "🐝 PSO + RF UI"
    ])

    # ---------------------------------------------------------
    # TAB 1: BASELINE UI
    # ---------------------------------------------------------
    with tab1:
        st.header("Baseline Machine Learning Model")
        st.info("Trains a standard Random Forest Regressor using all 164 available climate parameters.")
        
        if st.button("🚀 Train Baseline Model", key="btn_base"):
            with st.spinner("Training Random Forest on full feature set..."):
                r2, mae, duration = run_baseline_pipeline(X_train, X_test, y_train, y_test)
                
            st.success("Baseline Execution Complete!")
            col1, col2, col3 = st.columns(3)
            col1.metric("R-Squared (R²)", f"{r2:.4f}")
            col2.metric("Mean Absolute Error (MAE)", f"{mae:.2f}°C")
            col3.metric("Execution Time", f"{duration:.2f}s")

    # ---------------------------------------------------------
    # TAB 2: MI + RF UI
    # ---------------------------------------------------------
    with tab2:
        st.header("Mutual Information Selector + Random Forest")
        st.info("Filters independent features mathematically based on statistical dependency scores.")
        
        # UI controls for hyperparameter selection
        k_feat = st.slider("Select Number of Top Features (K)", min_value=5, max_value=50, value=15, step=5)
        
        if st.button("⚡ Run MI Feature Selection", key="btn_mi"):
            with st.spinner(f"Ranking features and training RF model on top {k_feat} inputs..."):
                r2, mae, duration, chosen_feats = run_mi_pipeline(X_train, X_test, y_train, y_test, feat_names, k=k_feat)
                
            st.success("MI Optimization Complete!")
            col1, col2, col3 = st.columns(3)
            col1.metric("MI + RF R² Score", f"{r2:.4f}")
            col2.metric("MI + RF MAE", f"{mae:.2f}°C")
            col3.metric("Execution Time", f"{duration:.2f}s")
            
            st.write("### 🔑 Selected Features:")
            st.write(chosen_feats)

    # ---------------------------------------------------------
    # TAB 3: PSO + RF UI
    # ---------------------------------------------------------
    with tab3:
        st.header("Particle Swarm Optimization + Random Forest")
        st.info("Uses a population of heuristic particles to dynamically hunt for the optimal multi-feature combination.")
        
        col_p1, col_p2 = st.columns(2)
        with col_p1:
            particles = st.slider("Swarm Size (Num Particles)", min_value=5, max_value=30, value=10, step=5)
        with col_p2:
            iterations = st.slider("Max Search Iterations", min_value=2, max_value=20, value=5, step=1)
            
        if st.button("🐝 Launch Swarm Optimization Loop", key="btn_pso"):
            # Setup an empty placeholder to capture text outputs from the console print statements
            status_container = st.empty()
            status_container.info("Swarm initialized. Watch your terminal console window for live iteration outputs...")
            
            with st.spinner("Particles running fitness evaluations across feature space..."):
                r2, mae, duration, pso_feats, history = run_pso_pipeline(
                    X_train, X_test, y_train, y_test, feat_names, num_particles=particles, max_iter=iterations
                )
                
            status_container.empty()
            st.success("Swarm Convergence Achieved!")
            
            col1, col2, col3 = st.columns(3)
            col1.metric("PSO + RF R² Score", f"{r2:.4f}")
            col2.metric("PSO + RF MAE", f"{mae:.2f}°C")
            col3.metric("Total Swarm Time", f"{duration:.2f}s")
            
            st.write(f"### 🎯 Swarm Optimization Selection ({len(pso_feats)} features found):")
            st.write(pso_feats)
            
            # Plot the Convergence Curve
            st.write("### 📈 Swarm Fitness Convergence Curve")
            fig, ax = plt.subplots(figsize=(8, 3))
            ax.plot(history, marker='o', color='orange', linestyle='-')
            ax.set_title("PSO Feature Selection Convergence")
            ax.set_xlabel("Iteration Index")
            ax.set_ylabel("Best Swarm R² Score")
            ax.grid(True)
            st.pyplot(fig)