import streamlit as st
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt

# Fungsi untuk memuat dataset
@st.cache_data
def load_data(filepath):
    return pd.read_csv(filepath)

# Memuat dataset
st.title("Flowshop Scheduling Optimization with ACO")

uploaded_file = st.file_uploader("Upload Flowshop Scheduling Dataset", type=["csv"])
if uploaded_file is not None:
    data = load_data(uploaded_file)
    st.write("Dataset Preview:")
    st.dataframe(data.head())

    # Parameter ACO (pindahkan ke paparan utama)
    col1, col2 = st.columns(2)

    with col1:
        NUM_ANTS = st.number_input("Number of Ants", min_value=10, max_value=200, value=50, step=10)
        NUM_ITERATIONS = st.number_input("Number of Iterations", min_value=10, max_value=500, value=100, step=10)
        ALPHA = st.slider("Pheromone Importance (Alpha)", min_value=0.1, max_value=5.0, value=1.0, step=0.1)
        BETA = st.slider("Heuristic Importance (Beta)", min_value=0.1, max_value=5.0, value=2.0, step=0.1)

    with col2:
        EVAPORATION_RATE = st.slider("Evaporation Rate", min_value=0.1, max_value=1.0, value=0.5, step=0.1)
        Q = st.number_input("Pheromone Deposit Factor (Q)", min_value=10, max_value=500, value=100, step=10)
        MUT_RATE = st.slider("Mutation Rate", min_value=0.0, max_value=1.0, value=0.2, step=0.05)

    # Define bounds for optimization
    bounds = {
        "Processing_Time_Machine_1": (data["Processing_Time_Machine_1"].min(), data["Processing_Time_Machine_1"].max()),
        "Processing_Time_Machine_2": (data["Processing_Time_Machine_2"].min(), data["Processing_Time_Machine_2"].max()),
        "Setup_Time_Machine_1": (data["Setup_Time_Machine_1"].min(), data["Setup_Time_Machine_1"].max()),
        "Setup_Time_Machine_2": (data["Setup_Time_Machine_2"].min(), data["Setup_Time_Machine_2"].max()),
    }

    # Fungsi-fungsi lain masih sama...

    # Menunjukkan Aliran Kerja (Workflow) dalam bentuk teks dan imej
    if st.button("Run ACO Optimization"):
        best_solution, fitness_trends = ant_colony_optimization()

        # Paparan hasil terbaik
        st.subheader("Best Solution")
        st.write(best_solution)
