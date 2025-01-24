import streamlit as st
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt

# Fungsi untuk memuat dataset
@st.cache
def load_data(filepath):
    data = pd.read_csv(filepath)
    return data

# Load dataset
data = load_data('/content/flowshop_scheduling_dataset.csv')

# Paparkan data
st.title("Ant Colony Optimization for Flow Shop Scheduling")
st.write("### Dataset Preview")
st.dataframe(data.head())

# Parameter
NUM_ANTS = st.sidebar.slider("Number of Ants", min_value=10, max_value=100, value=50, step=10)
NUM_ITERATIONS = st.sidebar.slider("Number of Iterations", min_value=10, max_value=200, value=100, step=10)
ALPHA = st.sidebar.slider("Pheromone Importance (Alpha)", min_value=0.1, max_value=3.0, value=1.0, step=0.1)
BETA = st.sidebar.slider("Heuristic Importance (Beta)", min_value=0.1, max_value=3.0, value=2.0, step=0.1)
EVAPORATION_RATE = st.sidebar.slider("Evaporation Rate", min_value=0.1, max_value=1.0, value=0.5, step=0.1)
Q = st.sidebar.slider("Pheromone Deposit Factor (Q)", min_value=10, max_value=200, value=100, step=10)
MUT_RATE = st.sidebar.slider("Mutation Rate", min_value=0.0, max_value=1.0, value=0.2, step=0.05)

# Tentukan sempadan untuk pengoptimuman
bounds = {
    "Processing_Time_Machine_1": (data["Processing_Time_Machine_1"].min(), data["Processing_Time_Machine_1"].max()),
    "Processing_Time_Machine_2": (data["Processing_Time_Machine_2"].min(), data["Processing_Time_Machine_2"].max()),
    "Setup_Time_Machine_1": (data["Setup_Time_Machine_1"].min(), data["Setup_Time_Machine_1"].max()),
    "Setup_Time_Machine_2": (data["Setup_Time_Machine_2"].min(), data["Setup_Time_Machine_2"].max()),
}

# Inisialisasi pheromone
def initialize_pheromones():
    pheromones = {key: np.ones(int(bounds[key][1] - bounds[key][0] + 1)) for key in bounds}
    return pheromones

# Fungsi Fitness
def fitness_cal(solution):
    total_time = (
        solution["Processing_Time_Machine_1"] + solution["Processing_Time_Machine_2"] +
        solution["Setup_Time_Machine_1"] + solution["Setup_Time_Machine_2"]
    )
    due_date = solution["Due_Date"]
    weight = solution["Weight"]
    lateness_penalty = max(0, total_time - due_date) * weight
    return -lateness_penalty if lateness_penalty > 0 else -1  # Elakkan fitness kosong

# Fungsi Pemilihan Berdasarkan Pheromone dan Heuristic
def select_value(pheromone, heuristic):
    probabilities = (pheromone ** ALPHA) * (heuristic ** BETA)
    probabilities /= probabilities.sum()
    return np.random.choice(len(probabilities), p=probabilities)

# Fungsi Mutasi
def mutate(solution):
    if random.random() < MUT_RATE:
        key = random.choice(list(bounds.keys()))
        solution[key] = random.randint(*bounds[key])
    return solution

# Algoritma ACO
@st.cache_data
def ant_colony_optimization():
    pheromones = initialize_pheromones()
    best_solution = None
    best_fitness = float('-inf')
    fitness_trends = []

    for iteration in range(NUM_ITERATIONS):
        solutions = []
        fitness_values = []

        for _ in range(NUM_ANTS):
            solution = {
                key: random.randint(*bounds[key]) for key in bounds
            }
            solution["Due_Date"] = random.choice(data["Due_Date"].values)
            solution["Weight"] = random.choice(data["Weight"].values)
            fitness = fitness_cal(solution)
            solutions.append(solution)
            fitness_values.append(fitness)

        # Cari penyelesaian terbaik dalam iterasi ini
        best_iteration_fitness = max(fitness_values)
        best_iteration_solution = solutions[fitness_values.index(best_iteration_fitness)]

        # Kemas kini penyelesaian terbaik global jika perlu
        if best_iteration_fitness > best_fitness:
            best_fitness = best_iteration_fitness
            best_solution = best_iteration_solution

        # Kemas kini pheromone berdasarkan penyelesaian
        for solution, fitness in zip(solutions, fitness_values):
            for key in pheromones:
                index = int(solution[key] - bounds[key][0])
                if fitness != 0:
                    pheromones[key][index] += Q / (-fitness)
                else:
                    pheromones[key][index] += Q

        # Sejat pheromone
        for key in pheromones:
            pheromones[key] *= (1 - EVAPORATION_RATE)

        # Terapkan mutasi
        best_solution = mutate(best_solution)

        # Rekod trend fitness untuk visualisasi
        fitness_trends.append(best_fitness)

    return best_solution, fitness_trends

# Jalankan ACO
if st.button("Run ACO Optimization"):
    with st.spinner("Running Ant Colony Optimization..."):
        best_solution, fitness_trends = ant_colony_optimization()

    # Paparkan hasil
    st.write("### Best Solution")
    for key, value in best_solution.items():
        st.write(f"{key}: {value}")

    st.write(f"### Best Fitness: {max(fitness_trends)}")

    # Paparkan graf
    st.write("### Fitness Trends Over Iterations")
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(fitness_trends, label="Fitness")
    ax.set_xlabel("Iterations")
    ax.set_ylabel("Fitness")
    ax.set_title("Fitness Trends Over Iterations")
    ax.legend()
    st.pyplot(fig)
