import streamlit as st
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt

# Parameter ACO
NUM_ANTS = 50
NUM_ITERATIONS = 100
ALPHA = 1  # Kepentingan pheromone
BETA = 2   # Kepentingan heuristik
EVAPORATION_RATE = 0.5
Q = 100  # Faktor deposit pheromone
MUT_RATE = 0.2  # Kadar mutasi

# Fungsi untuk inisialisasi pheromone
def initialize_pheromones(bounds):
    pheromones = {key: np.ones(int(bounds[key][1] - bounds[key][0] + 1)) for key in bounds}
    return pheromones

# Fungsi kecergasan
def fitness_cal(solution):
    total_time = (
        solution["Processing_Time_Machine_1"] + solution["Processing_Time_Machine_2"] +
        solution["Setup_Time_Machine_1"] + solution["Setup_Time_Machine_2"]
    )
    due_date = solution["Due_Date"]
    weight = solution["Weight"]
    lateness_penalty = max(0, total_time - due_date) * weight
    return -lateness_penalty if lateness_penalty > 0 else -1  # Hindari kecergasan nol

# Fungsi mutasi
def mutate(solution, bounds):
    if random.random() < MUT_RATE:
        key = random.choice(list(bounds.keys()))
        solution[key] = random.randint(*bounds[key])
    return solution

# Fungsi utama ACO
def ant_colony_optimization(data, bounds):
    pheromones = initialize_pheromones(bounds)
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

        # Cari kecergasan terbaik iterasi
        best_iteration_fitness = max(fitness_values)
        best_iteration_solution = solutions[fitness_values.index(best_iteration_fitness)]

        # Kemaskini penyelesaian global terbaik
        if best_iteration_fitness > best_fitness:
            best_fitness = best_iteration_fitness
            best_solution = best_iteration_solution

        # Kemaskini pheromone
        for solution, fitness in zip(solutions, fitness_values):
            for key in pheromones:
                index = int(solution[key] - bounds[key][0])
                if fitness != 0:
                    pheromones[key][index] += Q / (-fitness)
                else:
                    pheromones[key][index] += Q  # Nilai asas

        # Penyejatan pheromone
        for key in pheromones:
            pheromones[key] *= (1 - EVAPORATION_RATE)

        # Terapkan mutasi
        best_solution = mutate(best_solution, bounds)

        # Catat tren kecergasan untuk plot
        fitness_trends.append(best_fitness)

        # Papar kecergasan terbaik iterasi
        st.write(f"Iterasi {iteration + 1}, Kecergasan Terbaik: {best_fitness}")

    return best_solution, fitness_trends

# Antaramuka Streamlit
st.title("Simulasi Ant Colony Optimization")

# Muat naik dataset
uploaded_file = st.file_uploader("Muat naik dataset flowshop scheduling (CSV)", type="csv")
if uploaded_file:
    data = pd.read_csv(uploaded_file)
    st.write("Data yang dimuat naik:")
    st.dataframe(data)

    # Tetapkan julat
    bounds = {
        "Processing_Time_Machine_1": (data["Processing_Time_Machine_1"].min(), data["Processing_Time_Machine_1"].max()),
        "Processing_Time_Machine_2": (data["Processing_Time_Machine_2"].min(), data["Processing_Time_Machine_2"].max()),
        "Setup_Time_Machine_1": (data["Setup_Time_Machine_1"].min(), data["Setup_Time_Machine_1"].max()),
        "Setup_Time_Machine_2": (data["Setup_Time_Machine_2"].min(), data["Setup_Time_Machine_2"].max()),
    }

    # Jalankan ACO
    best_solution, fitness_trends = ant_colony_optimization(data, bounds)

    # Papar hasil terbaik
    st.write("Penyelesaian Terbaik:")
    st.json(best_solution)

    # Plot tren kecergasan
    st.line_chart(fitness_trends)
