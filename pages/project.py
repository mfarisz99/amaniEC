import streamlit as st
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt

# Fungsi untuk memuat dataset
@st.cache_data
def load_data(filepath):
    return pd.read_csv(filepath)

# Fungsi untuk menjana penyelesaian (solusi) berdasarkan feromon
def generate_solution(pheromone, num_tasks, num_machines):
    solution = []
    for task in range(num_tasks):
        probabilities = pheromone[task, :]
        total_pheromone = sum(probabilities)
        probabilities = [p / total_pheromone for p in probabilities]  # Normalisasi kepada kebarangkalian
        chosen_machine = np.random.choice(num_machines, p=probabilities)  # Pilih mesin
        solution.append(chosen_machine)
    return solution

# Fungsi untuk mengira kecergasan (fitness)
def calculate_fitness(solution, data):
    processing_time_machine_1 = 0
    processing_time_machine_2 = 0
    for i, machine in enumerate(solution):
        if machine == 0:  # Mesin 1
            processing_time_machine_1 += data["Processing_Time_Machine_1"][i] + data["Setup_Time_Machine_1"][i]
        else:  # Mesin 2
            processing_time_machine_2 += data["Processing_Time_Machine_2"][i] + data["Setup_Time_Machine_2"][i]
    total_time = max(processing_time_machine_1, processing_time_machine_2)
    return total_time, processing_time_machine_1, processing_time_machine_2

# Fungsi untuk mengemas kini feromon
def update_pheromone(pheromone, solutions, fitness_values, EVAPORATION_RATE, Q):
    pheromone *= (1 - EVAPORATION_RATE)
    for solution, fitness_value in zip(solutions, fitness_values):
        for task, machine in enumerate(solution):
            pheromone[task, machine] += Q / fitness_value
    return pheromone

# Fungsi utama untuk ACO
def ant_colony_optimization(data, NUM_ANTS, NUM_ITERATIONS, ALPHA, BETA, EVAPORATION_RATE, Q):
    num_tasks = len(data)
    num_machines = 2
    pheromone = np.ones((num_tasks, num_machines))
    best_solution = None
    best_fitness = np.inf
    fitness_trends = []

    for iteration in range(NUM_ITERATIONS):
        iteration_solutions = []
        iteration_fitness = []

        # Jana penyelesaian oleh setiap semut
        for _ in range(NUM_ANTS):
            solution = generate_solution(pheromone, num_tasks, num_machines)
            fitness_value, proc_time_1, proc_time_2 = calculate_fitness(solution, data)
            iteration_solutions.append(solution)
            iteration_fitness.append(fitness_value)

            # Periksa jika penyelesaian ini terbaik
            if fitness_value < best_fitness:
                best_solution = solution
                best_fitness = fitness_value

        # Kemas kini feromon
        pheromone = update_pheromone(pheromone, iteration_solutions, iteration_fitness, EVAPORATION_RATE, Q)

        # Rekod tren kecergasan
        fitness_trends.append(best_fitness)

    return best_solution, fitness_trends

# Paparan Streamlit
st.title("Flowshop Scheduling Optimization with ACO")

uploaded_file = st.file_uploader("Upload Flowshop Scheduling Dataset", type=["csv"])
if uploaded_file:
    data = load_data(uploaded_file)
    st.write("Dataset Preview:")
    st.dataframe(data.head())

    # Parameter ACO
    st.sidebar.header("ACO Parameters")
    NUM_ANTS = st.sidebar.slider("Number of Ants", 10, 200, 50, step=10)
    NUM_ITERATIONS = st.sidebar.slider("Number of Iterations", 10, 500, 100, step=10)
    EVAPORATION_RATE = st.sidebar.slider("Evaporation Rate", 0.1, 1.0, 0.5, step=0.1)
    Q = st.sidebar.number_input("Pheromone Deposit Factor (Q)", 10, 500, 100, step=10)

    # Jalankan algoritma ACO
    if st.button("Run ACO Optimization"):
        best_solution, fitness_trends = ant_colony_optimization(data, NUM_ANTS, NUM_ITERATIONS, 1.0, 2.0, EVAPORATION_RATE, Q)

        # Papar penyelesaian terbaik
        st.subheader("Best Solution")
        solution_df = pd.DataFrame({
            "Task": [f"Task {i+1}" for i in range(len(best_solution))],
            "Machine Assigned": ["Machine 1" if x == 0 else "Machine 2" for x in best_solution]
        })
        st.table(solution_df)

        # Visualisasi tren kecergasan
        st.subheader("Fitness Trend Over Iterations")
        st.line_chart(fitness_trends)

        # Maklumat masa pemprosesan
        _, proc_time_1, proc_time_2 = calculate_fitness(best_solution, data)
        st.subheader("Processing Times")
        st.write(f"Machine 1: {proc_time_1} units")
        st.write(f"Machine 2: {proc_time_2} units")
