import streamlit as st
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
import networkx as nx

# Fungsi untuk memuat dataset
@st.cache_data
def load_data(filepath):
    return pd.read_csv(filepath)

# Fungsi untuk menjana penyelesaian (solusi) berdasarkan feromon
def generate_solution(pheromone, num_tasks, num_machines):
    solution = []
    for task in range(num_tasks):
        probabilities = pheromone[task, :]  # Dapatkan nilai feromon bagi tugas
        total_pheromone = sum(probabilities)
        probabilities = [p / total_pheromone for p in probabilities]  # Normalisasi kepada kebarangkalian
        chosen_machine = np.random.choice(num_machines, p=probabilities)  # Pilih mesin berdasarkan kebarangkalian
        solution.append(chosen_machine)
    return solution

# Fungsi untuk mengira kecergasan (fitness) penyelesaian
def calculate_fitness(solution, data):
    total_time = 0
    processing_time_machine_1 = 0
    processing_time_machine_2 = 0
    for i, machine in enumerate(solution):
        if machine == 0:
            processing_time_machine_1 += data["Processing_Time_Machine_1"][i] + data["Setup_Time_Machine_1"][i]
        else:
            processing_time_machine_2 += data["Processing_Time_Machine_2"][i] + data["Setup_Time_Machine_2"][i]
    total_time = processing_time_machine_1 + processing_time_machine_2
    return total_time, processing_time_machine_1, processing_time_machine_2

# Fungsi untuk mengemas kini feromon
def update_pheromone(pheromone, solutions, fitness_values, EVAPORATION_RATE, Q):
    # Penguapan feromon
    pheromone *= (1 - EVAPORATION_RATE)

    # Deposit feromon berdasarkan penyelesaian terbaik
    for solution, fitness_value in zip(solutions, fitness_values):
        for task, machine in enumerate(solution):
            pheromone[task, machine] += Q / fitness_value
    return pheromone

# Fungsi untuk pengoptimuman ACO
def ant_colony_optimization(data, NUM_ANTS, NUM_ITERATIONS, ALPHA, BETA, EVAPORATION_RATE, Q, MUT_RATE):
    num_tasks = len(data)  # Bilangan tugas dalam jadual
    num_machines = 2  # Bilangan mesin dalam masalah Flowshop

    # Inisialisasi feromon
    pheromone = np.ones((num_tasks, num_machines))  # Matrix feromon awal
    best_solution = None
    best_fitness = np.inf
    fitness_trends = []

    for iteration in range(NUM_ITERATIONS):
        iteration_solutions = []
        iteration_fitness = []

        # Generate solutions for each ant
        for ant in range(NUM_ANTS):
            solution = generate_solution(pheromone, num_tasks, num_machines)
            fitness_value, processing_time_machine_1, processing_time_machine_2 = calculate_fitness(solution, data)
            iteration_solutions.append(solution)
            iteration_fitness.append(fitness_value)

            # Update best solution
            if fitness_value < best_fitness:
                best_solution = solution
                best_fitness = fitness_value

        # Update pheromone levels based on fitness values
        pheromone = update_pheromone(pheromone, iteration_solutions, iteration_fitness, EVAPORATION_RATE, Q)

        # Store fitness trends for visualization
        fitness_trends.append(best_fitness)

    return best_solution, fitness_trends, processing_time_machine_1, processing_time_machine_2, pheromone

# Fungsi untuk visualisasi perjalanan ant colony
def visualize_ant_colony(solution, num_tasks, pheromone):
    # Create a graph to visualize the ant colony
    G = nx.Graph()
    
    for task in range(num_tasks):
        G.add_node(task, label=f"Task {task+1}")
    
    # Add edges based on the solution and pheromone values
    for i in range(len(solution) - 1):
        if solution[i] == solution[i+1]:
            # Highlight pheromone strength
            pheromone_strength = pheromone[i, solution[i]]
            G.add_edge(i, i+1, weight=pheromone_strength)
    
    pos = nx.spring_layout(G)
    edge_weights = nx.get_edge_attributes(G, 'weight')

    # Plot the graph
    plt.figure(figsize=(8, 6))
    node_labels = nx.get_node_attributes(G, 'label')
    nx.draw(G, pos, with_labels=True, node_size=500, node_color='lightblue', font_size=10, font_weight='bold', edge_color='gray', width=2)
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_weights, font_size=8)
    plt.title("Ant Colony Path Visualization")
    plt.show()

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

    # Menunjukkan Aliran Kerja (Workflow) dalam bentuk teks dan imej
    if st.button("Run ACO Optimization"):
        best_solution, fitness_trends, processing_time_machine_1, processing_time_machine_2, pheromone = ant_colony_optimization(data, NUM_ANTS, NUM_ITERATIONS, ALPHA, BETA, EVAPORATION_RATE, Q, MUT_RATE)

        # Paparan hasil terbaik dalam bentuk jadual
        st.subheader("Best Solution (Machine Allocation for Each Task)")
        solution_df = pd.DataFrame({
            "Task": [f"Task {i+1}" for i in range(len(best_solution))],
            "Machine Assigned": best_solution
        })
        st.table(solution_df)

        # Paparan masa pemprosesan bagi setiap mesin
        st.subheader("Processing Time for Each Machine")
        processing_time_df = pd.DataFrame({
            "Machine 1 Processing Time": [processing_time_machine_1],
            "Machine 2 Processing Time": [processing_time_machine_2]
        })
        st.table(processing_time_df)

        # Visualisasi perjalanan ant colony
        visualize_ant_colony(best_solution, len(best_solution), pheromone)

        # Plotting Fitness Trends
        st.subheader("Fitness Trend Over Iterations")
        st.line_chart(fitness_trends)
