import streamlit as st
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
import graphviz

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

    # Bahagi paparan utama untuk parameter ACO
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

    # Fungsi untuk inisialisasi feromon
    def initialize_pheromones():
        pheromones = {key: np.ones(int(bounds[key][1] - bounds[key][0] + 1)) for key in bounds}
        return pheromones

    # Fungsi fitness
    def fitness_cal(solution):
        total_time = (
            solution["Processing_Time_Machine_1"] + solution["Processing_Time_Machine_2"] +
            solution["Setup_Time_Machine_1"] + solution["Setup_Time_Machine_2"]
        )
        due_date = solution["Due_Date"]
        weight = solution["Weight"]
        lateness_penalty = max(0, total_time - due_date) * weight
        return -lateness_penalty if lateness_penalty > 0 else -1  # Avoid zero fitness

    # Fungsi mutasi
    def mutate(solution):
        if random.random() < MUT_RATE:
            key = random.choice(list(bounds.keys()))
            solution[key] = random.randint(*bounds[key])
        return solution

    # Fungsi utama ACO
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

            # Cari fitness terbaik untuk iterasi ini
            best_iteration_fitness = max(fitness_values)
            best_iteration_solution = solutions[fitness_values.index(best_iteration_fitness)]

            # Kemaskini solusi terbaik global jika perlu
            if best_iteration_fitness > best_fitness:
                best_fitness = best_iteration_fitness
                best_solution = best_iteration_solution

            # Kemaskini feromon berdasarkan solusi
            for solution, fitness in zip(solutions, fitness_values):
                for key in pheromones:
                    index = int(solution[key] - bounds[key][0])
                    if fitness != 0:
                        pheromones[key][index] += Q / (-fitness)
                    else:
                        pheromones[key][index] += Q  # Assign base value

            # Penyejatan feromon
            for key in pheromones:
                pheromones[key] *= (1 - EVAPORATION_RATE)

            # Terapkan mutasi
            best_solution = mutate(best_solution)

            # Rekodkan tren fitness untuk carta
            fitness_trends.append(best_fitness)

        return best_solution, fitness_trends

    # Menunjukkan hasil dan workflow
    if st.button("Run ACO Optimization"):
        best_solution, fitness_trends = ant_colony_optimization()

        # Paparan hasil terbaik
        st.subheader("Best Solution")
        st.write(best_solution)

        # Paparan aliran kerja
        st.subheader("Workflow for Ant Colony Optimization")
        st.markdown("""
        **Step 1: Initialize Pheromones**  
        Pheromones are initialized randomly to create a starting point for the optimization process.
        
        **Step 2: Generate Solutions**  
        Ants explore different possible solutions based on pheromone levels and the heuristics.

        **Step 3: Evaluate Solutions**  
        Each solution is evaluated using a fitness function that considers the total time and penalty for lateness.

        **Step 4: Update Pheromones**  
        Pheromone levels are updated based on the fitness of the solutions, rewarding better solutions with stronger pheromones.

        **Step 5: Evaporate Pheromones**  
        Over time, pheromones evaporate to simulate the diminishing influence of previous decisions.

        **Step 6: Apply Mutation**  
        Mutation is applied to introduce random changes to the solutions, encouraging exploration of new possibilities.

        **Step 7: Iterate**  
        The process repeats over multiple iterations, with the goal of finding the best solution that minimizes lateness penalties.

        **Final Step: Output Best Solution**  
        The best solution found after all iterations is presented.
        """)

        # Menambah gambaran visual menggunakan Graphviz
        st.subheader("Graphical Workflow Visualization")
        dot = graphviz.Digraph(format='png')
        dot.node('A', 'Initialize Pheromones')
        dot.node('B', 'Generate Solutions')
        dot.node('C', 'Evaluate Solutions')
        dot.node('D', 'Update Pheromones')
        dot.node('E', 'Evaporate Pheromones')
        dot.node('F', 'Apply Mutation')
        dot.node('G', 'Iterate')
        dot.node('H', 'Output Best Solution')
        
        dot.edges(['AB', 'BC', 'CD', 'DE', 'EF', 'FG', 'GH'])
        st.image(dot.render(), caption="ACO Workflow Diagram")

