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

    # Parameter ACO
    NUM_ANTS = st.sidebar.number_input("Number of Ants", min_value=10, max_value=200, value=50, step=10)
    NUM_ITERATIONS = st.sidebar.number_input("Number of Iterations", min_value=10, max_value=500, value=100, step=10)
    ALPHA = st.sidebar.slider("Pheromone Importance (Alpha)", min_value=0.1, max_value=5.0, value=1.0, step=0.1)
    BETA = st.sidebar.slider("Heuristic Importance (Beta)", min_value=0.1, max_value=5.0, value=2.0, step=0.1)
    EVAPORATION_RATE = st.sidebar.slider("Evaporation Rate", min_value=0.1, max_value=1.0, value=0.5, step=0.1)
    Q = st.sidebar.number_input("Pheromone Deposit Factor (Q)", min_value=10, max_value=500, value=100, step=10)
    MUT_RATE = st.sidebar.slider("Mutation Rate", min_value=0.0, max_value=1.0, value=0.2, step=0.05)

    # Define bounds for optimization
    bounds = {
        "Processing_Time_Machine_1": (data["Processing_Time_Machine_1"].min(), data["Processing_Time_Machine_1"].max()),
        "Processing_Time_Machine_2": (data["Processing_Time_Machine_2"].min(), data["Processing_Time_Machine_2"].max()),
        "Setup_Time_Machine_1": (data["Setup_Time_Machine_1"].min(), data["Setup_Time_Machine_1"].max()),
        "Setup_Time_Machine_2": (data["Setup_Time_Machine_2"].min(), data["Setup_Time_Machine_2"].max()),
    }

    # Initialize pheromones
    def initialize_pheromones():
        pheromones = {key: np.ones(int(bounds[key][1] - bounds[key][0] + 1)) for key in bounds}
        return pheromones

    # Fitness function
    def fitness_cal(solution):
        total_time = (
            solution["Processing_Time_Machine_1"] + solution["Processing_Time_Machine_2"] +
            solution["Setup_Time_Machine_1"] + solution["Setup_Time_Machine_2"]
        )
        due_date = solution["Due_Date"]
        weight = solution["Weight"]
        lateness_penalty = max(0, total_time - due_date) * weight
        return -lateness_penalty if lateness_penalty > 0 else -1  # Avoid zero fitness

    # Mutation function
    def mutate(solution):
        if random.random() < MUT_RATE:
            key = random.choice(list(bounds.keys()))
            solution[key] = random.randint(*bounds[key])
        return solution

    # Main ACO loop
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

            # Find the best fitness of the current iteration
            best_iteration_fitness = max(fitness_values)
            best_iteration_solution = solutions[fitness_values.index(best_iteration_fitness)]

            # Update global best solution if necessary
            if best_iteration_fitness > best_fitness:
                best_fitness = best_iteration_fitness
                best_solution = best_iteration_solution

            # Update pheromones based on solutions
            for solution, fitness in zip(solutions, fitness_values):
                for key in pheromones:
                    index = int(solution[key] - bounds[key][0])
                    if fitness != 0:
                        pheromones[key][index] += Q / (-fitness)
                    else:
                        pheromones[key][index] += Q  # Assign base value

            # Evaporate pheromones
            for key in pheromones:
                pheromones[key] *= (1 - EVAPORATION_RATE)

            # Apply mutation
            best_solution = mutate(best_solution)

            # Log fitness trends for plotting
            fitness_trends.append(best_fitness)

        return best_solution, fitness_trends

    # Run ACO
    if st.button("Run ACO Optimization"):
        best_solution, fitness_trends = ant_colony_optimization()

        # Display the best solution
        st.subheader("Best Solution")
        st.write(best_solution)

        # Plot fitness trends
        st.subheader("Fitness Trends Over Iterations")
        fig, ax = plt.subplots()
        ax.plot(fitness_trends, label="Fitness")
        ax.set_xlabel("Iterations")
        ax.set_ylabel("Fitness")
        ax.set_title("Fitness Trends Over Iterations")
        ax.legend()
        st.pyplot(fig)
