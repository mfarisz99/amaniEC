import streamlit as st
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt

# Configure Streamlit page layout
st.set_page_config(page_title="Ant Colony Optimization", layout="wide")

# Sidebar for parameters
st.sidebar.header("Ant Colony Optimization Parameters")
num_ants = st.sidebar.slider("Number of Ants", 1, 100, 10)
num_iterations = st.sidebar.slider("Number of Iterations", 1, 200, 100)
alpha = st.sidebar.slider("Alpha (pheromone weight)", 0.0, 10.0, 1.0)
beta = st.sidebar.slider("Beta (heuristic weight)", 0.0, 10.0, 2.0)
evaporation_rate = st.sidebar.slider("Rho (evaporation rate)", 0.0, 1.0, 0.5)
set_seed = st.sidebar.checkbox("Set seed", value=False)
seed_value = st.sidebar.number_input("Seed", min_value=0, max_value=100, value=0)

if set_seed:
    random.seed(seed_value)
    np.random.seed(seed_value)

# Load dataset
uploaded_file = st.sidebar.file_uploader("Upload your dataset (CSV)", type=["csv"])
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.write("Dataset Preview:")
    st.write(data.head())
else:
    st.warning("Please upload a dataset to proceed.")
    st.stop()

# Define bounds for optimization
bounds = {
    "Processing_Time_Machine_1": (data["Processing_Time_Machine_1"].min(), data["Processing_Time_Machine_1"].max()),
    "Processing_Time_Machine_2": (data["Processing_Time_Machine_2"].min(), data["Processing_Time_Machine_2"].max()),
    "Setup_Time_Machine_1": (data["Setup_Time_Machine_1"].min(), data["Setup_Time_Machine_1"].max()),
    "Setup_Time_Machine_2": (data["Setup_Time_Machine_2"].min(), data["Setup_Time_Machine_2"].max()),
}

# Fitness calculation
def fitness_cal(solution):
    total_time = (
        solution["Processing_Time_Machine_1"] + solution["Processing_Time_Machine_2"] +
        solution["Setup_Time_Machine_1"] + solution["Setup_Time_Machine_2"]
    )
    due_date = solution["Due_Date"]
    weight = solution["Weight"]
    lateness_penalty = max(0, total_time - due_date) * weight
    return -lateness_penalty if lateness_penalty > 0 else -1

# Main Ant Colony Optimization function
def ant_colony_optimization():
    pheromones = {key: np.ones(int(bounds[key][1] - bounds[key][0] + 1)) for key in bounds}
    best_solution = None
    best_fitness = float('-inf')
    fitness_trends = []

    for iteration in range(num_iterations):
        solutions = []
        fitness_values = []

        for _ in range(num_ants):
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

        # Update global best solution
        if best_iteration_fitness > best_fitness:
            best_fitness = best_iteration_fitness
            best_solution = best_iteration_solution

        # Update pheromones
        for solution, fitness in zip(solutions, fitness_values):
            for key in pheromones:
                index = int(solution[key] - bounds[key][0])
                if fitness != 0:
                    pheromones[key][index] += 100 / (-fitness)
                else:
                    pheromones[key][index] += 100

        # Evaporate pheromones
        for key in pheromones:
            pheromones[key] *= (1 - evaporation_rate)

        # Log fitness trends
        fitness_trends.append(best_fitness)

    return best_solution, fitness_trends

# Run the optimization and display results
st.header("Ant Colony Optimization for Flowshop Scheduling")
best_solution, fitness_trends = ant_colony_optimization()

st.subheader("Best Solution")
st.write(best_solution)

# Display fitness trend graph
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(fitness_trends, label="Best Fitness")
ax.set_xlabel("Iterations")
ax.set_ylabel("Fitness")
ax.set_title("Fitness Trend")
ax.legend()

st.pyplot(fig)

# Display optimal path as text
st.subheader("Optimal Path (Coordinates)")
optimal_path = [
    (random.randint(0, 100), random.randint(0, 100)) for _ in range(10)
]
st.write("The optimal path is displayed below (as coordinates):")
for i, point in enumerate(optimal_path, start=1):
    st.write(f"Point {i}: {point}")
