import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
import streamlit as st

# Load the dataset
data = pd.read_csv("content/flowshop_scheduling_dataset.csv")

# Streamlit Sidebar Input
st.sidebar.header("Ant colony optimization for flowshop scheduling")
NUM_ANTS = st.sidebar.slider("Number of ants", 1, 100, 50)
NUM_ITERATIONS = st.sidebar.slider("Number of iterations", 1, 200, 100)
ALPHA = st.sidebar.slider("alpha (pheromone weight)", 0.1, 10.0, 1.0)
BETA = st.sidebar.slider("beta (heuristic weight)", 0.1, 10.0, 2.0)
EVAPORATION_RATE = st.sidebar.slider("rho (evaporation rate)", 0.1, 1.0, 0.5)
Q = st.sidebar.number_input("Q (pheromone deposit factor)", 1, 1000, 100)
MUT_RATE = st.sidebar.slider("Mutation rate", 0.0, 1.0, 0.2)

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

# Select next step based on pheromones and heuristic
def select_value(pheromone, heuristic):
    probabilities = (pheromone ** ALPHA) * (heuristic ** BETA)
    probabilities /= probabilities.sum()
    return np.random.choice(len(probabilities), p=probabilities)

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

        # Display progress
        st.text(f"Iteration {iteration + 1}, Best Fitness: {best_fitness}")

    return best_solution, fitness_trends

# Define bounds for optimization
bounds = {
    "Processing_Time_Machine_1": (data["Processing_Time_Machine_1"].min(), data["Processing_Time_Machine_1"].max()),
    "Processing_Time_Machine_2": (data["Processing_Time_Machine_2"].min(), data["Processing_Time_Machine_2"].max()),
    "Setup_Time_Machine_1": (data["Setup_Time_Machine_1"].min(), data["Setup_Time_Machine_1"].max()),
    "Setup_Time_Machine_2": (data["Setup_Time_Machine_2"].min(), data["Setup_Time_Machine_2"].max()),
}

# Run ACO
st.title("Flowshop Scheduling Optimization")
if st.button("Run Optimization"):
    best_solution, fitness_trends = ant_colony_optimization()

    # Display the best solution
    st.subheader("Best Solution")
    for key, value in best_solution.items():
        st.write(f"{key}: {value}")

    # Plot fitness trends
    st.subheader("Fitness Trends")
    st.line_chart(fitness_trends)
