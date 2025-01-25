import pandas as pd
import numpy as np
import random
import streamlit as st
import matplotlib.pyplot as plt
import networkx as nx

# Load the dataset
@st.cache
def load_data():
    return pd.read_csv("/content/flowshop_scheduling_dataset.csv")

data = load_data()

# Parameters (Input by user)
NUM_ANTS = st.number_input("Number of Ants", min_value=10, max_value=100, value=50)
NUM_ITERATIONS = st.number_input("Number of Iterations", min_value=10, max_value=200, value=100)
ALPHA = st.slider("Pheromone Importance (Alpha)", min_value=1, max_value=10, value=1)
BETA = st.slider("Heuristic Importance (Beta)", min_value=1, max_value=10, value=2)
EVAPORATION_RATE = st.slider("Pheromone Evaporation Rate", min_value=0.0, max_value=1.0, value=0.5)
Q = st.slider("Pheromone Deposit Factor (Q)", min_value=1, max_value=200, value=100)
MUT_RATE = st.slider("Mutation Rate", min_value=0.0, max_value=1.0, value=0.2)

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

        # Print fitness of best solution in current iteration
        print(f"Iteration {iteration + 1}, Best Fitness: {best_fitness}")

    return best_solution, fitness_trends

# Function for visualizing the best solution as a graph
def visualize_best_solution(best_solution):
    G = nx.DiGraph()  # Directed graph to show process flow

    # Add nodes representing each shop/job
    for i, key in enumerate(best_solution):
        if key not in ['Due_Date', 'Weight']:
            G.add_node(key, label=f"{key}\n{best_solution[key]}")

    # Add edges based on the process order (e.g., job sequence)
    for i in range(len(best_solution)-1):
        G.add_edge(list(best_solution.keys())[i], list(best_solution.keys())[i+1])

    # Set up the plot
    pos = nx.spring_layout(G, seed=42)  # Set position for the nodes
    labels = nx.get_node_attributes(G, 'label')

    # Plot the graph
    plt.figure(figsize=(10, 6))
    nx.draw(G, pos, with_labels=True, node_color="skyblue", node_size=2000, font_size=10, font_weight="bold", arrows=True)
    nx.draw_networkx_labels(G, pos, labels, font_size=12)
    st.pyplot()

# Run ACO and Visualize
if st.button('Run ACO'):
    best_solution, fitness_trends = ant_colony_optimization()

    # Display the best solution
    st.write("Best Solution:")
    st.write(best_solution)

    # Visualize best solution as a graph
    st.write("Visualizing Best Solution as Graph:")
    visualize_best_solution(best_solution)

    # Display fitness trends graph
    st.write("Fitness Trends over Iterations:")
    st.line_chart(fitness_trends)

    # Additional Information like lateness penalty can be displayed
    lateness_penalty = max(0, best_solution["Processing_Time_Machine_1"] + best_solution["Processing_Time_Machine_2"] - best_solution["Due_Date"])
    st.write(f"Lateness Penalty: {lateness_penalty}")
