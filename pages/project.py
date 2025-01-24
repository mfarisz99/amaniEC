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

# Generate random points as an example dataset
num_points = st.sidebar.slider("Number of Points", 5, 20, 10)
coordinates = np.random.rand(num_points, 2) * 100  # Random (x, y) coordinates
points = {i: tuple(coordinates[i]) for i in range(num_points)}

st.sidebar.write("Generated Points:")
st.sidebar.write(points)

# ACO initialization
pheromone = np.ones((num_points, num_points))
best_path = None
best_distance = float("inf")

def distance(point1, point2):
    return np.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)

def calculate_total_distance(path):
    return sum(distance(points[path[i]], points[path[i + 1]]) for i in range(len(path) - 1))

# Ant Colony Optimization function
def ant_colony_optimization():
    global pheromone, best_path, best_distance
    for iteration in range(num_iterations):
        all_paths = []
        all_distances = []
        
        for ant in range(num_ants):
            path = [random.randint(0, num_points - 1)]  # Start point
            for _ in range(num_points - 1):
                current = path[-1]
                probabilities = []
                for next_point in range(num_points):
                    if next_point not in path:
                        prob = (
                            pheromone[current][next_point] ** alpha
                            / (distance(points[current], points[next_point]) ** beta)
                        )
                        probabilities.append((next_point, prob))
                probabilities.sort(key=lambda x: x[1], reverse=True)
                path.append(probabilities[0][0])  # Add the next point with highest probability
            path.append(path[0])  # Return to start point
            total_distance = calculate_total_distance(path)
            all_paths.append(path)
            all_distances.append(total_distance)

        # Update the best path
        min_distance = min(all_distances)
        if min_distance < best_distance:
            best_distance = min_distance
            best_path = all_paths[all_distances.index(min_distance)]

        # Update pheromones
        pheromone *= (1 - evaporation_rate)
        for path, total_distance in zip(all_paths, all_distances):
            for i in range(len(path) - 1):
                pheromone[path[i]][path[i + 1]] += 1 / total_distance

    return best_path, best_distance

# Run the optimization
st.header("Ant Colony Optimization for Optimal Path")
best_path, best_distance = ant_colony_optimization()

# Display best path and its total distance
st.subheader("Optimal Path")
optimal_coordinates = [points[node] for node in best_path]
st.write(f"The optimal path is: {best_path}")
st.write(f"Coordinates for the optimal path: {optimal_coordinates}")
st.write(f"Best distance for the optimal path: {best_distance:.2f} meters")

# Visualize the optimal path
fig, ax = plt.subplots(figsize=(10, 5))
for i, coord in points.items():
    ax.scatter(*coord, label=f"Point {i}")
for i in range(len(best_path) - 1):
    start, end = best_path[i], best_path[i + 1]
    ax.plot(
        [points[start][0], points[end][0]],
        [points[start][1], points[end][1]],
        "r-"
    )
ax.set_title("Optimal Path Visualization")
ax.legend()
st.pyplot(fig)
