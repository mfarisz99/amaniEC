import numpy as np
import matplotlib.pyplot as plt
import random
import streamlit as st

# Parameter global
def initialize_params():
    num_shops = st.slider("Number of Shops", min_value=5, max_value=20, value=8)
    iterations = st.sidebar.slider("Number of Iterations", min_value=10, max_value=100, value=50)
    num_ants = st.sidebar.slider("Number of Ants", min_value=5, max_value=20, value=10)
    alpha = st.sidebar.slider("Alpha (Pheromone Importance)", min_value=0.1, max_value=5.0, value=1.0)
    beta = st.sidebar.slider("Beta (Heuristic Importance)", min_value=0.1, max_value=5.0, value=2.0)
    rho = st.sidebar.slider("Evaporation Rate (Rho)", min_value=0.1, max_value=1.0, value=0.5)
    return num_shops, iterations, num_ants, alpha, beta, rho

# Lokasi kedai (random untuk contoh ini)
def generate_shops(num_shops):
    return {
        f"Shop_{i}": (round(random.uniform(0, 10), 2), round(random.uniform(0, 10), 2)) for i in range(1, num_shops + 1)
    }

# Jarak antara kedai
def calculate_distance(shop1, shop2, shops):
    x1, y1 = shops[shop1]
    x2, y2 = shops[shop2]
    return np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

def create_distance_matrix(shops):
    shop_list = list(shops.keys())
    num_shops = len(shop_list)
    matrix = np.zeros((num_shops, num_shops))
    for i in range(num_shops):
        for j in range(num_shops):
            if i != j:
                matrix[i][j] = calculate_distance(shop_list[i], shop_list[j], shops)
    return matrix

# Fungsi heuristik (1 / jarak)
def heuristic(i, j, distance_matrix):
    return 1 / (distance_matrix[i][j] + 1e-10)

# Fungsi ACO utama
def ant_colony_optimization(shops, distance_matrix, num_shops, iterations, num_ants, alpha, beta, rho):
    pheromone = np.ones((num_shops, num_shops))
    best_distance = float("inf")
    best_path = None
    shop_list = list(shops.keys())

    for _ in range(iterations):
        all_paths = []
        all_distances = []

        for ant in range(num_ants):
            visited = []
            current = random.randint(0, num_shops - 1)
            visited.append(current)

            while len(visited) < num_shops:
                probabilities = []
                for next_shop in range(num_shops):
                    if next_shop not in visited:
                        prob = (pheromone[current][next_shop] ** alpha) * (heuristic(current, next_shop, distance_matrix) ** beta)
                        probabilities.append((next_shop, prob))

                probabilities = sorted(probabilities, key=lambda x: x[1], reverse=True)
                next_shop = probabilities[0][0]
                visited.append(next_shop)
                current = next_shop

            all_paths.append(visited)
            distance = sum(
                distance_matrix[visited[i]][visited[i + 1]] for i in range(len(visited) - 1)
            )
            all_distances.append(distance)

        min_distance = min(all_distances)
        if min_distance < best_distance:
            best_distance = min_distance
            best_path = all_paths[all_distances.index(min_distance)]

        for path in all_paths:
            for i in range(len(path) - 1):
                pheromone[path[i]][path[i + 1]] += 1 / min_distance
        pheromone *= (1 - rho)

    return best_path, best_distance

# Visualisasi hasil
def plot_shops_and_path(shops, best_path):
    plt.figure(figsize=(10, 6))
    for shop, (x, y) in shops.items():
        plt.scatter(x, y, label=shop)
        plt.text(x + 0.1, y + 0.1, shop)

    shop_list = list(shops.keys())
    for i in range(len(best_path) - 1):
        shop1, shop2 = shop_list[best_path[i]], shop_list[best_path[i + 1]]
        x1, y1 = shops[shop1]
        x2, y2 = shops[shop2]
        plt.plot([x1, x2], [y1, y2], 'r-')

    plt.title("Optimized Shop Path with ACO")
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.legend()
    st.pyplot(plt)

# Paparan hasil dalam bentuk jadual
def display_results_in_table(shops, best_path, best_distance):
    shop_list = list(shops.keys())
    path_names = [shop_list[i] for i in best_path]
    distances = [round(calculate_distance(shop_list[best_path[i]], shop_list[best_path[i + 1]], shops), 2) for i in range(len(best_path) - 1)]
    total_distance = sum(distances)

    st.subheader("Machine Results")
    data = {
        "Shop": path_names[:-1],
        "Next Shop": path_names[1:],
        "Distance": distances,
    }
    st.table(data)
    st.write("Total Distance:", total_distance)

# Main function for Streamlit
def main():
    st.title("Ant Colony Optimization for Shop Path Optimization")

    # Sidebar for parameters
    num_shops, iterations, num_ants, alpha, beta, rho = initialize_params()

    # Generate shops and distance matrix
    shops = generate_shops(num_shops)
    distance_matrix = create_distance_matrix(shops)

    # Run ACO
    best_path, best_distance = ant_colony_optimization(
        shops, distance_matrix, num_shops, iterations, num_ants, alpha, beta, rho
    )

    # Display results
    st.subheader("Optimized Path")
    st.write("Best Path:", [list(shops.keys())[i] for i in best_path])
    st.write("Best Distance:", best_distance)

    # Plot results
    plot_shops_and_path(shops, best_path)

    # Display results in a table
    display_results_in_table(shops, best_path, best_distance)

if __name__ == "__main__":
    main()
