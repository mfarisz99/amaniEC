import numpy as np
import matplotlib.pyplot as plt
import random

# Parameter global
num_shops = 8  # Bilangan kedai
iterations = 50  # Iterasi algoritma
num_ants = 10  # Bilangan semut
alpha = 1  # Kepentingan pheromone
beta = 2  # Kepentingan heuristik
rho = 0.5  # Kadar penyejatan pheromone

# Lokasi kedai (random untuk contoh ini)
shops = {
    f"Shop_{i}": (random.uniform(0, 10), random.uniform(0, 10)) for i in range(1, num_shops + 1)
}

# Jarak antara kedai
def calculate_distance(shop1, shop2):
    x1, y1 = shops[shop1]
    x2, y2 = shops[shop2]
    return np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

def create_distance_matrix():
    shop_list = list(shops.keys())
    matrix = np.zeros((num_shops, num_shops))
    for i in range(num_shops):
        for j in range(num_shops):
            if i != j:
                matrix[i][j] = calculate_distance(shop_list[i], shop_list[j])
    return matrix

shop_list = list(shops.keys())
distance_matrix = create_distance_matrix()

# Inisialisasi pheromone
pheromone = np.ones((num_shops, num_shops))

# Fungsi heuristik (1 / jarak)
def heuristic(i, j):
    return 1 / (distance_matrix[i][j] + 1e-10)

# Fungsi ACO utama
def ant_colony_optimization():
    global pheromone
    best_distance = float("inf")
    best_path = None

    for _ in range(iterations):
        all_paths = []
        all_distances = []

        for ant in range(num_ants):
            # Setiap semut memilih laluan
            visited = []
            current = random.randint(0, num_shops - 1)
            visited.append(current)

            while len(visited) < num_shops:
                probabilities = []
                for next_shop in range(num_shops):
                    if next_shop not in visited:
                        prob = (pheromone[current][next_shop] ** alpha) * (heuristic(current, next_shop) ** beta)
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

        # Cari laluan terbaik
        min_distance = min(all_distances)
        if min_distance < best_distance:
            best_distance = min_distance
            best_path = all_paths[all_distances.index(min_distance)]

        # Update pheromone
        for path in all_paths:
            for i in range(len(path) - 1):
                pheromone[path[i]][path[i + 1]] += 1 / min_distance
        pheromone *= (1 - rho)  # Evaporasi

    return best_path, best_distance

# Jalankan ACO
best_path, best_distance = ant_colony_optimization()

# Visualisasi hasil
plt.figure(figsize=(10, 6))
for shop, (x, y) in shops.items():
    plt.scatter(x, y, label=shop)
    plt.text(x + 0.1, y + 0.1, shop)

for i in range(len(best_path) - 1):
    shop1, shop2 = shop_list[best_path[i]], shop_list[best_path[i + 1]]
    x1, y1 = shops[shop1]
    x2, y2 = shops[shop2]
    plt.plot([x1, x2], [y1, y2], 'r-')

plt.title("Optimized Shop Path with ACO")
plt.xlabel("X Coordinate")
plt.ylabel("Y Coordinate")
plt.legend()
plt.show()

print("Best Path:", [shop_list[i] for i in best_path])
print("Best Distance:", best_distance)
