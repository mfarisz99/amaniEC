import networkx as nx
import matplotlib.pyplot as plt

def visualize_ant_colony(best_solution, num_tasks, pheromone):
    G = nx.DiGraph()  # Graff berarah untuk menunjukkan hubungan antara tugas dan mesin
    positions = {}  # Posisi untuk visualisasi

    # Menambah nodes untuk setiap tugas
    for i in range(num_tasks):
        G.add_node(i, label=f'Task {i+1}')
        positions[i] = (i, 0)  # Susun tugas dalam satu garis mendatar (y = 0)

    # Menambah mesin sebagai node tambahan
    for i in range(2):
        G.add_node(f'Machine {i+1}', label=f'Machine {i+1}')
        positions[f'Machine {i+1}'] = (i, 1)  # Susun mesin dalam garis y = 1

    # Menambah edges berdasarkan penyelesaian terbaik dan feromon
    for i in range(num_tasks):
        machine = best_solution[i]
        pheromone_strength = pheromone[i, machine]  # Dapatkan kekuatan feromon untuk mesin
        G.add_edge(i, f'Machine {machine + 1}', weight=pheromone_strength)

    # Visualisasi menggunakan NetworkX dan Matplotlib
    plt.figure(figsize=(10, 8))
    nx.draw(G, pos=positions, with_labels=True, node_size=3000, node_color="skyblue", font_size=10, font_weight="bold", edge_color='gray', width=2)

    # Tambah ketebalan garis berdasarkan feromon
    edge_weights = nx.get_edge_attributes(G, 'weight')
    nx.draw_networkx_edge_labels(G, pos=positions, edge_labels=edge_weights)

    # Menambah sedikit pembezaan untuk membezakan kekuatan feromon
    for u, v, d in G.edges(data=True):
        d['weight'] = d['weight'] * 2  # Memperbesar ketebalan garis mengikut feromon

    plt.title("Ant Colony Path Visualization")
    st.pyplot(plt)
    plt.close()  # Tutup gambar untuk elakkan duplikasi jika streamlit dipanggil berulang kali
