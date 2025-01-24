import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Fungsi untuk memuat dataset
@st.cache_data
def load_data(filepath):
    return pd.read_csv(filepath)

# Fungsi untuk memisahkan tugas kepada dua mesin berdasarkan penyelesaian terbaik
def separate_tasks(best_solution, task_coordinates):
    machine_1_tasks = [i for i, m in enumerate(best_solution) if m == 0]
    machine_2_tasks = [i for i, m in enumerate(best_solution) if m == 1]
    
    machine_1_coords = [task_coordinates[task] for task in machine_1_tasks]
    machine_2_coords = [task_coordinates[task] for task in machine_2_tasks]
    
    return machine_1_coords, machine_2_coords, machine_1_tasks, machine_2_tasks

# Visualisasi laluan untuk penyelesaian terbaik
def plot_best_solution(task_coordinates, machine_1_coords, machine_2_coords, machine_1_tasks, machine_2_tasks):
    fig, ax = plt.subplots(figsize=(8, 6))

    # Plot laluan untuk Machine 1
    x1, y1 = zip(*machine_1_coords)
    ax.plot(x1, y1, '-o', color='red', label='Machine 1')

    # Plot laluan untuk Machine 2
    x2, y2 = zip(*machine_2_coords)
    ax.plot(x2, y2, '-o', color='blue', label='Machine 2')

    # Plot semua tugas dengan label nombor
    for task, (x, y) in task_coordinates.items():
        ax.text(x, y, str(task + 1), fontsize=10, ha='center', va='center', 
                bbox=dict(facecolor='white', edgecolor='black', boxstyle='circle'))

    # Tetapan graf
    ax.set_title("Best Solution Visualization: Task Assignment by Machine")
    ax.set_xlabel("X Coordinate")
    ax.set_ylabel("Y Coordinate")
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.7)

    # Paparkan graf
    st.pyplot(fig)

# Streamlit UI
st.title("Flowshop Scheduling Optimization - Best Solution Visualization")

uploaded_file = st.file_uploader("Upload Flowshop Scheduling Dataset", type=["csv"])

if uploaded_file is not None:
    data = load_data(uploaded_file)
    st.write("Dataset Preview:")
    st.dataframe(data.head())

    # Masukkan penyelesaian terbaik dan koordinat tugas secara manual atau berdasarkan output ACO
    best_solution = st.text_input("Enter Best Solution (e.g., 0,1,0,1,0,1,0):", "0,1,0,1,1,0,1,0,0,1")
    best_solution = [int(x) for x in best_solution.split(',')]

    # Masukkan koordinat tugas secara manual
    st.subheader("Enter Task Coordinates:")
    num_tasks = len(best_solution)
    task_coordinates = {}
    for i in range(num_tasks):
        x = st.number_input(f"Task {i+1} X-coordinate:", min_value=0.0, value=float(i+1))
        y = st.number_input(f"Task {i+1} Y-coordinate:", min_value=0.0, value=float(i+2))
        task_coordinates[i] = (x, y)

    # Pisahkan tugas kepada dua mesin
    machine_1_coords, machine_2_coords, machine_1_tasks, machine_2_tasks = separate_tasks(best_solution, task_coordinates)

    # Visualisasi
    if st.button("Visualize Best Solution"):
        plot_best_solution(task_coordinates, machine_1_coords, machine_2_coords, machine_1_tasks, machine_2_tasks)
