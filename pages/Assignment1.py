import streamlit as st
import random
import csv
import pandas as pd

# Function to read the CSV file and convert it to the desired format
def read_csv_to_dict(file_path):
    program_ratings = {}
    with open(file_path, mode='r', newline='') as file:
        reader = csv.reader(file)
        # Skip the header
        header = next(reader)
        for row in reader:
            program = row[0]
            ratings = [float(x) for x in row[1:]]
            program_ratings[program] = ratings
    return program_ratings

# Path to the CSV file
file_path = 'program_ratings.csv'

# Get the data in the required format
program_ratings_dict = read_csv_to_dict(file_path)

# Fitness function
def fitness_function(schedule):
    total_rating = 0
    for time_slot, program in enumerate(schedule):
        total_rating += ratings[program][time_slot]
    return total_rating

# Crossover
def crossover(schedule1, schedule2):
    crossover_point = random.randint(1, len(schedule1) - 2)
    child1 = schedule1[:crossover_point] + schedule2[crossover_point:]
    child2 = schedule2[:crossover_point] + schedule1[crossover_point:]
    return child1, child2

# Mutation
def mutate(schedule):
    mutation_point = random.randint(0, len(schedule) - 1)
    new_program = random.choice(all_programs)
    schedule[mutation_point] = new_program
    return schedule

# Genetic algorithm
def genetic_algorithm(initial_schedule, generations, population_size, crossover_rate, mutation_rate, elitism_size):
    population = [initial_schedule]
    for _ in range(population_size - 1):
        random_schedule = initial_schedule.copy()
        random.shuffle(random_schedule)
        population.append(random_schedule)

    for generation in range(generations):
        new_population = []

        # Elitism
        population.sort(key=lambda schedule: fitness_function(schedule), reverse=True)
        new_population.extend(population[:elitism_size])

        while len(new_population) < population_size:
            parent1, parent2 = random.choices(population, k=2)
            if random.random() < crossover_rate:
                child1, child2 = crossover(parent1, parent2)
            else:
                child1, child2 = parent1.copy(), parent2.copy()

            if random.random() < mutation_rate:
                child1 = mutate(child1)
            if random.random() < mutation_rate:
                child2 = mutate(child2)

            new_population.extend([child1, child2])

        population = new_population

    return population[0]

# Streamlit UI
st.title("Program Scheduling Optimization")
st.sidebar.header("Genetic Algorithm Parameters")

# Input parameters
GEN = 100
POP = 50
crossover_rate = float(input("Crossover Rate (0.0 to 0.95, default 0.8): ") or 0.8)
mutation_rate = float(input("Mutation Rate (0.01 to 0.05, default 0.2): ") or 0.2)
EL_S = 2

if st.sidebar.button("Run Genetic Algorithm"):
    # Create initial schedule
    initial_schedule = all_programs[:len(all_time_slots)]
    random.shuffle(initial_schedule)

    # Run the algorithm
    best_schedule = genetic_algorithm(initial_schedule, generations, population_size, crossover_rate, mutation_rate, elitism_size)

    # Display results in a table format
    st.subheader("Final Optimal Schedule")
    schedule_data = {
        "Time Slot": [f"{hour}:00" for hour in all_time_slots],
        "Program": best_schedule
    }
    df = pd.DataFrame(schedule_data)
    st.table(df)

    # Display total ratings
    total_rating = fitness_function(best_schedule)
    st.write(f"**Total Ratings: {total_rating:.2f}**")
