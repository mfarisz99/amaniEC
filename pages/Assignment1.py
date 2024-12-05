import streamlit as st
import random
import csv
import pandas as pd

# Function to read the CSV file and convert it to the desired format
def read_csv_to_dict(file_path):
    program_ratings = {}
    time_slots = []  # To store time slots
    with open(file_path, mode='r', newline='') as file:
        reader = csv.reader(file)
        # Read header (time slots)
        header = next(reader)
        time_slots = header[1:]  # Skip the first column (program names)
        
        for row in reader:
            program = row[0]
            ratings = [float(x) for x in row[1:]]
            program_ratings[program] = ratings
    return program_ratings, time_slots

# Path to the CSV file
file_path = 'content/program_ratings.csv'

# Read CSV and extract program ratings and time slots
program_ratings_dict, all_time_slots = read_csv_to_dict(file_path)
all_programs = list(program_ratings_dict.keys())

# Ensure there are enough programs
if len(all_programs) < len(all_time_slots):
    all_programs = all_programs * (len(all_time_slots) // len(all_programs) + 1)
    all_programs = all_programs[:len(all_time_slots)]

# Fitness function
def fitness_function(schedule):
    total_rating = 0
    for time_slot, program in enumerate(schedule):
        total_rating += program_ratings_dict[program][time_slot]
    return total_rating

# Crossover
def crossover(schedule1, schedule2):
    crossover_point = random.randint(1, len(schedule1) - 2)
    child1 = schedule1[:crossover_point] + schedule2[crossover_point:]
    child2 = schedule2[:crossover_point] + schedule1[crossover_point:]
    return child1, child2

# Mutation
def mutate(schedule):
    mutate_point = random.randint(0, len(schedule) - 1)
    new_program = random.choice(all_programs)
    schedule[mutate_point] = new_program
    return schedule

# Genetic algorithm
def genetic_algorithm(initial_schedule, generations, population_size, CO_R, MUT_R, elitism_size):
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
            if random.random() < CO_R:
                child1, child2 = crossover(parent1, parent2)
            else:
                child1, child2 = parent1.copy(), parent2.copy()

            if random.random() < MUT_R:
                child1 = mutate(child1)
            if random.random() < MUT_R:
                child2 = mutate(child2)

            new_population.extend([child1, child2])

        population = new_population

    return population[0]

# Input parameters in Streamlit
st.write("Genetic Algorithm Parameters")

# Default values
default_CO_R = 0.8
default_MUT_R = 0.2

# Get user inputs and validate them
try:
    # Input for crossover rate
    CO_R = st.number_input(
        "Enter Crossover Rate (0.0 to 0.95):",
        min_value=0.0,
        max_value=0.95,
        value=default_CO_R,  # Default value
        step=0.01
    )
    
    # Input for mutation rate
    MUT_R = st.number_input(
        "Enter Mutation Rate (0.0 to 0.05):",
        min_value=0.0,
        max_value=0.05,
        value=default_MUT_R,  # Default value
        step=0.01
    )
except ValueError as e:
    st.error(f"Error: {e}")

# Display the chosen values for user confirmation
st.write(f"**Selected Parameters:**")
st.write(f"- Crossover Rate (CO_R): {CO_R}")
st.write(f"- Mutation Rate (MUT_R): {MUT_R}")

# Default parameters for genetic algorithm
generations = 100
population_size = 50
elitism_size = 2

# Create initial schedule
initial_schedule = all_programs[:len(all_time_slots)]
random.shuffle(initial_schedule)

# Run the algorithm
st.write("Running Genetic Algorithm...")
best_schedule = genetic_algorithm(
    initial_schedule, 
    generations, 
    population_size, 
    CO_R, 
    MUT_R, 
    elitism_size
)

# Display results
st.subheader("Final Optimal Schedule:")

# Ensure lists are the same length before creating DataFrame
schedule_data = {
    "Time Slot": [f"{hour}:00" for hour in range(6, 6 + len(best_schedule))],
    "Program": best_schedule
}
df = pd.DataFrame(schedule_data)
st.table(df)

# Display total ratings
total_rating = fitness_function(best_schedule)
st.write(f"Total Ratings: {total_rating:.2f}")
