
import random
import datetime
import csv
import copy
from simulation_core import Simulation, Train

# --- 1. GENETIC ALGORITHM PARAMETERS ---

POPULATION_SIZE = 50
NUM_GENERATIONS = 100
MUTATION_RATE = 0.1
CROSSOVER_RATE = 0.8

# --- 2. FITNESS FUNCTION ---

def calculate_fitness(simulation_log):
    """Calculates the fitness of a schedule based on the simulation log."""
    if simulation_log.empty:
        return 0

    # 1. Average Delay (lower is better)
    average_delay = simulation_log.groupby('train_id')['delay_minutes'].max().mean()

    # 2. Track Utilization (higher is better)
    line_utilization = simulation_log['line'].nunique() / 4 # 4 is the number of lines

    # 3. Fairness (lower std dev is better)
    avg_delay_by_type = simulation_log.groupby('train_type')['delay_minutes'].max().mean()
    fairness = simulation_log.groupby('train_type')['delay_minutes'].max().std()

    # 4. Congestion (lower is better)
    congestion = len(simulation_log[simulation_log['event'].isin(['halted', 'delayed'])])

    # Combine metrics into a single fitness score (weights can be tuned)
    fitness = (1 / (1 + average_delay)) + \
              (line_utilization) + \
              (1 / (1 + fairness)) - \
              (congestion / len(simulation_log))

    return fitness

def selection(population, fitness_scores):
    selected = []
    for _ in range(len(population)):
        tournament = random.sample(list(zip(population, fitness_scores)), k=3)
        winner = max(tournament, key=lambda x: x[1])
        selected.append(winner[0])
    return selected

# --- 3. GENETIC ALGORITHM ---

def create_initial_population(num_trains):
    population = []
    for _ in range(POPULATION_SIZE):
        simulation = Simulation(num_trains=num_trains)
        population.append(simulation.trains)
    return population



def crossover(parent1, parent2):
    if random.random() < CROSSOVER_RATE:
        crossover_point = random.randint(1, len(parent1) - 1)
        child1 = copy.deepcopy(parent1[:crossover_point]) + copy.deepcopy(parent2[crossover_point:])
        child2 = copy.deepcopy(parent2[:crossover_point]) + copy.deepcopy(parent1[crossover_point:])
        return child1, child2
    return copy.deepcopy(parent1), copy.deepcopy(parent2)

def mutate(schedule, stations):
    mutated_schedule = copy.deepcopy(schedule)
    for i in range(len(mutated_schedule)):
        if random.random() < MUTATION_RATE:
            # Mutate scheduled departure time
            mutated_schedule[i].scheduled_departure_time += datetime.timedelta(minutes=random.randint(-30, 30))

        # Mutate planned platform
        if random.random() < MUTATION_RATE:
            train = mutated_schedule[i]
            if train.station: # Only mutate if the train is at a station
                station = stations.get(train.station)
                if station:
                    new_platform = random.choice(list(station.platforms.keys()))
                    train.planned_platform = new_platform

    return mutated_schedule

def run_optimizer():
    """Runs the genetic algorithm to find the optimal train schedule."""
    # Create a simulation to get station data
    base_sim = Simulation()
    stations = base_sim.stations

    population = create_initial_population(num_trains=80)
    best_schedule = None
    best_fitness = 0

    for generation in range(NUM_GENERATIONS):
        print(f"Generation {generation + 1}/{NUM_GENERATIONS}")

        # Evaluate fitness
        fitness_scores = []
        for schedule in population:
            sim = Simulation(trains=schedule)
            log = sim.run()
            fitness = calculate_fitness(log)
            fitness_scores.append(fitness)

            if fitness > best_fitness:
                best_fitness = fitness
                best_schedule = copy.deepcopy(schedule)

        # Select parents
        parents = selection(population, fitness_scores)

        # Create next generation
        next_generation = []
        for i in range(0, len(parents), 2):
            parent1 = parents[i]
            parent2 = parents[i+1]
            child1, child2 = crossover(parent1, parent2)
            next_generation.append(mutate(child1, stations))
            next_generation.append(mutate(child2, stations))

        population = next_generation

    print("Optimizer finished.")
    return best_schedule

if __name__ == "__main__":
    best_schedule = run_optimizer()

    # Save the best schedule
    with open('optimized_schedule.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["train_id", "train_type", "direction", "scheduled_departure_time"])
        for train in best_schedule:
            writer.writerow([train.id, train.type, train.direction, train.scheduled_departure_time.strftime('%Y-%m-%d %H:%M:%S')])

    print("Optimized schedule saved to optimized_schedule.csv")
