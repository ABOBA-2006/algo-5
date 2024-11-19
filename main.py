import random
import json
import matplotlib.pyplot as plt

P = 500
N = 100
MIN_PRICE = 2
MAX_PRICE = 30
MIN_WEIGHT = 1
MAX_WEIGHT = 20
MAX_DUPLICATE = 5
CROSSOVER_OPERATOR = 50
CROSSOVER_OPERATOR2 = [33, 66]
MUTATION_PROBABILITY = 5
ITERATIONS = 1000
LOAD_FROM_FILE = True


class Object:
    def __init__(self, price, weight):
        self.price = price
        self.weight = weight


# initializing backpack and fulfill parallel the first population
def init():
    backpack = []
    population = []
    fitness = []
    individual = [0]*N
    for i in range(N):
        price = random.randint(MIN_PRICE, MAX_PRICE)
        weight = random.randint(MIN_WEIGHT, MAX_WEIGHT)
        backpack.append(Object(price, weight))

        new_individual = individual.copy()
        new_individual[i] = random.randint(1, MAX_DUPLICATE)
        population.append(new_individual)

        current_fitness = backpack[i].price * new_individual[i]
        if backpack[i].weight * new_individual[i] > P:
            current_fitness = 0
        fitness.append(current_fitness)

    return backpack, population, fitness


# functions for work with file
def save_to_txt(backpack, population, fitness, filename):
    data = {
        'backpack': [{'price': obj.price, 'weight': obj.weight} for obj in backpack],
        'population': population,
        'fitness': fitness
    }
    with open(filename, 'w') as f:
        # Save data as JSON to make it easy to read and write
        json.dump(data, f, indent=4)


def load_from_txt(filename):
    with open(filename, 'r') as f:
        data = json.load(f)

    backpack = [Object(obj['price'], obj['weight']) for obj in data['backpack']]
    population = data['population']
    fitness = data['fitness']

    return backpack, population, fitness

if LOAD_FROM_FILE:
    Backpack, Population, Fitness = load_from_txt('data.txt')
else:
    Backpack, Population, Fitness = init()
    save_to_txt(Backpack, Population, Fitness, 'data.txt')


# helpful functions
def calc_fitness_weight(individual):
    total_price = 0
    total_weight = 0
    for i in range(N):
        if individual[i] != 0:
            total_price += (Backpack[i].price * individual[i])
            total_weight += (Backpack[i].weight * individual[i])
    return total_price, total_weight


# function to get the random individual via inversely proportional fitness func
def roulette_wheel_selection(fitness):
    total_fitness = sum(fitness)
    pick = random.uniform(0, total_fitness)

    current = 0
    for i, fitness in enumerate(fitness):
        current += fitness
        if current > pick:
            return fitness, i


# first local_improvement
def local_improvement(individual, weights, values, child_weight, child_fitness):

    best_item = -1
    best_ratio = 0

    for i in range(len(individual)):
        if individual[i] < MAX_DUPLICATE:
            ratio = values[i] / weights[i]
            if ratio > best_ratio and child_weight + weights[i] <= P:
                best_ratio = ratio
                best_item = i

    if best_item != -1:
        while individual[best_item] < MAX_DUPLICATE:
            if child_weight + weights[best_item] <= P:
                individual[best_item] += 1
                child_weight += weights[best_item]
                child_fitness += values[best_item]
            else:
                break

    return individual, child_weight, child_fitness


def evolution():
    # selecting parents
    max_fitness = max(Fitness)
    max_index = Fitness.index(max_fitness)
    random_fitness, random_index = roulette_wheel_selection(Fitness)

    # crossover #1
    first_half = Population[max_index][:CROSSOVER_OPERATOR]
    second_half = Population[random_index][CROSSOVER_OPERATOR:]
    child = first_half + second_half
    child_fitness, child_weight = calc_fitness_weight(child)
    print(child_weight)

    # mutation
    random_number = random.randint(1, 100)
    if random_number <= MUTATION_PROBABILITY:
        random_gen = random.randint(0, N-1)
        mutant = child.copy()

        choice = random.randint(0, 1)
        if (choice == 0 and mutant[random_gen] != 0) or mutant[random_gen] == MAX_DUPLICATE:
            mutant[random_gen] -= 1
            mutant_fitness = child_fitness - Backpack[random_gen].price
            mutant_weight = child_weight - Backpack[random_gen].weight
        else:
            mutant[random_gen] += 1
            mutant_fitness = child_fitness + Backpack[random_gen].price
            mutant_weight = child_weight + Backpack[random_gen].weight

        if mutant_fitness > child_fitness and mutant_weight < P:
            child = mutant
            child_fitness = mutant_fitness
            child_weight = mutant_weight

    # local upgrading
    child, child_weight, child_fitness = local_improvement(child, [obj.weight for obj in Backpack],
                                                           [obj.price for obj in Backpack], child_weight, child_fitness)
    # # updating population
    min_fitness = min(Fitness)
    min_index = Fitness.index(min_fitness)

    if child_fitness > min_fitness and child_weight < P:
        Population[min_index] = child
        Fitness[min_index] = child_fitness


fitness_history = []
def main():
    for i in range(ITERATIONS):
        max_fitness = max(Fitness)
        fitness_history.append(max_fitness)
        evolution()


    max_fitness = max(Fitness)
    max_index = Fitness.index(max_fitness)
    print("INDIVIDUAL = ")
    for i in range(0, N, 10):
        print(Population[max_index][i:i + 10])  # Print the next 10 elements
    print()
    print("MAX_PRICE = " + str(max_fitness) + "\n")
    weight = 0
    for i in range(N):
        if Population[max_index][i] != 0:
            weight += (Backpack[i].weight * Population[max_index][i])
    print("WEIGHT = " + str(weight) + "\n")


    plt.plot(range(ITERATIONS), fitness_history, label="Max Fitness")
    plt.xlabel("Iteration (i)")
    plt.ylabel("Max Fitness")
    plt.title("Max Fitness Over Iterations")
    plt.legend()
    plt.grid()
    plt.show()

if __name__=="__main__":
    main()


