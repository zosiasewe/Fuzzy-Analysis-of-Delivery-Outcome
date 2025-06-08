import numpy as np
import matplotlib.pyplot as plt
import math

# ----------------------------------------------------------------------
# Load data
# I think that the X in ES should be a ‚mu’ vectors of p(i)+delta parameters? And then you mutate them to find best


d = np.loadtxt("model20.txt", dtype=float)
i = np.array(d[:, 0])
o = np.array(d[:, 1])
N = 101
mi = 100  # parent population size -> trzeba zwiekszyc pewnie
l = 5 * mi  # offspring size
n = 2  # number of parameters (a, b, c)
Tmax = 200 #-> trzeba zwiekszyc


# ----------------------------------------------------------------------
# zmiana na p values; delta ; sigma p values i sigma delta

def suspicious_count():
    statuses = [-1.0, 0.5, 1.0] # kolejnosc ma znaczenie ; max 3 wartosci ; moga sie powtarzac
    p_i = []

    # -1 05 -1
    # 1 1 0.5
    # ..... wszystkie kombinacje zawierajace przynajmn iej 1 suspiocous

    for a in statuses:
        if -0.5 in [a]:
            p_i.append([a])

    for a in statuses:
        for b in statuses:
            if -0.5 in [a,b]:
                p_i.append([a,b])

    for a in statuses:
        for b in statuses:
            for c in statuses:
                if -0.5 in [a,b,c]:
                    p_i.append([a,b,c])
    return p_i

def initial_population(mi, p_i):
    n_values = p_i + 1 # bo mamy rozne p i jedna delte
    population = np.zeros((mi, 2 * n_values))  # [p1,p2,...pn delta sigma_p1, ... sigma_pn, sigma_delta]

    for idx in range(mi):
        population[idx, :n_values] = np.random.uniform(-0.5, 0.5, size=n_values)
        population[idx, n_values:] = np.random.uniform(0.1, 0.5, size=n_values)           #wstepnie takie wart ale do zmiany
    return population


def fitness_function(population, X_val, y_val, p_i):                #objective function to bd zmienioone z szukaniem p i delta ( 
    fitness_values = np.zeros(population.shape[0])
    n_suspicious = len(p_i)

    for idx, chrom in enumerate(population):
        p_values = chrom[:n_suspicious]
        delta = chrom[n_suspicious]

    #...
    # expert values
    # expert - predicted


    # fitness_values[idx] = np.mean(error)
    return fitness_values


def mating_pool(l, fitness_values, population):
    fitness_scores = 1 / (1 + fitness_values)
    probabilities = fitness_scores / np.sum(fitness_scores)
    selected_indices = np.random.choice(len(population), size=l, p=probabilities)
    pool = population[selected_indices]
    np.random.shuffle(pool)
    return pool


def mutation(mating_pool_result, n):
    tau_1 = 1 / math.sqrt(2 * n)
    tau_2 = 1 / math.sqrt(2 * math.sqrt(n))
    mutants = []

    for individual in mating_pool_result:
        r = np.exp(tau_1 * np.random.normal(0, 1))
        sigma = individual[n:] * r * np.exp(tau_2 * np.random.normal(0, 1, size=n))
        sigma = np.clip(sigma, 1e-8, 10)

        values = individual[:n] + sigma * np.random.normal(0, 1, size=n)
        values = np.clip(values, -10, 10)

        mutant = np.concatenate([values, sigma])
        mutants.append(mutant)

    return np.array(mutants)


def selection_mi_lambda(fitness, mutants):
    top = np.argsort(fitness)[:mi]
    return mutants[top]


def selection_mi_plus_lambda(fitness_mutants, fitness_parents, mutants, parents):
    new_population = np.concatenate((mutants, parents))
    new_fitness = np.concatenate((fitness_mutants, fitness_parents))
    top = np.argsort(new_fitness)[:mi]
    return new_population[top]


# ----------------------------------------------------------------------
if __name__ == "__main__":
    selection_mode = "mi_plus_lambda"  # or "mi_lambda"

    P_mi = initial_population(mi)
    fitness_parents = fitness_function(i, P_mi, o)

    t = 0
    min_fitness_log = []
    convergence_threshold = 1e-5

    print(f"Initial best fitness: {np.min(fitness_parents):.6f}")

    while t < Tmax:
        prev_fitness_parents = np.min(fitness_parents) #parents przed selekcja

        mating_pool_result = mating_pool(l, fitness_parents, P_mi)
        mutation_results = mutation(mating_pool_result, n)
        fitness_offspring = fitness_function(i, mutation_results, o)

        min_fitness_offspring = np.min(fitness_offspring) #offsping przed selekcja

        delta = abs(prev_fitness_parents - min_fitness_offspring) #!

        min_fitness_log.append(prev_fitness_parents)
        print(f"Generation {t}: Parent best = {prev_fitness_parents:.6f}, "
              f"Offspring best = {min_fitness_offspring:.6f}, "
              f"Delta = {delta:.2e}")

        if delta < convergence_threshold:
            print(f"\nStopped after {t + 1} generations")
            print(f"Final difference delta: {delta:.2e}")
            break

        if selection_mode == "mi_lambda":
            P_mi = selection_mi_lambda(fitness_offspring, mutation_results)
        elif selection_mode == "mi_plus_lambda":
            P_mi = selection_mi_plus_lambda(fitness_offspring, fitness_parents, mutation_results, P_mi)

        fitness_parents = fitness_function(i, P_mi, o)
        t += 1
