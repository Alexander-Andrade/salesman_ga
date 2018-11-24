import numpy as np
import itertools
import random
import matplotlib.pyplot as plt
import time


def pairs(iterable, n):
    size = len(iterable)
    count = 0

    while True:
        ids = np.random.permutation(size).tolist() + np.random.permutation(size).tolist()
        ids_len = size << 1
        for i in range(0, ids_len, 2):
            yield (iterable[ids[i]], iterable[ids[i+1]])
            count += 1
            if count >= n:
                return


def read_cities_coords(file_name):
    with open(file_name) as f:
        content = f.readlines()

    content = content[6: -2]
    coords = np.empty(shape=(len(content), 2))

    for i, data in enumerate(content):
        numbers = [float(s) for s in data.rstrip("\n\r").split()]
        coords[i] = numbers[1:]

    return coords


def euclidean_distance(c1, c2):
    return ((c1[0] - c2[0])**2 + (c1[1] - c2[1])**2)**0.5


def distance_martix(coords):
    dist_matr_top = np.zeros(shape=(coords.shape[0], coords.shape[0]))
    for i in range(coords.shape[0]):
        for j in range(i-1, -1, -1):
            dist_matr_top[i][j] = euclidean_distance(coords[i], coords[j])

    dist_matr_down = np.transpose(dist_matr_top)
    return dist_matr_top + dist_matr_down


def generate_population(population_size, n_cities):
    pop = np.empty(shape=(population_size, n_cities), dtype=np.uint16)
    for i in range(population_size):
        pop[i] = np.random.permutation(n_cities)
    return pop


def calc_distance(tour, dist_matr):
    prev_city = tour[0]
    distance = 0
    for city in itertools.islice(tour, 1, None):
        distance += dist_matr[prev_city][city]
        prev_city = city

    return distance


def calc_fitness(population, dist_matr):
    distances = np.zeros(shape=(population.shape[0]))
    for i, tour in enumerate(population):
        distances[i] = calc_distance(tour, dist_matr)

    return distances


def pmx_child(parent1, parent2, p1, p2):
    child = parent1.copy()
    p2_mid = parent2[p1:p2]
    child[p1:p2] = p2_mid
    conflicts_ids_beg = np.nonzero(np.isin(parent1[:p1], p2_mid, assume_unique=True))[0]
    conflicts_ids_end = np.nonzero(np.isin(parent1[p2:], p2_mid, assume_unique=True))[0]
    conflicts_ids = np.append(conflicts_ids_beg, conflicts_ids_end+p2)
    conflict_values = parent1[conflicts_ids]
    for ind, value in zip(conflicts_ids, conflict_values):
        while True:
            i = np.nonzero(parent2 == value)[0][0]
            if parent1[i] in child:
                value = parent1[i]
            else:
                child[ind] = parent1[i]
                break

    return child


def pmx_crossover(parents, n_children, dist_matr):
    n_cities = parents.shape[1]
    children = np.empty(shape=(n_children, n_cities), dtype=np.uint16)

    for i, pair in enumerate(pairs(parents, n_children)):
        cutting_planes = np.random.randint(low=1, high=n_cities-2, size=2)
        cutting_planes.sort()
        child1 = pmx_child(pair[0], pair[1], cutting_planes[0], cutting_planes[1])
        child2 = pmx_child(pair[1], pair[0], cutting_planes[0], cutting_planes[1])

        if calc_distance(child1, dist_matr) > calc_distance(child2, dist_matr):
            children[i] = child2
        else:
            children[i] = child1

    return children


def tournament(population, n_parents):
    m = random.randint(2, population.shape[0]-1)
    parents = np.empty(shape=(n_parents, population.shape[1]), dtype=np.uint16)
    for i in range(n_parents):
        participants_ids = random.sample(range(population.shape[0]), m)
        parents[i] = population[min(participants_ids)]
    return parents


def mutation(children, mutation_prob):
    for child in children:
        if random.random() < mutation_prob:
            ids = np.random.choice(child.shape[0], size=2)
            child[ids[0]], child[ids[1]] = child[ids[1]], child[ids[0]]


def reduce_population(parents, children, pop_size):
    n_grandparents = pop_size - children.shape[0]
    new_population = np.empty((pop_size, parents.shape[1]), dtype=np.uint16)
    new_population[:n_grandparents, :] = parents[:n_grandparents]
    new_population[n_grandparents:, :] = children
    return new_population


def init_path_plot(cities_coords):
    axes = plt.gca()
    axes.plot(cities_coords[:, 0], cities_coords[:, 1], 'x', zorder=2)
    solutions, = axes.plot([], [], 'r-', zorder=1)
    plt.draw()
    return solutions


def draw_solutions(population, graph_data, cities_coords):
    coords = cities_coords[population[0]]
    graph_data.set_xdata(coords[:, 0])
    graph_data.set_ydata(coords[:, 1])
    plt.draw()
    plt.pause(1e-17)
    # time.sleep(0.1)


def find_tour(cities_coords, num_generations=100000, population_size=16, n_parents=8, n_grandparents=2,
              mutation_prob=0.01):
    graph_data = init_path_plot(cities_coords)
    dist_matr = distance_martix(cities_coords)
    population = generate_population(population_size, dist_matr.shape[0])

    for generation in range(num_generations):
        distances = calc_fitness(population, dist_matr)
        population = population[distances.argsort()]

        print(calc_distance(population[0], dist_matr))
        draw_solutions(population=population, graph_data=graph_data, cities_coords=cities_coords)

        parents = tournament(population, n_parents)
        children = pmx_crossover(parents=parents,
                                 n_children=population_size-n_grandparents,
                                 dist_matr=dist_matr)
        mutation(children, mutation_prob)
        population = reduce_population(parents=parents,
                                       children=children,
                                       pop_size=population_size)


if __name__ == '__main__':
    coords = read_cities_coords('berlin52.tsp')
    find_tour(coords)
    plt.show()

