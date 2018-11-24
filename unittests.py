import unittest
from main import *


class Salesman(unittest.TestCase):

    def test_distance_martix(self):
        coords = read_cities_coords('berlin52.tsp')
        dist_matr = distance_martix(coords)
        dist = euclidean_distance(coords[0], coords[3])
        self.assertEqual(dist_matr[0][3], dist)
        self.assertEqual(dist_matr[3][0], dist)
        self.assertEqual(dist_matr[-1][-2], euclidean_distance(coords[-1], coords[-2]))
        self.assertEqual(dist_matr[3][3], 0.)

    def test_pmx_child(self):
        parent1 = np.arange(1, 10, dtype=np.uint16)
        parent2 = np.array([4, 5, 2, 1, 8, 7, 6, 9, 3])
        child = pmx_child(parent1, parent2, 3, 7)
        self.assertSequenceEqual(child.tolist(), [4, 2, 3, 1, 8, 7, 6, 5, 9])

        parent1 = np.array([5, 3, 9, 6, 1, 8, 0, 4, 7, 2])
        parent2 = np.array([2, 7, 6, 4, 8, 0, 1, 9, 3, 5])
        child1 = pmx_child(parent1, parent2, 2, 5)
        self.assertEqual(np.unique(child1).shape[0] == child1.shape[0], True)
        child2 = pmx_child(parent1, parent2, 2, 5)
        self.assertEqual(np.unique(child2).shape[0] == child2.shape[0], True)
        self.assertEqual(parent1.shape, child1.shape)

    def test_pmx_uniqueness(self):
        coords = read_cities_coords('berlin52.tsp')
        dist_matr = distance_martix(coords)
        population = generate_population(2, coords.shape[0])
        children = pmx_crossover(parents=population,
                                 n_children=population.shape[0],
                                 dist_matr=dist_matr)
        a, counts = np.unique(children[0], return_counts=True)
        self.assertEqual(np.unique(children[0]).shape[0] == children[0].shape[0], True)

if __name__ == '__main__':
    unittest.main()
