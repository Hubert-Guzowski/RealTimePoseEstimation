from csv_reader import CsvReader
import numpy as np


class Triangle:
    def __init__(self, V0: np.ndarray, V1: np.ndarray, V2: np.ndarray):
        self.V0 = V0
        self.V1 = V1
        self.V2 = V2


class Ray:
    def __init__(self, P0: np.ndarray, P1: np.ndarray):
        self.P0 = P0
        self.P1 = P1


class Mesh:
    def __init__(self, num_vertex=0, num_triangles=0, list_vertex=None, list_triangles=None):
        if list_triangles is None:
            list_triangles = []
        if list_vertex is None:
            list_vertex = []

        self.num_vertex = num_vertex
        self.num_triangles = num_triangles
        self.list_vertex = list_vertex
        self.list_triangles = list_triangles

    def load(self, path):
        csv_reader = CsvReader(path)
        self.list_vertex, self.list_triangles = csv_reader.read_ply()
        self.list_triangles = np.array(self.list_triangles)
        self.list_vertex = np.array(self.list_vertex)

        self.num_vertex = len(self.list_vertex)
        self.num_triangles = len(self.list_triangles)


# M = Mesh()
# M.load('test.ply')
# print(M.list_vertex)
# print(M.list_triangles)
