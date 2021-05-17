from csv_reader import CsvReader


class Triangle:
    def __init__(self, V0, V1, V2):
        self.V0 = V0
        self.V1 = V1
        self.V2 = V2


class Ray:
    def __init__(self, P0, P1):
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

        self.num_vertex = len(self.list_vertex)
        self.num_triangles = len(self.list_triangles)


# M = Mesh()
# M.load('test.ply')
# print(M.list_vertex)
# print(M.list_triangles)