import csv
import numpy as np


class CsvReader:
    def __init__(self, path, separator=' '):
        self.path = path
        self.separator = separator

    def read_ply(self):
        list_vertex = []
        list_triangles = []
        num_vertex = 0
        num_triangles = 0
        count = 0
        post_end_header = False
        end_vertex = False

        with open(self.path) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=self.separator)

            for row in csv_reader:
                if row[0] == 'element':
                    if row[1] == 'vertex':
                        num_vertex = int(row[2])
                    if row[1] == 'face':
                        num_triangles = int(row[2])

                if row[0] == 'end_header':
                    post_end_header = True
                    continue

                if post_end_header:
                    if not end_vertex and count < num_vertex:
                        vertex = [float(row[0]), float(row[1]), float(row[2])]
                        list_vertex.append(vertex)

                        count += 1
                        if count == num_vertex:
                            count = 0
                            end_vertex = True
                    elif end_vertex and count < num_triangles:
                        triangle = np.array([int(row[0]), int(row[1]), int(row[2])])
                        list_triangles.append(triangle)

                        count += 1
                        if count == num_vertex:
                            count = 0

        return list_vertex, list_triangles
