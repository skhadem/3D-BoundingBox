import numpy as np

#TODO save to file and read from it
"""
Class will hold the average dimension for a class, regressed value is the residual
"""
class ClassAverages:
    def __init__(self, classes):
        self.dimension_map = {}

        for detection_class in classes:
            self.dimension_map[detection_class] = {}
            self.dimension_map[detection_class]['count'] = 0
            self.dimension_map[detection_class]['average'] = np.zeros(3, dtype=np.double)


    def add_item(self, class_, dimension):
        self.dimension_map[class_]['count'] += 1
        self.dimension_map[class_]['average'] += dimension
        # self.dimension_map[class_]['average'] /= self.dimension_map[class_]['count']

    def get_item(self, class_):
        return self.dimension_map[class_]['average'] / self.dimension_map[class_]['count']
