import numpy as np
from util.parse_utils import Scale
from shapely.geometry.polygon import Polygon
from shapely.geometry import Point
from operator import xor


class RegionOfInterest:
    """
    Class for defining a region of interest
    """

    def __init__(self, vertices_):
        self.vertices = vertices_
        self.ROI = np.array([0, 0, 1, 1], dtype=np.float)  # xmin ymin xmax ymax

        # with open(filename, 'r') as f:
        #     file_lines = f.readlines()
        #     for ll in file_lines:
        #         line_parts = ll.split(' ')
        #         if len(line_parts) == 5 and line_parts[0] == 'ROI':
        #             self.ROI[0] = float(line_parts[1])
        #             self.ROI[1] = float(line_parts[2])
        #             self.ROI[2] = float(line_parts[3])
        #             self.ROI[3] = float(line_parts[4])
        #             break

        # ROI_Polygon = np.loadtxt(conf['Dataset']['ROI'])
        # self.vertices = np.loadtxt(filename)
        # self.ROI[:2] = scale.normalize(self.ROI[:2])  # tl
        # self.ROI[2:] = scale.normalize(self.ROI[2:])  # br

        # self.vertices = scale.normalize(vertices_)
        self.polygon = Polygon(vertices_)

    def contains(self, pnt):
        return self.polygon.contains(Point(pnt[0], pnt[1]))
        # return xor(self.ROI[0] < pnt[0, 0] < self.ROI[2], self.ROI[1] < pnt[0, 1] < self.ROI[3])

