from typing import List
import matplotlib
import matplotlib.pyplot as plt
import copclib as copc
import pymeepcl as mee
import numpy as np
from mpl_toolkits.mplot3d.art3d import Line3DCollection

class MeePlotter():
    def __init__(self, struct: mee.MeeStruct) -> None:
        self.struct = struct

    def draw_roots(self, ax=None, color="b",):
        boxes = []
        for file in self.struct.files:
            boxes.append(copc.Box(list(file.global_bounds_min), list(file.global_bounds_max)))

        if not ax:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
        self._draw_boxes(boxes, ax=ax, color=color)

    def draw_nodes(self, depth, ax=None, color="b"):
        boxes = []
        for file in self.struct.files:
            nodes = self.struct.files[0].node_entries
            nodes = [node for node in nodes if node.key.d <= depth]

            for node in nodes:
                boxes.append(copc.Box(node.key, self.struct.files[0].las_header))

            if not ax:
                fig = plt.figure()
                ax = fig.add_subplot(111, projection='3d')
            self._draw_boxes(boxes, ax=ax, color=color)
            return ax

    def _draw_boxes(self, boxes, ax, color="b"):
        """
        Optimized box plotter using Line3DCollection.
        """

        segments = []

        # Define the 12 edge connections (indices) once, outside the loop
        # 0-3: Bottom, 4-7: Top, 8-11: Vertical connectors
        edge_indices = [
            (0, 1), (1, 2), (2, 3), (3, 0), # Bottom face
            (4, 5), (5, 6), (6, 7), (7, 4), # Top face
            (0, 4), (1, 5), (2, 6), (3, 7)  # Vertical lines
        ]

        # Collect min/max for auto-scaling later
        all_x, all_y, all_z = [], [], []

        for box in boxes:
            # Define the 8 corners of the current box
            # (x, y, z) tuples
            corners = [
                (box.x_min, box.y_min, box.z_min), # 0
                (box.x_max, box.y_min, box.z_min), # 1
                (box.x_max, box.y_max, box.z_min), # 2
                (box.x_min, box.y_max, box.z_min), # 3
                (box.x_min, box.y_min, box.z_max), # 4
                (box.x_max, box.y_min, box.z_max), # 5
                (box.x_max, box.y_max, box.z_max), # 6
                (box.x_min, box.y_max, box.z_max)  # 7
            ]

            # Extend lists for scaling
            all_x.extend([box.x_min, box.x_max])
            all_y.extend([box.y_min, box.y_max])
            all_z.extend([box.z_min, box.z_max])

            # Create the 12 line segments for this box
            for start_idx, end_idx in edge_indices:
                segments.append([corners[start_idx], corners[end_idx]])

        # Create the collection (ONE object for ALL lines)
        # colors='b' sets the color for all lines
        collection = Line3DCollection(segments, colors=color, linewidths=1)
        
        # Add to axes
        ax.add_collection3d(collection)

        # CRITICAL: Collections do not auto-update the axis limits.
        # We must manually set the limits to fit the data.
        if all_x: # check if list is not empty
            ax.set_xlim3d(min(all_x), max(all_x))
            ax.set_ylim3d(min(all_y), max(all_y))
            ax.set_zlim3d(min(all_z), max(all_z))

        ax.set_xlabel('X axis')
        ax.set_ylabel('Y axis')
        ax.set_zlabel('Z axis')

        ax.set_aspect('equal', adjustable='box')

    
        return ax
    
    def draw_ray(self, ray, lenght=100.0, ax=None):
        if not ax:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
        ax.scatter([ray.origin[0]], [ray.origin[1]], [ray.origin[2]], color='r')
        ax.plot([ray.origin[0], ray.origin[0]+ray.direction[0]*lenght], [ray.origin[1], ray.origin[1]+ray.direction[1]*lenght], [ray.origin[2], ray.origin[2]+ray.direction[2]*lenght], color='r')

