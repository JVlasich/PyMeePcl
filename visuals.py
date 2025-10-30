from typing import List
import matplotlib
import matplotlib.pyplot as plt
import copclib as copc

def draw_boxes(boxes: List[copc.Box], ax) -> None:
    """ Nimmt eine Liste von copc.Box Instanzen entgegen und plottet sie auf eine vordefinierte mpl axes"""

    for box in boxes:
        # Punkte an denen sich die Kanten der voxel berühren 
        verts = [
            (box.x_min, box.y_min, box.z_min),
            (box.x_max, box.y_min, box.z_min),
            (box.x_max, box.y_max, box.z_min),
            (box.x_min, box.y_max, box.z_min),
            (box.x_min, box.y_min, box.z_max),
            (box.x_max, box.y_min, box.z_max),
            (box.x_max, box.y_max, box.z_max),
            (box.x_min, box.y_max, box.z_max)
        ]

        # 12 Kanten
        edges = [
            (0, 1), (1, 2), (2, 3), (3, 0),  # Grundfläche
            (4, 5), (5, 6), (6, 7), (7, 4),  # Deckfläche
            (0, 4), (1, 5), (2, 6), (3, 7)   # Verbindungen
        ]

        for edge in edges:
            p1, p2 = edge
            x = [verts[p1][0], verts[p2][0]]
            y = [verts[p1][1], verts[p2][1]]
            z = [verts[p1][2], verts[p2][2]]
            ax.plot(x, y, z, c='b')

    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')