import copclib as copc
import numpy as np
import time
import os
import warnings
import argparse

warnings.filterwarnings('ignore') # numpy devide by zero -> inf, gewolltes Verhalten

class Ray():
    """
    Klasse die einen Strahl anhand Ursprung und Richtung beschreibt.

    Attribute:
        direction, origin: np.ndarray
    """
    def __init__(self, origin: np.ndarray, direction: np.ndarray) -> None:
        if not ((origin.shape == (3,)) and (direction.shape == (3,))):
            raise ValueError("Argumente müssen jeweils 3 Dimensionen haben") 

        self.origin = origin
        self.direction = direction

    @property
    def origin(self) -> np.ndarray:
        return self._origin

    @origin.setter
    def origin(self, value: np.ndarray):
        if not isinstance(value, np.ndarray):
            raise TypeError("Ursprung muss ein NumPy ndarray sein.")
        if value.shape != (3,):
            raise ValueError("Ursprung muss ein 3D vector (shape (3,)) sein")
        self._origin = value

    @property
    def direction(self) -> np.ndarray:
        return self._direction

    @direction.setter
    def direction(self, value: np.ndarray):
        if not isinstance(value, np.ndarray):
            raise TypeError("Richtung muss ein NumPy ndarray sein")
        if value.shape != (3,):
            raise ValueError("Richtung muss ein 3D vector (shape (3,)) sein")

        # Normalisiere den Richtungsvektor
        norm = np.linalg.norm(value)
        if np.isclose(norm, 0):
            raise ValueError("Richtung darf kein Nullvektor sein")
        self._direction = value / norm
        self._inverse_direction = (value / norm)**(-1) 

    def __str__(self):
        return f"Ray( Ursprung={self.origin}, direction={self.direction})"
        
    def slab_test(self, box: copc.Box) -> bool:
        """Führt einen vektorisierten slab-Test mit einer copc.Box Instanz aus"""
        t_min, t_max = 0.0, np.inf
        box_min = np.array([box.x_min - args.radius, box.y_min - args.radius, box.z_min - args.radius]) # berücksichtigt radius
        box_max = np.array([box.x_max + args.radius, box.y_max + args.radius, box.z_max + args.radius]) 
        
        t1 = (box_min - self.origin) * self._inverse_direction
        t2 = (box_max - self.origin) * self._inverse_direction

        # minimaler und maximaler t-Wert pro Achse
        # np.minimum/maximum vergleicht Elementweise
        t_near = np.minimum(t1, t2)
        t_far = np.maximum(t1, t2)

        t_min = np.max(t_near) # spätest möglicher Eintrittszeitpunkt
        t_max = np.min(t_far) # frühest möglicher Austrittszeitpunkt
        
        return (t_min < t_max)
    
    def slab_test_vectorized(self, boxes_min: np.ndarray, boxes_max: np.ndarray, radius: float) -> np.ndarray:
        """
        Tests intersection against given boxes simultaneously.
        Returns a boolean mask.
        """
        # Expand boxes by radius
        b_min = boxes_min - radius
        b_max = boxes_max + radius

        # Vectorized calculation (N, 3)
        t1 = (b_min - self.origin) * self._inverse_direction
        t2 = (b_max - self.origin) * self._inverse_direction

        # Find t_near and t_far
        t_near = np.minimum(t1, t2)
        t_far = np.maximum(t1, t2)

        # Max of nearest, Min of farthest (Collapse to shape (N,))
        t_enter = np.max(t_near, axis=1)
        t_exit = np.min(t_far, axis=1)

        # Valid intersection if entry < exit and exit > 0 (box is not behind ray)
        return (t_enter < t_exit) & (t_exit > 0)

# Argument parser erstellen
parser = argparse.ArgumentParser(description="Ein Script welches alle Punkte in einem Radius um einen Bildstrahl aus einer copc Datei extrahiert")

# Input file path
parser.add_argument("-i", "--input", required=True, type=str,
                    help="Pfad zur copc Datei")
# Projektionszentrum des Strahls als 3 floats
parser.add_argument("-p", "--projektionszentrum", type=float, nargs=3, metavar=("X", "Y", "Z"),
                    required=True, help="Die x, y, z Koordinaten des Projektionszentrums des Strahls")
# Richtung des Strahls
parser.add_argument("-d", "--direction", type=float, nargs=3, metavar=("DX", "DY", "DZ"),
                    required=True, help="Die x, y, z Werte des Richtungsvektors des Strahls")
# Zylinderradius
parser.add_argument("-r", "--radius", type=float, default=10,
                    help="Der Radius des Zylinders um den Bildstrahl herum (default: 10m)")
# outputfile
parser.add_argument("-o", "--output", dest="outputfile", type=str, default="punkte.xyz",
                    help="Pfad zum outputfile (default: punkte.xyz)")
# sortieren nach entfernung vom Strahl                    
parser.add_argument("-s", "--sorted", dest="sorted", type=bool, default=True,
                    help="Punkte nach Entfernung entlang des Strahls sortieren")

args = parser.parse_args()

print("\nStarte Einlesen der Copc Datei...")
start1 = time.time()

reader = copc.FileReader(args.input)

# Las Header (für konstruieren der Boxen)
las_header = reader.copc_config.las_header

# Alle Nodes auslesen und Bounding Boxen konstruieren
all_entries = reader.GetAllNodes()
#boxes = []
#for entry in all_entries:
#    boxes.append(copc.Box(entry.key, las_header)) 

print(f"Einlesen erfolgreich!\nDauer: {round(1000*(time.time()-start1),1)} ms\n")

print("Starte Schnitttest des Strahls mit den Nodes...")
start = time.time()

Strahl = Ray(np.array(args.projektionszentrum), np.array(args.direction))

#intersected_nodes = []
#intersected_boxes = []
#
#for entry, box in zip(all_entries, boxes):
##for entry in all_entries:
#    #box = copc.Box(entry.key, las_header)
#    if Strahl.slab_test(box):
#        intersected_nodes.append(entry)
#        intersected_boxes.append(box)

# 
count = len(all_entries)
boxes_min = np.zeros((count, 3))
boxes_max = np.zeros((count, 3))

for i, entry in enumerate(all_entries):
    box = copc.Box(entry.key, las_header)
    boxes_min[i] = [box.x_min, box.y_min, box.z_min]
    boxes_max[i] = [box.x_max, box.y_max, box.z_max]

intersect_mask = Strahl.slab_test_vectorized(boxes_min, boxes_max, args.radius)
intersected_nodes = [all_entries[i] for i in range(count) if intersect_mask[i]]

print(f"Schnitttest erfolgreich!\nDurchdrungene Nodes: {len(intersected_nodes)} / {len(all_entries)}\nDauer: {round(1000*(time.time()-start),1)} ms\n")

print(f"Starte Einlesen der Punkte aus den gefundenen Nodes...")
start = time.time()

# Punkte aus den geschnittenen Nodes extrahieren
#intersected_points_packed = []

#for node in intersected_nodes:
    #intersected_points_packed.append(reader.GetPoints(node))

# Punkte in ein Numpy array überführen
#all_x_arrays = [points.x for points in intersected_points_packed]
#all_y_arrays = [points.y for points in intersected_points_packed]
#all_z_arrays = [points.z for points in intersected_points_packed]

#all_x = np.concatenate(all_x_arrays)
#all_y = np.concatenate(all_y_arrays)
#all_z = np.concatenate(all_z_arrays)

#punkte = np.column_stack((all_x, all_y, all_z))

# Anzahl Punkte
total_points = sum(node.point_count for node in intersected_nodes)

# Pre-allocate array (N, 3)
punkte = np.empty((total_points, 3), dtype=np.float64)

# Füll das Array Node für Node
cursor = 0
for node in intersected_nodes:
    points = reader.GetPoints(node)
    count = len(points.x)
    
    punkte[cursor:cursor+count, 0] = points.x
    punkte[cursor:cursor+count, 1] = points.y
    punkte[cursor:cursor+count, 2] = points.z
    
    cursor += count

print(f"""Punkte Erfolgreich eingelesen!
Anzahl der eingelesenen Punkte: {len(punkte)}
Durchschnittliche Anzahl an Punkten pro Node: {round(len(punkte)/len(intersected_nodes))}
Dauer: {round(1000*(time.time()-start),1)} ms\n""")

print(f"Starte Koordinatentransformation in das lokale System des Strahls...")
start=time.time()

# neues koordinatensystem definieren
ursprung = Strahl.origin
x_achse = Strahl.direction

# Zufälliges "Up" array für Kreuzprodukt, überprüfen ob er zu parallel ist
up_array = np.array([0,1,0]) if abs(np.dot(x_achse, np.array([1,0,0]))) > 0.99 else np.array([1,0,0])

y_achse = np.cross(x_achse, up_array) # Kreuzprodukt steht auf beide vektoren normal
y_achse = y_achse / np.linalg.norm(y_achse) # Normalisieren

z_achse = np.cross(x_achse, y_achse) #nochmal
z_achse = z_achse / np.linalg.norm(z_achse)

rotation = np.column_stack((x_achse,y_achse,z_achse))

punkte_transformiert = punkte - ursprung
punkte_transformiert = punkte_transformiert @ rotation

print(f"""Transformation erfolgreich!
Dauer: {round(1000*(time.time()-start),1)} ms\n""")

print(f"Starte Extraktion der Punkte im Umkreis von {args.radius} m um den Strahl...")
start = time.time()

# Abstandsquadrate Test
y_koordinaten = punkte_transformiert[:,1]
z_koordinaten = punkte_transformiert[:,2]

abstandsquadrate = (y_koordinaten**2) + (z_koordinaten**2)
abstandsquadrate_maske = abstandsquadrate < (args.radius**2)

# anwenden der Maske
punkte_im_zylinder_global = punkte[abstandsquadrate_maske]


print(f"""Punkte erfolgreich extrahiert!
Gefundene Punkte: {len(punkte_im_zylinder_global)}
Gespeichert in: {args.outputfile}
Dauer: {round(1000*(time.time()-start),1)} ms

Dauer des gesamten Programms: {round(1000*(time.time()-start1),1)} ms
""")

# Sortieren falls gewünscht
if args.sorted:
    punkte_im_zylinder_lokal = punkte_transformiert[abstandsquadrate_maske]
    punkte_im_zylinder_lokal = punkte_im_zylinder_lokal[punkte_im_zylinder_lokal[:, 0].argsort()]

    # Rücktransformation
    punkte_rücktransformiert = punkte_im_zylinder_lokal @ rotation.T
    punkte_rücktransformiert = punkte_rücktransformiert + ursprung

    np.savetxt(args.outputfile, punkte_rücktransformiert)
else:
    np.savetxt(args.outputfile, punkte_im_zylinder_global)