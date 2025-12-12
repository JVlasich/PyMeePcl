from __future__ import annotations
import copclib as copc
import numpy as np
import os
from os import path
import glob
from typing import List
from collections import OrderedDict
import time
import argparse

class MeeStruct():
    ## Methods:
    # trace_ray(Ray)
    # add_file(filepath)
    # del_file(filepath)

    #TODO
    # make_statistic()
    # draw_boxes()

    ## Properties
    # cache (NodeCache Klasse)
    # files (List[MeeFiles])
    # bounds (combined global bounds of all files )
    
    def __init__(self, source: str | List[str], pattern: str = "*", cache_size: int = 25):
        """
        Initialize with a single path string, a list of strings, or a directory string.
        arg pattern: glob search
        arg cache_size: maximum nodes to be loaded
        """
        self.cache = NodeCache(max_size=cache_size) # Key: (filepath, node_key_string), Value: PointData object
        self.bounds = None
        self.files: List[MeeFile] = []
        
        # Make into list if single input
        if isinstance(source, str):
            sources = [source]
        else:
            sources = source

        # process list
        for entry in sources:

            if not os.path.exists(entry):
                raise FileNotFoundError(f"Path does not exist: {entry}")

            if os.path.isdir(entry):
                # construct the search string
                search_path = os.path.join(entry, pattern)
                # Use glob to search dir
                matches = glob.glob(search_path)
                # filter for files
                for match in matches:
                    if os.path.isfile(match):
                        self.files.append(MeeFile(match))
                        
            elif os.path.isfile(entry):
                self.files.append(MeeFile(entry))

        self.update_bounds()
    
    def update_bounds(self):
        """Updates the combined global bounds of all files."""
        if len(self.files) == 0:
            self.bounds = None
            return

        # prealocate empty array
        count = len(self.files)
        bounds_min = np.zeros((count, 3))
        bounds_max = np.zeros((count, 3))

        # populate array
        for i, file in enumerate(self.files):
            bounds_min[i] = file.global_bounds_min
            bounds_max[i] = file.global_bounds_max

        # get minimum/maximum of each column
        minimums = bounds_min.min(axis=0)
        maximums = bounds_max.max(axis=0)
        self.bounds = (minimums, maximums)

    def add_file(self, filepath: str):
        """Adds a COPC file to the structure."""
        try:
            self.files.append(MeeFile(filepath))
            self.update_bounds()
        except Exception as e:
            print(f"Failed to add file {filepath}: {e}")

    def del_file(self, filepath: str):
        """Removes a file and clears associated cache entries."""
        self.files = [f for f in self.files if f.filepath != filepath]
        
        # Clean cache for this file
        self.cache.remove_file(filepath)

    def _transform_and_filter(self, points_np: np.ndarray, ray: Ray, radius: float, return_sorted: bool = True) -> np.ndarray:
        """
        Transforms points to Local system (Ray) and filters by cylinder radius.
        """
        if len(points_np) == 0:
            return np.empty((0, 3))

        # coordinate system definition
        x_axis = ray.direction
        
        # "Up" vector check to avoid parallel cross product
        up_array = np.array([0, 1, 0]) if abs(np.dot(x_axis, np.array([1, 0, 0]))) > 0.99 else np.array([1, 0, 0])
        
        y_axis = np.cross(x_axis, up_array)
        y_axis = y_axis / np.linalg.norm(y_axis)
        
        z_axis = np.cross(x_axis, y_axis)
        z_axis = z_axis / np.linalg.norm(z_axis)
        
        rotation = np.column_stack((x_axis, y_axis, z_axis))

        # Transform
        points_transformed = points_np - ray.origin
        points_transformed = points_transformed @ rotation

        # Cylinder Filter
        y_coords = points_transformed[:, 1]
        z_coords = points_transformed[:, 2]
        
        dist_sq = (y_coords**2) + (z_coords**2)
        mask = dist_sq < (radius**2)

        if return_sorted:
            points_transformed = points_transformed[mask]
            points_transformed = points_transformed[points_transformed[:, 0].argsort()]

            point_backtransformed = points_transformed @ rotation.T
            point_backtransformed = point_backtransformed + ray.origin

            return point_backtransformed
        else:
            return points_np[mask]

    def trace_ray(self, ray: Ray, radius: float, return_sorted: bool = True, timer: bool = False) -> np.ndarray:
        """ 
        1. Checks global bounds of all files.
        2. Checks node bounds within hit files.
        3. Retrieves points (from Cache or Disk).
        4. Filters points within cylinder radius.
        
        Args:
            ray: Ray object defining origin and direction
            radius: Cylinder radius around the ray (must be positive)
            return_sorted: If True, returns points sorted along ray direction
            timer: If True, prints timing information
            
        Returns:
            np.ndarray: Array of shape (N, 3) containing filtered points
        """
        if radius <= 0:
            raise ValueError("Radius must be positive")
        
        #TODO transform and filter for each node instead of end, dont break sort logic
        all_points_list = []

        start1, start = 0.0, 0.0
        if timer:
            start1 = time.time()
            start = time.time()
        
        for file in self.files:
            # 1. Global Box Check
            if not file.intersect_global(ray, radius):
                continue

            # 2. Node Check
            hit_nodes = file.intersect_nodes(ray, radius)

            if timer:
                mid1 = round((time.time() - start) * 1000, 1)
                start = time.time()
                print(f"Nodes of file {path.basename(file.filepath)} intersected in {mid1} ms")
            
            # 3. Get points
            for node in hit_nodes:
                cache_key = (file.filepath, str(node.key))
                
                # Try to get from Cache
                points_data = self.cache.get(cache_key)
                
                if points_data is None:
                    # If not found, load and put in cache
                    points_data = file.get_points(node)
                    self.cache.put(cache_key, points_data)
                
                # to numpy (N, 3)
                node_pts = np.column_stack((points_data.x, points_data.y, points_data.z))
                all_points_list.append(node_pts)

            if timer:
                mid2 = round((time.time() - start) * 1000, 1)
                start = time.time()
                print(f"Points of intersected nodes in file {file.filepath} loaded in {mid2} ms")

        if not all_points_list:
            return np.empty((0, 3))

        # Concatenate all found points
        total_cloud = np.vstack(all_points_list)

        # 4. Transform and Filter (Cylinder check)
        filtered_points = self._transform_and_filter(total_cloud, ray, radius, return_sorted=return_sorted)

        if timer:
            mid3 = round((time.time() - start) * 1000, 1)
            end = round((time.time() - start1) * 1000, 1)
            print(f"Points transformed and filtered in {mid3} ms\n\nTotal time: {end} ms")

        return filtered_points

            

        







class MeeFile():
    ## Properties:
        # reader
        # config
        # las_header
        # filepath

        # node_entries
            # GetAllNodes()
        # global_bounds
            # (2, 3) array containing Min/Max of the file
        # global_bounds
            # Min/Max of file, same as root_box
        # is_indexed
            # True / False
            # Boxes have been constructed
        # _node_bounds_min / _node_bounds_max
            # (N, 3) Node minimums / maximums
            # None until first hit with ray

    ## Methods:
        # init
            # reads header, set global bounds
        # intersect_global(ray)
            # True if ray hits root_box
            # use in MeeStruct class
        # intersect_nodes(ray, radius)
            # check if indexed
            # if not, call build_index
            # do slab test
        # get_points(nodes)
        # build_index()
            # fills numpy arrays
            # set indexed to true
    
    """
    Represents a single COPC file. Handles reading, indexing, and bounding box logic.
    """
    def __init__(self, filepath: str):
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File not found: {filepath}")
        
        self.filepath = filepath
        self.reader = copc.FileReader(filepath)
        self.config = self.reader.copc_config
        self.las_header = self.config.las_header
        
        # Initialize node storage
        self.node_entries = self.reader.GetAllNodes()
        self.is_indexed = False
        
        # Global bounds of the file (Min/Max from header)
        self.global_bounds_min = np.array([
            self.config.copc_info.center_x,
            self.config.copc_info.center_y,
            self.config.copc_info.center_z,]) - self.config.copc_info.halfsize

        self.global_bounds_max = np.array([
            self.config.copc_info.center_x,
            self.config.copc_info.center_y,
            self.config.copc_info.center_z,]) + self.config.copc_info.halfsize
        
        # Arrays for vectorized node checks (initialized in build_index)
        self._node_bounds_min = None
        self._node_bounds_max = None

    def intersect_global(self, ray: Ray, radius: float = 1) -> bool:
        """
        Checks if the ray hits the bounding box of the entire file.
        """
        # Wrap single bounds in array shape (1, 3) for vectorized function
        b_min = self.global_bounds_min.reshape(1, 3)
        b_max = self.global_bounds_max.reshape(1, 3)
        
        hit = ray.slab_test_vectorized(b_min, b_max, radius)
        return hit[0]

    def build_index(self) -> None:
        """
        Constructs the numpy arrays required for vectorized intersection tests.
        Only runs once per file instance.
        """
        count = len(self.node_entries)
        self._node_bounds_min = np.zeros((count, 3))
        self._node_bounds_max = np.zeros((count, 3))

        for i, entry in enumerate(self.node_entries):
            # Create a Box to calculate dimensions (requires copclib logic)
            box = copc.Box(entry.key, self.las_header)
            self._node_bounds_min[i] = [box.x_min, box.y_min, box.z_min]
            self._node_bounds_max[i] = [box.x_max, box.y_max, box.z_max]
            
        self.is_indexed = True

    def intersect_nodes(self, ray: Ray, radius: float) -> list:
        """
        Returns a list of node entries that the ray intersects.
        Automatically builds index if not present.
        """
        if not self.is_indexed:
            self.build_index()

        # Type assertion: after build_index(), these arrays are guaranteed to be set
        assert self._node_bounds_min is not None and self._node_bounds_max is not None
        
        intersect_mask = ray.slab_test_vectorized(
            self._node_bounds_min, #type: ignore
            self._node_bounds_max, #type: ignore
            radius
        )
        
        # Filter the original entries list using the mask
        intersected_nodes = [self.node_entries[i] for i in range(len(self.node_entries)) if intersect_mask[i]]
        
        return intersected_nodes

    def get_points(self, node) -> copc.PointData:  # type: ignore
        """
        Wrapper to get points for a specific node using the reader.
        """
        return self.reader.GetPoints(node)











class Ray():
    """
    Klasse die einen Strahl anhand Ursprung und Richtung beschreibt.

    Attribute:
        direction, origin: np.ndarray
    """
    # TODO: allow list in init
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
        with np.errstate(divide='ignore'):
            self._inverse_direction = 1.0 / self.direction 

    def __str__(self):
        return f"Ray( Ursprung={self.origin}, direction={self.direction})"
        
    def slab_test(self, box: copc.Box, radius) -> bool:
        """Tests intersection against a single copclib.Box object
        returns bool"""
        t_min, t_max = 0.0, np.inf
        box_min = np.array([box.x_min - radius, box.y_min - radius, box.z_min - radius]) # berücksichtigt radius
        box_max = np.array([box.x_max + radius, box.y_max + radius, box.z_max + radius]) 
        
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

        # exit > 0 (box is not behind ray)
        return (t_enter < t_exit) #& (t_exit > 0)











class NodeCache():
    def __init__(self, max_size: int = 1000):
        self.cache = OrderedDict()
        self.max_size = max_size

    def get(self, key: tuple) -> copc.PointData | None:  # type: ignore
        """Returns node data if exists, moves it to the 'fresh' end of the list."""
        if key not in self.cache:
            return None
        self.cache.move_to_end(key) # Mark as recently used
        return self.cache[key]

    def put(self, key: tuple, data: copc.PointData) -> None:  # type: ignore
        """Adds data to cache. Removes oldest item if full."""
        if key in self.cache:
            self.cache.move_to_end(key)
        self.cache[key] = data
        
        # If exceed capacity, remove the oldest item
        if len(self.cache) > self.max_size:
            self.cache.popitem(last=False) # pops the oldest item

    def clear(self) -> None:
        """Clears all cache entries."""
        self.cache.clear()

    def remove_file(self, filepath: str) -> None:
        """Removes all cache entries associated with a specific file."""
        # Keys are tuples: (filepath, node_key)
        keys_to_delete = [k for k in self.cache.keys() if k[0] == filepath]
        for k in keys_to_delete:
            del self.cache[k]

    def __len__(self) -> int:
        return len(self.cache)











if __name__ == "__main__":

    # Argument parser erstellen
    parser = argparse.ArgumentParser(description="Ein Script welches alle Punkte in einem Radius um einen Bildstrahl aus einer copc Datei extrahiert")

    # Input file path
    parser.add_argument("-i", "--input", required=True, type=str,
                        help="Pfad zur copc Datei / Verzeichnis")
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

    struct = MeeStruct(args.input)

    punkte = struct.trace_ray(
        Ray(np.array(args.projektionszentrum),
        np.array(args.direction)),
        radius = args.radius,
        return_sorted=args.sorted,
        timer=True)
        
    np.savetxt(args.outputfile, punkte)