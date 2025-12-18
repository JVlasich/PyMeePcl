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
    """
    Manages multiple COPC files for efficient ray tracing operations.
    
    This class provides a unified interface to work with one or more COPC files, enabling
    efficient spatial queries using ray tracing. It maintains a cache of loaded point data
    and supports dynamic addition/removal of files.
    
    Attributes:
        cache (NodeCache): Cache for storing loaded node point data to reduce disk reads.
            Keys are tuples of (filepath, node_key_string), values are PointData objects.
        files (List[MeeFile]): List of MeeFile objects representing the loaded COPC files.
        bounds (tuple | None): Combined global bounding box of all files as a tuple of
            (minimums, maximums), each as numpy arrays of shape (3,). None if no files loaded.
    
    Methods:
        trace_ray(ray, radius, return_sorted, timer, recursive):
            Traces a ray through all loaded files and returns points within a cylindrical radius.
        add_file(filepath):
            Adds a new COPC file to the structure.
        del_file(filepath):
            Removes a file from the structure and clears associated cache entries.
        update_bounds():
            Recalculates the combined global bounds of all loaded files.
    """
    
    def __init__(self, source: str | List[str], cache_size: int = 25):
        """
        Initialize with a single path string, a list of strings, or a directory string.
        arg cache_size: maximum nodes to be loaded at once
        """
        self.cache = NodeCache(max_size=cache_size) # Key: (filepath, node_key_string), Value: PointData object
        self.bounds_copc = None
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
                pattern = "*.copc.laz"
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
        """Updates the combined global octree-bounds of all files."""
        if len(self.files) == 0:
            self.bounds_copc = None
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
        self.bounds_copc = (minimums, maximums)

    def add_file(self, filepath: str):
        """Adds a COPC file to the structure."""
        if filepath in [file.filepath for file in self.files]:
            print("File already added to Struct")
            return
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

    def trace_ray(self, ray: Ray, radius: float, return_sorted: bool = True, timer: bool = False, recursive: bool = False) -> np.ndarray:
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
            recursive: If True, uses recursive octree traversal
            
        Returns:
            np.ndarray: Array of shape (N, 3) containing filtered points
        """
        if radius <= 0:
            raise ValueError("Radius must be positive")
        
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
            if recursive:
                hit_nodes = file.intersect_nodes_octree(ray, radius)
            else:
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
                    if recursive:
                        points_data = file.get_points(node.copclib_node)
                    else:
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
    """
    Represents a single COPC file. Handles reading, indexing, and bounding box logic.

    Attributes:
        filepath (str): The path to the COPC file.
        reader (copclib.FileReader): The reader object for the COPC file.
        config (copclib.FileConfig): The configuration object for the COPC file.
        las_header (copclib.LasHeader): The header object for the LAS file.
        node_entries (List[copclib.Node]): A list of all nodes in the file.
        is_indexed (bool); is_octree_indexed (bool): True if the file has been indexed.
        octree_index (OctreeIndex | None): Octree structure. None until build_index_octree is called
        _node_bounds_* (np.ndarray): Node bounds of the file (min/max), None until build_index is called
        global_bounds_* (np.ndarray): Global bounds of the file. Differentiated between octree bounds and las bounds.

    Methods:
        intersect_global(ray: Ray, radius: float = 1) -> bool: Checks if ray hits the root box.
        intersect_nodes(ray: Ray, radius: float = 1) -> List[Node]: Returns nodes intersected by ray.
        intersect_nodes_octree(ray: Ray, radius: float = 1) -> List[OktreeNode]: Returns nodes intersected by ray.
        get_points(node: copclib.Node) -> PointData: Retrieves points for nodes.
        build_index(): Constructs numpy arrays for vectorized intersection tests.
        build_index_octree(): Constructs octree structure.
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

        self.global_bounds_min_las = np.array([self.las_header.min.x, self.las_header.min.y, self.las_header.min.z])
        self.global_bounds_max_las = np.array([self.las_header.max.x, self.las_header.max.y, self.las_header.max.z])
        
        # Arrays for vectorized node checks (initialized in build_index)
        self._node_bounds_min = None
        self._node_bounds_max = None
        
        # Octree structure (initialized in build_index_octree)
        self.is_octree_indexed = False
        self.octree_index = None

    def intersect_global(self, ray: Ray, radius: float = 1) -> bool:
        """
        Checks if the ray hits the bounding box of the entire file.
        """
        # Wrap single bounds in array shape (1, 3) for vectorized function
        b_min = self.global_bounds_min.reshape(1, 3)
        b_max = self.global_bounds_max.reshape(1, 3)
        
        hit = slab_test_vectorized(ray, b_min, b_max, radius)
        return hit[0]

    def build_index(self) -> None:
        """
        Constructs the numpy arrays required for vectorized intersection tests.
        Only runs once per file instance.
        """
        if self.is_indexed:
            return

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
        
        intersect_mask = slab_test_vectorized(
            ray,
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

    def build_index_octree(self) -> None:
        """
        Builds an octree structure using OctreeNode class.
        Constructs parent-child relationships from node_entries.
        Only runs once per file instance.
        """
        if self.is_octree_indexed:
            return

        self.octree_index = {}
        for node in self.node_entries:
            key = node.key
            octree_node = OctreeNode(key, self.las_header, self.reader)
            self.octree_index[key] = octree_node
        
        for octree_node in self.octree_index.values():
            key = octree_node.key
            if key.d == 0:
                continue

            parent_key = (key.d-1, key.x//2, key.y//2, key.z//2)
            parent_key = copc.VoxelKey(parent_key)
            
            # find parent in index and link to child
            parent_node = self.octree_index[parent_key]
            parent_node.children.append(octree_node)
            parent_node.children_bbox_min.append(octree_node.bounds_min)
            parent_node.children_bbox_max.append(octree_node.bounds_max)

        for octree_node in self.octree_index.values():
            octree_node.children_bbox_min = np.array(octree_node.children_bbox_min)
            octree_node.children_bbox_max = np.array(octree_node.children_bbox_max)
        
        
        self.is_octree_indexed = True

    def intersect_nodes_octree(self, ray: Ray, radius: float) -> list:
        """
        Hierarchical intersection test using octree structure.
        Skips checking child nodes if parent node doesn't intersect.
        Automatically builds octree index if not present.
        
        Returns a list of OktreeNode entries that the ray intersects.
        """
        if not self.is_octree_indexed:
            self.build_index_octree()
        assert self.octree_index

        # Create List to return, Root Node is always hit if this function is called
        result_entries = [self.octree_index[copc.VoxelKey((0,0,0,0))]]

        # Recursively traverse the children of hit nodes, start with root, populates list
        self._traverse_octree(result_entries[0], ray, radius, result_entries)

        return result_entries

    def _traverse_octree(self, node: OctreeNode, ray: Ray, radius: float, result_entries: list) -> None:
        assert (isinstance(node.children_bbox_min, np.ndarray) and isinstance(node.children_bbox_max, np.ndarray))
        # Tests the ray against all the children
        hits = slab_test_vectorized(
            ray,
            node.children_bbox_min,
            node.children_bbox_max,
            radius
        )

        intersected_nodes = [node.children[i] for i in range(len(node.children)) if hits[i]]
        for node in intersected_nodes:
            # Append to result list and call the function again if not a leaf
            result_entries.append(node)
            if len(node.children) > 0:
                self._traverse_octree(node, ray, radius, result_entries)

    def __repr__(self) -> str:
        return f"File({self.filepath})"












class Ray():
    """
    Klasse die einen Strahl anhand Ursprung und Richtung beschreibt.

    Attribute:
        direction, origin: np.ndarray
    """
    def __init__(self, origin: List[float] | np.ndarray, direction: List[float] | np.ndarray) -> None:
        """
        Initialize a Ray with origin and direction.
        
        Args:
            origin: Origin point as list [x, y, z] or numpy array of shape (3,)
            direction: Direction vector as list [dx, dy, dz] or numpy array of shape (3,)
        """
        # Convert lists to numpy arrays if needed
        if isinstance(origin, list):
            origin = np.array(origin)
        if isinstance(direction, list):
            direction = np.array(direction)
            
        if not ((origin.shape == (3,)) and (direction.shape == (3,))):
            raise ValueError("Argumente müssen jeweils 3 Dimensionen haben") 

        self.origin = origin
        self.direction = direction

    @property
    def origin(self) -> np.ndarray:
        return self._origin

    @origin.setter
    def origin(self, value: List[float] | np.ndarray):
        # Convert list to numpy array if needed
        if isinstance(value, list):
            value = np.array(value)
        if not isinstance(value, np.ndarray):
            raise TypeError("Ursprung muss ein NumPy ndarray oder eine Liste sein.")
        if value.shape != (3,):
            raise ValueError("Ursprung muss ein 3D vector (shape (3,)) sein")
        self._origin = value

    @property
    def direction(self) -> np.ndarray:
        return self._direction

    @direction.setter
    def direction(self, value: List[float] | np.ndarray):
        # Convert list to numpy array if needed
        if isinstance(value, list):
            value = np.array(value)
        if value.shape != (3,):
            raise ValueError("Richtung muss ein 3D vector (shape (3,)) sein")

        # Normalisiere den Richtungsvektor
        norm = np.linalg.norm(value)
        if np.isclose(norm, 0):
            raise ValueError("Richtung darf kein Nullvektor sein")
        self._direction = value / norm
        with np.errstate(divide='ignore'):
            self._inverse_direction = 1.0 / self.direction 

    def find_step_to_value(self, value: float, xyz: int) -> float:
        """
        Calculates the distance in meters along the ray until a given value is reached on the x/y/z axis.
        Parameters:
            value (float): The desired value to be reached.
            xyz (int): The dimension represented by the axis (0 for x, 1 for y, 2 for z).
        Returns:
            float: The number of steps (in meters) the ray needs to reach the given value.
        """
        assert xyz in (0,1,2)
        assert self.direction[xyz] != 0
        ergebnis = (value - self.origin[xyz]) / self.direction[xyz]
        return ergebnis 

    def __str__(self):
        return f"Ray( Ursprung={self.origin}, direction={self.direction})"
    
    def __repr__(self):
        return f"Ray( Ursprung={self.origin}, direction={self.direction})"











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
    
    def __repr__(self) -> str:
        return f"NodeCache({len(self.cache)})"











class OctreeNode():
    def __init__(self, key: tuple | copc.VoxelKey, las_header: copc.LasHeader, reader: copc.FileReader):
        if isinstance(key, (tuple, list)):
            self.key = copc.VoxelKey(key)
        elif isinstance(key, copc.VoxelKey):
            self.key = key
        else: raise TypeError("key has to be a tuple, list or copc.Voxelkey")

        if isinstance(las_header, copc.LasHeader):
            self.las_header = las_header
        else: raise TypeError("las_header has to be of type: copc.LasHeader")

        self.copclib_node = reader.FindNode(key)
        
        self.box = copc.Box(key, las_header)
        self.bounds_min = [self.box.x_min, self.box.y_min, self.box.z_min]
        self.bounds_max = [self.box.x_max, self.box.y_max, self.box.z_max]
        
        self.children = []
        self.children_bbox_min = []
        self.children_bbox_max = []


    def __str__(self) -> str:
        return f"OctreeNode({self.key}) nChildren: {len(self.children)}"

    def __repr__(self) -> str:
        return f"OctreeNode({self.key}) nChildren: {len(self.children)}"











def slab_test(ray, box: copc.Box, radius) -> bool:
    """Tests intersection against a single copclib.Box object
    returns bool"""
    t_min, t_max = 0.0, np.inf
    box_min = np.array([box.x_min - radius, box.y_min - radius, box.z_min - radius]) # berücksichtigt radius
    box_max = np.array([box.x_max + radius, box.y_max + radius, box.z_max + radius]) 
    
    t1 = (box_min - ray.origin) * ray._inverse_direction
    t2 = (box_max - ray.origin) * ray._inverse_direction

    # minimaler und maximaler t-Wert pro Achse
    # np.minimum/maximum vergleicht Elementweise
    t_near = np.minimum(t1, t2)
    t_far = np.maximum(t1, t2)

    t_min = np.max(t_near) # spätest möglicher Eintrittszeitpunkt
    t_max = np.min(t_far) # frühest möglicher Austrittszeitpunkt
    
    return (t_min < t_max)
    
def slab_test_vectorized(ray, boxes_min: np.ndarray, boxes_max: np.ndarray, radius: float) -> np.ndarray:
    """
    Tests intersection against given boxes simultaneously.
    Returns a boolean mask.
    """
    # Expand boxes by radius
    b_min = boxes_min - radius
    b_max = boxes_max + radius

    # Vectorized calculation (N, 3)
    t1 = (b_min - ray.origin) * ray._inverse_direction
    t2 = (b_max - ray.origin) * ray._inverse_direction

    # Find t_near and t_far
    t_near = np.minimum(t1, t2)
    t_far = np.maximum(t1, t2)

    # Max of nearest, Min of farthest (Collapse to shape (N,))
    t_enter = np.max(t_near, axis=1)
    t_exit = np.min(t_far, axis=1)

    # exit > 0 (box is not behind ray)
    return (t_enter < t_exit) #& (t_exit > 0)











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
        Ray(np.array(args.projektionszentrum), np.array(args.direction)),
        radius = args.radius,
        return_sorted=args.sorted,
        timer=True)
        
    np.savetxt(args.outputfile, punkte)