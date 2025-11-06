import copclib as copc

def find_nodes_r(reader, key, nodes_with_points):

    node = reader.FindNode(key)
    if node and (node.point_count > 0): # Node existiert und ist nicht leer
        nodes_with_points.append(key)

    # Rekursiv Children Nodes suchen
    if node:
        for i in range(8): # 8 potentielle children
            child_key = key.GetChildren()[i]
            if child_key.d <= (reader.GetMaxDepth()): # bis zur maximalen tiefe
                 find_nodes_r(reader, child_key, nodes_with_points)


def get_all_nodes(file_path):
    """Sucht rekursiv alle Nodes einer Copc Datei und gibt sie als Liste[copc.Voxelkey] zurück"""
    reader = copc.FileReader(file_path)
    nodes_with_points = []
    root_key = copc.VoxelKey(0, 0, 0, 0)
    find_nodes_r(reader, root_key, nodes_with_points)
    return nodes_with_points