import numpy as np
from .line_representation import BoundingBox, Line3D

class BVHNode:
    """
    Node in a Bounding Volume Hierarchy (BVH) tree.
    Each node has a bounding box that contains all lines in its subtree.
    Leaf nodes contain a small number of lines, while internal nodes have two children.
    """
    def __init__(self, lines=None, index=None):
        """
        Initialize a BVH node.
        
        Args:
            lines (list, optional): List of Line3D objects contained in this node
            index (int, optional): Index of this node in the BVH
        """
        self.box = None
        self.left = None
        self.right = None
        self.lines = lines
        self.index = index
        self.is_leaf = False
    
    def __repr__(self):
        if self.is_leaf:
            return f"Leaf(lines={len(self.lines) if self.lines else 0})"
        return f"Node(left={self.left}, right={self.right})"

class BVH:
    """
    Bounding Volume Hierarchy (BVH) for efficiently querying line intersections.
    """
    def __init__(self, lines, max_lines_per_leaf=10, volume_bounds=None):
        """
        Build a BVH for a set of lines.
        
        Args:
            lines (list): List of Line3D objects
            max_lines_per_leaf (int): Maximum number of lines in a leaf node
            volume_bounds (tuple, optional): Bounds of the volume (min_bound, max_bound)
        """
        self.lines = lines
        self.max_lines_per_leaf = max_lines_per_leaf
        
        # If volume bounds are not provided, use a default volume
        if volume_bounds is None:
            volume_min = np.zeros(3, dtype=np.float32)
            volume_max = np.ones(3, dtype=np.float32) * 100.0  # 100mm in each dimension
        else:
            volume_min, volume_max = volume_bounds
        
        self.volume_min = volume_min
        self.volume_max = volume_max
        
        # Build line bounds
        self.line_bounds = []
        for line in lines:
            min_bound, max_bound = line.get_bounds(volume_min, volume_max)
            self.line_bounds.append((min_bound, max_bound))
        
        # Build the BVH tree
        self.root = self._build(list(range(len(lines))))
    
    def _build(self, line_indices, depth=0):
        """
        Recursively build the BVH tree.
        
        Args:
            line_indices (list): Indices of lines to include in this subtree
            depth (int): Current depth in the tree
            
        Returns:
            BVHNode: Root node of the subtree
        """
        node = BVHNode()
        
        # Compute bounding box for all lines
        min_bound = np.ones(3, dtype=np.float32) * float('inf')
        max_bound = np.ones(3, dtype=np.float32) * float('-inf')
        
        for idx in line_indices:
            line_min, line_max = self.line_bounds[idx]
            min_bound = np.minimum(min_bound, line_min)
            max_bound = np.maximum(max_bound, line_max)
        
        node.box = BoundingBox(min_bound, max_bound)
        
        # If few enough lines, make a leaf node
        if len(line_indices) <= self.max_lines_per_leaf:
            node.lines = [self.lines[idx] for idx in line_indices]
            node.is_leaf = True
            return node
        
        # Choose the axis with the largest spread to split on
        extents = max_bound - min_bound
        axis = np.argmax(extents)
        
        # Sort line indices by their centroid along the chosen axis
        def get_centroid(idx):
            line_min, line_max = self.line_bounds[idx]
            return (line_min[axis] + line_max[axis]) / 2.0
        
        line_indices.sort(key=get_centroid)
        
        # Split at the median
        mid = len(line_indices) // 2
        
        # Recursively build children
        node.left = self._build(line_indices[:mid], depth + 1)
        node.right = self._build(line_indices[mid:], depth + 1)
        
        return node
    
    def query_intersections(self, other_bvh, tolerance=1.0):
        """
        Find all potential line intersections between this BVH and another.
        
        Args:
            other_bvh (BVH): Other BVH to intersect with
            tolerance (float): Tolerance for intersection testing
            
        Returns:
            list: List of tuples (line1, line2) representing potential intersections
        """
        candidates = []
        self._query_recursive(self.root, other_bvh.root, candidates, tolerance)
        return candidates
    
    def _query_recursive(self, node1, node2, candidates, tolerance):
        """
        Recursively query for intersections between two BVH nodes.
        
        Args:
            node1 (BVHNode): Node from this BVH
            node2 (BVHNode): Node from the other BVH
            candidates (list): List to store candidates in
            tolerance (float): Tolerance for intersection testing
        """
        # Expand boxes by tolerance and check if they intersect
        box1 = BoundingBox(node1.box.min_bound, node1.box.max_bound)
        box2 = BoundingBox(node2.box.min_bound, node2.box.max_bound)
        
        expanded_box1 = box1.expand(tolerance)
        expanded_box2 = box2.expand(tolerance)
        
        if not expanded_box1.intersects(expanded_box2):
            return
        
        # If both are leaves, check all pairs of lines
        if node1.is_leaf and node2.is_leaf:
            for line1 in node1.lines:
                for line2 in node2.lines:
                    # Only consider lines from different planes
                    if line1.plane_id is not None and line2.plane_id is not None and line1.plane_id == line2.plane_id:
                        continue
                    candidates.append((line1, line2))
            return
        
        # Recursively descend into smaller node first (heuristic)
        if node1.is_leaf:
            if node2.left:
                self._query_recursive(node1, node2.left, candidates, tolerance)
            if node2.right:
                self._query_recursive(node1, node2.right, candidates, tolerance)
        elif node2.is_leaf:
            if node1.left:
                self._query_recursive(node1.left, node2, candidates, tolerance)
            if node1.right:
                self._query_recursive(node1.right, node2, candidates, tolerance)
        else:
            # Both are internal nodes, traverse all four combinations
            if node1.left and node2.left:
                self._query_recursive(node1.left, node2.left, candidates, tolerance)
            if node1.left and node2.right:
                self._query_recursive(node1.left, node2.right, candidates, tolerance)
            if node1.right and node2.left:
                self._query_recursive(node1.right, node2.left, candidates, tolerance)
            if node1.right and node2.right:
                self._query_recursive(node1.right, node2.right, candidates, tolerance)

def create_bvh_for_lines(lines, max_lines_per_leaf=10, volume_bounds=None):
    """
    Create a BVH for a list of lines.
    
    Args:
        lines (list): List of Line3D objects
        max_lines_per_leaf (int): Maximum number of lines in a leaf node
        volume_bounds (tuple, optional): Bounds of the volume (min_bound, max_bound)
        
    Returns:
        BVH: Bounding Volume Hierarchy for the lines
    """
    return BVH(lines, max_lines_per_leaf, volume_bounds)

def find_potential_intersections(lines_by_plane, tolerance=1.0, volume_bounds=None):
    """
    Find potential intersections between lines from different planes.
    
    Args:
        lines_by_plane (dict): Dictionary mapping plane_id to list of Line3D objects
        tolerance (float): Tolerance for intersection testing
        volume_bounds (tuple, optional): Bounds of the volume (min_bound, max_bound)
        
    Returns:
        list: List of tuples (line1, line2) representing potential intersections
    """
    # Create BVHs for each plane
    bvhs = {}
    for plane_id, lines in lines_by_plane.items():
        bvhs[plane_id] = create_bvh_for_lines(lines, volume_bounds=volume_bounds)
    
    # Find potential intersections between planes
    candidates = []
    plane_ids = sorted(bvhs.keys())
    
    for i in range(len(plane_ids)):
        for j in range(i + 1, len(plane_ids)):
            plane_id1 = plane_ids[i]
            plane_id2 = plane_ids[j]
            
            bvh1 = bvhs[plane_id1]
            bvh2 = bvhs[plane_id2]
            
            plane_candidates = bvh1.query_intersections(bvh2, tolerance)
            candidates.extend(plane_candidates)
    
    return candidates 