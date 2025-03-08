import numpy as np
import torch

class Line3D:
    """
    Class representing a 3D line in the form of a point P and a direction vector d.
    The line is defined as all points X where X = P + t * d for some scalar t.
    """
    def __init__(self, point, direction, plane_id=None):
        """
        Initialize a 3D line.
        
        Args:
            point (np.ndarray or torch.Tensor): A point on the line (3D vector)
            direction (np.ndarray or torch.Tensor): The direction vector of the line (3D vector)
            plane_id (int, optional): The ID of the wire plane this line comes from
        """
        # Convert inputs to numpy arrays if they are not already
        if isinstance(point, torch.Tensor):
            self.point = point.cpu().numpy() if point.is_cuda else point.numpy()
        else:
            self.point = np.asarray(point, dtype=np.float32)
            
        if isinstance(direction, torch.Tensor):
            self.direction = direction.cpu().numpy() if direction.is_cuda else direction.numpy()
        else:
            self.direction = np.asarray(direction, dtype=np.float32)
        
        # Normalize the direction vector
        norm = np.linalg.norm(self.direction)
        if norm > 0:
            self.direction /= norm
        
        self.plane_id = plane_id
    
    def point_at(self, t):
        """
        Get a point on the line at parameter t.
        
        Args:
            t (float): Parameter value
            
        Returns:
            np.ndarray: Point on the line
        """
        return self.point + t * self.direction
    
    def closest_point_to(self, point):
        """
        Find the closest point on the line to the given point.
        
        Args:
            point (np.ndarray): 3D point
            
        Returns:
            np.ndarray: Closest point on the line to the given point
        """
        v = point - self.point
        t = np.dot(v, self.direction)
        return self.point_at(t)
    
    def distance_to_point(self, point):
        """
        Calculate the minimum distance from the line to a point.
        
        Args:
            point (np.ndarray): 3D point
            
        Returns:
            float: Minimum distance from the line to the point
        """
        closest = self.closest_point_to(point)
        return np.linalg.norm(point - closest)
    
    def get_bounds(self, volume_min, volume_max):
        """
        Get the bounding box of the line segment within the given volume.
        
        Args:
            volume_min (np.ndarray): Minimum corner of the volume (3D vector)
            volume_max (np.ndarray): Maximum corner of the volume (3D vector)
            
        Returns:
            tuple: (min_bound, max_bound) representing the bounding box of the line segment
        """
        # Calculate the intersection of the line with each of the 6 faces of the volume
        t_values = []
        
        for i in range(3):  # For x, y, z dimensions
            # Check intersection with min face
            if self.direction[i] != 0:
                t = (volume_min[i] - self.point[i]) / self.direction[i]
                p = self.point_at(t)
                if all(volume_min[j] <= p[j] <= volume_max[j] for j in range(3)):
                    t_values.append(t)
            
            # Check intersection with max face
            if self.direction[i] != 0:
                t = (volume_max[i] - self.point[i]) / self.direction[i]
                p = self.point_at(t)
                if all(volume_min[j] <= p[j] <= volume_max[j] for j in range(3)):
                    t_values.append(t)
        
        # If the line doesn't intersect the volume, return an empty bounding box
        if not t_values:
            return np.array([0, 0, 0]), np.array([0, 0, 0])
        
        # Sort t values to get the segment of the line inside the volume
        t_values.sort()
        t_min, t_max = t_values[0], t_values[-1]
        
        # Calculate the endpoints of the line segment inside the volume
        p_min = self.point_at(t_min)
        p_max = self.point_at(t_max)
        
        # Calculate the bounding box
        min_bound = np.minimum(p_min, p_max)
        max_bound = np.maximum(p_min, p_max)
        
        return min_bound, max_bound

class BoundingBox:
    """
    Class representing an axis-aligned bounding box (AABB) in 3D.
    """
    def __init__(self, min_bound, max_bound):
        """
        Initialize a bounding box.
        
        Args:
            min_bound (np.ndarray): Minimum corner of the bounding box (3D vector)
            max_bound (np.ndarray): Maximum corner of the bounding box (3D vector)
        """
        self.min_bound = np.asarray(min_bound, dtype=np.float32)
        self.max_bound = np.asarray(max_bound, dtype=np.float32)
    
    def expand(self, amount):
        """
        Expand the bounding box by a fixed amount in all directions.
        
        Args:
            amount (float): Amount to expand the bounding box by
            
        Returns:
            BoundingBox: Expanded bounding box
        """
        return BoundingBox(
            self.min_bound - amount,
            self.max_bound + amount
        )
    
    def intersects(self, other):
        """
        Check if this bounding box intersects with another.
        
        Args:
            other (BoundingBox): Other bounding box
            
        Returns:
            bool: True if the bounding boxes intersect, False otherwise
        """
        return all(self.min_bound[i] <= other.max_bound[i] and 
                   other.min_bound[i] <= self.max_bound[i] for i in range(3))
    
    def contains_point(self, point):
        """
        Check if this bounding box contains a point.
        
        Args:
            point (np.ndarray): 3D point
            
        Returns:
            bool: True if the bounding box contains the point, False otherwise
        """
        return all(self.min_bound[i] <= point[i] <= self.max_bound[i] for i in range(3)) 