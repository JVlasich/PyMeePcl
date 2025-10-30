import copclib as copc
import numpy as np


class Ray():
    """
    Klasse die einen Strahl anhand Ursprung und Richtung beschreibt.

    Attribute:
        direction, strahl: np.ndarray
    """
    def __init__(self, origin: np.ndarray, direction: np.ndarray) -> None:
        if not ((origin.shape == (3,)) and (direction.shape == (3,))):
            raise ValueError("Argumente müssen jeweils 3 Dimensionen haben") 

        self.origin = origin
        self.direction = direction

    @property
    def origin(self) -> np.ndarray:
        """Get origin"""
        return self._origin

    @origin.setter
    def origin(self, value: np.ndarray):
        """Set origin und validiere"""
        if not isinstance(value, np.ndarray):
            raise TypeError("Ursprung muss ein NumPy ndarray sein.")
        if value.shape != (3,):
            raise ValueError("Ursprung muss ein 3D vector (shape (3,)) sein")
        self._origin = value

    @property
    def direction(self) -> np.ndarray:
        """Get direction"""
        return self._direction

    @direction.setter
    def direction(self, value: np.ndarray):
        """Set, validiere und normalisiere direction"""
        if not isinstance(value, np.ndarray):
            raise TypeError("direction muss ein NumPy ndarray sein")
        if value.shape != (3,):
            raise ValueError("direction muss ein 3D vector (shape (3,)) sein")

        # Normalisiere den directionsvektor
        norm = np.linalg.norm(value)
        if np.isclose(norm, 0):
            raise ValueError("direction darf kein Nullvektor sein")
        self._direction = value / norm

    def point_at_parameter(self, t: float) -> np.ndarray:
        """
        Gibt den Punkt entlang des Strahles nach t metern wieder
        """
        return self.origin + t * self.direction

    def __repr__(self):
        return f"Ray( Ursprung={self.origin}, direction={self.direction})"

    def draw_ray(self, ax, t):
        """Plottet den Strahl als t-meter langes Liniensegment auf eine übergebene axes"""
        start_point = self.origin
        end_point = self.point_at_parameter(t)
    
        # Plotte den Ursprung als Punkt
        ax.scatter(start_point[0], start_point[1], start_point[2], color='green', s=100, label='Origin')
    
        # Plotte den Strahl als Liniensegment
        ax.plot([start_point[0], end_point[0]], 
                [start_point[1], end_point[1]], 
                [start_point[2], end_point[2]], 
                color='red', label=f'Ray (length={t})')