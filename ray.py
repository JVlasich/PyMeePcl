import copclib as copc
import numpy as np


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

    def point_at_parameter(self, t: float) -> np.ndarray:
        """Gibt den Punkt entlang des Strahles nach t metern wieder"""
        return self.origin + t * self.direction

    def __str__(self):
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
        
    def slab_test(self, box: copc.Box) -> bool:
        t_min, t_max = 0.0, np.inf
        box_min = np.array([box.x_min, box.y_min, box.z_min])
        box_max = np.array([box.x_max, box.y_max, box.z_max])

        #Für jede der drei Dimensionen, nicht vektorisiert
        #for d in range(3):
        #    t1 = (box_min[d] - self.origin[d]) * self._inverse_direction[d]
        #    t2 = (box_max[d] - self.origin[d]) * self._inverse_direction[d]
        #    t_min = max(t_min, min(t1, t2))
        #    t_max = min(t_max, max(t1, t2))
        
        t1 = (box_min - self.origin) * self._inverse_direction
        t2 = (box_max - self.origin) * self._inverse_direction

        # minimaler und maximaler t-Wert pro Achse
        # np.minimum/maximum vergleicht Elementweise
        t_near = np.minimum(t1, t2)
        t_far = np.maximum(t1, t2)

        t_min = np.max(t_near) # spätest möglicher Eintrittszeitpunkt
        t_max = np.min(t_far) # frühest möglicher Austrittszeitpunkt
        
        return (t_min < t_max)