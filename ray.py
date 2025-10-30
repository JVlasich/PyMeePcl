import copclib as copc
import numpy as np


class Ray():
    """
    Klasse die einen Strahl anhand Ursprung und Richtung beschreibt.

    Attribute:
        richtung, strahl: np.ndarray
    """
    def __init__(self, ursprung: np.ndarray, richtung: np.ndarray) -> None:
        if not ((ursprung.shape == (3,)) and (richtung.shape == (3,))):
            raise ValueError("Argumente müssen jeweils 3 Dimensionen haben") 

        self.ursprung = ursprung
        self.richtung = richtung

    @property
    def ursprung(self) -> np.ndarray:
        """Get ursprung"""
        return self._ursprung

    @ursprung.setter
    def ursprung(self, value: np.ndarray):
        """Set ursprung und validiere"""
        if not isinstance(value, np.ndarray):
            raise TypeError("ursprung muss ein NumPy ndarray sein.")
        if value.shape != (3,):
            raise ValueError("ursprung muss ein 3D vector (shape (3,)) sein")
        self._ursprung = value

    @property
    def richtung(self) -> np.ndarray:
        """Get richtung"""
        return self._richtung

    @richtung.setter
    def richtung(self, value: np.ndarray):
        """Set, validiere und normalisiere richtung"""
        if not isinstance(value, np.ndarray):
            raise TypeError("richtung muss ein NumPy ndarray sein")
        if value.shape != (3,):
            raise ValueError("richtung muss ein 3D vector (shape (3,)) sein")

        # Normalisiere den Richtungsvektor
        norm = np.linalg.norm(value)
        if np.isclose(norm, 0):
            raise ValueError("Richtung darf kein Nullvektor sein")
        self._richtung = value / norm

    def point_at_parameter(self, t: float) -> np.ndarray:
        """
        Gibt den Punkt entlang des Strahles nach t metern wieder
        """
        return self.ursprung + t * self.richtung

    def __repr__(self):
        return f"Ray( Ursprung={self.ursprung}, Richtung={self.richtung})"

    def draw_ray(self, ax, t):
        """Plottet den Strahl als t-meter langes Liniensegment auf eine übergebene axes"""
        start_point = self.ursprung
        end_point = self.point_at_parameter(t)
    
        # Plotte den Ursprung als Punkt
        ax.scatter(start_point[0], start_point[1], start_point[2], color='green', s=100, label='Origin')
    
        # Plotte den Strahl als Liniensegment
        ax.plot([start_point[0], end_point[0]], 
                [start_point[1], end_point[1]], 
                [start_point[2], end_point[2]], 
                color='red', label=f'Ray (length={t})')

        

# test = Ray(np.array([1,2,3]), np.array([10,10,10]))
# print(test)
