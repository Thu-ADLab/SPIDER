import numpy as np

def cart2polar(x, y):
    r = np.sqrt(x**2 + y**2)
    theta = np.atan2(y, x)
    return r, theta

def polar2cart(r, theta):
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return x, y



class PolarTransformer:
    def __init__(self):
        pass
    
    def cart2polar(self, x, y, vx=None, vy=None, ax=None, ay=None, *, order=0):
        r = np.sqrt(x**2 + y**2)
        theta = np.atan2(y, x)
        return r, theta
    
    def polar2cart(self, r, theta):
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        return x, y
