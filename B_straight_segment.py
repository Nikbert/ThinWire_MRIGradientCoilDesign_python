import numpy as np

def B_straight_segment(Pstart, Pend, Points):
    """
    Calculate the magnetic field of a straight wire defined by its start and end points at the given locations.
    
    Args:
        Pstart (ndarray): Start point of the wire
        Pend (ndarray): End point of the wire
        Points (ndarray): Locations where the magnetic field is calculated
        
    Returns:
        B (ndarray): Magnetic field at the given locations
        
    """
    # Constants
    mu_0 = 4 * np.pi * 1e-7
    unit_scale = mu_0 / (4 * np.pi)  # SI units
    epsilon = 1e-12
    
    if np.linalg.norm(Pstart - Pend) < epsilon:
        B = np.zeros(3)
    else:
        a = Pend[np.newaxis, :] - Points
        b = Pstart[np.newaxis, :] - Points
        c = Pend[np.newaxis, :] - Pstart[np.newaxis, :]
        
        an = normalize(a)
        bn = normalize(b)
        cn = normalize(c)
        
        d = np.linalg.norm(np.cross(a, cn), axis=1)
        IndNon0 = np.where(d > epsilon)[0]
        inv_d = np.zeros_like(d)
        inv_d[IndNon0] = 1. / d[IndNon0]
        
        Babs = unit_scale * (np.dot(an, cn.T) - np.dot(bn, cn.T)) * inv_d
        
        Dir_B = np.cross(an, cn)
        LenDir_B = np.linalg.norm(Dir_B, axis=1)
        
        # Fix zero norm cases
        NormDir_B = np.zeros_like(Dir_B)
        NormDir_B[IndNon0] = Dir_B[IndNon0] / LenDir_B[IndNon0, np.newaxis]
        
        B = NormDir_B * Babs[:, np.newaxis]
    
    return B

#Are the following functions used at all?  normalize is! replace?
def vnorm(vin):
    """
    Norm of the vector (assuming the second dimension to be the vector coordinates).
    """
    return np.sqrt(np.dot(vin, vin.T))

def vdot(v1, v2):
    """
    Dot product of two vectors (assuming the second dimension to be the vector coordinates).
    """
    return np.dot(v1, v2.T)

def vcross(v1, v2):
    """
    Cross product of two vectors (assuming the second dimension to be the vector coordinates).
    """
    return np.cross(v1, v2)

def normalize(vin):
    """
    Normalized vector (assuming the second dimension to be the vector coordinates).
    """
    vn = np.sqrt(np.dot(vin, vin.T))
    return vin / vn[:, np.newaxis]

