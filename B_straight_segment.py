import numpy as np

def B_straight_segment(Pstart, Pend, Points):
    #print('Pstart',Pstart)
    #print('Pend',Pend)
   # print('Points.shape',Points.shape)
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
        # Calculate the difference matrix 'a'
        a = np.ones((Points.shape[0], 1)) * Pend - Points
    
        # Calculate the difference matrix 'b'
        b = np.ones((Points.shape[0], 1)) * Pstart - Points
    
        # Calculate the difference matrix 'c'
        c = np.ones((Points.shape[0], 1)) * Pend - np.ones((Points.shape[0], 1)) * Pstart

        #print('Pend[np.newaxis, :]',Pend[np.newaxis, :])
        #print('Pstart[np.newaxis, :]',Pstart[np.newaxis, :])
        #print('c',c.shape)
        
        an = normalize(a)
        #an = np.norm
        #print('an',an.shape)
        bn = normalize(b)
        #print('bn',bn.shape)
        cn = normalize(c)
        #print('cn',cn.shape)
       
    
        # Assuming you have numpy arrays 'a', 'bn', 'cn', and 'an' defined already.
        # If not, make sure to define them before this code block.
    
        # Calculate 'd'
        d = np.linalg.norm(np.cross(a, cn), axis=1)
    
        # Find indices where 'd' is greater than 'epsilon'
        IndNon0 = np.where(d > epsilon)[0]
    
        # Initialize 'inv_d' with zeros
        inv_d = np.zeros(d.shape)
    
        # Set non-zero values of 'inv_d' based on 'IndNon0'
        inv_d[IndNon0] = 1.0 / d[IndNon0]
    
        # Calculate 'Babs'
        Babs = unit_scale * (vdot(an,cn) - vdot(bn,cn)) * inv_d #; matlab
        #Babs = unit_scale * (np.dot(an, cn.T) - np.dot(bn, cn.T)) * inv_d
        #Babs = unit_scale * (np.dot(an, cn) - np.dot(bn, cn)) * inv_d
    
        # Calculate 'Dir_B'
        Dir_B = np.cross(an, cn)
    
        # Calculate 'LenDir_B'
        LenDir_B = np.linalg.norm(Dir_B, axis=1)
    
        # Initialize 'NormDir_B' with zeros
        NormDir_B = np.zeros(Dir_B.shape)
    
        # Fix zero norm cases
        NormDir_B[IndNon0] = Dir_B[IndNon0] / LenDir_B[IndNon0, None]
    
        # Calculate 'B'
        B = NormDir_B * Babs[:, None]

    return B

#Are the following functions used at all?  normalize is! replace?
def vnorm(vin):
    """
    Norm of the vector (assuming the second dimension to be the vector coordinates).
    """
    return np.sqrt(np.dot(vin, vin.T))

#def vdot(v1, v2):
#    """
#    Dot product of two vectors (assuming the second dimension to be the vector coordinates).
#    """
#    return np.dot(v1, v2.T)

def vcross(v1, v2):
    """
    Cross product of two vectors (assuming the second dimension to be the vector coordinates).
    """
    return np.cross(v1, v2)

def normalize(x):
    norm = x / np.linalg.norm(x)
    return norm

def vdot(A, B):
    D = np.sum(A * B, axis=1)
    return D
