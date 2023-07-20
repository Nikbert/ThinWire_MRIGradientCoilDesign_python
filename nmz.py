import numpy as np

def nmz(data):
    """
    Normalize the input data between 0 and 1.
    
    Args:
        data (ndarray): Input data
        
    Returns:
        out (ndarray): Normalized data
        
    """
    dat = np.reshape(data, 1, -1)
    minData = np.min(dat)
    dat = dat - minData
    newMaxData = np.max(dat)
    out = (data - minData) / newMaxData
    
    return out

