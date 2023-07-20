import numpy as np

def TikhonovReg(ElementFields, TargetField, lambda_val):
    """
    Tikhonov Regularization
    
    Args:
        ElementFields (ndarray): Element fields
        TargetField (ndarray): Target field
        lambda_val (float): Regularization parameter
    
    Returns:
        Currents (ndarray): Computed currents
        
    """
    AtA = ElementFields.T @ ElementFields
    G = lambda_val * np.linalg.norm(ElementFields) * np.eye(ElementFields.shape[1])
    GtG = G.T @ G
    Currents = np.linalg.pinv(AtA + GtG) @ ElementFields.T @ TargetField
    
    return Currents

