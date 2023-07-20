import numpy as np

def TikhonovReg_Weigh(ElementFields, TargetField, lambda_val, w):
    """
    Tikhonov Regularization with Weighting
    
    Args:
        ElementFields (ndarray): Element fields
        TargetField (ndarray): Target field
        lambda_val (float): Regularization parameter
        w (float): Weight
    
    Returns:
        Currents (ndarray): Computed currents
        
    """
    AtA = ElementFields.T @ ElementFields
    G = lambda_val * np.linalg.norm(ElementFields) * w
    GtG = G.T @ G
    Currents = np.linalg.pinv(AtA + GtG) @ ElementFields.T @ TargetField
    
    return Currents

