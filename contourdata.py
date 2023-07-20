import numpy as np

def contourdata(c):
    """
    Extract Contour Data from Contour Matrix C.
    
    Args:
        c (ndarray): Contour matrix
        
    Returns:
        s (list): List of contour data
        
    """
    if c.ndim != 2 or c.shape[0] != 2 or c.shape[1] < 4 or not np.issubdtype(c.dtype, np.floating):
        raise ValueError("Input must be a 2-by-N contour matrix 'c'")
    
    tol = 1e-12
    k = 0  # Contour line number
    col = 0  # Index of column containing contour level and number of points
    s = []  # List of contour data
    
    while col < c.shape[1]:
        s.append({
            'level': c[0, col],
            'numel': c[1, col],
            'xdata': c[0, col + 1:col + 1 + c[1, col]],
            'ydata': c[1, col + 1:col + 1 + c[1, col]],
            'isopen': (np.abs(np.diff(c[0, col + 1:col + 2 + c[1, col]])) > tol) or
                      (np.abs(np.diff(c[1, col + 1:col + 2 + c[1, col]])) > tol)
        })
        k += 1
        col += c[1, col] + 1
    
    return s

