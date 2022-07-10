import numpy as np
def ImageDistance(im1,im2,order = 2):
    return np.linalg.norm(im1 - im2,ord=order)