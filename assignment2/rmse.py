import numpy as np

def rmse(predictions, targets):
    pred = np.array(predictions)
    tar = np.array(targets)
    #rmse = None # TODO: Implement RMSE Calculation here...
    n = len(tar)
    rmse = np.sqrt(np.sum((pred-tar)**2)/n)
    return rmse