import numpy as np

def linear_kernel(x, y=None, params={}):
    
    if y is not None:
        output = np.dot(x, y)
    else:
        output = np.dot(x, x.T)   
    return output

def polynomial_kernel(x, y=None, params={}):
    
    delta = params.get('delta', 2.)
    gamma = params.get('gamma', 1.2)
    
    if y is not None:
        output = (gamma + np.dot(x, y)) ** delta
    else:
        output = (gamma + np.dot(x, x.T)) ** delta
    return output

def rbf_kernel(x, y=None, params={}):
    
    gamma = params.get('gamma', 1.)
    
    if y is not None:
        d = x - y
        output = np.exp(-gamma * np.dot(d, d))
    else: 
        x_norm = np.sum(x ** 2., axis = -1)
        output = np.exp(-gamma * (x_norm[:,None] + x_norm[None,:] - 2. * np.dot(x, x.T)))
    return output

def gaussian_kernel(x, y=None, params={}):
    
    gamma = params.get('gamma', 1.)
    
    if y is not None:
        output = np.exp(-np.linalg.norm(x - y) / (2. * gamma ** 2.))
    else: 
        x_norm = -np.linalg.norm(x[:, None] - x[None, :], axis=2)
        output = np.exp(x_norm / (2. * gamma ** 2.))
    return output