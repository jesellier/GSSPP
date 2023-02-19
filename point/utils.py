
import numpy as np
import numbers

from numpy import genfromtxt
import matplotlib.pyplot as plt

import tensorflow as tf

from data import coal_dataset



def check_random_state_instance(seed):
    """Turn seed into a np.random.RandomState instance

    Parameters
    ----------
    seed : None | int | instance of RandomState
        If seed is None, return the RandomState singleton used by np.random.
        If seed is an int, return a new RandomState instance seeded with seed.
        If seed is already a RandomState instance, return it.
        Otherwise raise ValueError.
    """
    if seed is None or seed is np.random:
        return np.random.mtrand._rand
    if isinstance(seed, numbers.Integral):
        return np.random.RandomState(seed)
    if isinstance(seed, np.random.RandomState):
        return seed
    raise ValueError('%r cannot be used to seed a numpy.random.RandomState'
                     ' instance' % seed)



def build_coal_data():
    dim = 1
    events = coal_dataset.get_events_distributed(rng=np.random.RandomState(42))
    events = np.array(events, float).reshape(-1, dim)
    domain = np.array(coal_dataset.domain, float).reshape(dim, 2)

    return events, domain[0]


def build_λ1_data():
    directory = "D:\GitHub\point\data\synth1"
    data = []
    
    for i in range(10):
        name = "sample" + str(i+1)
        d= genfromtxt(directory + "/" + name + ".csv", delimiter=',')
        d = d.reshape(len(d),1)
        data.append(d)
        
    return data

def build_λ2_data():
    directory = "D:\GitHub\point\data\synth2"
    data = []
    
    for i in range(10):
        name = "sample" + str(i+1)
        d= genfromtxt(directory + "/" + name + ".csv", delimiter=',')
        d = d.reshape(len(d),1)
        data.append(d)
        
    return data


def build_λ2_data_g():
    directory = "D:\GitHub\point\data\synth2"
    data = []
    
    for i in range(1):
        name = "sgraph" + str(i+1)
        d= genfromtxt(directory + "/" + name + ".csv", delimiter=',')
        d = d.reshape(len(d),1)
        data.append(d)
        
    return data


def build_λ3_data():
    directory = "D:\GitHub\point\data\synth3"
    data = []
    
    for i in range(10):
        name = "sample" + str(i+1)
        d= genfromtxt(directory + "/" + name + ".csv", delimiter=',')
        d = d.reshape(len(d),1)
        data.append(d)
        
    return data


def get_taxi_data_set(scale = 1.0):
    rng  = np.random.RandomState(200)
    directory = "D:\GitHub\point\data"
    name = "porto_trajectories"
    data = genfromtxt(directory + "/" + name + ".csv", delimiter=',')
    data = data[data[:,1] <= 41.18]
    data = data[data[:,1] >= 41.147]
    data = data[data[:,0] <= -8.58]
    data = data[data[:,0] >= -8.65]

    data = data[rng.choice(data.shape[0], 7000, replace=False), :]

    X1 =  normalize(data[:,0],  [-8.58, -8.65], scale = 1.0)
    X2 =  normalize(data[:,1],  [41.18, 41.147], scale = 1.0)
    data = scale * np.column_stack((X1,X2))
    
    return data

def get_bei_data_set(scale = 1.0):
    name = 'bei'
    directory = "D:\GitHub\point\data"
    data = genfromtxt(directory + "/" + name + ".csv", delimiter=',')
    data = scale * data[1:, 0:2]
    return data


def get_data_set(name):
    directory = "D:\GitHub\point\data"
    data = genfromtxt(directory + "/" + name + ".csv", delimiter=',')
    data = data[1:, 0:2]
    return data

def normalize(X, domain, scale = 1.0) :
    center = np.mean(domain)
    norm = domain[1] - center
    return scale * (X - center) / norm

def print_contour_2D(grid, lambda_sample, X = None, X2 = None, markersize=1, fmt = None, vmin = None, vmax = None, cmap = 'Reds'):
    n = int(np.sqrt(lambda_sample.shape[0]))
    lambda_sample = lambda_sample.numpy()  
    lambda_matrix = lambda_sample.reshape(n,n)
    
    plt.figure(figsize=(5,4.5))
    #cmaps = 'viridis'
    plt.contour(np.unique(grid[:,0]), np.unique(grid[:,1]), lambda_matrix, colors=('k',), vmin = vmin, vmax = vmax, linestyles='--', linewidths= 0.5)
    plt.contourf(np.unique(grid[:,0]), np.unique(grid[:,1]), lambda_matrix, cmap= cmap, vmin = vmin, vmax = vmax, extend='both')

    if X is not None :
        x1fmt = 'ro'
        if fmt is not None : x1fmt = fmt[0] 
        plt.plot(X[:,0], X[:,1], x1fmt, markersize = markersize)
        
    if X2 is not None :
        x2fmt = 'ko'
        if fmt is not None : x2fmt = fmt[1] 
        plt.plot(X2[:,0], X2[:,1],x2fmt, markersize = markersize)
        
    plt.colorbar()
    plt.show()


def print_grid_2D(grid, lambda_sample, X = None, X2 = None, vmin = None, vmax = None, name = None, colorbar = False, figsize = (5,4.5)):
    n = int(np.sqrt(lambda_sample.shape[0]))
    
    if tf.is_tensor(lambda_sample):
         lambda_sample = lambda_sample.numpy()
         
    lambda_matrix = lambda_sample.reshape(n,n)
    cmap = 'plasma'
    #cmap = 'viridis'
    #cmap = 'Reds'
    
    plt.figure(figsize=figsize)
    plt.xlim(grid.min(), grid.max())
    plt.pcolormesh(np.unique(grid[:,0]), np.unique(grid[:,1]), lambda_matrix, vmin = vmin, vmax = vmax, shading='auto', cmap= cmap)
    
    if X is not None :
        plt.plot(X[:,0], X[:,1], 'wo', markersize = 0.5)
        
    if X2 is not None :
        plt.plot(X2[:,0], X2[:,1],'ro', markersize = 0.5)
    
    if name is not None : 
        plt.title(name)

    #plt.xticks(fontsize=8)
    #plt.yticks(fontsize=8)
    plt.axis('off')

    if colorbar is True :
        cb = plt.colorbar()
        cb.ax.tick_params(labelsize=8)

    plt.show()
    
    
def print_grid_2D_2(grid, lambda_sample, X = None, X2 = None, vmin = None, vmax = None, name = None, colorbar = False, figsize = (5,4.5)):
    n = int(np.sqrt(lambda_sample.shape[0]))
    
    if tf.is_tensor(lambda_sample):
         lambda_sample = lambda_sample.numpy()
         
    lambda_matrix = lambda_sample.reshape(n,n)
    #cmap = 'plasma'
    #cmap = 'viridis'
    cmap = 'Reds'
    
    plt.figure(figsize=figsize)
    plt.xlim(grid.min(), grid.max())
    plt.pcolormesh(np.unique(grid[:,0]), np.unique(grid[:,1]), lambda_matrix, vmin = vmin, vmax = vmax, shading='auto', cmap= cmap)
    
    if X is not None :
        plt.plot(X[:,0], X[:,1], 'ko', markersize = 0.5)
        
    if X2 is not None :
        plt.plot(X2[:,0], X2[:,1],'ro', markersize = 0.5)
    
    if name is not None : 
        plt.title(name)

    #plt.xticks(fontsize=8)
    #plt.yticks(fontsize=8)
    plt.axis('off')

    if colorbar is True :
        cb = plt.colorbar()
        cb.ax.tick_params(labelsize=8)

    plt.show()
    
    
def plot_data_set_2D(X):
    plt.plot(X[:,0], X[:,1],'ro', markersize = 0.5)
    plt.show()
    

def domain_grid_1D(bound, num_points):
    grid = np.linspace(bound[0], bound[1], num_points)
    return grid.reshape(grid.shape[0], 1)


def domain_grid_2D(bound, step = 0.1):
    stop = bound[1] + 1e-05
    x_mesh = y_mesh = np.arange(bound[0], stop, step)
    X, Y = np.meshgrid(x_mesh, y_mesh)
    grid = np.zeros((X.shape[0]**2,2))
    grid[:,0] = np.ravel(X)
    grid[:,1] = np.ravel(Y)
    return grid, x_mesh, y_mesh


def inducing_grid_2D(num_inducing, bound = (-10,10)):

    num = int(np.sqrt(num_inducing))
    x_array = np.linspace(bound[0], bound[1], num)
    X, Y = np.meshgrid(x_array, x_array)
    grid = np.zeros((X.shape[0]**2,2))
    grid[:,0] = np.ravel(X)
    grid[:,1] = np.ravel(Y)

    return grid




 
 
