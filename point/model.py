
import abc
import numpy as np
import math
import matplotlib.pyplot as plt

from point.utils import check_random_state_instance, domain_grid_1D
from point.misc import Space
from scipy.interpolate import griddata


def get_generative_model(lambda_sample, grid, dims = 1, random_state = None):
    random_state = check_random_state_instance(random_state)
    bound = (int(grid.min()), int(math.ceil(grid.max())))
    return InHomogeneousModelGrid(grid, lambda_sample, bound, dims, random_state) 

def get_generative_λ1(random_state = None) :
    random_state = check_random_state_instance(random_state)
    func = lambda x : 2 * np.exp(- x / 15) + np.exp(-( (x - 25) / 10)**2)
    return InHomogeneousModelLambda(func, lambdaMax = 2.2, space = Space([0,50]), random_state = random_state)


def get_generative_λ2(random_state = None) :
    random_state = check_random_state_instance(random_state)
    func = lambda x : 5 * np.sin( x **2) + 6
    return InHomogeneousModelLambda(func, lambdaMax = 12, space = Space([0,5]), random_state = random_state)


def get_generative_λ3(random_state = None) :
    random_state = check_random_state_instance(random_state)
    
    def piecewise_linear(x):
        condlist = [x < 25, (x >= 25) & (x < 50), (x >= 50) & (x < 75), (x >= 75)]
        funclist = [lambda x: 0.04 * x + 2, lambda x: -0.08 * x + 5, lambda x: 0.06 * x + -2, lambda x: 0.02 * x + 1]
        return np.piecewise(x, condlist, funclist)
    
    func = lambda x : piecewise_linear(x)
    
    return InHomogeneousModelLambda(func, lambdaMax = 3, space = Space([0,100]), random_state = random_state)






class PointsData():
    
    def __init__(self, sizes, points, space):
        self.space = space
        self.points = points
        self.sizes = sizes
    
    @property
    def n_samples(self):
        return self.points.shape[0]
    
    @property
    def size(self, index  = 0):
        return len(self.points[index])
    
    def __getitem__(self, item):
        return self.points[item][0:self.sizes[item],:]




class  CoxLowRankSpatialModel() :
    
    def __init__(self, low_rank_gp, name = None, random_state = None):
        self.lrgp =  low_rank_gp
        self.space = low_rank_gp.space
        self._name = name
        self._random_state = random_state

    @property
    def parameters(self):
        return self.lrgp.parameters
    
    @property
    def name(self):
        return self._name
        
    @property
    def trainable_variables(self):
        return self.lrgp.trainable_variables

    
    def log_likelihood(self, X): 
        return self.lrgp.maximum_log_likelihood_objective(X)
    
    
    
class HomogeneousModel() :

    def __init__(self, lam, space, dims = 2, random_state = None):
        
        self.space = space
        self.n_dimension = dims 
        
        self._lambda = lam
        self._random_state = random_state
        
    @property
    def bound(self):
        return self.space.bound1D
    
    @property
    def space_measure(self):
        return self.space.measure(self.n_dimension)

    def generate(self):
        random_state = check_random_state_instance(self._random_state)
        lam = self._lambda *  self.space_measure
       
        n_points = random_state.poisson(size= 1, lam = lam)
        points = random_state.uniform( self.bound[0], self.bound[1], size=(n_points[0], self.n_dimension))
        return points
    
    
class InHomogeneousModelBase(metaclass=abc.ABCMeta) :
    
    def __init__(self, space = Space(), dims = 2, random_state = None):
        
        self.space = space
        self.n_dimension = dims 
        self._random_state = random_state
        
    @property
    def bound(self):
        return self.space.bound1D
    
    @property
    def space_measure(self):
        return self.space.measure(self.n_dimension)

    @property
    def lambdaMax(self):
        return None

    @abc.abstractmethod 
    def get_lambda(self, points):
        raise NotImplementedError()

    def generate(self, n_samples = 1):
        random_state = check_random_state_instance(self._random_state)
        lambdaMax = self.lambdaMax
        
        hsm = HomogeneousModel(lambdaMax, self.space, self.n_dimension, random_state= random_state)
        
        points_lst = []
        sizes = []
        n_max = 0
        
        for i in range(n_samples):
            full_points  = hsm.generate()
            lambdas =  self.get_lambda(full_points)
            
            n_lambdas = lambdas.shape[0]
            u = random_state.uniform(0, 1, n_lambdas)
            tmp = (u < lambdas.reshape(n_lambdas)/lambdaMax)
            
            retained_points = full_points[tmp]
            n_points = retained_points.shape[0]
            
            n_max = max(n_max , n_points)  
            points_lst.append(full_points[tmp])
            sizes.append(n_points)
            
        if n_samples == 1 :
            return retained_points
        
        #padding for the output
        points = np.zeros((n_samples, n_max, self.n_dimension))
        for b in range(n_samples):
            points[b, :points_lst[b].shape[0]] = points_lst[b]

        return PointsData(
            sizes = sizes, 
            points = points, 
            space = self.bound
        )
    
    
    def plot_1D(self, X = None):
        
        if self.dims !=1 :
            raise ValueError("model must be of dimension 1")
        
        grid =  domain_grid_1D(self.bound, 100) 
        lambdas = self.get_lambda(grid)

        plt.xlim(grid.min(), grid.max())
        plt.plot(grid, lambdas)
        
        if X is not None :
            plt.plot(X, np.zeros_like(X), "|")
        plt.show()
   
     
    
    
class InHomogeneousModelGrid( InHomogeneousModelBase) :
    
    def __init__(self, grid, lam, space, dims = 2, random_state = None):
        
        super().__init__(space, dims, random_state)
        self._grid = grid
        self._lambda = lam
    
    @property
    def lambdaMax(self):
        return max(self._lambda)

    def get_lambda(self, points):
        lambdas =  griddata(self._grid, self._lambda, points, method='nearest')
        return lambdas
    
    
class InHomogeneousModelLambda( InHomogeneousModelBase) :
    
    def __init__(self, func, lambdaMax, space, random_state = None):
        
        super().__init__(space, 1, random_state)
        
        self._func = func
        self._lambdaMax = lambdaMax
    
    @property
    def lambdaMax(self):
        return self._lambdaMax

    def get_lambda(self, points):
        lambdas =  self._func(points)
        return lambdas
        






    


    









