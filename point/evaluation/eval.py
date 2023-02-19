
import numpy as np
import copy
import statistics
import math

import pandas as pd  
from sklearn.model_selection import train_test_split
from point.utils import check_random_state_instance, domain_grid_1D
from point.misc import Space
from point.laplace import opt_method_to_str

import matplotlib.pyplot as plt


def normalize_x(X, xdomain, rdomain) :
    center = np.mean(xdomain)
    norm = xdomain[1] - center
    Xn = (X - center) / norm
    return Xn * rdomain[1]
    



class EvalTuple():
    def __init__(self, model, optim_func):
        self.model =  model
        self.ofunc = optim_func
        
    @property
    def name(self):
        return self.model.name

    @property
    def p(self):
        return self.model.p
    
    def set_random_state(self, random_state):
        self.model.set_random_state(random_state)
        
    
        
        
class Results():
        
    def __init__(self, tag, model_names, entries = None, args = None):
        
        if entries is None :
            entries = ['llp', 'time']
        
        self._tag = tag
        self._names = model_names
        self._entries = entries
        self._args = args
        
        self.llp = {}
        self.llm = {}
        self.l2 = {}
        self.time = {}
        self.std_lvl = 1.0
        
        
        for n in self._names:
            for e in entries :
                getattr(self, e)[n] = []


    def _compile(self, valid_names = None, valid_entries = None, **kwargs):
        
        if valid_names is None :
            valid_names = self._names
            
        if valid_entries is None :
            valid_entries = self._entries

        data = {}
        self._args = {**self._args, **kwargs}
        
        idx = [self._names.index(vn) for vn in valid_names]
        for key, value in self._args.items():
            v = [value[i] for i in idx]
            data[key] = v

        for e in valid_entries :
            if e == 'time' :
                data[e] =  [statistics.median(getattr(self, e)[key]) for key in valid_names]
            else :
                data[e] =  [sum(getattr(self, e)[key]) / len(getattr(self, e)[key]) for key in valid_names]
                
            data[e + ".std"] =  [ np.std(getattr(self, e)[key]) for key in valid_names]

        self.df = pd.DataFrame(data, index = valid_names)  

    def __str__(self):
        pd.set_option('display.max_rows', len(self.df))
        pd.set_option('display.max_columns', len(self.df.columns))
        print(self.df)
        pd.reset_option('display.max_rows')
        pd.reset_option('display.max_columns')
        pass
    
    def add_results(self, res):
        
        self.llm = {**self.llm , **res.llm}
        self.llp = {**self.llp , **res.llp}
        self.l2 = {**self.l2 , **res.l2}
        self.time = {**self.time , **res.time}

        self._names = self._names + res._names
        
        for key in res._args :
            self._args[key] = self._args[key] + res._args[key]

            
            
class Evaluation():

    def __init__(self, models, X, space, tag = None):

        if not isinstance(models, list):
            models = np.array(models)

        names =  [m.model.name for m in models]
        entries = ['llm', 'llp', 'time']
        
        args = {}
        args['opt_methods'] =   [opt_method_to_str(m.model.default_opt_method) for m in models] 
        args['p'] =  [m.p for m in models]
        self.results = Results(tag, names, entries, args)
        
        self._tag = tag
        self._models =  models
        self._X = X
        self._log = []
        self._isTrained = False
        self._space = space
        self._num_grid = 50
        self._random_state = None
        self._llpl = -1 * math.inf
        self._n_restart = 0


    def run(self, test_size = 0.5, n_samples = 10, random_state = None, verbose = True, flag_llp = False):

        self._random_state = check_random_state_instance(random_state)

    
        #one_sample = (n_samples == 1 or test_size == 0)
        one_sample = False
        # grid =  domain_grid_1D(self._space.bound1D, 100)
        
        if one_sample :
            n_samples = 1
        
        n = 0

        while n < n_samples :
            
            if not one_sample :
                X_train, X_test = train_test_split(self._X, test_size= test_size,  random_state = random_state)
            else : X_train = X_test = self._X
            
            self._X2 = X_test
            
            # plt.figure()
            # plt.title("it:=" + str(n))
            # plt.xlim(grid.min(), grid.max())
            
            n_model = 0

            for i in range(len(self._models)) :

                model = self._models[i]
                n_model +=1
    
                #for attempt in range(3):
     
                m = copy.deepcopy(model)
                m.set_random_state(self._random_state)

                if verbose : 
                    print("start@" + m.name + "@iter." + str(n))

                #try:
                opt_time = m.ofunc(m.model, X_train, verbose = verbose)
  
                if not one_sample :
                    llp = m.model.predictive_log_likelihood(X_test)
                    llm = m.model.lambda_mean_log_likelihood(X_test)
                    
                    if flag_llp and llp < self._llpl and n_model == 1 :
                        n = n - 1
                        
                        if verbose :
                            print("llp:= %f" % (llp))
                            self._n_restart =  self._n_restart + 1 
                            msg_id = "model#"  + m.name + ": Eval ERROR stoped and re-attempt"
                            print(msg_id) 
                            print("")
                            
                        break

                    if verbose : 
                        print("llp:= %f" % (llp))

                    self.results.llp[m.name].append(llp.numpy())
                    self.results.llm[m.name].append(llm.numpy())
                    self.results.time[m.name].append(opt_time)
                else :
                    self._models[i] = m
  
                # except BaseException as err:
                #     msg_id = "model#"  + m.name + ": Eval ERROR stoped and re-attempt"
                #     msg_e = f"Unexpected {err=}, {type(err)=}"
                #     print(msg_id) 
                #     print(msg_e)
                #     self._log.append(msg_id + " : " + msg_e)
                #     #if attempt == 9 :
                #     raise ValueError("attempt ERROR")
                #     #continue
                # else:
                #     break
            
                #v = m.model.predict_lambda(grid)
                #plt.plot(grid, v, label= m.name)
    
                if verbose is True : print("")

            n = n+1 

            # plt.plot(X_train, np.zeros_like(X_train), "k|")
            # plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
            # plt.show()

        if not one_sample :
            self.results._compile( valid_entries = ['llp', 'time'])
        self._isTrained = False
        
        
        
        
        
class EvaluationSynthetic():

    def __init__(self, models, data, generator, space = Space(), tag = None):

        if not isinstance(models, list):
            models = np.array(models)

        names =  [m.model.name for m in models]
        entries = ['l2', 'llp', 'time']
        
        args = {}
        args['opt_methods'] =  [opt_method_to_str(m.model.default_opt_method) for m in models] 
        args['p'] =  [m.p for m in models]
        
        self.results = Results(tag, names, entries, args)
        
        self._tag = tag
        self._num_grid = 100
        self._models =  models
        self._gen =  generator
        self._data = data
        self._space = space
        self._log = []
        self._isTrained = False
        self._random_state = None


    def run(self, n_testing = 10, random_state = None, verbose = True):

        self._random_state = check_random_state_instance(random_state)
        x_grid = domain_grid_1D(self._gen.bound, self._num_grid)
        n_grid = normalize_x(x_grid, self._gen.bound, self._space.bound1D)
        λtruth = (self._gen.space_measure / self._space.measure(1)) * self._gen.get_lambda(x_grid)
 
        for n in range(len(self._data)):
            
            X_train = normalize_x(self._data[n], self._gen.bound, self._space.bound1D)
            X_test = self._gen.generate(n_samples = n_testing)
  
            for model in self._models :

                if verbose is True : print("start@" + model.name + "@iter." + str(n))
                
                for attempt in range(10):
                    
                    try:
                        m = copy.deepcopy(model)
                        m.set_random_state(self._random_state)
                        opt_time = m.ofunc(m.model, X_train, verbose = verbose)

                        #compute l2
                        λ = m.model.predict_lambda(n_grid)
                        l2 = sum(((λ - λtruth)**2).numpy())
                        l2 = np.sqrt(l2[0]) / self._num_grid
                        
                        print("l2:=" + str(l2))
                        if l2 > 20 : raise ValueError("l2: error")
                        
                        self.results.l2[m.name].append(l2)
                        
                        #compute E.likelihood
                        for x in X_test :
                            x = normalize_x(x, self._gen.bound, self._space.bound1D)
                            llp = m.model.predictive_log_likelihood(x)
                            self.results.llp[m.name].append(llp.numpy())
         
                        self.results.time[m.name].append(opt_time)
    
                    except BaseException as err:
                        msg_id = "model#"  + m.name + ": ERROR stooped and re-attempt"
                        msg_e = f"Unexpected {err=}, {type(err)=}"
                        print(msg_id) 
                        print(msg_e)
                        self._log.append(msg_id + " : " + msg_e)
                        if attempt == 9 :
                            raise ValueError("attempt ERROR")
                        continue
                    else:
                        break
            
                if verbose is True : print("")
            
        p =  [m.p for m in self._models]
        self.results._compile()
        self._isTrained = False



