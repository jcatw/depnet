import pdb
import numpy as np
from sklearn.linear_model import LogisticRegression
import networkx as nx
import multiprocessing as mp
import itertools

def all_same_value(vec):
    #assumes that vec is binary
    return vec.all() or not vec.any()

def dn_fit_parallel(dn, n_proc):
    """
    Map dn_fit_parallel_worker over (dn, k) pairs
    this is ugly but gets around the restriction that Pool.map funcs
    be callable from the top of the module.

    This needs to sit outside the class for pool.map to work.

    """
    pool = mp.Pool(processes = n_proc)
    
    cons = itertools.izip((dn for x in xrange(dn.n_features)),xrange(dn.n_features))
    conditional_models = pool.map(dn_fit_parallel_worker, cons)
    pool.close()
    return conditional_models

def dn_fit(x,y):
    """
    If we only observe one y, x will completely seperate y,
    and the logistic regression parameters will be meaningless.
    We will condition on any such features, so return the value.
    
    Otherwise, fit a logistic regression, and return the model.
    """
    if (all_same_value(y)):
        return y[0]
    else:
        model = LogisticRegression(penalty="l1")
        model.fit(x, y)    
        return model


def dn_fit_parallel_worker(cons):
    dn, k = cons
    x = dn._get_x(k)
    y = dn._get_y(k)
    
    return dn_fit(x, y)
    

# like dn_fit_parallel, this needs to sit outside the class for pool.map to work
def dn_sample_parallel(dn, n_samples, n_chains, burn_in, burn_interval):
    pool = mp.Pool(processes = n_chains)  # one processor per chain
    samples_per_chain = np.zeros(n_chains, dtype="int64")
    for i in xrange(n_samples):
        samples_per_chain[i % n_chains] += 1
    # map dn_sample_parallel_worker over (dn, n_samples_chain, burn_in, burn_interval) tuples
    # this is ugly but gets around the restriction that Pool.map funcs
    # be callable from the top of the module
    cons = itertools.izip((dn for x in xrange(n_chains)),
                          samples_per_chain,
                          (burn_in for x in xrange(n_chains)),
                          (burn_interval for x in xrange(n_chains)))
    chain_samples = pool.map(dn_sample_parallel_worker, cons)
    pool.close()
    samples = np.zeros((n_samples, dn.n_features))
    i = 0
    for n_chain_samples, chain_sample in itertools.izip(samples_per_chain, chain_samples):
        samples[i:(i+n_chain_samples),:] = chain_sample
        i += n_chain_samples
    return samples

def dn_sample_parallel_worker(cons):
    dn, n_samples_chain, burn_in, burn_interval = cons
    return dn.sample(n_samples_chain, burn_in, burn_interval)
    
class depnet:
    def __init__(self, X):
        self.X = X
        self.n_instances, self.n_features = self.X.shape
        self._col_indices = range(self.n_features)
        self._conditioned_feature = [False] * self.n_features
        self._conditioned_feature_val = {}

    def _get_x(self, k):
        return self.X[:,filter(lambda z: z != k, self._col_indices)]
    def _get_y(self, k):
        return self.X[:,k]
    

    def fit(self):
        self.conditional_models = []
        for k in xrange(self.n_features):
            x = self._get_x(k)
            y = self._get_y(k)
            fit_return = dn_fit(x,y)
            if isinstance(fit_return,int) or isinstance(fit_return,float):
                self.conditional_models.append(None)
                self._conditioned_feature[k] = True
                self._conditioned_feature_val[k] = fit_return
            else:
                self.conditional_models.append(fit_return)

    def fit_parallel(self, n_proc = 4):
        self.conditional_models = dn_fit_parallel(self, n_proc)
        # track conditioned models
        for k, model in enumerate(self.conditional_models):
            if isinstance(model,int) or isinstance(model,float):
                self.conditional_models[k] = None
                self._conditioned_feature[k] = True
                self._conditioned_feature_val[k] = model

    def _next_state(self, state, k):
        # sub in conditioned feature
        if self._conditioned_feature[k]:
            return self._conditioned_feature_val[k]
        # predict modeled feature
        else:
            conditioned_state = state[filter(lambda z: z != k, self._col_indices)]
            return self.conditional_models[k].predict(conditioned_state)

    def sample(self, n_samples, burn_in=100, burn_interval=5):
        samples = np.zeros((n_samples, self.n_features))
        
        # n_features binary random draws
        state = np.random.randint(0, 2, self.n_features)  
        
        for i in xrange(burn_in):
            for k in xrange(self.n_features):
                state[k] = self._next_state(state,k)

        for i in xrange(n_samples):
            for j in xrange(burn_interval):
                for k in xrange(self.n_features):
                    state[k] = self._next_state(state,k)
            samples[i] = state

        return samples

    def sample_parallel(self, n_samples, n_chains, burn_in=100, burn_interval=5):
        return dn_sample_parallel(self, n_samples, n_chains, burn_in, burn_interval)

    def dependency_net(self):
        adjacency_matrix = np.zeros((self.n_features,self.n_features))
        for i in xrange(self.n_features):
            m1 = 0
            for k in xrange(self.n_features):
                if k == i:
                    m1 = 1
                else:
                    adjacency_matrix[i,k] = int(self.conditional_models[i].coef_[0,k-m1] != 0.)
        #return adjacency_matrix
        D = nx.DiGraph(adjacency_matrix)
        return D
        
        
        
