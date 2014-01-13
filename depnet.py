import pdb
import numpy as np
from sklearn.linear_model import LogisticRegression
import networkx as nx
import multiprocessing as mp
import itertools

# this needs to sit outside the class for pool.map to work
def dn_fit_parallel(dn, n_proc):
    pool = mp.Pool(processes = n_proc)
    # map dn_fit_parallel_worker over (dn, k) pairs
    # this is ugly but gets around the restriction that Pool.map funcs
    # be callable from the top of the module
    cons = itertools.izip((dn for x in xrange(dn.n_features)),xrange(dn.n_features))
    conditional_models = pool.map(dn_fit_parallel_worker, cons)
    pool.close()
    return conditional_models
                    
def dn_fit_parallel_worker(cons):
    dn, k = cons
    model = LogisticRegression(penalty="l1")
    model.fit(dn._get_x(k), dn._get_y(k))
    return model

def dn_sample_parallel(dn, n_samples, n_chains, burn_in, burn_interval):
    pool = mp.Pool(processes = n_chains)  # one processor per chain
    samples_per_chain = np.zeros(n_chains, dtype="int64")
    for i in xrange(n_samples):
        samples_per_chain[i % n_chains] += 1
    cons = itertools.izip((dn for x in xrange(n_chains)),
                          samples_per_chain,
                          (burn_in for x in xrange(n_chains)),
                          (burn_interval for x in xrange(n_chains)))
    samples = pool.map(dn_sample_parallel_worker, cons)
    pool.close()
    samples = np.array( samples )
    samples = samples.reshape(n_samples, dn.n_features)
    return samples

def dn_sample_parallel_worker(cons):
    dn, n_samples_chain, burn_in, burn_interval = cons
    return dn.sample(n_samples_chain, burn_in, burn_interval)
    
class depnet:
    def __init__(self, X):
        self.X = X
        self.n_instances, self.n_features = self.X.shape
        self._col_indices = range(self.n_features)

    def _get_x(self, k):
        return self.X[:,filter(lambda z: z != k, self._col_indices)]
    def _get_y(self, k):
        return self.X[:,k]

    def fit(self):
        self.conditional_models = []
        for k in xrange(self.n_features):
            model = LogisticRegression(penalty="l1")
            model.fit(self._get_x(k), self._get_y(k))

    def fit_parallel(self, n_proc = 4):
        self.conditional_models = dn_fit_parallel(self, n_proc)

    def _next_state(self, state, k):
        conditioned_state = state[filter(lambda z: z != k, self._col_indices)]
        return self.conditional_models[k].predict(conditioned_state)

    def sample(self, n_samples, burn_in=100, burn_interval=5):
        samples = np.zeros((n_samples, self.n_features))
        
        # n_features binary random draws
        state = np.random.randint(0, 2, self.n_features)  
        
        for i in xrange(burn_in):
            for k in xrange(self.n_features):
                #conditioned_state = state[filter(lambda z: z != k, self._col_indices)]
                #state[k] = self.conditional_models[k].predict(conditioned_state)
                state[k] = self._next_state(state,k)

        for i in xrange(n_samples):
            for j in xrange(burn_interval):
                for k in xrange(self.n_features):
                    #conditioned_state = state[filter(lambda z: z != k, self._col_indices)]
                    #state[k] = self.conditional_models[k].predict(conditioned_state)
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
        
        
        
