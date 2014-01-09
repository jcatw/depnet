import numpy as np
from sklearn.linear_model import LogisticRegression

class depnet:
    def __init__(self, X):
        self.X = X
        self.n_instances, self.n_features = self.X.shape
        self._col_indices = range(self.n_features)

    def _get_x(self, k):
        return self.X[:,filter(k,self._col_indices)]
    def _get_y(self, k):
        return self.X[:,k]

    def fit(self):
        self.conditional_models = []
        for k in xrange(self.n_features):
            model = LogisticRegression(penalty="l1")
            model.fit(self._get_x(k), self._get_y(k))
            self.conditional_models.append(model)

    def sample(self, n_samples, burn_in=100, burn_interval=5):
        samples = np.zeros(n_samples, self.n_features)
        
        # n_features binary random draws
        state = np.random.randint(0, 2, self.n_features)  
        
        for i in xrange(burn_in):
            for k in xrange(self.n_features):
                state[k] = predict(self._get_x(k))

        for i in xrange(n_samples):
            for j in xrange(burn_interval):
                for k in xrange(self.n_features):
                    state[k] = predict(self._get_x(k))
            samples[i] = state

        return samples
            
        
        
