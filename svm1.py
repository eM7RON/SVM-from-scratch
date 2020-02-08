from sklearn.base import BaseEstimator, ClassifierMixin
from svm_kernels import linear_kernel, polynomial_kernel, rbf_kernel, gaussian_kernel
import numpy as np
import cvxpy as cp

class SVC(BaseEstimator, ClassifierMixin):
    '''
        ######################################################################
        # -A soft margin svm classifier capable of using non-linear kernels- #
        ######################################################################
          
    '''

    def __init__(
                 self,
                 C=None,
                 gamma=1, 
                 delta=1, 
                 kernel='linear', 
                 threshold=1e-7, 
                 dtype=np.float64, 
                 solver_params={'solver':'SCS', 'eps': 1e-14, 'acceleration_lookback': 0},
                 ):
        '''
        Args:
            C         - float or int, the soft margin penalty
            kernel    - function, takes two array-like objects as input and projects them
                        into a different space, returns a single array-like object
            gamma     - float or int, kernel hyperparameter
            delta     - float or int, kernel hyperparameter (polynomial only)
            threshold - float or int, support vectors have alphas that exceed this value
            dtype     - callable or string, the data type to use, effects precision,
                        i.e. np.float32 or 'float64' 

        '''
        self.kernel        = kernel
        self.C             = C
        self.gamma         = gamma
        self.delta         = delta
        self.threshold     = threshold
        self.dtype         = dtype
        self.solver_params = solver_params
        
    def _init(self):
        
        if self.C:
            self.C = self.dtype(self.C)
        
        if type(self.kernel) == str:
            self.kernel = globals()[self.kernel + '_kernel']
        
    def fit(self, X, y=None):
        ''' 
        Finds the optimal separating hyperplane by solving the dual constraint quadratic
        optimization problem of maximizing the margin by using cvxopt package
        
        Args:
            X - array-like, shape=[n_samples, n_features], the training data
            y - array-like, shape=[n_samples], training targets/labels
                
        '''
        self._init()
        X = X.astype(self.dtype)
        y = y.astype(self.dtype)
        n_samples, n_features = X.shape
    
        # Compute proto kernel/gram matrix
        K = self.kernel(X, params={'gamma': self.gamma, 'delta': self.delta})

        # Build the variables
        P = np.outer(y, y) * K # kernel/gram matrix
        q = -np.ones(n_samples, dtype=self.dtype) # negative makes it a minimization problem
        A = y.reshape(1, -1)
        b = self.dtype(0.0)
        x = cp.Variable(n_samples) # alphas
        
        # Constraints
        if self.C:  # If soft margin classifier...
            G = np.vstack((-np.eye(n_samples, dtype=self.dtype), np.eye(n_samples, dtype=self.dtype)))
            h = np.hstack((np.zeros(n_samples, dtype=self.dtype), np.ones(n_samples, dtype=self.dtype) * self.C))
        else:       # Hard margin...
            G = np.diag(-np.ones(n_samples, dtype=self.dtype))
            h = np.zeros(n_samples, dtype=self.dtype) # self.dtype(0.0) #
            
        objective   = cp.Minimize(0.5 * cp.quad_form(x, P) + q.T @ x)
        constraints = [G @ x <= h, 
                       A @ x == b]

        problem     = cp.Problem(objective, constraints)
        problem.solve(**self.solver_params)

        self.alphas = x.value

        # Support vectors have non zero lagrange multipliers
        mask = self.alphas > self.threshold # threshold otherwise we end up with everything being a support vector
        self.alphas = self.alphas[mask]
        self.support_vectors = X[mask]
        self.support_vector_labels = y[mask]

        # Calculate bias:
        # .. math::
        # b = rac{1}{N_S}\sum\limits_{v∈S} [α_u y_u k(x_u , x_v )]

        self.b   = self.dtype(0.0)
        self.idx = np.arange(len(X))[mask]
        n_alpha  = len(self.alphas)

        for i in range(n_alpha):
            self.b += self.support_vector_labels[i]
            self.b -= np.sum(self.alphas * self.support_vector_labels * K[self.idx[i], mask])
        self.b /= n_alpha

        return self

    def decision_function(self, X):
        '''
        Calculates the signed distance (d_u) of sample x_u using equation:
        
        d_{m} = \sum\limits_{n}lpha_{n} y_{n} k(x_{n}, x_{m}) + b
        
        Args:
        X         - array-like, shape[n_samples, n_features], new data
        
        Returns:
        distances - array-like, shape[n_samples, ]
        
        '''
        n_samples = X.__len__()
        distances = np.zeros(n_samples)
        
        for i in range(n_samples):
            for a, sv, sv_y in zip(
                                   self.alphas,
                                   self.support_vectors,
                                   self.support_vector_labels, 
                                   ):
                
                distances[i] += a * sv_y * self.kernel(X[i], sv, params={'gamma': self.gamma, 'delta': self.delta})
        distances += self.b
        return distances
    
    def predict(self, X):
        '''
        Returns the prediction, y_u, calculated from the signed distance d_u
        
        y_{m} = sgn( d_{m} )
        
        Args:
        X           - array-like, shape [n_train_samples, n_features], training data
        
        Returns:
        predictions - array-like, shape[n_samples, ]
        '''
        return np.sign(self.decision_function(X))