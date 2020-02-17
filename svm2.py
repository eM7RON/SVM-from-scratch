
from scipy.optimize import fmin_bfgs
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from svm_kernels import linear_kernel, polynomial_kernel, rbf_kernel, gaussian_kernel
import numpy as np
import cvxpy as cp

class PlattScaler(TransformerMixin):
    '''
    Perform Platt Scaling.
    Based on Platt 1999
    
    Platt, J.C., 1999. Probabilistic Outputs for Support Vector Machines and Comparisons
    to Regularized Likelihood Methods, Advances in Large Margin Classifiers, 61-74, 
    
    Original code: https://github.com/neal-o-r/platt/blob/master/platt.py
    '''

    def __init__(self):
        pass

    def fit(self, f, y):
        '''
        Fit Platt model.
        This method takes in the classifier outputs and the true labels,
        and fits a scaling model to convert classifier outputs to true
        probabilities. Sticks with Platt's weird notation throughout.

            f: classifier outputs
            y: true labels
        '''
        
        eps = np.finfo(np.float).tiny  # to avoid division by 0 warning

        # Bayes priors
        prior0 = float(np.sum(y <= 0))
        prior1 = y.shape[0] - prior0
        T = np.zeros(y.shape)
        T[y > 0] = (prior1 + 1.) / (prior1 + 2.)
        T[y <= 0] = 1. / (prior0 + 2.)
        T1 = 1. - T

        def objective(theta):
            A, B = theta
            E = np.exp(A * f + B)
            P = 1. / (1. + E)
            l = -(T * np.log(P + eps) + T1 * np.log(1. - P + eps))
            return l.sum()

        def grad(theta):
            A, B = theta
            E = np.exp(A * f + B)
            P = 1. / (1. + E)
            TEP_minus_T1P = P * (T * E - T1)
            dA = np.dot(TEP_minus_T1P, f)
            dB = np.sum(TEP_minus_T1P)
            return np.array([dA, dB])

        AB0 = np.array([0., np.log((prior0 + 1.) / (prior1 + 1.))])
        self.A_, self.B_ = fmin_bfgs(objective, AB0, fprime=grad, disp=False)


    def transform(self, f):
        '''
        Given a set of classifer outputs return probs.
        '''
        return 1. / (1. + np.exp(self.A_ * f + self.B_))

    def fit_transform(self, f, y):
        self.fit(f, y)
        return self.transform(f)
        

class SVC(BaseEstimator, ClassifierMixin):
    ''' 
        ######################################################################
        # -A soft margin svm classifier capable of using non-linear kernels- #
        ######################################################################
          
    '''

    proba_fit = False
    classes_  = [-1, 1] # Required

    def __init__(
             self,
             C=None,
             gamma=1, 
             delta=1, 
             kernel='linear', 
             threshold=1e-7, 
             dtype=np.float64,
             probability=False,
             solver_params={'solver':'SCS', 'eps': 1e-14, 'acceleration_lookback': 0},
             ):
        '''
        Args:
            C             - float or int, the soft margin penalty
            kernel        - function, takes two array-like objects as input and projects them
                            into a different space, returns a single array-like object
            gamma         - float or int, kernel hyperparameter
            delta         - float or int, kernel hyperparameter (polynomial only)
            threshold     - float or int, support vectors have alphas that exceed this value
            dtype         - callable or string, the data type to use, effects precision,
                            i.e. np.float32 or 'float64'
            solver_params - dictionary, kwargs for the cvxpy solver
            

        '''
        
        self.kernel        = kernel
        self.C             = C
        self.gamma         = gamma
        self.delta         = delta
        self.threshold     = threshold
        self.dtype         = dtype
        self.probability   = probability
        self.solver_params = solver_params
        
    def _init(self):
        
        if self.C:
            self.C = self.dtype(self.C)
            
        if type(self.kernel) == str:
            self.kernel = globals()[self.kernel + '_kernel']

    def fit(self, X, y=None):
        ''' 
        Finds the optimal separating hyperplane by solving the dual constraint quadratic
        optimization problem of maximizing the margin by using cvxpy package
        
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
        self.sv_idx = np.arange(len(X))[mask]
        n_alpha  = len(self.alphas)

        for i in range(n_alpha):
            self.b += self.support_vector_labels[i]
            self.b -= np.sum(self.alphas * self.support_vector_labels * K[self.sv_idx[i], mask])
        self.b /= n_alpha
        
        if self.probability:
            self.plattscaler = PlattScaler()
            distances = self.decision_function(X)
            self.plattscaler.fit(distances, y)
            self.proba_fit = True

        return self

    def predict_proba(self, X):
        if not self.proba_fit:
            raise Exception("SVC must be initialized with 'probability' keyword argument set to True             before calling fit method in order to use Platt Scaling and produce probabilistic outputs")
        distances = self.decision_function(X)
        n = distances.__len__()
        pos_p = self.plattscaler.transform(distances)
        neg_p = np.ones(n) - pos_p
        probabilities = np.array((neg_p, pos_p)).T.reshape(n, 2)
        return probabilities
    
    def decision_function(self, X):
        '''
        Calculates the signed distance (d_m) of sample x_m using equation:
        
        d_{m} = \sum_{n}alpha_{n} y_{n} k(x_{n}, x_{m}) + b
        
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
        Returns the prediction, y_u, calculated from the signed distance d_m
        
        y_{m} = sgn( d_{m} )
        
        Args:
        X           - array-like, shape [n_train_samples, n_features], training data
        
        Returns:
        predictions - array-like, shape[n_samples, ]
        '''
        return np.sign(self.decision_function(X))
