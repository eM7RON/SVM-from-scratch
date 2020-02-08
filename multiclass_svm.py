
import numpy as np
from scipy.stats import mode
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from svm2 import SVC

class MultiClassSVM:
    '''
            #########################################################
            #    An implementation of SVM for multiclass problems   #
            #########################################################
    '''
    
    def __init__(
                 self,
                 classifier=SVC,
                 param_grid={},
                 random_state=117,
                 clf_kwargs={},
                 ):
        '''
        Args:
            n_estimators - int, number of weak classifier instance in the ensemble
            classifier   - class with fit and predict method
            attr         - dict, string: value, attributes of the weak learners
            random_state - hashable, something to seed the random generator
            
        '''
        np.random.seed(random_state)
        self.classifier = classifier
        self.param_grid = param_grid
        #self.mode = __import__('scipy').stats.mode
        self.clf_kwargs = clf_kwargs
    
    def fit(self, X, y=None):
        
        '''
        Fits an svm for each pair of classes in the dataset
        
        Args:
            X - array-like, sample training data, shape=[n_samples, n_features]
            y - array-like, target labels, shape=[n_samples]
        '''

        self.ensemble = {}
        self.n_classes = len(np.unique(y))
        self.n_estimators = int(self.n_classes * (self.n_classes - 1) / 2)

        for i in range(self.n_classes - 1):
            for j in range(i + 1, self.n_classes):
            
                # get indices of classes
                idx_i = y==i
                idx_j = y==j
                
                # make new X and y with selected classes
                y_ = np.append(y[idx_i], y[idx_j])
                X_ = np.vstack((X[idx_i], X[idx_j]))

                # set labels to -1 and 1
                y_ = np.array([1 if k == i else -1 for k in y_])
                
                # shuffle
                idx = np.random.permutation(range(len(y_)))
                y_ = y_[idx]                  
                X_ = X_[idx]  

                # get best parameters

                cv = StratifiedKFold(n_splits=5, shuffle=True)

                grid = GridSearchCV(self.classifier(**self.clf_kwargs),
                                    param_grid=self.param_grid,
                                    #scoring=scorer,
                                    cv=cv, 
                                    n_jobs=-1)
                grid.fit(X_, y_)

                C = grid.best_params_['C']
                gamma = grid.best_params_['gamma']
                kernel = grid.best_params_['kernel']
                
#                 # create classifier and fit
#                 clf = self.classifier(C=C, gamma=gamma, kernel=kernel, **self.clf_kwargs)
#                 clf.fit(X_, y_)
                
                # store classifier and parameters
                self.ensemble[(i, j)] = {
                                         'clf': grid.best_estimator_, 
                                         'C': C, 
                                         'gamma': gamma, 
                                         'kernel': kernel,
                                         }
                
        return self

    def predict(self, X):
        '''
        Predict the class of each sample in X
    
        Args:
            X           - array-like, sample training data, shape[n_samples, n_features]
        
        Returns:
            predictions - array-like, predicted labels, shape[n_samples]

        '''
        n_samples = X.__len__()                        
        predictions = np.zeros([self.n_estimators, n_samples])
        i = 0

        for label, clf in self.ensemble.items():
            predictions[i] = [label[0] if j == 1 else label[1] for j in clf['clf'].predict(X)]
            i += 1

        return mode(predictions)[0][0].astype(int)