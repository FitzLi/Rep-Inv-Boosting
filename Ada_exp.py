import numpy as np
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

class Adaboost_exp():
    """ Replication of experiments from "Evidence Contrary to the Statistical View of Boosting" section 3
    
    Parameters
    ----------
    bayes_err:  float
                Bayes error rate of the simulated classification
            
    num_train:  int
                Training sample size
            
    num_test:   int
                Test sample size
            
    dim_input:  int
                Dimensions of total input data/features
            
    dim_rele:   int
                Dimensions of relevant input data/features
            
    num_iter:   int
                Number of iteration/stages in the AdaBoost model
            
    num_simu:   int
                Number of simulation run
            
    num_node:   list of int, optional (default=None)
                Number of nodes for the comparing base trees
            
    num_min_obs:list of int, optional (default=None)
                Number of minimum number of observations in the terminal node
            
    shrinkage:  list of float, optional (default=None)
                Degree of shrinkage for the
    
    """
    
    def __init__(self, bayes_err, num_train, num_test, dim_input, dim_rele, num_iter, num_simu,
                 clf_base=DecisionTreeClassifier, num_node=[2], num_min_obs=[1], shrinkage=[1], random_state=42):
        self.clf_base = clf_base
        self.bayes_err = bayes_err
        self.num_train = num_train
        self.num_test = num_test
        self.dim_input = dim_input
        self.dim_rele = dim_rele
        self.num_iter = num_iter
        self.num_simu = num_simu
        self.num_node = num_node
        self.num_min_obs = num_min_obs
        self.shrinkage = shrinkage
        self.random_state = random_state
        self.dict_clfs = {}
        self.dict_perf = {}
        
    def generate_input(self, num_obs):
        np.random.seed(self.random_state)
        return np.random.rand(num_obs, self.dim_input)
        
    def generate_label(self, X):
        np.random.seed(self.random_state)
        if self.dim_rele == 0:
            indicator = np.zeros(X.shape[0])
        else:
            indicator = (X[ : , : self.dim_rele].sum(axis=1) > self.dim_rele / 2).astype(int)
        class_prob = self.bayes_err + (1 - 2 * self.bayes_err) * indicator
        return (class_prob > np.random.rand(X.shape[0])).astype(int)
    
    def generate_dateset(self, dataset_type):
        self.random_state += 1
        if dataset_type == 'train':
            X = self.generate_input(self.num_train)
        elif dataset_type == 'test':
            X = self.generate_input(self.num_test)
        else:
            raise("What you want?")
        y = self.generate_label(X)
        return X, y
    
    def generate_clf(self):
        for n_node in self.num_node:
            for n_obs in self.num_min_obs:
                for shrink in self.shrinkage:
                    self.random_state += 1
                    name_clf = "node_" + str(n_node) + "_minobs_" + str(n_obs) + "_shrink_" + str(shrink)
                    clf_b = self.clf_base(max_leaf_nodes=n_node, min_samples_leaf=n_obs,
                                          random_state=self.random_state)
                    clf_ada = AdaBoostClassifier(base_estimator=clf_b, learning_rate=shrink,
                                                 n_estimators=self.num_iter, algorithm='SAMME')
                    self.dict_clfs[name_clf] = clf_ada
                    
    def run(self):
        X_train, y_train = self.generate_dateset('train')
        X_test, y_test = self.generate_dateset('test')
        self.generate_clf()
        for simu_tmp in range(self.num_simu):
            self.dict_perf[simu_tmp] = {}
            for key_tmp in self.dict_clfs.keys():
                name_tmp = key_tmp
                if name_tmp not in self.dict_perf[simu_tmp].keys():
                    self.dict_perf[simu_tmp][name_tmp] = []
                clf_tmp = self.dict_clfs[key_tmp]
                clf_tmp.fit(X_train, y_train)
                for pred in clf_tmp.staged_predict(X_test):
                    self.dict_perf[simu_tmp][name_tmp].append(1 - accuracy_score(y_test, pred))