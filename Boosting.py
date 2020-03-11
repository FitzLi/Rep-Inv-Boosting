import numpy as np
from copy import deepcopy


class Adaboost():
    def __init__(self, base_clf, num_esti, learning_rate=1):
        self.base_clf = base_clf
        self.num_esti = num_esti
        self.learning_rate = learning_rate
        self.dict_clf = {}
        self.dict_err = {}
        self.dict_alpha = {}
        self.dict_weight = {}
        
    def fit(self, X_train, y_train):
        weight_tmp = np.ones(X_train.shape[0])
        weight_tmp /= X_train.shape[0]
        for num_tmp in range(self.num_esti):
            clf_tmp = self.base_clf
            clf_tmp.fit(X_train, y_train, sample_weight=weight_tmp)
            self.dict_clf[num_tmp] = deepcopy(clf_tmp)

            pred = clf_tmp.predict(X_train)
            err_tmp = ((y_train != pred) * weight_tmp).sum()
            self.dict_err[num_tmp] = err_tmp

            alpha_tmp = 0.5 * np.log((1 - err_tmp) / err_tmp)
            self.dict_alpha[num_tmp] = alpha_tmp

            weight_tmp = weight_tmp * (np.e ** (-1 * alpha_tmp * pred * y_train))
            weight_tmp /= weight_tmp.sum()
            self.dict_weight[num_tmp] = weight_tmp
            
    def predict(self, X_test):
        pred_ada = 0
        for num_tmp in range(self.num_esti):
            pred_ada += self.dict_clf[num_tmp].predict(X_test) * self.dict_alpha[num_tmp]
        return ((pred_ada>0).astype(int) - 0.5) * 2
        
    def staged_predict(self, X_test):
        list_pred = []
        pred_ada = 0
        for num_tmp in range(self.num_esti):
            pred_ada += self.dict_clf[num_tmp].predict(X_test) * self.dict_alpha[num_tmp] *\
                        self.learning_rate
            list_pred.append(((pred_ada>0).astype(int) - 0.5) * 2)
        return list_pred