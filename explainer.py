import numpy as np
from collections import deque
from sklearn.ensemble import ExtraTreesRegressor
from GPy.models import GPClassification
from GPy.kern import RBF



class BaselineExplainer():

    def __init__(self, cost_type='norm', plausibility_constraint=1.0, fit_verifier=False):
        self.cost_type = cost_type
        self.plausibility_constraint = plausibility_constraint
        self.fit_verifier = fit_verifier


    def initialize(self, n_counterfactuals, n_features, n_rounds, n_timeout):
        self.round = 0 
        self.n_counterfactuals = n_counterfactuals
        return self


    def select(self, feasibility, probability):
        k_chosen = feasibility[np.argmax(probability[feasibility])]        
        return k_chosen


    def update(self, feedbacks):
        self.round = self.round + 1
        return self


    
class LinucbExplainer():

    def __init__(self, cost_type='norm', plausibility_constraint=1.0, alpha=20.0, delta=0.05):
        self.cost_type = cost_type
        self.plausibility_constraint = plausibility_constraint
        self.fit_verifier = False
        self.alpha = alpha
        self.delta = delta
        

    def initialize(self, n_counterfactuals, n_features, n_rounds, n_timeout):
        self.round = 0
        self.n_counterfactuals = n_counterfactuals
        self.xy = np.zeros(n_counterfactuals)
        self.invcov = (1 / self.alpha) * np.eye(n_counterfactuals)     
        self.bias = deque(maxlen=n_timeout)
        self.theta = np.zeros(n_counterfactuals)
        K = n_counterfactuals
        rounds = np.arange(1, n_rounds+1)
        self.f = np.sqrt(self.alpha) + np.sqrt(2 * np.log(1 / self.delta) + K * np.log((K * self.alpha + rounds) / (K * self.alpha)))
        return self


    def select(self, feasibility, probability):
        A_norm = (probability ** 2) * np.diag(self.invcov)
        kappa = (2 * self.f[self.round] + sum(self.bias))
        obj = self.theta * probability + kappa * A_norm
        k_chosen = feasibility[np.argmax(obj[feasibility])]           
        self.bias.append(A_norm[k_chosen])
        self.invcov = self.invcov - ((probability[k_chosen] ** 2) / (1 + A_norm[k_chosen])) * np.outer(self.invcov[k_chosen], self.invcov[k_chosen])
        return k_chosen


    def update(self, feedbacks):
        for feedback in feedbacks:
            k_chosen, prob_k, reward = feedback
            self.xy[k_chosen] = self.xy[k_chosen] + reward * prob_k    
        self.theta = self.invcov @ self.xy
        self.round = self.round + 1
        return self   



class BwoucbExplainer():

    def __init__(self, cost_type=None, plausibility_constraint=1.0, gamma=1.0, n_estimators=50, sample_ratio=4.0, n_init=1):
        self.cost_type = None
        self.plausibility_constraint = plausibility_constraint
        self.fit_verifier = False
        self.gamma = gamma
        self.n_estimators = n_estimators
        self.sample_ratio = sample_ratio
        self.n_init = n_init


    def initialize(self, n_counterfactuals, n_features, n_rounds, n_timeout):
        self.round = 0 
        self.Z = np.zeros((n_rounds, 2 * n_features))
        self.R = -1 * np.ones(n_rounds)
        self.estimator = ExtraTreesRegressor(n_estimators=self.n_estimators, bootstrap=True, max_samples=float(1 / self.sample_ratio), n_jobs=-1)
        self.unlearned = True
        self.beta = 1.0
        return self


    def select(self, feasibility, Z):
        if self.unlearned:
            k_chosen = np.random.choice(feasibility)
        else:
            self.beta = self.beta * self.gamma
            kappa = np.sqrt(self.beta * np.log(self.round+1) / (self.round+1))
            mu, sigma = self._predict(Z)            
            obj = mu + kappa * sigma
            k_chosen = feasibility[np.argmax(obj[feasibility])]
        self.Z[self.round] = Z[k_chosen]
        self.R[self.round] = 0
        return k_chosen


    def update(self, feedbacks):
        for feedback in feedbacks:
            round, reward = feedback
            self.R[round] = reward
        if self.unlearned:
            if np.sum(self.R > 0) >= self.n_init:
                self.unlearned = False
        if len(feedbacks) > 0:
            is_feasible = (self.R >= 0)
            self.estimator = self.estimator.fit(self.Z[is_feasible], self.R[is_feasible])
        self.round = self.round + 1
        return self


    def _predict(self, Z, min_variance=0.0):
        mu = np.zeros(Z.shape[0])
        std = np.zeros(Z.shape[0])
        for tree in self.estimator.estimators_:
            var_tree = tree.tree_.impurity[tree.apply(Z)]
            var_tree[var_tree < min_variance] = min_variance
            mean_tree = tree.predict(Z)
            mu += mean_tree
            std += var_tree + mean_tree ** 2
        mu /= self.n_estimators
        std /= self.n_estimators
        std -= mu ** 2.0
        std[std < 0.0] = 0.0
        sigma = std ** 0.5
        return mu, sigma



class CgpucbExplainer():

    def __init__(self, cost_type=None, plausibility_constraint=1.0, gamma=1.0, kernels=None, composite_type='prod', n_init=1):
        self.cost_type = None
        self.plausibility_constraint = plausibility_constraint
        self.fit_verifier = False
        self.gamma = gamma
        self.kernels = kernels
        self.composite_type = composite_type        
        self.n_init = n_init


    def initialize(self, n_counterfactuals, n_features, n_rounds, n_timeout):
        self.round = 0 
        self.Z = np.zeros((n_rounds, 2 * n_features))
        self.R = -1 * np.ones(n_rounds)
        if self.kernels is None:
            ker1 = RBF(input_dim=n_features, variance=1., lengthscale=1., active_dims=np.arange(n_features))
            ker2 = RBF(input_dim=n_features, variance=1., lengthscale=1., active_dims=np.arange(n_features, n_features*2))
        else:
            ker1 = self.kernels[0]
            ker2 = self.kernels[1]
        if self.composite_type == 'prod':
            self.composite_kernel = ker1 * ker2
        elif self.composite_type == 'add':
            self.composite_kernel = ker1 + ker2
        self.beta = 1.0
        return self


    def select(self, feasibility, Z):
        if self.round == 0:
            k_chosen = np.random.choice(feasibility)
        else:
            self.beta = self.beta * self.gamma
            kappa = np.sqrt(self.beta * np.log((self.round+1)) / (self.round+1))
            mu, sigma = self._predict(Z)            
            obj = mu + kappa * sigma
            k_chosen = feasibility[np.argmax(obj[feasibility])]
        self.Z[self.round] = Z[k_chosen]
        self.R[self.round] = 0
        return k_chosen


    def update(self, feedbacks):
        for feedback in feedbacks:
            round, reward = feedback
            self.R[round] = reward
        is_feasible = (self.R >= 0)
        self.gp = GPClassification(self.Z[is_feasible], self.R[is_feasible].reshape(-1, 1), self.composite_kernel)
        if self.round % 10 == 0:
            self.gp.optimize(messages=False)
        self.round = self.round + 1
        return self


    def _predict(self, Z):
        mu, _ = self.gp.predict(Z)
        K_inv = self.gp.posterior.woodbury_inv
        K_test = self.gp.kern.K(Z, self.gp.X)
        k = self.gp.kern.Kdiag(Z)
        variance = k - np.sum(K_test @ K_inv * K_test, axis=1)
        sigma = np.sqrt(variance.reshape(-1, 1))
        return mu, sigma
    
    
    