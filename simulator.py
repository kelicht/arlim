import numpy as np
from time import time
from tqdm import tqdm
from scipy.interpolate import interp1d
from scipy.stats import gaussian_kde as kde
from scipy.stats import bernoulli, geom
from sklearn.base import clone
from sklearn.ensemble import IsolationForest, RandomForestClassifier



COST_TYPES = [
    'mps',
    'norm',
    'mahalanobis',
]

class Simulator():
    def __init__(
        self,
        estimator,
        dataset,
        n_rounds=1000,
        n_timeout=10,
        max_counterfactuals=200,
        cost_type='norm',
        delay_type=0.2,
        improvement_noise=0.0,
        probability_noise=0.0,
        verbose=False
    ):
        self.estimator = estimator
        self.dataset = dataset
        self.n_rounds = n_rounds
        self.n_timeout = n_timeout
        self.max_counterfactuals = max_counterfactuals
        self.cost_type = cost_type
        self.delay_type = delay_type
        self.improvement_noise = improvement_noise
        self.probability_noise = probability_noise
        self.verbose = verbose


    def initialize(self, fit_verifier=False):
        self.dataset = self.dataset.initialize()

        X_tr, y_tr = self.dataset.getTrainingSamples()
        self.estimator = self.estimator.fit(X_tr, y_tr)
        self.n_features_ = X_tr.shape[1]

        counterfactuals, theta = self.dataset.getCounterfactuals(self.estimator)
        if counterfactuals.shape[0] <= self.max_counterfactuals:
            self.counterfactuals_ = counterfactuals
            self.theta_ = theta
        else:
            selected = np.random.choice(counterfactuals.shape[0], self.max_counterfactuals, replace=False)
            self.counterfactuals_ = counterfactuals[selected]
            self.theta_ = theta[selected]
        self.n_counterfactuals_ = self.counterfactuals_.shape[0]
        self.theta_with_noise_ = np.clip(self.theta_ + np.random.randn(self.n_counterfactuals_) * self.improvement_noise, 0, 1)

        X = self.dataset.getInstances(self.estimator)      
        selected = np.random.choice(X.shape[0], self.n_rounds)
        self.X_ = X[selected]

        self.weights_ = np.ones(self.n_features_)       
        self.percentiles_ = []
        for d in range(X_tr.shape[1]):
            if self.dataset.is_immutable[d]:
                self.percentiles_.append(None)
                continue
            kde_estimator = kde(X_tr[:, d])
            grid = np.linspace(np.quantile(X_tr[:, d], 0.001), np.quantile(X_tr[:, d], 0.999), 100)
            pdf = kde_estimator(grid)
            cdf_raw = np.cumsum(pdf)
            total = cdf_raw[-1] + 1e-6 + 1e-6
            cdf = (1e-6 + cdf_raw) / total
            p_d = interp1d(x=grid, y=cdf, copy=False, fill_value=(1e-6, 1.0 - 1e-6), bounds_error=False, assume_sorted=False)
            self.percentiles_.append(p_d)
        self.invcov_ = np.linalg.pinv(np.cov(X_tr.T))
        
        self.outlier_scores_ = -1 * IsolationForest().fit(X_tr[y_tr==1]).score_samples(self.counterfactuals_)

        if fit_verifier:
            left, right = np.random.choice(X_tr.shape[0], 10000), np.random.choice(X_tr.shape[0], 10000)
            X_left, X_right = X_tr[left], X_tr[right]
            y_left, y_right = y_tr[left], y_tr[right]
            Z = np.concatenate([X_left, X_right], axis=1)
            c = (y_left == y_right).astype(int)
            self.verifier_ = clone(self.estimator)
            self.verifier_ = self.verifier_.fit(Z, c)
            
            is_different = (c == 0)
            verifier_score = self.verifier_.predict_proba(Z[is_different])[:, 1]
            confidence_left, confidence_right = self.estimator.predict_proba(X_left[is_different])[:, 1], self.estimator.predict_proba(X_right[is_different])[:, 1]
            trust_score = abs(verifier_score - (confidence_left * confidence_right + (1-confidence_left) * (1-confidence_right)))
            self.threshold_ = np.quantile(trust_score, 0.3)
            
            Z_left, Z_right = np.repeat(self.X_, self.n_counterfactuals_, axis=0), np.tile(self.counterfactuals_, (self.n_rounds, 1))
            Z = np.concatenate([Z_left, Z_right], axis=1)
            self.verifier_score_ = self.verifier_.predict_proba(Z)[:, 1].reshape(self.n_rounds, self.n_counterfactuals_)
            self.confidence_x_ = self.estimator.predict_proba(self.X_)[:, 1]
            self.confidence_cf_ = self.estimator.predict_proba(self.counterfactuals_)[:, 1]
            
        else:
            self.verifier_ = None

        return self


    def getCandidateActions(self, t, x, plausibility_constraint):
        action = self.counterfactuals_ - x
        
        is_feasible = (abs(action[:, (self.dataset.is_immutable)]) == 0).all(axis=1)
        is_feasible = is_feasible * (self.outlier_scores_ <= plausibility_constraint)
        if self.verifier_ is not None:
            trust_score = abs(self.verifier_score_[t] - (self.confidence_x_[t] * self.confidence_cf_ + (1-self.confidence_x_[t]) * (1-self.confidence_cf_)))
            is_trustworthy = (trust_score <= self.threshold_)
            is_feasible = is_feasible * is_trustworthy
        feasibility = np.where(is_feasible)[0]

        cost_dict = {}
        cost_dict['norm'] = np.sum(abs(action) * self.weights_, axis=1)
        cost = np.zeros(self.n_counterfactuals_)
        for d in range(x.shape[0]):
            if self.dataset.is_immutable[d]: continue
            diffs = abs(self.percentiles_[d](self.counterfactuals_[:, d]) - self.percentiles_[d](x[d]))
            cost = np.maximum(cost, diffs)                
        cost_dict['mps'] = cost
        cost_dict['mahalanobis'] = np.sqrt(np.sum(action @ self.invcov_ * action, axis=1))
        
        probability_dict = {}
        for cost_type in COST_TYPES:
            if cost_type == 'mahalanobis':
                nu = 1 / (2 * self.n_features_)
            else:
                nu = 1 / self.n_features_
            probability_dict[cost_type] = np.exp(- nu * cost_dict[cost_type])

        return feasibility, cost_dict, probability_dict


    def getExpectedRewards(self, probability):
        return self.theta_with_noise_ * probability

    
    def getBestReward(self, feasibility, probability):
        reward = self.getExpectedRewards(probability)
        reward_best = np.max(reward[feasibility])
        return reward_best

    
    def play(self, k, prob_k):
        mean = self.theta_with_noise_[k] * prob_k
        y = bernoulli(mean).rvs()
        return y


    def run(self, explainer):
        
        self = self.initialize(explainer.fit_verifier)
        explainer = explainer.initialize(self.n_counterfactuals_, self.n_features_, self.n_rounds, self.n_timeout)
        feedbacks = [[] for t in range(self.n_rounds)]

        k = np.zeros(self.n_rounds)
        regrets = np.zeros(self.n_rounds)
        rewards = np.zeros(self.n_rounds)
        expected_rewards = np.zeros(self.n_rounds)
        costs = np.zeros(self.n_rounds)
        probabilities = np.zeros(self.n_rounds)
        improvements = np.zeros(self.n_rounds)        
        sparsity = np.zeros(self.n_rounds)
        plausibility = np.zeros(self.n_rounds)
        times = np.zeros(self.n_rounds)        
        
        for t in tqdm(range(self.n_rounds), disable=(not self.verbose)):
            start_time = time()
            x_t = self.X_[t]
            feasibility, cost_dict, probability_dict = self.getCandidateActions(t, x_t, explainer.plausibility_constraint)

            if len(feasibility) > 0:
                if self.cost_type == 'mix':
                    cost_type = 'mps' if self.dataset.getIsYounger(x_t) else 'mahalanobis'
                elif self.cost_type == 'random':
                    cost_type = np.random.choice(COST_TYPES)
                elif self.cost_type in COST_TYPES:
                    cost_type = self.cost_type

                cost = cost_dict[cost_type]
                probability = probability_dict[cost_type]
                expected_reward_best = self.getBestReward(feasibility, probability)
                
                if explainer.cost_type is None:
                    Z = np.concatenate([np.tile(x_t, (self.n_counterfactuals_, 1)), self.counterfactuals_ - x_t], axis=1)
                    k_chosen = explainer.select(feasibility, Z)
                else:
                    probability_with_noise = np.clip(probability_dict[explainer.cost_type] + np.random.randn(self.n_counterfactuals_) * self.probability_noise, 0, 1)
                    k_chosen = explainer.select(feasibility, probability_with_noise)
                
                prob_k = probability[k_chosen]
                cost_k = cost[k_chosen]
                improvement_k = self.theta_[k_chosen]
                plausibility_k = self.outlier_scores_[k_chosen]
                sparsity_k = (abs(self.counterfactuals_[k_chosen] - x_t) > 0.05).sum()
                expected_reward = self.theta_with_noise_[k_chosen] * prob_k
                reward = self.play(k_chosen, prob_k)           

                if self.delay_type == 'adaptive':
                    p = 0.2 if self.dataset.getIsYounger(x_t) else 0.05
                    delay = geom.rvs(p)
                elif self.delay_type > 0.0 and self.delay_type < 1.0:
                    delay = geom.rvs(self.delay_type)

                if reward != 0 and delay <= self.n_timeout and t + delay < self.n_rounds:
                    if explainer.cost_type is None:
                        feedbacks[t + delay - 1].append((t, reward))
                    else:
                        feedbacks[t + delay - 1].append((k_chosen, probability_dict[explainer.cost_type][k_chosen], reward))

            else:
                expected_reward_best = 0.0
                k_chosen = -1
                prob_k = 1.0
                cost_k = 0.0
                improvement_k = 0
                plausibility_k = 0.5
                sparsity_k = 0
                expected_reward = 0
                reward = 0                

            explainer = explainer.update(feedbacks[t])
            k[t] = k_chosen
            regrets[t] = expected_reward_best - expected_reward
            rewards[t] = reward
            expected_rewards[t] = expected_reward
            costs[t] = cost_k
            probabilities[t] = prob_k
            improvements[t] = improvement_k
            plausibility[t] = plausibility_k
            sparsity[t] = sparsity_k
            times[t] = time() - start_time
            
        cumulative_regrets = np.zeros(self.n_rounds)
        mean_rewards = np.zeros(self.n_rounds)
        mean_expected_rewards = np.zeros(self.n_rounds)
        cumulative_regrets[0] = regrets[0]
        mean_rewards[0] = rewards[0]
        mean_expected_rewards[0] = expected_rewards[0]
        for t in range(1, self.n_rounds):
            cumulative_regrets[t] = cumulative_regrets[t-1] + regrets[t]
            mean_rewards[t] = (mean_rewards[t-1] * t + rewards[t]) / (t+1)
            mean_expected_rewards[t] = (mean_expected_rewards[t-1] * t + expected_rewards[t]) / (t+1)
        
        results = {
            'k': k,
            'regret': regrets,
            'cumulative_regret': cumulative_regrets,
            'reward': rewards,
            'mean_reward': mean_rewards,
            'expected_reward': expected_rewards,
            'mean_expected_reward': mean_expected_rewards,
            'cost': costs,
            'probability': probabilities,
            'improvement': improvements,
            'plausibility': plausibility,
            'sparsity': sparsity,
            'time': times,
        }
        return results