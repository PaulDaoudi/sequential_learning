import numpy as np
from arms import Bernoulli

class KL_UCB:
    """
    KL-UCB algorithm using the bisection method.
    Args:
        T: number of time steps to perform
        K: number of arms
        list_p: list of each arm's mean
    """
    
    def __init__(self, T, K, list_p, eps=1e-3, max_iters=50):        
        # Global parameters
        self.T = T
        self.K = K
        self.list_p = list_p
        self.arms = [Bernoulli(p) for p in list_p]
        self.best_arm = np.argmax(self.list_p)
        
        # For bissection
        self.eps = eps
        self.max_iters = max_iters
            
    def kl_bernoulli(self, mu_1, mu_2):
        """
        kl between two bernouilli distributions
        """
        if abs(mu_2 - 1) < 1e-5:
            return 1e100
        if abs(mu_1) < 1e-5:
            return 0
        kl_b = mu_1 * np.log(mu_1/mu_2) + (1-mu_1) * np.log((1-mu_1)/(1-mu_2))
        return kl_b
            
    def upperbounds(self, t):
        """Returns the upperbound for all arms. Used in the bisection algorithm."""
        t += 1
        return np.log(1 + t*((np.log(t))**2))/(self.counts)
    
    def run(self):
        
        # Initialize
        self.counts = np.zeros(self.K)
        self.sum_rewards_arm = np.zeros(self.K)
        self.actions = np.zeros(self.T)
        self.rewards = np.zeros(self.T)
        self.chosen_mean = np.zeros(self.T)
        
        # Run for T time steps
        for t in range(self.T):
            
            # Select arm
            if t < self.K:
                action = t
            else:
                upperbounds = self.upperbounds(t)
                choices = [self.retrieve_inner_max(arm, upperbounds) for arm in range(self.K)]
                action = np.argmax(choices)

            # Sample arm and observe reward
            arm = self.arms[action]
            reward = arm.sample()
            
            # Update parameters
            self.counts[action] += 1
            self.sum_rewards_arm[action] += reward
            self.actions[t] = action
            self.rewards[t] = reward
            self.chosen_mean[t] = self.list_p[action]
            
        # Compute regret
        time_steps = np.arange(self.T)+1
        self.regret = np.max(self.list_p)*time_steps - np.cumsum(self.chosen_mean)
            
    def retrieve_inner_max(self, arm, upperbounds):
        """
        Compute the maximum mean with the bisection method.
        """
        
        # Initialize parameters
        mean = self.sum_rewards_arm[arm]/self.counts[arm]
        
        # print('mean:', mean)
        upperbound = upperbounds[arm]
        u = 1
        l = mean
        n = 0

        # Bisection method
        while n < self.max_iters and u - l > self.eps:
            q = (l + u)/2
            if self.kl_bernoulli(mean, q) > upperbound:
                u = q
            else:
                l = q
            n += 1

        return (l+u)/2