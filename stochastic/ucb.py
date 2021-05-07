import numpy as np
from arms import Bernoulli

class UCB:
    """
    UCB(sigma^2) algorithm.
    Args:
        T: number of time steps to perform
        K: number of arms
        list_p: list of each arm's mean
        sigma: standard deviation of the sub-gaussian random variables
    """
    
    def __init__(self, T, K, list_p, sigma):    
        self.T = T
        self.K = K
        self.list_p = list_p
        self.arms = [Bernoulli(p) for p in list_p]
        self.sigma = sigma
        
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
                bonus = self.ucb_bonus(t)
                means = self.sum_rewards_arm / self.counts
                optimistic_means =  means + bonus
                action = np.argmax(optimistic_means)

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
            
    def ucb_bonus(self, t):
        t += 1 # We need to start at one
        return np.sqrt(2*(self.sigma**2)*np.log(t)/self.counts)