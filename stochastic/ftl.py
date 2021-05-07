import numpy as np
from arms import Bernoulli

class FTL:
    """
    Follow the leader algorithm.
    Args:
        T: number of time steps to perform
        K: number of arms
        list_p: list of each arm's mean
    """
    
    def __init__(self, T, K, list_p):  
        self.T = T
        self.K = K
        self.list_p = list_p
        self.arms = [Bernoulli(p) for p in list_p]
        
    def run(self):
        
        # Initialize
        self.counts = np.zeros(self.K)
        self.sum_rewards_arm = np.zeros(self.K)
        self.actions = np.zeros(self.T)
        self.rewards = np.zeros(self.T)
        self.chosen_mean = np.zeros(self.T)
        
        # Run for T time steps
        for t in range(self.T):
            
            # Select arm: explore all arms first then select the highest mean
            if t < self.K:
                action = t
            else:
                means = self.sum_rewards_arm / self.counts
                action = np.argmax(means)

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