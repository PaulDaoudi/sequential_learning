import numpy as np
import matplotlib.pyplot as plt


class EXP3:
    """
    EXP3 algorithm.
    Args:
        eta
        T: number of time steps to perform
        K: number of arms
        L: losses
        q: aversarial weights
        optimize_adversary: if the adversarial learns or not
    """
  
    def __init__(self, eta, K, L, q, optimize_adversary=False):
        self.eta = eta
        self.K = K
        self.p = np.ones(K)/K
        self.L = L
        self.q = q
        self.loss_list = []
        self.p_list = [self.p]
        self.optimize_adversary = optimize_adversary
        self.q_list = [self.q]
        self.regrets = []
    
    def estimated_loss(self, i, L_ij, advers=False):
        l_hat = np.zeros(self.K)
        if advers:
            l_hat[i] = L_ij / self.q[i]
        else:
            l_hat[i] = L_ij / self.p[i]
        return l_hat

    def EXP3_update(self, i, L_ij):
        l_hat = self.estimated_loss(i, L_ij)
        return self.p*np.exp(-self.eta*l_hat)/np.sum(self.p*np.exp(-self.eta*l_hat))
    
    def adversary_update(self, i, L_ij):
        l_hat = self.estimated_loss(i, L_ij, advers=True)
        return self.q*np.exp(-self.eta*l_hat)/np.sum(self.q*np.exp(-self.eta*l_hat))
    
    def step(self):
        
        # Choose action
        action = np.random.choice(range(self.K), p=self.p)
        advers_a = np.random.choice(range(self.K), p=self.q)
        
        # Update player
        L_ij = self.L[action, advers_a]   
        self.p = self.EXP3_update(action, L_ij)
        
        # Update adversary if required
        if self.optimize_adversary:
            L_ij = self.L[advers_a, action]  
            self.q = self.adversary_update(advers_a, L_ij)
            self.q_list.append(self.q.copy())
        
        # Logs
        self.loss_list.append(self.L[action, advers_a])
        self.regrets.append(self.L[action, advers_a] - self.L[1, advers_a])
        self.p_list.append(self.p.copy())
    
    def run(self, T):
        for t in range(T):
            self.step()
    
