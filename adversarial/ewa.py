import numpy as np

class EWA:
    """
    Exponentially Weighted Average algorithm.
    Args:
        eta
        T: number of time steps to perform
        K: number of arms
        L: losses
        q: aversarial weights
        optimize_adversary: if the adversary learns or not
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

    def EWA_update(self, l):
        return self.p*np.exp(-self.eta*l)/np.sum(self.p*np.exp(-self.eta*l))
    
    def adversary_update(self, l):
        return self.q*np.exp(-self.eta*l)/np.sum(self.q*np.exp(-self.eta*l))
    
    def step(self):
        
        # Choose action
        action = np.random.choice(range(self.K), p=self.p)
        advers_a = np.random.choice(range(self.K), p=self.q)
        
        # Update player
        l = self.L[:, advers_a].copy()
        self.p = self.EWA_update(l)
        self.p_list.append(self.p.copy())
        
        # Update adversary if required
        if self.optimize_adversary:
            l_advers = self.L[:, action].copy()
            self.q = self.adversary_update(l_advers)
            self.q_list.append(self.q.copy())
        
        # Logs
        self.loss_list.append(self.L[action, advers_a])
        self.regrets.append(self.L[action,advers_a] - self.L[1,advers_a])
    
    def run(self, T):
        for t in range(T):
            self.step()
    


