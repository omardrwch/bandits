"""
Implements the Upper Confidence Bound (UCB) algorithm
"""

import numpy as np
import arms
import matplotlib.pyplot as plt


class UCB():
    """
    T (int): number of iterations
    """
    def __init__(self, arms_list,  T = 10000, store_time_data = True, plot = True):
        self.arms_list = arms_list
        self.T = T
        self.store_time_data = store_time_data
        self.plot = plot

    def run(self):
        K = len(self.arms_list)
        S = np.zeros(K) # sum of rewards for each arm
        N = np.zeros(K) # number of times each arm has been chosen

        if self.store_time_data:
            self.choices = -np.ones(self.T)
            self.rewards = np.zeros(self.T)

        t = 0 
        while t < self.T:
            if t < K:
                arm_idx = t
            else:
                f_t = self.f(t)
                means = S/N
                indexes =  means + np.sqrt( ( 2*np.log(f_t)/N ) )
                arm_idx = np.argmax(indexes)

            arm = self.arms_list[arm_idx]
            r_t = arm.sample() # reward
            S[arm_idx] += r_t
            N[arm_idx] += 1

            if self.store_time_data:
                self.choices[t] = arm_idx
                self.rewards[t] = r_t

            t += 1
            # if t%500 == 0:
            #     print("iteration = ", t)


    def estimate_regret(self, n_runs = 50):
        K = len(self.arms_list)
        means = np.array([arm.mean() for arm in self.arms_list])
        mu_star = means.max()
        deltas  = means.max() - means

        regret = np.zeros(self.T)
        for run in range(1, n_runs+1):
            self.run()
            R_n = np.zeros(self.T)
            for k in range(K):
                R_n += deltas[k]*np.cumsum( self.choices == k )

            regret = ((run-1)/run)*regret + (1/run)*R_n

        self.regret = regret
        if self.plot:
            self.plot_regret()

        return regret

    def f(self, t):
        """
        Get f(t) = 1/delta(t) for a confidence level of 1-delta(t) at time t
        """
        t = t + 1 # t starts from 0 in self.run, but f(t) needs t>0
        return 1.0 + t*np.power(np.log(t), 2.0)
        

    def plot_regret(self):
        plt.figure()
        plt.title('UCB regret')
        plt.plot( np.arange(1, ucb.T+1), self.regret )
        plt.grid()
        plt.show()



if __name__ == '__main__':
    arms_list = []
    arms_list += [ arms.Bernoulli(p) for p in [0.1, 0.9]]
    arms_list += [ arms.Uniform() ] 

    ucb = UCB(arms_list)
    regret = ucb.estimate_regret()
