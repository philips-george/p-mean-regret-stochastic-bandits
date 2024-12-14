import numpy as np
from abc import ABC, abstractmethod

from argparse import ArgumentParser

rng = np.random.default_rng(12479)

class StochasticArm(ABC):
    def __init__(self, mean, lb, ub):
        self.mean = mean
        self.lb = lb
        self.ub = ub
    # End fn __init__
    
    @abstractmethod
    def pull(self):
        pass
# End class StochasticArm

class BernoulliArm(StochasticArm):
    def __init__(self, param):
        super().__init__(mean=param, lb=0., ub=1.)
    # End fn __init__
    
    def pull(self):
        return 1. if rng.uniform(0., 1.) <= self.mean else 0.
    # End fn pull
    
    def __str__(self):
        return 'BernoulliArm(%.3f)' % self.mean
# End class BernoulliArm

class TriangularArm(StochasticArm):
    def __init__(self, mode):
        self.mode = mode
        super().__init__(mean=(1.+mode)/3, lb=0., ub=1.)
    # End fn __init__
    
    def pull(self):
        return rng.triangular(0., self.mode, 1.)
    # End fn pull
    
    def __str__(self):
        return 'TriangularArm(lower=%.2f, upper=%.2f, mode=%.2f)' % (self.lb, self.ub, self.mode)
# End class TriangularArm

class BetaArm(StochasticArm):
    def __init__(self, alpha, beta):
        self.alpha = alpha
        self.beta = beta
        super().__init__(mean=(1.*alpha/(alpha + beta)), lb=0., ub=1.)
    # End fn __init__
    
    def pull(self):
        return rng.beta(self.alpha, self.beta)
    # End fn pull
    
    def __str__(self):
        return 'BetaArm(alpha=%.3f, beta=%.3f)' % (self.alpha, self.beta)
# End class BetaArm

class UniformArm(StochasticArm):
    def __init__(self, lb, ub):
        super().__init__(mean=0.5*(lb + ub), lb=lb, ub=ub)
    # End fn __init__
    
    def pull(self):
        return rng.uniform(self.lb, self.ub)
    # End fn pull
    
    def __str__(self):
        return 'UniformArm(lb=%.3f, ub=%.3f)' % (self.lb, self.ub)
# End class UniformArm

class StochasticBandit:
    def __init__(self, k):
        self.k = k
        self.arms = []
    # End fn __init__
    
    def add_arm(self, arm):
        assert (len(self.arms) < k)
        self.arms.append(arm)
    # End fn add_arm
    
    def choose_arm(self, i):
        assert (i >= 0 and i < len(self.arms))
        return self.arms[i].pull()
    # End fn choose_arm
# End class StochasticBandit

def boost_reward(x, delta):
    return x + delta
# End fn boost_reward

def learn_ucb1(bandit, T, n):
    k = bandit.k
    rewards = [[] for _ in range(n)]
    for i in range(n):
        arm_freqs = np.zeros(shape=(k,), dtype=np.int32)
        arm_reward_cumsums = np.zeros(shape=(k,), dtype=np.float64)
        ihat = 0
        for t in range(k):
            reward = bandit.choose_arm(ihat)
            arm_reward_cumsums[ihat] += reward
            arm_freqs[ihat] += 1
            rewards[i].append(reward)
            ihat += 1
        # End for
        for t in range(k, T):
            reward_means = arm_reward_cumsums / arm_freqs
            ucb_widths = 6. * np.sqrt(np.log(T)) * np.sqrt(1./arm_freqs)
            ihat = int(np.argmax(reward_means + ucb_widths))
            reward = bandit.choose_arm(ihat)
            arm_reward_cumsums[ihat] += reward
            arm_freqs[ihat] += 1
            rewards[i].append(reward)
        # End for
    # End for i
    return np.array(rewards)
# End fn learn_ucb1

def learn_ncb(bandit, T, n):
    k = bandit.k
    rewards = [[] for _ in range(n)]
    Ttilde = int(16. * np.sqrt(T*np.log(T)*k/np.log(k)) + 1)
    print('NCB Ttilde = ', Ttilde)
    for i in range(n):
        arm_freqs = np.zeros(shape=(k,), dtype=np.int32)
        arm_reward_cumsums = np.zeros(shape=(k,), dtype=np.float64)
        for t in range(Ttilde):
            ihat = rng.integers(low=0, high=k, size=1)[0]
            reward = bandit.choose_arm(ihat)
            arm_reward_cumsums[ihat] += reward
            arm_freqs[ihat] += 1
            rewards[i].append(reward)
        # End for
        for t in range(Ttilde, T):
            reward_means = arm_reward_cumsums / arm_freqs
            ncb_widths = 4. * np.sqrt(np.log(T)) * np.sqrt(reward_means/arm_freqs)
            ihat = int(np.argmax(reward_means + ncb_widths))
            reward = bandit.choose_arm(ihat)
            arm_reward_cumsums[ihat] += reward
            arm_freqs[ihat] += 1
            rewards[i].append(reward)
        # End for
    # End for i
    return np.array(rewards)
# End fn learn_ncb

def learn_pmean_explore_then_ucb(bandit, p, T, n):
    k = bandit.k
    rewards = [[] for _ in range(n)]
    if p > 0 and p <= 1:
        Ttilde = int(16. * np.sqrt(T*np.log(T)*np.pow(k,p)/np.log(k)) + 1)
    elif p == 0:
        Ttilde = int(16. * np.sqrt(T*np.log(T)*k/np.log(k)) + 1)
    elif p < 0:
        assert(p >= -int(np.log(T)/(2.*np.log(k))+1))
        Ttilde = int(16. * np.sqrt(T*np.log(T)/np.pow(k,-p)) + 1)
    else:
        print('p = %.2f not supported.' % p)
        exit(1)
    # End if
    print('ETUCB Ttilde = ', Ttilde)
    for i in range(n):
        arm_freqs = np.zeros(shape=(k,), dtype=np.int32)
        arm_reward_cumsums = np.zeros(shape=(k,), dtype=np.float64)
        for t in range(Ttilde):
            ihat = rng.integers(low=0, high=k, size=1)[0]
            reward = bandit.choose_arm(ihat)
            arm_reward_cumsums[ihat] += reward
            arm_freqs[ihat] += 1
            rewards[i].append(reward)
        # End for
        for t in range(Ttilde, T):
            reward_means = arm_reward_cumsums / arm_freqs
            ucb_widths = 6. * np.sqrt(np.log(T)) * np.sqrt(1./arm_freqs)
            ihat = int(np.argmax(reward_means + ucb_widths))
            reward = bandit.choose_arm(ihat)
            arm_reward_cumsums[ihat] += reward
            arm_freqs[ihat] += 1
            rewards[i].append(reward)
        # End for
    # End for i
    return np.array(rewards)
# End fn learn_pmean_explore_then_ucb

def compute_pmean_regret(bandit, rewards, p):
    n = rewards.shape[0]
    T = rewards.shape[1]
    opt_reward_mean = max([arm.mean for arm in bandit.arms])
    if p == 0:
        return opt_reward_mean - np.pow(np.prod(np.mean(rewards, axis=0)), 1./T)
    else:
        return opt_reward_mean - np.pow(np.mean(np.pow(np.mean(rewards, axis=0), p)), 1./p)
# End fn compute_pmean_regret

if __name__ == '__main__':
    parser = ArgumentParser(prog="pmean_expts")
    parser.add_argument("instance_type", choices=["bernoulli", "beta", "triangular", "uniform"],)
    # Optional arguments
    parser.add_argument("-p", "--p-param", type=float, default=-1., help="The p parameter to use for computing p-mean-regret.")   
    parser.add_argument("-T", "--rounds", type=int, default=20000, help="The number of rounds (for the bandit algorithm).")
    parser.add_argument("-n", "--num-reps", type=int, default=30, help="The number of repetitions for the inner expectation (see p-mean regret definition).")
    parser.add_argument("-k", "--num-arms", type=int, default=50, help="The number of arms to have in bandit instance.")    
    args = parser.parse_args()

    p = args.p_param
    T = args.rounds
    n = args.num_reps
    k = args.num_arms
    instance_type = args.instance_type
    
    bandit = StochasticBandit(k)
    for a in range(k):
        if instance_type == 'bernoulli':
            # Bernoulli arms
            bandit.add_arm(BernoulliArm(rng.uniform(0.0005, 1.)))
        elif instance_type == 'beta':
            # Beta arms
            bandit.add_arm(BetaArm(alpha=rng.uniform(0.005, 0.995), beta=rng.uniform(0.005, 0.995)))
        elif instance_type == 'triangular':
            # Triangular arms
            bandit.add_arm(TriangularArm(mode=rng.uniform(0.005, 0.995)))
        elif instance_type == 'uniform':
            # Uniform arms
            lb = rng.uniform(0.005, 0.995)
            bandit.add_arm(UniformArm(lb=lb, ub=rng.uniform(lb+0.001, 1.)))
        else:
            print('Unsupported instance type: %s' % instance_type)
            exit(1)
    # End for
    
    print('SIMULATION\n-----------')
    print('Using %.2f-mean regret' % p)
    print('T = %d, num_reps = %d, num_arms = %d' % (T, n, k))
    print('BANDIT ARMS\n-----------')
    for a in bandit.arms: print(a)
    print('')
    
    rewards_eucb = learn_pmean_explore_then_ucb(bandit, p, T, n)
    rewards_ncb  = learn_ncb(bandit, T, n)
    rewards_ucb1 = learn_ucb1(bandit, T, n)
    
    reg_eucb = compute_pmean_regret(bandit, rewards_eucb, p)
    reg_ncb  = compute_pmean_regret(bandit, rewards_ncb, p)
    reg_ucb1 = compute_pmean_regret(bandit, rewards_ucb1, p)
    
    print('%.2f-mean-Regret[%d rounds, %d arms]: UCB1 %.3f, NCB %.3f, ETUCB %.3f' % (p, T, k, reg_ucb1, reg_ncb, reg_eucb))
# End main
