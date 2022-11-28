import numpy as np
from scipy.stats.distributions import norm
from scipy.special import softmax
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader


class SimplexEnvironment:
    def __init__(self, states=100, actions=20, bell_rank=10):
        self.S = states
        self.A = actions
        self.d = bell_rank
        self.initial_distrib = self.unif_over_states

        self.P = softmax(np.random.uniform(low=-self.d, high=self.d, size=(self.S, self.A, self.d)) , axis=2)
        self.U = softmax(np.random.uniform(-self.d, self.d, (self.S, self.d)), axis=0)
    
    
    def Phi(self, s, a):
        return self.P[s, a, :]
    
    def Mu(self, s):
        return self.U[s, :].T # s entre 0 y S-1 

    def T(self, s, a, s_):
        return np.dot(self.Phi(s, a), self.Mu(s_))

    def unif_over_states(self, states=100):
        return np.random.randint(low=0, high=states)

    def unif_over_actions(self, current_state, actions):
        return np.random.randint(low=0, high=actions)

    def next_step_distrib(self, s, a):
        return np.dot(self.Phi(s,a), self.U.T)

    def next_step_state(self, s, a):
        return np.random.choice(self.S, p=self.next_step_distrib(s,a))
    
    def next_step_reward(self, s, a):
        return norm.rvs()
    
    def next_step(self, s, a):
        """ 
        Gives the next state and reword for the given actions. 
        If `s` is the absorving state (i.e s == self.S - 1. Just by convention)
        Then this will return 0 as reward and a state following the initial state distrib.
        """
        if s == self.S - 1:
            return self.first_step(), 0
        else:
            return self.next_step_state(s, a), self.next_step_reward(s, a)
    
    def first_step(self):
        return self.initial_distrib(self.S)
    
    def get_transitions(self):
        transitions = []
        for a in tqdm(range(self.A)):
            transitions.append([self.P[:, a, :] @ self.U.T])
        return np.vstack(transitions)
        
    def simulate_n_steps(self, n):
        path = []
        prev_s = self.first_step()

        for _ in tqdm(range(n), desc=f"Simulating {n} Steps"):
            a = self.unif_over_actions(prev_s, self.A)
            s = self.next_step_state(prev_s, a)
            path.append((prev_s, a, s))
        
        return path

    def simulate_n_steps_policy(self, n, policy):
        path = []
        prev_s = self.first_step()

        for _ in tqdm(range(n), desc=f"Simulating {n} Steps and custom policy"):
            a = policy(prev_s, self.A)
            s = self.next_step_state(prev_s, a)
            path.append((prev_s, a, s))
        
        return path

    def simulate_n_steps_rewards(self, n):
        path = []
        prev_s = self.first_step()

        for i in tqdm(range(n), desc=f"Simulating {n} Steps with rewards"):
            a = self.unif_over_actions(prev_s, self.A)
            s, r = self.next_step(prev_s, a)
            path.append((prev_s, a, s, r))
        
        return path

    def simulate_n_steps_rewards_policy(self, n, policy=None):
        path = []
        prev_s = self.first_step()

        for i in tqdm(range(n), desc=f"Simulating {n} Steps with rewards and custom policy"):
            a = policy(prev_s, self.A)
            s, r = self.next_step(prev_s, a)
            path.append((prev_s, a, s, r))
        
        return path
            


class ListDataset():
    def __init__(self, l) -> None:
        # super().__init__()

        self.l = l

    def __len__(self):
        return len(self.l)

    def __getitem__(self, idx):
        return self.l[idx]
