import numpy as np
from scipy.stats.distributions import norm
from scipy.special import softmax
from tqdm import tqdm
from torch.utils.data import Dataset
import torch
from numpy.linalg import solve
from .policy import DeterministicPolicy


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
            


class ListDataset(Dataset):
    def __init__(self, l, batch_size=4) -> None:
        super().__init__()
        self.l = l
        self.batch_size = batch_size

    def __len__(self):
        return len(self.l)

    def __getitem__(self, idx):
        return (self.l[idx][:2]), self.l[idx][2]

    def __iter__(self):
        l0, l1, l2 = [], [], []
        for i in self.l:
            l0.append(i[0]); l1.append(i[1]); l2.append(i[2])
            if len(l0) == self.batch_size:
                yield torch.tensor(l0), torch.tensor(l1), torch.tensor(l2)
                l0, l1, l2 = [], [], []

class PolicyIteration():
    def __init__(self, states, actions, next_state_prob , reward_func, discount = 0.5, init_policy=None):

        # Number of states
        self.states = states
        # Number of actions
        self.actions = actions

        # Should receive n entries (s, a) should give all the probabilities for all the possible states
        self.next_state_prob = next_state_prob
        # Should receive (s, a, s') and return a reward
        self.reward_func = reward_func 

        self.discount = discount

        # Initial policiy Will be a Random policy if nothing else is given
        self.policy = DeterministicPolicy(self.states, self.actions) if init_policy is None else init_policy

        # Value function
        self.V = np.zeros(self.states)

        self.eval_policy = PolicyEvaluation(self.states, self.actions, self.next_state_prob, self.reward_func)

        self.improve_policy = PolicyImprovement(self.states, self.actions, self.next_state_prob, self.reward_func)


    def run(self, max_iter = 2000, print_progress = True):
        for _ in tqdm(range(max_iter), disable=(not print_progress)):
            self.V = self.eval_policy.evaluatePolicy(self.policy.get_action, self.discount)
            new_policy = self.improve_policy.improvePolicy(self.V, self.discount)

            if np.array_equal(self.policy.policy, new_policy):
                break

            self.policy.policy = new_policy

        
        return self.V, self.policy.policy




class PolicyEvaluation():
    def __init__(self, states, actions, next_state_prob, reward_func):

        # Number of states
        self.states = states
        # Number of actions
        self.actions = actions
        # Should receive n entries (s, a) should give all the probabilities for all the possible states
        self.next_state_prob = next_state_prob
        # Should receive (s, a, s') and return a reward
        self.reward_func = reward_func 


    def evaluatePolicy(self, policy, discount=0.5):
        """
        # Parameters:
        policy: Deterministic Policy: Should receive s or list of s and return a_s or list of a_s
        """
        
        s_arr = np.arange(self.states)
        a_arr = policy(s_arr) # Actions according to current policy

        P = discount * self.next_state_prob(s_arr, a_arr) # should return array of shape (states, states)

        s1 = np.repeat(s_arr, [self.states] * self.states)
        s2 = np.tile(s_arr, self.states)
        a1 = np.repeat(a_arr, [self.states] * self.states)

        R = self.reward_func(s1, a1, s2).reshape(self.states, self.states)
        PRs = np.sum(P * R, axis = -1)

        M = P - np.eye(self.states)

        return solve(M, -PRs)


class PolicyImprovement():
    def __init__(self, states, actions, next_state_prob, reward_func) -> None:
        # Number of states
        self.states = states
        # Number of actions
        self.actions = actions

        # Should receive n entries (s, a) should give all the probabilities for all the possible states
        self.next_state_prob = next_state_prob
        # Should receive (s, a, s') and return a reward
        self.reward_func = reward_func 

        s_l = self.states
        a_l = self.actions

        # All actions and states
        s_arr = np.arange(s_l)
        a_arr = np.arange(a_l)

        # Index for Calculating Probs of shape (s,a,s) where prob[s,a,s'] = P(s,a,s')
        self.s1_probs = np.repeat(s_arr, [a_l] * s_l).astype(np.int32)
        self.a1_probs = np.tile(a_arr, s_l).astype(np.int32)
        self.probs = self.next_state_prob(self.s1_probs, self.a1_probs).reshape(s_l, -1, s_l)

        # Indexes For Rewards and Value Functions of shape (s,a,s) where V[s, a, s'] = V(s') and R[s, a, s'] = R(s, a, s')
        self.s1 = np.repeat(s_arr, [s_l * a_l] * s_l).reshape(s_l, -1, s_l)
        self.a1 = np.tile(np.repeat(a_arr, [s_l]*a_l), s_l).reshape(s_l, -1, s_l)
        self.s2 = np.tile(s_arr, s_l * a_l).reshape(s_l, -1, s_l)

        self.R = self.reward_func(self.s1, self.a1, self.s2)
        

    def improvePolicy(self, V, discount=0.5) -> np.ndarray:

        V_index = V[self.s2]

        # Calculate Bellman Operator where Pi[s, a, s'] = p(s, a, s') * [R(s, a, s') + discount * V(s')]
        Pi = self.probs * (self.R + discount * V_index)
        # Sum over last index such that Pi_sum[s, a] = \sum_{s' \in S}  p(s, a, s') * [R(s, a, s') + discount * V(s')]
        Pi_sum = np.sum(Pi, axis=-1)

        # Calculate Optimal Policy for each state
        optimal_policy = np.argmax(Pi_sum, axis = -1)
        return optimal_policy




# TODO complete Value Iteration Algorithm 
class ValueIteration():
    def __init__(self, states, actions, next_state_prob , reward_func):

        # Number of states
        self.states = states
        # Number of actions
        self.actions = actions

        # Should receive n entries (s, a) should give all the probabilities for all the possible states
        self.next_state_prob = next_state_prob
        # Should receive (s, a, s') and return a reward
        self.reward_func = reward_func 

    
    def run(self, gamma = 1, delta = 0.1, max_iter = 500_000, values = None):


        V = np.array([0 for _ in range(self.states)]) if values is None else np.array(values)
        diffs = []

        s_l = self.states
        a_l = self.actions

        s_arr = np.arange(s_l)
        a_arr = np.arange(a_l)

        s1 = np.repeat(s_arr, [s_l * a_l] * s_l).reshape(s_l, -1, s_l)
        a1 = np.tile(np.repeat(a_arr, [s_l]*a_l), s_l).reshape(s_l, -1, s_l)
        s2 = np.tile(s_arr, s_l * a_l).reshape(s_l, -1, s_l)

        print(s1.shape, a1.shape, s2.shape)
        R  = self.reward_func(s1, a1, s2)
        print(R.shape)


        s1_probs = np.repeat(s_arr, [a_l] * s_l)
        a1_probs = np.tile(a_arr, s_l)
        print("Probs:" ,s1_probs.shape, a1_probs.shape)
        probs = self.next_state_prob(s1_probs, a1_probs).reshape(s_l, -1, s_l)



        with tqdm(enumerate(range(max_iter)), total=max_iter) as tqdm_iter:
            for i, _ in tqdm_iter:
                # max_diff = 0 # Current max diff between old and new values
                new_values = np.zeros(self.states)
                tqdm_iter.set_postfix(i=f'{i}')

                V_arr = np.tile(V, s_l * a_l).reshape(s_l, -1, s_l)
                operator = probs * (R + gamma * V_arr)
                operator_sum = np.sum(operator, axis = -1).reshape(s_l, a_l, 1)
                new_V = operator_sum.max(axis=1).reshape(-1)
                tqdm_iter.set_postfix(shape=f'{new_V.shape}')

                max_diff = np.max(np.abs(V - new_V))
                diffs.append(max_diff)
                # tqdm_iter.set_postfix(diff=f'{max_diff:.3f}')
                V = new_V

                # # for s in range(self.states - 1): # Since is episodic and (s-1) is the terminal state -> V(s-1) = 0 always
                # #     max_val = 0
                # #     # next_states = []

                # #     for a in range(self.actions):
                # #         states_probs = self.next_state_prob(s, a)
                # #         rewards = np.array([self.reward_func(s, a, s_) + gamma*V[s_] for s_ in range(self.states)])
                # #         new_val = np.dot(states_probs, rewards)
                # #         max_val = max(max_val, new_val)

                    # V_new[s] = max_val  # Update value with highest value
                #     max_diff = max(max_diff, abs(V[s] - max_val))
                
                if max_diff <= delta:
                    break
        
        return V, diffs





        
        

                








