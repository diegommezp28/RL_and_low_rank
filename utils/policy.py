from collections import defaultdict
import random
import numpy as np
from operator import itemgetter

class DeterministicPolicy():

    def __init__(self, states: int, actions: int, init_policy: np.ndarray = None) -> None:
        random.seed(0)
        self.states = states
        self.actions = actions
        self.policy = init_policy if init_policy is not None else np.random.randint(0, self.actions, size=self.states)
        


    def get_action(self, state):
        return self.policy[state]


class SimplexPlanner():

    def __init__(self, states, actions, rank, U) -> None:
        """
        # Parameters
        U: Tensor of shape (states, actions, rank) where U[s,a,:] = phi(s, a)
        """
        self.states = states
        self.actions = actions
        self.d = rank
        self.U = U

        self.polices = [np.argmax(self.U[..., i], axis=-1) for i in range(self.d)]


    def get_action(self, s):
        i = random.randint(0, (self.d - 1))
        return self.polices[i][s]

    
        



        
        