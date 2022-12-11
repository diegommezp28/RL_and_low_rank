import torch.nn as nn
import torch.nn.functional as F
import torch
from .policy import SimplexPlanner
from .utils import PolicyIteration
import numpy as np


class Net(nn.Module): 
    def __init__(self, states, actions, d, criterion, optimizer,  optim_args, real_transitions = None):
        super(Net, self).__init__()
        
        self.states = states
        self.actions = actions
        self.d = d
        self.device = 'cpu'
    
        self.l1 = nn.Linear(self.states + self.actions, 120)  #First Linear layers, Receives concat onehot enconding of state-action pair
        self.l2 = nn.Linear(120, 20)
        self.embedding = nn.Linear(20, self.d)

        self.mu_weights = nn.Parameter(torch.full((self.d, self.states), 1/self.states))

        self.criterion = criterion
        self.optimizer = optimizer(self.parameters(), **optim_args)
        self.t = real_transitions

        self.losses = []
        self.frob_diff = []
    
    def to(self, device):
        self.device = device
        super(Net, self).to(device)

    def encode_input(self, s, a):
         """
        # Parameters:
        s: State id or list like of state ids between 0 and self.states
        a: Action id or list like of actions ids between 0 and self.actions

        If s and a are list-like, both need to be the same lenght
         """
         input_len = len(s) if hasattr(s, '__len__') else 1
         actions_len = len(a) if hasattr(a, '__len__') else 1
         assert input_len == actions_len, f"The input lenghts do not coincide. Input States: {input_len}; Input Actions: {actions_len}"

        #  print(input_len)
        #  print(s.view(input_len, 1))
         s_hot = F.one_hot(s.view(input_len, 1), self.states).to(torch.float32)
         a_hot = F.one_hot(a.view(input_len, 1), self.actions).to(torch.float32)
         x = torch.cat((s_hot, a_hot), dim=-1) # Concat one-hot vectors 
         return x.to(self.device)

    def enconde_output(self, s):
        input_len = len(s) if hasattr(s, '__len__') else 1
        return F.one_hot(s.view(input_len, 1), self.states).to(torch.float32)


    def phi(self, s, a):
        """
        # Parameters:
        s: State id or list like of state ids between 0 and self.states
        a: Action id or list like of actions ids between 0 and self.actions

        If s and a are list-like, both need to be the same lenght
         """
        
        x = self.encode_input(s, a) 
        x = F.relu(self.l1(x))
        x = F.leaky_relu(self.l2(x))
        x = F.softmax(self.embedding(x), dim=-1) # Apply softmax row wise

        return x # This should be of shape (batch_size, 1, d)

    def mu(self):
        return F.softmax(self.mu_weights, dim=-1)

    def forward(self, s, a):
        """
        # Parameters:
        s: State id or list like of state ids between 0 and self.states
        a: Action id or list like of actions ids between 0 and self.actions

        If s and a are list-like, both need to be the same lenght
        """
        x = self.phi(s, a)
        soft_mu = self.mu()

        # We use bradcasting in here so the same parameters are used for every element of the batch
        x = torch.matmul(x, soft_mu) # Mat multiplication of (batch_size, 1, d) @ (d, states) --> (batch_size, 1, states) # Distribution over states

        return x.view(-1, self.states)

    
    def backward(self, dataset):
        """
        # Parameters:
        dataset: Batch iterator that for each enumerate iteration returns a batch of data
        """
        running_loss = 0.0
        n = 0
        # print(dataset.l)
        for epoch in range(3):
            for i, data in enumerate(dataset):

                # get the inputs; data is a list of [input1, input2, labels]
                input_states = data[0].to(self.device)
                input_actions = data[1].to(self.device)
                labels = data[2].to(self.device)

                # zero the parameter gradients
                self.optimizer.zero_grad()

                # forward + backward + optimize
                outputs = torch.log(self.forward(input_states, input_actions))
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

                running_loss += loss
                n += 1

            # Calculate stats once every epoch
            if self.t is not None:
                with torch.no_grad():
                    m0 = self.phi_a_matrix(torch.tensor(0)).detach().unsqueeze(0).to("cpu")
                    for i in range(1, self.actions):
                        ma = self.phi_a_matrix(torch.tensor(0)).detach().unsqueeze(0).to("cpu")
                        m0 = torch.cat((m0, ma), dim=0)
                        
                    frob = torch.norm((m0-self.t))
                    self.frob_diff.append(frob)
        
        return (running_loss / n).detach().to("cpu").numpy()


    def phi_a_matrix(self, a):
        """Calculates the transition kernes induces by phi when one action a is fixed.
        # Parameters:
        a: Action id between 0 and self.actions - 1
         """
        actions_list = a.repeat(self.states)
        state_list = torch.arange(0, (self.states), dtype=int)

        with torch.no_grad():
            phi_a_batch = self.phi(state_list, actions_list).view(self.states, self.d) # This should be of shape (states, d)

            return torch.matmul(phi_a_batch, self.mu()) # Should be size (states, states)

    def simplex_planner(self):
        """
        Returns the SimplexPlanner associated with the current state of the model `self`.
        """

        st = torch.tensor([i for i in range(self.states) for j in range(self.actions)])
        acts = torch.tensor(list(range(self.actions)) * self.states)
        x = self.phi(st, acts).detach().to("cpu").view(self.states, -1, self.d).numpy()
        planner = SimplexPlanner(self.states, self.actions, self.d, x)
        return planner

    def rep_ucb_planner(self, reward, inverse_covariance):
        # print("planner")

        def rep_ucb_reward(s_prev, a, s_next):
            # change
            # print("new")
            # print(s_prev.shape, s_prev.ravel().shape, s_prev.dtype)
            s_prev_t = torch.tensor(s_prev.ravel()).to(self.device).long()
            a_t = torch.tensor(a.ravel()).to(self.device).long()
            v = self.phi(s_prev_t, a_t).detach().to("cpu").numpy() #  shape (batch, 1, d)
            # change
            quad_form = (v @ inverse_covariance @ np.transpose(v, axes=[0, 2, 1])).reshape(s_prev.shape)
            return reward(s_prev, a, s_next) +  np.minimum(quad_form, 2)

       
        def next_state_prob(s, a):
            with torch.no_grad():
                return self.forward(torch.from_numpy(s.ravel()).to(self.device).long(), 
                                    torch.from_numpy(a.ravel()).to(self.device).long()).to("cpu").numpy()

        
        pol_iter = PolicyIteration( self.states, self.actions, next_state_prob , rep_ucb_reward)

        V, policy = pol_iter.run(print_progress=False, max_iter=300)

        return lambda s: policy[s]



