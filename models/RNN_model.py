import torch
import torch.nn as nn
from torch import optim

class RNNModule(nn.Module):
    def __init__(self):
        super(RNNModule, self).__init__()
        # TODO: Finish implementation for initializing the model

    def forward(self, X, hidden_state, memory):
        # TODO: Finish implementation for the forward function

    def initHidden(self, batch_size=1, max_seq_len=1):
        hidden_state = Variable(torch.zeros(max_seq_len, batch_size, self.hidden_dim))
        return hidden_state
    
    def initMemory(self, batch_size=1, max_seq_len=1):
        memory = Variable(torch.zeros(max_seq_len, batch_size, self.hidden_dim))
        return memory
