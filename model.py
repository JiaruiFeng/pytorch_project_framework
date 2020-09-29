import torch
import torch.nn as nn
import torch.nn.functional as F

class demo(nn.Module):
    def __init__(self,input_size):
        super(demo, self).__init__()
        self.proj=nn.Linear(input_size,2)

    def forward(self,x):
        x=self.proj(x)
        return F.softmax(x)

