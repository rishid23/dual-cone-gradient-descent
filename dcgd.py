import torch
import torch.nn as nn
import numpy as np
from typing import Callable, Optional, List, Tuple

class DCGD:
    

    def __init__(self,params,lr=1e-3,mode='center',weight_decay=0, eps=1e-8):
        if lr < 0.0:
            raise ValueError(f"Invalid Learning Rate: {lr}")
        
        if weight_decay < 0.0:
            raise ValueError(f"Invalid Weight Decay Val: {weight_decay}")
        

        self.params = list(params)
        self.lr = lr
        self.weight_decay = weight_decay
        self.mode = mode.lower()
        self.eps = eps


        if self.mode not in ["projection", "average", "center"]:
            raise ValueError(f"Invalid mode: {self.mode}. Choose from 'projection', 'average', or the 'center'")
        
        self.state = {}
    
    def zero_grad(self):
        for param in self.params:
            if parap.grad is not None:
                param.grad_zero_()

    
    def flatten_gradients(self,params: List[torch.Tensor]) -> torch.Tensor:
        grads = []
        for param in params:
            if param.grad is not None:
                grads.appen(param.grad.view(-1))

