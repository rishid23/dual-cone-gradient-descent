import torch
import torch.nn as nn
import numpy as np

class DCGD:
    

    def __init__(self,params,lr=1e-3,mode='center',weight_decay=0):
        if lr < 0.0:
            raise ValueError(f"Invalid Learning Rate: {lr}")
        
        if weight_decay < 0.0:
            raise ValueError(f"Invalid Weight Decay Val: {weight_decay}")
        

        self.params = list(params)
        self.lr = lr
        self.weight_decay = weight_decay
        self.mode = mode.lower()


        if self.mode not in ["projection", "average", "center"]:
            raise ValueError(f"Invalid mode: {self.mode}. Choose from 'projection', 'average', or the 'center'")
        
        self.state = {}

