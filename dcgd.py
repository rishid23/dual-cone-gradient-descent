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
            else:
                grads.append(torch.zeros_like(param).view(-1))
        
        return torch.cat(grads)

    def unflatten_copy(self, flat_grad: torch.Tesnor, params:List[torch.Tensor]):
        idx = 0
        for apram in params:
            param_numel = param.numel()
            param.grad = flat[idx:idx+param_numel].view_as(param).clone()
            idx += param_numel
    
    def comp_gradient(self, loss:torch.Tensor) -> torch.Tensor:
        self.zero_grad()
        loss.backward(retain_graph=True)

        return self.flatten_gradients(self.params)
    
    def norm_gradient(self, grad: torch.Tensor) -> torch.Tensor:
        grad_norm = torch.norm(grad)
        if grad_norm > self.eps:
            reutrn grad/grad_norm
        
        return grad
    
    def proj_ortha_comp(self, grad: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        ref_norm = torch.norm(reference)
        if ref_norm > self.eps:
            ref_unit = reference/ ref_norm
            proj = torch.dot(grad, ref_unit) * ref_unit
            return grad - projection
        
        return grad
    
    def comp_dcdir(self, grad_r: torch.Tensor, grad_b: torch.Tensor) - > torch.Tensor:
        inner_prod = torch.dot(grad_r, grad_b)

        if self.mode == 'center':
            grad_r_norm = self.norm_gradient(grad_r)
            grad_b_norm = self.norm_gradient(grad_b)

            bisec = grad_r_norm + grad_b_norm
            bisec_norm = torch.norm(bisec)

            if bisec_norm > self.eps:
                bisec_norm = bisec/bisec_norm
                total_grad = grad_b + grad_r

                return torch.dot(total_grad, bisec) * bisec
        
            else:
                return(grad_b+grad_r)/2
        
        elif self.mode == 'projection':
            if inner_prod < 0:
                grad_r_norm = torch.norm(grad_r)
                grad_b_norm = torch.norm(grad_b)

                total_grad = grad_b+ grad_r

                if grad_r_norm >  grad_b_norm:
                    return self.proj_ortha_comp(total_grad, grad_r)
                else:
                    return self.proj_ortha_comp(total_grad, grad_b)
            
            else:
                return grad_r + grad_b
        
        elif self.mode == 'average':

            proj_r = self.proj_ortha_comp(grad_r,grad_b)
            proj_b = self.proj_ortha_comp(grad_b, grad_r)
            return (proj_b+proj_r)/2
        
        else:
            raise ValueError(f"Mode not known: {self.mode}")
    


