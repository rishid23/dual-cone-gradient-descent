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
            if param.grad is not None:
                param.grad.zero_()

    
    def flatten_gradients(self,params: List[torch.Tensor]) -> torch.Tensor:
        grads = []
        for param in params:
            if param.grad is not None:
                grads.append(param.grad.view(-1))
            else:
                grads.append(torch.zeros_like(param).view(-1))
        
        return torch.cat(grads)

    def unflatten_copy(self, flat_grad: torch.Tensor, params:List[torch.Tensor]):
        idx = 0
        for param in params:
            param_numel = param.numel()
            param.grad = flat_grad[idx:idx+param_numel].view_as(param).clone()
            idx += param_numel
    
    def comp_gradient(self, loss:torch.Tensor) -> torch.Tensor:
        self.zero_grad()
        loss.backward(retain_graph=True)

        return self.flatten_gradients(self.params)
    
    def norm_gradient(self, grad: torch.Tensor) -> torch.Tensor:
        grad_norm = torch.norm(grad)
        if grad_norm > self.eps:
            return grad/grad_norm
        
        return grad
    
    def proj_ortha_comp(self, grad: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        ref_norm = torch.norm(reference)
        if ref_norm > self.eps:
            ref_unit = reference/ ref_norm
            proj = torch.dot(grad, ref_unit) * ref_unit
            return grad - proj
        
        return grad
    
    def comp_dcdir(self, grad_r: torch.Tensor, grad_b: torch.Tensor) -> torch.Tensor:
        inner_prod = torch.dot(grad_r, grad_b)

        if self.mode == 'center':
            grad_r_norm = self.norm_gradient(grad_r)
            grad_b_norm = self.norm_gradient(grad_b)

            bisec = grad_r_norm + grad_b_norm
            bisec_norm = torch.norm(bisec)

            if bisec_norm > self.eps:
                bisec = bisec/bisec_norm
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
    

    def step(self, model:nn.Module, loss_r:torch.Tensor, loss_b:torch.Tensor):

        if not loss_r.requires_grad:
            raise ValueError("loss_r must require gradients.")
        if not loss_b.requires_grad:
            raise ValueError("loss_b must require gradients.")
        
        grad_r = self.comp_gradient(loss_r)
        grad_b = self.comp_gradient(loss_b)

        if self.weight_decay != 0:
            flat_params = torch.cat([p.view(-1) for p in self.params])
            grad_r += self.weight_decay * flat_params
            grad_b += self.weight_decay * flat_params

        update_direction = self.comp_dcdir(grad_r, grad_b)

        update_direction = -self.lr*update_direction
        self.unflatten_copy(update_direction, self.params)

        with torch.no_grad():
            for param in self.params:
                if param.grad is not None:
                    param.add_(param.grad)
        
        self.zero_grad()

    def state_dict(self):
        return {
            'lr':self.lr,
            'weight_decay':self.weight_decay,
            'mode':self.mode,
            'eps':self.eps
        }
    
    def load_state_dict(self, state_dict):
        self.lr = state_dict.get('lr')
        self.weight_decay = state_dict.get('weight_decay')
        self.mode = state_dict.get('mode')
        self.eps = state_dict.get('eps', 1e-8)


if __name__ == "__main__":

    class SimpleModule(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(2,10)
            self.fc2 = nn.Linear(10,1)

        def forward(self,x):
            x = torch.tanh(self.fc1(x))
            return self.fc2(x)
        
    model = SimpleModule()
    optimizer = DCGD(model.parameters(), lr=0.01, mode='center')

    x=torch.randn(100,2,requires_grad=True)
    y_true = torch.randn(100,1)

    y_pred = model(x)

    loss_r = torch.mean((y_pred-y_true)**2)
    loss_b = torch.mean(y_pred**2)

    print(f"Initial loss_r: {loss_r.item():.4f}")
    print(f"Initial loss_b: {loss_b.item():.4f}")

    optimizer.step(model, loss_r, loss_b)

    y_pred_new = model(x)
    loss_r_new = torch.mean((y_pred_new - y_true)**2)
    loss_b_new = torch.mean(y_pred_new**2)

    print(f"Updated loss_r: {loss_r_new.item():.4f}")
    print(f"Updated loss_b: {loss_b_new.item():.4f}")

    print(f"Optmizer mode: {optimizer.mode}")
    print("Finished")   




