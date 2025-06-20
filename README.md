# dual cone gradient descent

Over the past few weeks/month (i had exams i swear) I have been working and learning on this is pytorch implementation of this project. It is based on the Dual Cone Gradient Descent (DCGD) method from a NeurIPS 2024 paper by Youngsik Hwang, Dong-Young Lim, and their team. They came up with this to help train PINNs when gradients from different losses clash. 

It is designed to optimize two losses at the same time, which often happens in Physics-Informed Neural Networks. When the gradients from these losses point in different or conflicting directions, it can make training unecessarily unstable or less effective. DCGD solves this issue through combining the gradients in a way that avoid such conflicts. This helps the model improve both losses together more smoothly + more efficiently.

the optimizer witholds three modes:  
- `center`: bisects normalized gradients  
- `projection`: removes conflicting directions based on magnitude  
- `average`: orthogonal projection average  

---

### Example

```python
model = SimpleModule()
optimizer = DCGD(model.parameters(), lr=0.01, mode="center")

x = torch.randn(100, 2, requires_grad=True)
y_true = torch.randn(100, 1)

y_pred = model(x)
loss_r = torch.mean((y_pred - y_true) ** 2)
loss_b = torch.mean(y_pred ** 2)

print(f"Initial loss_r: {loss_r.item():.4f}")
print(f"Initial loss_b: {loss_b.item():.4f}")

optimizer.step(model, loss_r, loss_b)

y_pred_new = model(x)
loss_r_new = torch.mean((y_pred_new - y_true) ** 2)
loss_b_new = torch.mean(y_pred_new ** 2)

print(f"Updated loss_r: {loss_r_new.item():.4f}")
print(f"Updated loss_b: {loss_b_new.item():.4f}")
````

---

### Requirements

* python 3.8 or higher
* pyTorch ≥ 2.0
* numpy (for the data handling/testing)
* extra credit: GPU with CUDA support for faster training 
* no other dependencies necessary

---

### Results

On a toy regression task (2D → 1D), the optimizer reduced both `loss_r` and `loss_b`:

```
Initial loss_r: 1.1158
Initial loss_b: 0.0721
Updated loss_r: 1.1092
Updated loss_b: 0.0648
Optmizer mode: center
```
Below is Figure 1 from the original paper; it shows how DCGD outperforms Adam in minimizing conflicting losses in a PINN-type setup:

<img width="312" alt="image" src="https://github.com/user-attachments/assets/d9893bcf-e3ef-40a2-888b-97a39c4bce72" />




you can change the mode to `"projection"` or `"average"` to see how the gradient conflict resolution changes training.

---

### Code Layout

* `dcgd.py`: full implementation of the optimizer + toy test at the bottom
* no extra files, all self-contained

---

### Paper - Credit to all visualizations
**Dual Cone Gradient Descent for Training Physics-Informed Neural Networks**
NeurIPS 2024 (https://arxiv.org/pdf/2409.18426)



---

### How it works

* gradients end up flattened → then resolved using chosen mode → then proceed to be unflattened and applied
* weight decay support has been included
* `eps` avoids division by zero when normalizing

The figures below (2–4 from the paper) illustrate how DCGD resolves conflicting gradients using the dual cone projections:

<img width="581" alt="image" src="https://github.com/user-attachments/assets/ed55900e-818c-420e-8a2c-3503c212287a" style="border: 1px solid #ccc; padding: 4px; border-radius: 6px;"/>
<img width="499" alt="image" src="https://github.com/user-attachments/assets/e6593e62-c05f-4dec-9fac-910477fc338e" />
<img width="586" alt="image" src="https://github.com/user-attachments/assets/ce5b9afc-0a9f-477d-91c9-b93eb20322e7" />


---

### What I Learned

When I first started this project, I mainly focused on getting the optimizer to run on some fake data. I wouldn’t say I was super confident with all the math behind it, but working through the implementation pushed me to understand how PyTorch handles gradients — especially when you’re dealing with multiple losses at once. Flattening gradients, reassigning them, and handling cases similar to missing or conflicting gradients was tricky but truly eye-opening.

Since I’m interested in physics, building this project gave me a chance to connect what I’ve learned in class to real-world problems where physics & machine learning meet. Even if I didn’t fully grasp every single equation, the hands-on coding helped me get the core ideas and the significance of the algorithm in a greater setting.

---

### Footnotes

* maybe integrate with torch.optim if needed
* Flattening and unflattening gradients was a bit tricky but helped me understand how optimization works better.
* Since I didn’t have a good GPU, I tested on simple synthetic data to check if the optimizer works.

