"""
2.1 - Mandelbrot set
"""

import torch
import numpy as np

#device config
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#Numpy 2D array of complex numbers on [-2,2]x[-2,2]
Y, X = np.mgrid[-1.3:1.3:0.0005, -2:1:0.0005]

#load into PyTorch tensors
x = torch.Tensor(X)
y = torch.Tensor(Y)
z = torch.complex(x,y)
zs = z.clone()
ns = torch.zeros_like(z)

#transfer to GPU device
z = z.to(device)
zs = zs.to(device)
ns = ns.to(device)

#mandelbrot set
# Mandelbrot set
for i in range(200):
    # compute new values of z: z^2 + x
    zs_ = zs * zs + z
    # have we diverged with this new value?
    not_diverged = torch.abs(zs_) < 4.0
    # update variables to compute
    ns += not_diverged
    zs = zs_

def julia():
    """
    Julia set (example with c = -0.7 + 0.27015j)
    Reset zs and ns for Julia computation
    """
    zs = z.clone()
    ns = torch.zeros_like(z)
    c = torch.complex(torch.tensor(-0.7, device=device), torch.tensor(0.27015, device=device))
    for i in range(200):
        zs_ = zs * zs + c
        not_diverged = torch.abs(zs_) < 4.0
        ns += not_diverged
        zs = zs_

#plot result via n counter
import matplotlib.pyplot as plt


fig = plt.figure(figsize=(16,10))
def processFractal(a):
    """Display an array of iteration counts as a
    colorful picture of a fractal."""
    a_cyclic = (6.28*a/20.0).reshape(list(a.shape)+[1])
    img = np.concatenate([10+20*np.cos(a_cyclic),
    30+50*np.sin(a_cyclic),
    155-80*np.cos(a_cyclic)], 2)
    img[a==a.max()] = 0
    a = img
    a = np.uint8(np.clip(a, 0, 255))
    return a

plt.imshow(processFractal(ns.cpu().numpy()))
plt.tight_layout(pad=0)
plt.show()

"""
2.2 - There were no problems and it was fast. 
"""