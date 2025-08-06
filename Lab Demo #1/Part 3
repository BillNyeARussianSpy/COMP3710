import torch
import matplotlib.pyplot as plt
import numpy as np

#device config
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#Sierpiński triangle generation
def sierpinski(iterations=100000, size=1000):
    #vertices
    vertices = torch.tensor([[0, 0], [size-1, 0], [size//2, int(size*np.sqrt(3)/2)]], dtype=torch.float32, device=device)
    #random start point
    point = torch.rand(2, device=device) * size
    #iterate and store points
    points = torch.zeros((iterations, 2), device=device)
    for i in range(iterations):
        vertex = vertices[torch.randint(0, 3, (1,))]
        point = (point + vertex) / 2
        points[i] = point
    return points.cpu().numpy()

#plot
points = sierpinski(iterations=100000, size=1000)
plt.figure(figsize=(8,8))
plt.scatter(points[:,0], points[:,1], s=0.1, color='purple')
plt.axis('off')
plt.show()