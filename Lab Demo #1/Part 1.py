import torch
import numpy as np

print("PyTorch Version:", torch.__version__)


#device config
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#grid for image computation, subdividing space
X, Y = np.mgrid[-4.0:4:0.01, -4.0:4:0.01]

#load into pytorch sensors
x = torch.Tensor(X)
y = torch.Tensor(Y)

#transfer to GPU
x = x.to(device)
y = y.to(device)

#Compute Gaussian
z_g = torch.exp(-(x**2+y**2)/2.0)

#2D sine
z_s = torch.sin(x) + torch.sin(y)

#Multiply and Visualise
z = z_g * z_s


def lab_plot():
    #plot
    import matplotlib.pyplot as plt

    plt.imshow(z.cpu().numpy())

    plt.tight_layout()
    plt.show()

def AI_plot():
    """
    Note: had to say 'its 2D, not 3D after ChatGPT
    gave me a 3D one (which was kewl, i guess)
    """
    import numpy as np
    import matplotlib.pyplot as plt

    # Parameters for the 2D Gaussian
    mu_x = 0        # Mean in x-direction
    mu_y = 0        # Mean in y-direction
    sigma_x = 1     # Standard deviation in x-direction
    sigma_y = 1     # Standard deviation in y-direction

    # Create a grid of (x, y) coordinates
    x = np.linspace(-5, 5, 200)
    y = np.linspace(-5, 5, 200)
    X, Y = np.meshgrid(x, y)

    # Compute the 2D Gaussian function
    Z = (1 / (2 * np.pi * sigma_x * sigma_y)) * np.exp(
        -(((X - mu_x) ** 2) / (2 * sigma_x ** 2) + ((Y - mu_y) ** 2) / (2 * sigma_y ** 2))
    )

    # Plot the 2D Gaussian using contour and imshow
    plt.figure(figsize=(8, 6))
    plt.contour(X, Y, Z, levels=15, cmap='viridis')
    plt.imshow(Z, extent=[x.min(), x.max(), y.min(), y.max()], origin='lower', cmap='viridis', alpha=0.6)
    plt.colorbar(label='Probability Density')
    plt.title('2D Gaussian Distribution')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.tight_layout()
    plt.show()


lab_plot()