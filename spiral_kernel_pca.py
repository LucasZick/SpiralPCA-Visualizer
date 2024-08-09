import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import KernelPCA, PCA

a = 0.1
theta_init = 0.5
theta_fin = 2.05 * np.pi
theta_step = 0.2
z_values = np.linspace(-1, 1, 11)

theta = np.arange(theta_init, theta_fin, theta_step)

x = a * theta * np.cos(theta)
y = a * theta * np.sin(theta)

spiral_points = np.array([[xi, yi, zi] for xi, yi in zip(x, y) for zi in z_values])

gamma_values = [0.1, 1, 10,100]

def plot_pca(kernel_pca=True, n_components=2):
    plt.figure(figsize=(15, 5))

    if kernel_pca:
        for i, gamma in enumerate(gamma_values, 1):
            kpca = KernelPCA(n_components=n_components, kernel='rbf', gamma=gamma)
            spiral_kpca = kpca.fit_transform(spiral_points)
            
            plt.subplot(1, len(gamma_values), i)
            if n_components == 1:
                plt.scatter(spiral_kpca[:, 0], np.zeros_like(spiral_kpca[:, 0]), c=spiral_points[:, 2], cmap='viridis')
                plt.xlabel('Componente Principal 1')
                plt.ylabel('Valor Fixo')
            else:
                plt.scatter(spiral_kpca[:, 0], spiral_kpca[:, 1], c=spiral_points[:, 2], cmap='viridis')
                plt.xlabel('Componente Principal 1')
                plt.ylabel('Componente Principal 2')
            plt.title(f'Kernel PCA com γ = {gamma}')
    else:
        pca = PCA(n_components=n_components)
        spiral_pca = pca.fit_transform(spiral_points)
        
        plt.scatter(spiral_pca[:, 0], np.zeros_like(spiral_pca[:, 0]) if n_components == 1 else spiral_pca[:, 1], c=spiral_points[:, 2], cmap='viridis')
        plt.title('PCA (2 Componentes Principais)' if n_components == 2 else 'PCA (1 Componente Principal)')
        plt.xlabel('Componente Principal 1')
        if n_components == 2:
            plt.ylabel('Componente Principal 2')
    
    plt.colorbar(label='Valor de Z')
    plt.show()

plot_pca(kernel_pca=True, n_components=2)
plot_pca(kernel_pca=True, n_components=1)
plot_pca(kernel_pca=False, n_components=2)
plot_pca(kernel_pca=False, n_components=1)
