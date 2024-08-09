import numpy as np
import matplotlib.pyplot as plt

a = 0.1
theta_init = 0.5
theta_fin = 2.05 * np.pi
theta_step = 0.2
z_values = np.linspace(-1, 1, 11)

theta = np.arange(theta_init, theta_fin, theta_step)

x = a * theta * np.cos(theta)
y = a * theta * np.sin(theta)

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

cor = 'r'

for z in z_values:
    ax.plot(x, y, z, marker='o', linestyle='', color=cor)

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Espiral Tridimensional de Arquimedes (11 Camadas)')

plt.show()
