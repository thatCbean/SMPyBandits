import matplotlib.pyplot as plt
import numpy as np

data_x = [5/5, 5/10, 5/20, 5/50, 5/100]
data_y_linUCB = np.array([81.568, 83.181, 94.913, 55.599, 71.267])
data_y_SALasso = np.array([86.671, 83.094, 96.066, 49.223, 43.351])
std_SALasso = np.array([6.6914, 5.2485, 6.533, 3.881, 4.8957])
std_linUCB = np.array([3.7806, 10.47, 8.0558, 5.9122, 20.293])


ratio = data_y_linUCB / data_y_SALasso
std_ratio = np.abs(ratio) * np.sqrt((std_linUCB / data_y_linUCB) ** 2 + (std_SALasso / data_y_SALasso) ** 2)

data_y = (ratio * 100) - 100
print(data_y)
std_data_y = std_ratio * 100

fig, ax = plt.subplots()
ax.bar(data_x, data_y, width=0.05)
ax.errorbar(data_x, data_y, yerr=2*std_data_y, fmt='-o', color='cyan', markersize=5, capsize=3,)
#ax.axhline(y=0, color='gray', linestyle='--', label='y = 0')
title_text = r'Obtained cumulative regret of LinUCB compared to SALasso' + '\n' + r'in environment with $s_0 = 5$ and $K=20$'
ax.set_title(title_text)
ax.set_xlabel(r'Sparsity to density ratio $\frac{s_0}{d}$')
ax.set_ylabel('Obtained regret compared to SALasso [%]')

plt.grid(True)
plt.tight_layout()
plt.show()