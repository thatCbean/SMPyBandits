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
# plt.savefig('cumulative_regret_plot.png', dpi=300)
plt.show()

data_x = [5/50, 10/50, 15/50, 20/50, 25/50, 30/50, 35/50, 40/50, 45/50, 50/50]

data_y_SALasso = np.array([67.835, 78.154, 123.32, 128.52, 107.35, 144.09, 149.23, 135.73, 128.7, 119.71])
std_SALasso = np.array([8.9769, 8.432, 22.522, 10.248, 15.8, 8.8992, 20.284, 23.669, 29.811, 23.014])

data_y_linUCB = np.array([76.683, 83.264, 116.36, 137.32, 113.92, 149.18, 149.21, 138.01, 132.23, 126.68])
std_linUCB = np.array([17.161, 12.063, 9.37, 16.358, 10.386, 17.204, 19.977, 21.596, 28.201, 33.005])


ratio = data_y_linUCB / data_y_SALasso
std_ratio = np.abs(ratio) * np.sqrt((std_linUCB / data_y_linUCB) ** 2 + (std_SALasso / data_y_SALasso) ** 2)

data_y = (ratio * 100) - 100
print(data_y)
std_data_y = std_ratio * 100

fig1, ax = plt.subplots()
ax.bar(data_x, data_y, width=0.05)
ax.errorbar(data_x, data_y, yerr=2*std_data_y, fmt='-o', color='cyan', markersize=5, capsize=3,)
#ax.axhline(y=0, color='gray', linestyle='--', label='y = 0')
title_text = r'Obtained cumulative regret of LinUCB compared to SALasso' + '\n' + r'in environment with $d = 50$ and $K=20$'
ax.set_title(title_text)
ax.set_xlabel(r'Sparsity to density ratio $\frac{s_0}{d}$')
ax.set_ylabel(r'Obtained regret compared to SALasso $\frac{\mathcal{R}^{LinUCB}(1000)}{\mathcal{R}^{SALasso}(1000)}$ [%]')

plt.grid(True)
plt.tight_layout()
plt.savefig('cumulative_regret_plot.png')
plt.show()
#
# data_x = [5/50, 10/50, 15/50, 20/50, 25/50, 30/50, 35/50, 40/50, 45/50, 50/50]
#
# data_y_SALasso = np.array([3.3696, 7.394, 2.491, 3.4929, 1.7536, 144.09, 149.23, 135.73, 128.7, 119.71])
# std_SALasso = np.array([1.0829, 1.5137, 0.40484, 0.22723, 0.43656, 8.8992, 20.284, 23.669, 29.811, 23.014])
#
# data_y_linUCB = np.array([8.2109, 13.522, 2.3315, 2.4191, 0.091563, 149.18, 149.21, 138.01, 132.23, 126.68])
# std_linUCB = np.array([1.3747, 1.0016, 0.32362, 0.27776, 10.386, 17.204, 19.977, 21.596, 28.201, 33.005])
#
#
# ratio = data_y_linUCB / data_y_SALasso
# std_ratio = np.abs(ratio) * np.sqrt((std_linUCB / data_y_linUCB) ** 2 + (std_SALasso / data_y_SALasso) ** 2)
#
# data_y = (ratio * 100) - 100
# print(data_y)
# std_data_y = std_ratio * 100
#
# fig1, ax = plt.subplots()
# ax.bar(data_x, data_y, width=0.05)
# ax.errorbar(data_x, data_y, yerr=2*std_data_y, fmt='-o', color='cyan', markersize=5, capsize=3,)
# #ax.axhline(y=0, color='gray', linestyle='--', label='y = 0')
# title_text = r'Obtained cumulative regret of LinUCB compared to SALasso' + '\n' + r'in environment with $d = 50$ and $K=20$'
# ax.set_title(title_text)
# ax.set_xlabel(r'Sparsity to density ratio $\frac{s_0}{d}$')
# ax.set_ylabel('Obtained regret compared to SALasso [%]')
#
# plt.grid(True)
# plt.tight_layout()
# # plt.savefig('cumulative_regret_plot.png', dpi=300)
# plt.show()