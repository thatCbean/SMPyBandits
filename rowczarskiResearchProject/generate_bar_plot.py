import matplotlib.pyplot as plt
import numpy as np

data_x = [1, 0.5, 5/20, 5/50, 5/100]
data_y_linUCB = np.array([81.568, 83.181, 94.913, 55.599, 71.267])
data_y_SALasso = np.array([86.671, 83.094, 96.066, 49.223, 43.351])
data_y = (data_y_linUCB / data_y_SALasso * 100) - 100

fig, ax = plt.subplots()
ax.bar(data_x, data_y, width=0.1)
ax.set_xlabel('Sparsity to density ratio')
ax.set_ylabel('Obtained regret compared to SALasso [%]')

plt.show()