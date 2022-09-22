import matplotlib.pyplot as plt
import numpy as np

obs = (1, 5, 10, 25, 50)
rel_max = (0.44, 0.61, 0.64, 0.68, 0.55)
rel_average = (0.07, 0.13, 0.05, 0.32, 0.35)
abs_max = (0.61, 0.68, 0.71, 0.6, 0.56)
abs_average = (0.24, 0.47, 0.25, 0.42, 0.38)

plt.plot(obs, np.array((rel_max, rel_average, abs_max, abs_average)).T)
plt.ylim(0, 1)
plt.xlabel("number of observations")
plt.ylabel("overall performance")
plt.legend(("relative max", "relative average", "absolute max", "absolute average"))
plt.savefig("n_obs", dpi=400, transparent=True)
plt.show()
