import matplotlib.pyplot as plt
import numpy as np

with open("DCP_normal.csv", "r") as f:
    lines = [list(eval(i.strip("\n"))) for i in f.readlines()]

arr1 = np.array(lines, np.float32)
length = arr1.shape[0]
t = np.array(range(length))

target = "mae_ba"
plt.title(f"{target}")
plt.grid(True)
plt.plot(t, arr1[:, 12])
plt.xlabel("epochs")
plt.ylabel(f"{target}")
plt.show()