import numpy as np
import matplotlib.pyplot as plt

"""
1. Test에 해당하는 부분만 추출한다.

"""

tgt_lines_AB = []
tgt_lines_BA = []

with open("DCP_origin.csv", "r") as f:
    lines = f.readlines()  # 8 -> 23 -> 39 (16)

    for i in range(7, len(lines), 15):
        A2B = lines[i].strip("\n")
        B2A = lines[i+2].strip("\n")
        # print(A2B.split(", "))
        A2B_list = [j.split(": ")[1] for j in A2B.split(", ") if j[0] != "C"]
        B2A_list = [j.split(": ")[1] for j in B2A.split(", ")]

        if A2B_list[0] == "100":
            break

        tgt_lines_AB.append(A2B_list[1:])
        tgt_lines_BA.append(B2A_list[1:])

tgt_lines_AB = np.array(tgt_lines_AB, np.float32)
tgt_lines_BA = np.array(tgt_lines_BA, np.float32)
t = np.array(range(len(tgt_lines_AB)))

with open("DCP_normal.csv", "r") as f:
    lines = [list(eval(i.strip("\n"))) for i in f.readlines()]

arr1 = np.array(lines, np.float32)

target = "mae_ba"
plt.title(f"{target}")
plt.grid(True)
plt.plot(t, arr1[:, 12])
plt.plot(t, tgt_lines_BA[:, 3])
plt.legend(["my code", "origin"])
plt.xlabel("epochs")
plt.ylabel(f"{target}")

plt.show()
