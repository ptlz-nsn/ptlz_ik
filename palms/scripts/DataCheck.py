import numpy as np
import matplotlib.pyplot as plt
import  os

cwd = os.getcwd()
data1 = np.loadtxt(cwd + "/data/validate.txt")
plt.figure()
xs = data1[:, 0].T
ys1 = data1[:, 1:5]
yst = ys1[:, 0]
plt.plot(xs, yst, color='b', ls='--', lw='2', label='direction pitch')
yst = ys1[:, 1]
plt.plot(xs, yst, color='g', ls='--', lw='2', label='direction roll')
yst = ys1[:, 2]
plt.plot(xs, yst, color='r', ls='--', lw='2', label='IK-FK pitch')
yst = ys1[:, 3]
plt.plot(xs, yst, color='c', ls='--', lw='2', label='IK-FK roll')

plt.xlabel("t(s)")
plt.ylabel(r"$\theta$ (rad)")
plt.legend(loc='best')


plt.figure()
xs = data1[:, 0].T
ys1 = data1[:, 5:9]
yst = ys1[:, 0]
plt.plot(xs, yst, color='b', ls='--', lw='2', label='IK Linear_1')
yst = ys1[:, 1]
plt.plot(xs, yst, color='g', ls='--', lw='2', label='IK Linear_2')
yst = ys1[:, 2]
plt.plot(xs, yst, color='r', ls='--', lw='2', label='IF Linear_1')
yst = ys1[:, 3]
plt.plot(xs, yst, color='c', ls='--', lw='2', label='IF Linear_2')


plt.xlabel("t(s)")
plt.ylabel(r"$l$ (mm)")
plt.legend(loc='best')

plt.figure()
xs = data1[:, 0].T
ys1 = data1[:, 1:5]
yst = ys1[:, 2] - ys1[:, 0]
plt.plot(xs, yst, color='r', ls='--', lw='2', label='IK-FK pitch error')
yst = ys1[:, 3] - ys1[:, 1]
plt.plot(xs, yst, color='c', ls='--', lw='2', label='IK-FK roll error')

plt.ticklabel_format(style='sci', axis='both', scilimits=(0,0))
plt.xlabel("t(s)")
plt.ylabel(r"$\theta$ (rad)")
plt.legend(loc='best')

plt.figure()
xs = data1[:, 0].T
ys1 = data1[:, 5:9]
yst = ys1[:, 0] - ys1[:, 2]
plt.plot(xs, yst, color='b', ls='--', lw='2', label='IK-IF Linear_1 error')
yst = ys1[:, 1] - ys1[:, 3]
plt.plot(xs, yst, color='g', ls='--', lw='2', label='IK-IF Linear_2 error')

plt.ticklabel_format(style='sci', axis='both', scilimits=(0,0))
plt.xlabel("t(s)")
plt.ylabel(r"l (mm))")
plt.legend(loc='best')

plt.show()