import matplotlib.pyplot as plt

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')  # 自动加载Axes3D
ax.scatter([0], [0], [0])
plt.show()