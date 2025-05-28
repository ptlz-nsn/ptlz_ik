"""
    绘制3D曲面图
"""
import numpy as np
import matplotlib.pyplot as mp
from mpl_toolkits.mplot3d import Axes3D 
import  os
import matplotlib
matplotlib.use('QtAgg')
cwd = os.getcwd()
basePath = cwd + "/data/"
def MeanDistance():
    rdata = np.loadtxt(basePath + "joint1.txt")
    rdata = rdata[1:, 1:]
    pdata = rdata.copy()
    pN = 0
    for i in range(rdata.shape[0]):
        for j in range(rdata.shape[1]):
            if(i > 0 and i < (rdata.shape[0]-1)):
                if(i%2 == 1 and j%2 == 0):
                    pdata[i, j] = (rdata[i+1, j] + rdata[i-1, j]) / 2
                    pN += 1

            if( j > 0 and j < (rdata.shape[1]-1)):
                if(j%2 == 1 and i%2 == 0):
                    pdata[i, j] = (rdata[i, j+1] + rdata[i, j-1]) / 2
                    pN += 1

    for i in range(rdata.shape[0]):
        for j in range(rdata.shape[1]):
            if(i > 0 and i < (rdata.shape[0]-1) and j > 0 and j < (rdata.shape[1]-1)):
                if(i%2 == 1 and j%2 == 1):
                    pdata[i, j] = (pdata[i+1, j] + pdata[i-1, j] + pdata[i, j+1] + pdata[i, j-1]) / 4
                    pN += 1


    rdata1 = rdata.copy()
    pdata1 = pdata.copy()
    rdata = np.loadtxt(basePath + "joint2.txt")
    pdata = rdata
    for i in range(rdata.shape[0]):
        for j in range(rdata.shape[1]):
            if(i > 0 and i < (rdata.shape[0]-1)):
                if(i%2 == 1 and j%2 == 0):
                    pdata[i, j] = (rdata[i+1, j] + rdata[i-1, j]) / 2
                    pN += 1

            if( j > 0 and j < (rdata.shape[1]-1)):
                if(j%2 == 1 and i%2 == 0):
                    pdata[i, j] = (rdata[i, j+1] + rdata[i, j-1]) / 2
                    pN += 1

    for i in range(rdata.shape[0]):
        for j in range(rdata.shape[1]):
            if(i > 0 and i < (rdata.shape[0]-1) and j > 0 and j < (rdata.shape[1]-1)):
                if(i%2 == 1 and j%2 == 1):
                    pdata[i, j] = (pdata[i+1, j] + pdata[i-1, j] + pdata[i, j+1] + pdata[i, j-1]) / 4
                    pN += 1


    rdata2 = rdata.copy()
    pdata2 = pdata.copy()

    dis = ((rdata1 - pdata1)**2 + (rdata1 - pdata1)**2)**0.5
    print("Max Val", dis.max())
    print("Mean Val", dis.sum() / pN)


def RelationshipMapping():
    fig = mp.figure("3D Surface1", facecolor="lightgray")
    mp.title("3D Surface", fontsize=18)
    mp.tick_params(labelsize=10)
    # 设置为3D图片类型
    # ax3d = fig.add_subplot(111, projection='3d') 

    ax3d = fig.add_subplot(projection='3d') 
    
    ax3d.set_xlabel("pitch (rad)")
    ax3d.set_ylabel("roll(rad)")
    ax3d.set_zlabel("linear_1 (m)")
    data = np.loadtxt(basePath + "joint1.txt")
    x = data[1:, 0]
    y = data[0, 1:]
    x, y = np.meshgrid(x, y)
    z = data[1:, 1:].T
    ax3d.plot_surface(x, y, z, cstride=20, rstride=20, cmap="jet")


    fig = mp.figure("3D Surface2", facecolor="lightgray")
    mp.title("3D Surface", fontsize=18)
    mp.tick_params(labelsize=10)
    # 设置为3D图片类型
    # ax3d = Axes3D(fig)
    ax3d = fig.add_subplot(projection='3d')  # 自动加载Axes3D
    ax3d.set_xlabel("pitch (rad)")
    ax3d.set_ylabel("roll(rad)")
    ax3d.set_zlabel("linear_2 (m)")
    data = np.loadtxt(basePath + "joint2.txt")
    x = data[1:, 0]
    y = data[0, 1:]
    x, y = np.meshgrid(x, y)
    z = data[1:, 1:].T
    ax3d.plot_surface(x, y, z, cstride=20, rstride=20, cmap="jet")
    mp.show()


MeanDistance()
RelationshipMapping()