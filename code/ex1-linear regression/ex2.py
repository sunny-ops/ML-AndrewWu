import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

path =  'ex1data2.txt'
data = pd.read_csv(path, header=None, names=['Size', 'Beds','Price'])
# print(data.head())
# print(data.describe())


# (3.0) 创建三维图形
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter(data.iloc[:, 0], data.iloc[:, 1], data.iloc[:, 2])
# ax.set_xlabel('X轴')
# ax.set_ylabel('Y轴')
# ax.set_zlabel('Z轴')
# plt.show()

# (3.1) Feature Normalization
# X_normalized = (X - X.mean()) / X.std()
mean_vals = data.mean(axis=0)
std_vals = data.std(axis=0)
normalized_data = (data - mean_vals) / std_vals
# print("mean_vals\n", mean_vals)
# print("std_vals\n", std_vals)
# print("normalized_data\n",normalized_data.head())

# (3.2) Gradient Descent
normalized_data.insert(0, 'Ones', 1) # 最左边插入一列
data.insert(0, 'Ones', 1)
cols = normalized_data.shape[1]
X = normalized_data.iloc[:,0:cols-1] #左闭右开 col[0:cols-1]
y = normalized_data.iloc[:,cols-1:]
print(X.head())
print(y.head())
X = np.matrix(X.values)
y = np.matrix(y.values)
theta = np.matrix(np.array([0,0,0], dtype=np.float128))
print("theta", theta)

def computeCost(X, y, theta):
    m = X.shape[0]
    cost = np.power(X * theta.T - y, 2)
    # cost = np.multiply((X * theta.T - y), (X * theta.T - y)) 
    cost = np.sum(cost) / (2 * m)
    return cost

def gradientDescent(X, y, theta, alpha, iters):
    m = X.shape[0]
    temp = np.array(theta)
    costs = np.zeros(iters)
    for i in range(iters):
        for j in range(theta.shape[1]):
            temp[0,j] = temp[0,j] - np.sum(np.multiply((X * temp.T - y), X[:,j])) * alpha / m
        costs[i] = computeCost(X, y, temp)
        
    return costs, temp

print("cost1", computeCost(X, y, theta))
alpha = 0.3
iters = 1000
costs, theta = gradientDescent(X, y, theta, alpha, iters)
print("theta\n", theta)

x = np.arange(len(costs))

# 绘制曲线cost-iteration曲线
plt.plot(x, costs)
plt.title('')
plt.xlabel('iteration')
plt.ylabel('cost')
# plt.show()

# (3.3) Normal Equations
thetaN = np.linalg.inv(X.T * X)*(X.T)*y
print(thetaN)



