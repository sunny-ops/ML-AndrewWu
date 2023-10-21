import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math 
from scipy.optimize import minimize
from scipy import optimize as opt


path =  'ex2data1.txt'
data = pd.read_csv(path, header=None, names=['Exam 1', 'Exam 2', 'Admitted'])
# print(data.head())
# print(data.describe())
# 1.1 Visualizing the data
positive = data[data['Admitted'].isin([1])]
negative = data[data['Admitted'].isin([0])]
# fig, ax = plt.subplots(figsize=(12,8))
# ax.scatter(positive['Exam 1'], positive['Exam 2'], s=50, c='b', marker='o', label='Admitted')
# ax.scatter(negative['Exam 1'], negative['Exam 2'], s=50, c='r', marker='x', label='Not Admitted')
# ax.legend()
# ax.set_xlabel('Exam 1')
# ax.set_ylabel('Exam 2')
# plt.xlabel("Exam 1") 
# plt.ylabel("Exam 2") 
# plt.show()

# 1.2.1 sigmoid function
# x = np.linspace(-10, 10, 100) 
# z = 1/(1 + np.exp(-x)) 
# plt.plot(x, z) 
# plt.xlabel("x") 
# plt.ylabel("Sigmoid(X)") 
# plt.show() 
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# 1.2.2 cost function
data.insert(0, 'Ones', 1)
print(data.head())
cols = data.shape[1]
X = data.iloc[:, 0: cols - 1]
y = data.iloc[:, cols - 1:]
X = np.matrix(X.values)
y = np.matrix(y.values)
theta = np.matrix(np.array([0,0,0]))
# theta = np.zeros((1, 3))
# theta = np.matrix(theta)

def cost(theta, X, y):
    m = X.shape[0]
    # theta = np.matrix(theta)
    X = np.matrix(X)
    y = np.matrix(y)
    cost = np.sum(np.multiply(-y, np.log(sigmoid(X*theta.T))) - np.multiply(1 - y, np.log(1 - sigmoid(X*theta.T)))) / m
    return cost
print(cost(theta, X, y)) # 0.6931

# 1.2.3 Learning parameters using fminunc(minimize)
# result = minimize(cost, np.zeros((0, X.shape[1])))
def gradient(theta, X, y):
    theta = np.matrix(theta)
    X = np.matrix(X)
    y = np.matrix(y)
    m = X.shape[0]
    n = X.shape[1]
    g = np.zeros(n)
    for j in range(n):
        g[j] = sum(np.multiply(sigmoid(X*theta.T) -  y, X[:,j])) / m
    return g


print(gradient(theta, X, y))

theta = np.matrix(theta)
result = opt.fmin_tnc(func=cost, x0=np.matrix(theta), fprime=gradient, args=(X, y))

print("result0", result[0])
ans = sigmoid(np.sum(np.multiply([1,45,85],result[0])))
print("ans", ans)
x1 = np.linspace(30, 100, 100)
x2 = - (result[0][0] + result[0][1] * x1) / result[0][2] 

fig, ax = plt.subplots(figsize=(12,8))
ax.plot(x1, x2, 'g', label='Prediction')
ax.scatter(positive['Exam 1'], positive['Exam 2'], s=50, c='b', marker='o', label='Admitted')
ax.scatter(negative['Exam 1'], negative['Exam 2'], s=50, c='r', marker='x', label='Not Admitted')
ax.legend()
ax.set_xlabel('Exam 1')
ax.set_ylabel('Exam 2')
plt.xlabel("Exam 1") 
plt.ylabel("Exam 2") 
plt.show()


