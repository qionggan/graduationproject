import matplotlib.pyplot as plt
import numpy as np
def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-1.0 * z))
z = np.linspace(-10,10,100)
y_ = sigmoid(z)
plt.plot(z, y_, color="black", linewidth=2, linestyle="--" )
# y=1时的原损失函数
def L_p_p(z):
    return 1 -  sigmoid(z)
# y=1时的替代损失函数
def L_p(z):
    return -1.0 * np.log(sigmoid(z))

z_p = np.linspace(-2,5,100)
Lp = L_p(z_p)
plt.plot(z_p, Lp, color="black", linewidth=2, linestyle="--" )
Lpp = L_p_p(z_p)
plt.plot(z_p, Lpp, color="red", linewidth=2)
# y=0时的原损失函数
def L_n_n(z):
    return sigmoid(z)
# y=0时的替代损失函数
def L_n(z):
    return -1.0 * np.log(1.0 - sigmoid(z))

z_n = np.linspace(-5,2,100)
Ln = L_n(z_n)
plt.plot(z_n, Ln, color="black", linewidth=2, linestyle="--" )
Lnn = L_n_n(z_n)
plt.plot(z_n, Lnn, color="red", linewidth=2)