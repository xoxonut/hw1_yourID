import numpy as np
import pandas as pd
from problem_1 import get_gaussian_basis_func,MSE,get_sigma_values
import matplotlib.pyplot as plt
# Files path setting (don't change this)
TRAIN_PATH = "inputs/training_dataset.csv"
TEST_PATH = "inputs/testing_dataset.csv"
SAVE_PATH = "outputs/result_3.csv"
def plot_3d(x, y, filename):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(x[:, 0], x[:, 1], y, c=y, cmap='viridis', marker='o')
    ax.set_xlabel('X1 Label')
    ax.set_ylabel('X2 Label')
    ax.set_zlabel('Y Label')
    fig.colorbar(scatter, ax=ax, shrink=0.5, aspect=5)
    plt.savefig(filename)
    plt.close()

data = pd.read_csv(TRAIN_PATH)
x_train = data.iloc[:,:-1].values

t_train = data.iloc[:,-1].values
alpha = 0.0001         # prior precision
beta = 10  #noise precision
O1, O2 =  25, 30
s, e = 0.02,0.1
mu_x1 = np.linspace(0, 1, O1)
mu_x2 = np.linspace(0, 1, O2)
Mu = np.array(np.meshgrid(mu_x1, mu_x2)).T.reshape(-1, 2)
sigma_values = get_sigma_values(O1, O2,s,e)
PHI = get_gaussian_basis_func(x_train, Mu, sigma_values)
PHI = np.column_stack([np.ones(x_train.shape[0]), PHI])  # Add bias term
N,M = PHI.shape
S_0_inv = alpha * np.eye(M)
S_N_inv = S_0_inv + beta * PHI.T @ PHI
S_N = np.linalg.inv(S_N_inv)
m_0 = np.zeros(M)
m_N = (S_N @ (S_0_inv @ m_0 + beta * PHI.T @ t_train)).reshape(-1)        
pred = np.maximum(PHI @ m_N,0)
print(MSE(t_train, pred))
plot_3d(x_train, t_train.ravel(), "outputs/3d_scatter_train.png")
plot_3d(x_train, pred, "outputs/3d_scatter_pred.png")
TEST_PATH = "inputs/(additional_small)testing_dataset.csv"
data = pd.read_csv(TEST_PATH)
x_test = data.iloc[:,:-1].values
t_test = data.iloc[:,-1].values
PHI = get_gaussian_basis_func(x_test, Mu, sigma_values)
PHI = np.column_stack([np.ones(x_test.shape[0]), PHI])  # Add bias term
pred = np.maximum(PHI @ m_N,0)
print(MSE(t_test, pred))
plot_3d(x_test, t_test.ravel(), "outputs/3d_scatter_test.png")
plot_3d(x_test, pred, "outputs/3d_scatter_pred_test.png")