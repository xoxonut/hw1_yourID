import numpy as np
import pandas as pd
from problem_1 import get_gaussian_basis_func,MSE,get_sigma_values,plot_3d,split_data,predict,test
import matplotlib.pyplot as plt
# Files path setting (don't change this)
TRAIN_PATH = "inputs/training_dataset.csv"
TEST_PATH = "inputs/testing_dataset.csv"
SAVE_PATH = "outputs/result_3.csv"
def bayesian_linear_regression(PHI, t, alpha, beta):
    _,M = PHI.shape
    S_0_inv = alpha * np.eye(M)
    S_N_inv = S_0_inv + beta * PHI.T @ PHI
    S_N = np.linalg.inv(S_N_inv)
    m_0 = np.zeros(M)
    m_N = (S_N @ (S_0_inv @ m_0 + beta * PHI.T @ t)).reshape(-1)
    return m_N     


def train(O1,O2,s,e,alpha,beta):
    mse = np.inf
    N = 5
    best_w = None
    datas = split_data(TRAIN_PATH,N)
    for i in range(N):
        train_data = pd.concat([datas[j] for j in range(N) if j != i])
        x_train = train_data.iloc[:,:-1].values
        y_train = train_data.iloc[:,-1].values
        val = datas[i]
        x_val = val.iloc[:,:-1].values
        y_val = val.iloc[:,-1].values
        sigma_values = get_sigma_values(O1, O2,s,e)
        mu_x1 = np.linspace(0, 1, O1)
        mu_x2 = np.linspace(0, 1, O2)
        Mu = np.array(np.meshgrid(mu_x1, mu_x2)).T.reshape(-1, 2)
        Phi = get_gaussian_basis_func(x_train, Mu, sigma_values)
        Phi = np.column_stack([np.ones(x_train.shape[0]), Phi])
        w = bayesian_linear_regression(Phi, y_train, alpha, beta)
        pred_train = predict(Phi, w)
        print("train MSE: ", MSE(y_train, pred_train))
    
        Phi = get_gaussian_basis_func(x_val, Mu, sigma_values)
        Phi = np.column_stack([np.ones(x_val.shape[0]), Phi])
        pred_val = predict(Phi, w)
        print("val MSE: ", MSE(y_val, pred_val))
        if mse > MSE(y_val, pred_val):
            mse = MSE(y_val, pred_val)
            best_w = w
    return best_w
def save_result(preds: np.ndarray, weights: np.ndarray):
    """
    Save prediction and weights to a CSV file
    - `preds`: predicted values with shape (n_samples,)
    - `weights`: model weights with shape (n_basis,)
    """

    max_length = max(len(preds), len(weights))

    result = np.full((max_length, 2), "", dtype=object)
    result[:len(preds), 0] = preds.astype(str)
    result[:len(weights), 1] = weights.astype(str)

    np.savetxt(SAVE_PATH, result, delimiter=",", fmt="%s")

def main(O1,O2,s,e,alpha,beta):
    w = train(O1,O2,s,e,alpha,beta)
    pred_test = test(w,O1,O2,s,e,TEST_PATH)
    save_result(pred_test, w)
if __name__ == "__main__":
    main(25,30,0.01,0.05,0.0001,10)
# data = pd.read_csv(TRAIN_PATH)
# x_train = data.iloc[:,:-1].values

# t_train = data.iloc[:,-1].values
# alpha = 0.0001         # prior precision
# beta = 10  #noise precision
# O1, O2 =  25, 30
# s, e = 0.02,0.1
# mu_x1 = np.linspace(0, 1, O1)
# mu_x2 = np.linspace(0, 1, O2)
# Mu = np.array(np.meshgrid(mu_x1, mu_x2)).T.reshape(-1, 2)
# sigma_values = get_sigma_values(O1, O2,s,e)
# PHI = get_gaussian_basis_func(x_train, Mu, sigma_values)
# PHI = np.column_stack([np.ones(x_train.shape[0]), PHI])  # Add bias term
# _,M = PHI.shape
# S_0_inv = alpha * np.eye(M)
# S_N_inv = S_0_inv + beta * PHI.T @ PHI
# S_N = np.linalg.inv(S_N_inv)
# m_0 = np.zeros(M)
# m_N = (S_N @ (S_0_inv @ m_0 + beta * PHI.T @ t_train)).reshape(-1)        
# pred = np.maximum(PHI @ m_N,0)
# print(MSE(t_train, pred))
# plot_3d(x_train, t_train.ravel(), "outputs/3d_scatter_train.png")
# plot_3d(x_train, pred, "outputs/3d_scatter_pred11.png")
# data = pd.read_csv(TEST_PATH)
# x_test = data.iloc[:,:-1].values
# t_test = data.iloc[:,-1].values
# PHI = get_gaussian_basis_func(x_test, Mu, sigma_values)
# PHI = np.column_stack([np.ones(x_test.shape[0]), PHI])  # Add bias term
# pred = np.maximum(PHI @ m_N,0)
# print(MSE(t_test, pred))
# plot_3d(x_test, t_test.ravel(), "outputs/3d_scatter_test.png")
# plot_3d(x_test, pred, "outputs/3d_scatter_pred_test.png")