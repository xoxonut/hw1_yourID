import numpy as np
import pandas as pd
from problem_1 import get_gaussian_basis_func, get_sigma_values,split_data, predict,MSE,test
# Files path setting (don't change this)
TRAIN_PATH = "inputs/training_dataset.csv"
TEST_PATH = "inputs/testing_dataset.csv"
SAVE_PATH = "outputs/result_2.csv"

def MAP(Phi,y,lam):
    return np.linalg.inv(Phi.T @ Phi + lam*np.eye(Phi.shape[1])) @ Phi.T @ y


def train(O1,O2,s,e,lam):
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
        w = MAP(Phi, y_train, lam)
        pred_train = predict(Phi, w)
        print("train MSE: ", MSE(y_train, pred_train))
    
        Phi = get_gaussian_basis_func(x_val, Mu, sigma_values)
        Phi = np.column_stack([np.ones(x_val.shape[0]), Phi])
        pred_val = predict(Phi, w)
        tmp = MSE(y_val, pred_val)
        print("val MSE: ", tmp)
        if tmp < mse:
            mse = tmp
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

def main(O1,O2,s,e,lam):
    w = train(O1,O2,s,e,lam)
    preds = test(w,O1,O2,s,e,TEST_PATH)
    save_result(preds, w)

if __name__ == "__main__":
    main(25,30,0.02,0.1,0.00000006)
