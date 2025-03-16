import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# Files path setting (don't change this)
TRAIN_PATH = "inputs/training_dataset.csv"
TEST_PATH = "inputs/testing_dataset.csv"
SAVE_PATH = "outputs/result_1.csv"

def get_gaussian_basis_func(X, Mu, sigma_values):
   return np.exp(-np.sum(((X[:, None, :] - Mu[None, :, :]) ** 2) / (2 * sigma_values[None, :, :] ** 2), axis=2))
def get_sigma_values(O1, O2,s,e):
    return np.tile(np.linspace(s, e, O1 * O2).reshape(-1, 1), (1, 2))
def MLE(X, y):
    return np.linalg.pinv(X) @ y
def predict(X, w):
    y_pred = X @ w
    y_pred = np.maximum(y_pred, 0)
    return y_pred
def MSE(y, y_pred):
    return np.mean((y - y_pred) ** 2)

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

def split_data(path,n_splits):
    train = pd.read_csv(path,header=None)
    train = [train.sample(frac=1/n_splits,random_state=i).reset_index(drop=True) for i in range(n_splits)]
    return train
def train(O1,O2,s,e):
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
        w = MLE(Phi, y_train)
        pred_train = predict(Phi, w)
        print("train MSE: ", MSE(y_train, pred_train))
    
        Phi = get_gaussian_basis_func(x_val, Mu, sigma_values)
        pred_val = predict(Phi, w)
        tmp = MSE(y_val, pred_val)
        print("val MSE: ", tmp)
        if tmp < mse:
            best_w = w
            mse = tmp
    return best_w
def test(w,O1,O2,s,e,path):
    test_data = pd.read_csv(path,header=None)
    x_test = test_data.iloc[:,:-1].values
    y_test = test_data.iloc[:,-1].values
    sigma_values = get_sigma_values(O1, O2,s,e)
    mu_x1 = np.linspace(0, 1, O1)
    mu_x2 = np.linspace(0, 1, O2)
    Mu = np.array(np.meshgrid(mu_x1, mu_x2)).T.reshape(-1, 2)
    Phi = get_gaussian_basis_func(x_test, Mu, sigma_values)
    pred_test = predict(Phi, w)
    print("test MSE: ", MSE(y_test, pred_test))
    plot_3d(x_test, y_test, "outputs/3d_scatter.png")
    plot_3d(x_test, pred_test, "outputs/3d_scatter_pred.png")
    return pred_test
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

def main(O1,O2,s,e):
    w = train(O1,O2,s,e)
    pred_test = test(w,O1,O2,s,e,"TEST_PATH")
    save_result(pred_test, w)
if __name__ == "__main__":
    main(33,36,0.05,0.27)