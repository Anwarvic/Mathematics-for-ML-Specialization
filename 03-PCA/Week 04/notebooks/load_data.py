from six.moves import urllib
from sklearn.datasets import fetch_mldata

def load_mnist():
    mnist_path = "./mnist-original.mat"
    from scipy.io import loadmat
    mnist_raw = loadmat(mnist_path)
    mnist = {
        "data": mnist_raw["data"].T,
        "target": mnist_raw["label"][0],
        "COL_NAMES": ["label", "data"],
        "DESCR": "mldata.org dataset: mnist-original",
    }
    return mnist