from sklearn import datasets
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import numpy as np

linnerud = datasets.load_linnerud()
linnerud_X = linnerud.data[:, np.newaxis, 0]   # Use only one feature

# Split to train and test
linnerud_X_train = linnerud_X[:10]
linnerud_X_test = linnerud_X[10:]

Y = linnerud.target[: np.newaxis, 0]
linnerud_y_train = Y[:10]
linnerud_y_test = Y[10:]

lm = linear_model.LinearRegression()
lm.fit(linnerud_X_train, linnerud_y_train)

print("迴歸係數:",lm.coef_)
print("截距:",lm.intercept_)
