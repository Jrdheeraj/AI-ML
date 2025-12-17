import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_regression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split

x, y = make_regression(
    n_samples=100, n_features=1, noise=15, random_state=42
)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

knn = KNeighborsRegressor(n_neighbors=5)
knn.fit(x_train, y_train)

x_plot = np.linspace(x.min(), x.max(), 200).reshape(-1, 1)
y_plot = knn.predict(x_plot)

plt.scatter(x_train, y_train, color='blue', label='Training Data')
plt.plot(x_plot, y_plot, color='red', label='KNN Regression Prediction(k=5)')
plt.title('KNN Regression on Synthetic Data')
plt.xlabel('Feature')
plt.ylabel('Target')    
plt.legend()
plt.show()
