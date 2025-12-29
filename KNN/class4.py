import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_classification
from sklearn.neighbors import KNeighborsClassifier

# 1. Create synthetic dataset (2D for visualization)
X, y = make_classification(
    n_samples=369, n_features=2, n_classes=2,
    n_clusters_per_class=1, n_redundant=0, random_state=42
)

# 2. Train KNN classifier
knn = KNeighborsClassifier(n_neighbors=19)
knn.fit(X, y)

# 3. Create mesh grid for plotting decision boundary
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                     np.linspace(y_min, y_max, 200))  # Fixed incomplete line

# 4. Predict class for each point in the grid
Z = knn.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# 5. Plot decision boundary and data points
plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.coolwarm)
plt.scatter(X[:, 0], X[:, 1], c=y, edgecolor='k', cmap=plt.cm.coolwarm)
plt.title("KNN Classification (k=19)")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()
