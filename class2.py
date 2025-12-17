import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_classification
from sklearn.neighbors import KNeighborsClassifier
x,y=make_classification(
    n_samples=200,n_features=2,n_classes=2,n_clusters_per_class=1,n_redundant=0,random_state=42
)
knn=KNeighborsClassifier(n_neighbors=5)
knn.fit(x,y)
x_plot=np.linspace(x.min(),x.max(),200).reshape(-1,2)
y_plot=knn.predict(x_plot)  
plt.scatter(x[:,0],x[:,1],c=y,cmap='bwr',alpha=0.7,label='Data Points')
plt.scatter(x_plot[:,0],x_plot[:,1],c=y_plot,cmap='bwr',alpha=0.2,marker='.',s=100,label='KNN Prediction(k=5)')
plt.title('KNN Classification on Synthetic Data')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.show()
