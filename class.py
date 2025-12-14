from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report

# 1. Load real dataset
iris = load_iris()
X = iris.data          # features: sepal length/width, petal length/width
y = iris.target        # labels: 0,1,2 -> iris species

# 2. Train–test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 3. Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 4. Create and train KNN classifier
knn_clf = KNeighborsClassifier(
    n_neighbors=5,
    metric="minkowski",
    p=2              # p=2 → Euclidean
)
knn_clf.fit(X_train_scaled, y_train)

# 5. Predict
y_pred = knn_clf.predict(X_test_scaled)

# 6. Evaluate
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification report:\n", classification_report(y_test, y_pred, target_names=iris.target_names))
