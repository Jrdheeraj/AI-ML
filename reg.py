from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score

# 1. Load real regression dataset
diabetes = load_diabetes()
X = diabetes.data      # 10 medical features
y = diabetes.target    # disease progression (continuous)

# 2. Train–test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 3. Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 4. Create and train KNN regressor
knn_reg = KNeighborsRegressor(
    n_neighbors=5,
    metric="minkowski",
    p=2              # Euclidean
)
knn_reg.fit(X_train_scaled, y_train)

# 5. Predict
y_pred = knn_reg.predict(X_test_scaled)

# 6. Evaluate
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("MSE:", mse)
print("R² score:", r2)
