# Import libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_circles
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Step 1: Generate Non-Linearly Separable Data
X, y = make_circles(n_samples=1000, noise=0.1, factor=0.5, random_state=42)

# Visualize the data
plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', edgecolors='k')
plt.title("Non-Linearly Separable Data")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()

# Step 2: Feature Engineering (Square the features)
# Add squared features
X_squared = np.c_[X, X[:, 0]**2, X[:, 1]**2]

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_squared, y, test_size=0.3, random_state=42)

# Step 3: Logistic Regression
model = LogisticRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")
print("Classification Report:\n", classification_report(y_test, y_pred))

# Visualize the decision boundary
def plot_decision_boundary_with_squared_features(X, y, model, ax):
    h = 0.02
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    # Add squared features to the grid
    grid = np.c_[xx.ravel(), yy.ravel(), xx.ravel()**2, yy.ravel()**2]
    Z = model.predict(grid)
    Z = Z.reshape(xx.shape)
    ax.contourf(xx, yy, Z, alpha=0.8, cmap='viridis')
    ax.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', cmap='viridis')

fig, ax = plt.subplots(figsize=(8, 6))
plot_decision_boundary_with_squared_features(X, y, model, ax)
ax.set_title("Decision Boundary with Squared Features")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()
