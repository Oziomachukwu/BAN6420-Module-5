# pca_cancer_analysis.py
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

# automatically set the working directory to the script's directory
script_dir = os.path.dirname(os.path.abspath(__file__))

# output files to the same directory
pca_csv_path = os.path.join(script_dir, "pca_components.csv")
predictions_csv_path = os.path.join(script_dir, "predictions.csv")
plot_path = os.path.join(script_dir, "pca_scatter.png")

# Load the cancer dataset
data = load_breast_cancer()
X = data.data
y = data.target

# Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply PCA
pca = PCA()
X_pca = pca.fit_transform(X_scaled)

# Explained variance ratio
explained_variance = pca.explained_variance_ratio_
print("Explained variance by each component:", explained_variance)

# Reduce to 2 components
pca_2 = PCA(n_components=2)
X_reduced = pca_2.fit_transform(X_scaled)

# Create a DataFrame for visualization
pca_df = pd.DataFrame(data=X_reduced, columns=['PC1', 'PC2'])
pca_df['Target'] = y

# Plot the 2D PCA result
plt.figure(figsize=(8,6))
sns.scatterplot(data=pca_df, x='PC1', y='PC2', hue='Target', palette='Set1')
plt.title('PCA: 2 Component Projection')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend(title='Target')
plt.grid(True)
plt.tight_layout()
plt.savefig('pca_scatter.png')
plt.show()

# BONUS: Logistic Regression on reduced data
X_train, X_test, y_train, y_test = train_test_split(X_reduced, y, test_size=0.2, random_state=42)
model = LogisticRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Evaluation
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

print("\nLogistic Regression Accuracy on PCA-reduced data:", accuracy)
print("Confusion Matrix:\n", conf_matrix)


# Save predictions and components for review
predictions_df = pd.DataFrame({'True Label': y_test, 'Predicted': y_pred})
predictions_df.to_csv(predictions_csv_path, index=False)
np.savetxt(pca_csv_path, X_reduced, delimiter=",", header="PC1,PC2", comments="")

# print the paths of the saved files
print("✅ Files saved successfully in script directory:\n")
print(" - PCA components:", pca_csv_path)
print(" - Predictions:", predictions_csv_path)
print(" - PCA scatter plot:", plot_path)
