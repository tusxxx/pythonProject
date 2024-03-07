import pandas as pd
import sklearn.model_selection
from sklearn import datasets
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import scale

data = datasets.load_wine(as_frame=True)
print(data.data)
print(data.target)
kf = sklearn.model_selection.KFold(shuffle=True, random_state=42)
knn = KNeighborsClassifier()

X_train = data.data[:]
y_train = data.target[:]
best_accuracy = 0
best_k = 0

for k in range(1, 51):
    knn = KNeighborsClassifier(n_neighbors=k)
    accuracy = cross_val_score(knn, X_train, y_train, cv=kf, scoring='accuracy').mean()
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_k = k

X_scaled = scale(X_train)
best_accuracy_scaled = 0
best_k_scaled = 0

for k in range(1, 51):
    knn_scaled = KNeighborsClassifier(n_neighbors=k)
    accuracy_scaled = cross_val_score(knn_scaled, X_scaled, y_train, cv=kf, scoring='accuracy').mean()
    if accuracy_scaled > best_accuracy_scaled:
        best_accuracy_scaled = accuracy_scaled
        best_k_scaled = k

print("Optimal k before scaling features:", best_k)
print("Accuracy before scaling features:", best_accuracy)
print("Optimal k after scaling features:", best_k_scaled)
print("Accuracy after scaling features:", best_accuracy_scaled)
