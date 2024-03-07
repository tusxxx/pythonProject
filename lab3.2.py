import pandas as pd
from sklearn.preprocessing import scale
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import cross_val_score
import numpy as np

# Загрузка выборки Boston
data_url = "http://lib.stat.cmu.edu/datasets/boston"
raw_df = pd.read_csv(data_url, sep="\s+", skiprows=22, header=None)
data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
target = raw_df.values[1::2, 2]

X = data
y = target

# Приведение признаков к одному масштабу
X_scaled = scale(X)

# Перебор параметра p
p_values = np.linspace(1, 10, 200)
best_p = None
best_score = float('-inf')

for p in p_values:
    knn = KNeighborsRegressor(n_neighbors=5, weights='distance', p=p)
    scores = cross_val_score(knn, X_scaled, y, scoring='neg_mean_squared_error', cv=5)
    mean_score = scores.mean()

    if mean_score > best_score:
        best_score = mean_score
        best_p = p

print("Оптимальное значение параметра p:", best_p)
