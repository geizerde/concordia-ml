import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np
# Загрузка данных
df = pd.read_csv("dataset.csv")
df.drop(columns=['sex'], inplace=True)
df = df.dropna()
# Преобразование категориальных данных с помощью one-hot encoding
df = pd.get_dummies(df, columns=['body_type', 'diet', 'drinks', 'education', 'pets', 'smokes'], drop_first=True)

# Нормализация числовых данных
scaler = StandardScaler()
df[['age', 'height']] = scaler.fit_transform(df[['age', 'height']])

# Если нужно, добавьте другие столбцы

from sklearn.neighbors import NearestNeighbors

# Применение KNN для нахождения ближайших соседей
count_neighbors = 100
knn = NearestNeighbors(n_neighbors=count_neighbors, metric='cosine')
knn.fit(df)

distances, indices = knn.kneighbors(df.iloc[0].values.reshape(1, -1))

print("Похожие пользователи для пользователя 0:", indices[0][1:])  # Исключаем самого себя
d_min = np.min(distances)
d_max = np.max(distances)
# Нормализуем расстояния, вычитаем из 1 для получения схожести от 0 до 1 и округляем до одного знака после запятой
normalized_distances = np.round(1 - (distances - d_min) / (d_max - d_min), 1)
"""
print("Расстояния до ближайших соседей:", normalized_distances)  # Вывод расстояний до ближайших соседей
for i in indices[0]:
    print(df.iloc[i])  # Вывод данных для каждого из ближайших соседей
"""