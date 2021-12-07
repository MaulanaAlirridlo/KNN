import pandas as pd
import numpy as np
from os import listdir
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

df = pd.read_excel("extraction.xlsx")

x = np.array(df.iloc[:, 0:23])
y = np.array(df['Class'])

xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size=0.3)

knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(xTrain, yTrain)

yPredict = knn.predict(xTest)

print("akurasi = ", np.mean(yPredict == yTest))
