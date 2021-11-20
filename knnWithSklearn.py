import pandas as pd
import numpy as np
from os import listdir
from sklearn.neighbors import KNeighborsClassifier
from hsvExtraction import hsvExtraction

df = pd.read_excel("hsv.xlsx")

x = np.array(df.iloc[:, 0:3])
y = np.array(df['Class'])

classifier = KNeighborsClassifier(n_neighbors=3)
classifier.fit(x, y)

path = './images/testing'
for file in listdir(path) :
    classified = classifier.predict([hsvExtraction(path+"/"+file)])
    print(classified)