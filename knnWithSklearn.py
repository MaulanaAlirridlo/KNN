import pandas as pd
import numpy as np
from os import listdir
from sklearn.neighbors import KNeighborsClassifier
from imgExtraction import imgExtraction

df = pd.read_excel("extraction.xlsx")

x = np.array(df.iloc[:, 0:23])
y = np.array(df['Class'])

classifier = KNeighborsClassifier(n_neighbors=3)
classifier.fit(x, y)

path = './images/testing'
for file in listdir(path) :
    classified = classifier.predict([imgExtraction(path+"/"+file)])
    print(classified)