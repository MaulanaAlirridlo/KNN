import pandas as pd
import numpy as np
from os import listdir
import math
from sklearn.model_selection import train_test_split

df = pd.read_excel("hsv.xlsx")

x = np.array(df.iloc[:, 0:3])
y = np.array(df['Class'])

def predict(k, xTrain, yTrain, attributes) :
  ed = []
  for v in xTrain :
    res = ((v[0]-attributes[0])**2)+((v[1]-attributes[1])**2)+((v[2]-attributes[2])**2)
    ed.append(math.sqrt(res))
  sortedK = [ed for yTrain, ed in sorted(zip(ed, yTrain))]
  return max(set(sortedK[:k]), key=sortedK[:k].count)

xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size=0.3)

yPredict = []

for v in xTest :
  yPredict.append(predict(3, xTrain, yTrain, v))

print("akurasi = ", np.mean(yPredict == yTest))
