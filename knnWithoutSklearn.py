import pandas as pd
import numpy as np
from os import listdir
import math
from hsvExtraction import hsvExtraction

df = pd.read_excel("hsv.xlsx")

x = np.array(df.iloc[:, 0:3])
y = np.array(df['Class'])

def predict(k, attributes) :
  ed = []
  for v in x :
    res = ((v[0]-attributes[0])**2)+((v[1]-attributes[1])**2)+((v[2]-attributes[2])**2)
    ed.append(math.sqrt(res))
  sortedK = [ed for y, ed in sorted(zip(ed, y))]
  return max(set(sortedK[:k]), key=sortedK[:k].count)

path = './images/testing'
for file in listdir(path) :
    kelas = predict(3, hsvExtraction(path+"/"+file))
    print(kelas)
