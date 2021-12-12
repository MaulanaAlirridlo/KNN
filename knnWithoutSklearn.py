import pandas as pd
import numpy as np
from os import listdir
import math
from imgExtraction import imgExtraction
import time
start = time.time()
df = pd.read_excel("extraction.xlsx")
col = 23

x = np.array(df.iloc[:, 0:col])
y = np.array(df['Class'])

def predict(k, attributes) :
    ed = []
    res = 0
    for v in x :
        for i in range(col) :
            res += ((v[i]-attributes[i])**2)
        ed.append(math.sqrt(res))
        res = 0
    sortedK = [ed for y, ed in sorted(zip(ed, y))]
    print(sortedK)
    return max(set(sortedK[:k]), key=sortedK[:k].count)

path = './images/testing'
for file in listdir(path) :
    img = imgExtraction(path+"/"+file)
    print(img)
    classified = predict(3, img)
    print(classified)

print(time.time()-start)
