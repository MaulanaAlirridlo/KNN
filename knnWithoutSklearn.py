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


def calculate_rank(arr):
    a = {}
    rank = 1
    for num in sorted(arr):
        if num not in a:
            a[num] = rank
            rank = rank+1
    return [a[i] for i in arr]


def predict(k, attributes):
    ed = []
    res = 0
    for v in x:
        for i in range(col):
            res += ((v[i]-attributes[i])**2)
        ed.append(math.sqrt(res))
        res = 0
    # sortedK = [y for ed, y in sorted(zip(ed, y))]
    # print(sortedK)
    # return max(set(sortedK[:k]), key=sortedK[:k].count)

    rank = calculate_rank(ed)
    selected = [y for rank, y in zip(rank, y) if rank <= k]
    print(tuple(sorted(zip(rank, y, ed))))
    print(selected)
    return max(set(selected[:k]), key=selected[:k].count)


path = './images/testing'
for file in listdir(path):
    img = imgExtraction(path+"/"+file)
    classified = predict(3, img)
    print(classified)

print(time.time()-start)
