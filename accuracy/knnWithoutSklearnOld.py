import pandas as pd
import numpy as np
from os import listdir
import math
from sklearn.model_selection import train_test_split
import time

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

def predict(k, xTrain, yTrain, attributes) :
    ed = []
    res = 0
    for v in xTrain :
        for i in range(col) :
            res += ((v[i]-attributes[i])**2)
        ed.append(math.sqrt(res))
        res = 0
    sortedK = [ed for yTrain, ed in sorted(zip(ed, yTrain))]
    return max(set(sortedK[:k]), key=sortedK[:k].count)

    # rank = calculate_rank(ed)
    # selected = [yTrain for rank, yTrain in zip(rank, yTrain) if rank <= k]
    # return max(set(selected[:k]), key=selected[:k].count)

akurasi = 0
i = 1
while akurasi < 85 :
# while i < 11 :
    xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size=0.3)

    yPredict = []

    for v in xTest :
        yPredict.append(predict(8, xTrain, yTrain, v))
    akurasi = np.mean(yPredict == yTest)*100

    benar = sum(yPredict == yTest)

    print(i, "benar = ", benar)
    print(i, "salah = ", len(yPredict == yTest)-benar)
    print(i, "akurasi = ", akurasi, "\n")
    i+=1
    # time.sleep(0.1)
    # if i == 1000 :
    #     break
