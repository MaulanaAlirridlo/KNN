import xlsxwriter
from concurrent.futures import ThreadPoolExecutor
from os import listdir
from os.path import isdir, isfile, splitext, join
from statistics import mean
from glcm import glcm
import cv2
import time

start_time1 = time.time()
hsvArray = []
path = "./images/training"
valid_images = [".jpg", ".jpeg", ".png"]

hsvArray.append(["Hue", "Saturation", "Value", "ASM0", "ASM45", "ASM90", "ASM135", "kontras0", "kontras45", "kontras90", "kontras135", "IDM0", "IDM45", "IDM90",
                 "IDM135", "Entropi0", "Entropi45", "Entropi90", "Entropi135", "Korelasi0", "Korelasi45", "Korelasi90", "Korelasi135", "Class"])


def process(file, folder, folderPath):
    print(file, folder)
    start_time = time.time()
    img = cv2.imread(join(folderPath, file))
    hsvImg = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    hue = hsvImg[:, :, 0].mean()
    saturation = hsvImg[:, :, 1].mean()
    value = hsvImg[:, :, 2].mean()
    [g0, g45, g90, g135] = glcm(img)
    hsvArray.append([hue, saturation, value, g0['asm'], g45['asm'], g90['asm'], g135['asm'], g0['kontras'], g45['kontras'], g90['kontras'], g135['kontras'], g0['idm'], g45['idm'], g90['idm'],
                    g135['idm'], g0['entropi'], g45['entropi'], g90['entropi'], g135['entropi'], g0['korelasi'], g45['korelasi'], g90['korelasi'], g135['korelasi'], folder])
    print(file, folder, time.time() - start_time)


with ThreadPoolExecutor(max_workers=10) as executor:
    # for folder in listdir(path):
    #     folderPath = join(path, folder)
    #     if isdir(folderPath):
    #         for file in listdir(folderPath):
    #             ext = splitext(file)[1]
    #             if ext.lower() in valid_images:
    #                 task = executor.submit(process, *[file, folder, folderPath])

    path = "./backup/fadhil"
    folder = "matang"
    for file in listdir(path):
        ext = splitext(file)[1]
        if ext.lower() in valid_images:
            task = executor.submit(process, *[file, folder, path])


workbook = xlsxwriter.Workbook('x.xlsx')
worksheet = workbook.add_worksheet()
for row, hsv in enumerate(hsvArray):
    worksheet.write_row(row, 0, hsv)
workbook.close()
print(time.time() - start_time1)
