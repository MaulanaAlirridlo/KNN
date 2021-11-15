import xlsxwriter
from os import listdir
from os.path import isdir, isfile, splitext
from statistics import mean
import cv2

hsvArray = []
path = "./images/training"
valid_images = [".jpg", ".jpeg", ".png"]

workbook = xlsxwriter.Workbook('classification.xlsx')
worksheet = workbook.add_worksheet()

loop = 0

for folder in listdir(path):
    folderPath = path+"/"+folder
    if isdir(folderPath) :
        for file in listdir(folderPath) :
            ext = splitext(file)[1]
            if ext.lower() in valid_images:
                img = cv2.imread(folderPath+"/"+file)
                hsvImg = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
                hue = hsvImg[:, :, 0].mean()
                saturation = hsvImg[:, :, 1].mean()
                value = hsvImg[:, :, 2].mean()
                hsvArray.append([hue, saturation, value])
                
                worksheet.write(loop, 0, folder)
                loop += 1
workbook.close()

workbook = xlsxwriter.Workbook('hsv.xlsx')
worksheet = workbook.add_worksheet()
for row, hsv in enumerate(hsvArray):
    worksheet.write_row(row, 0, hsv)
workbook.close()
