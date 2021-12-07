import cv2
from glcm import glcm

def imgExtraction(path):
    img = cv2.imread(path)
    hsvImg = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    hue = hsvImg[:, :, 0].mean()
    saturation = hsvImg[:, :, 1].mean()
    value = hsvImg[:, :, 2].mean()
    [g0, g45, g90, g135] = glcm(img)
    return [hue, saturation, value, g0['asm'], g45['asm'], g90['asm'], g135['asm'], g0['kontras'], g45['kontras'], g90['kontras'], g135['kontras'], g0['idm'], g45['idm'], g90['idm'],
            g135['idm'], g0['entropi'], g45['entropi'], g90['entropi'], g135['entropi'], g0['korelasi'], g45['korelasi'], g90['korelasi'], g135['korelasi'],]
