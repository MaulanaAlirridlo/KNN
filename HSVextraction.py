import cv2

def hsvExtraction(path):
    img = cv2.imread(path)
    hsvImg = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    hue = hsvImg[:, :, 0].mean()
    saturation = hsvImg[:, :, 1].mean()
    value = hsvImg[:, :, 2].mean()
    return [[hue, saturation, value]]
