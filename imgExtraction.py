import cv2
from glcm import glcm
import numpy as np

def RGBtoHSV(image):
    sample = (image * (1/255.0))
    B, G, R = cv2.split(sample)

    rows, cols, channels = sample.shape

    V = np.zeros(sample.shape[:2], dtype=np.float64)
    S = np.zeros(sample.shape[:2], dtype=np.float64)
    H = np.zeros(sample.shape[:2], dtype=np.float64)

    for i in range(rows):
        for j in range(cols):
            V[i, j] = max(B[i, j], G[i, j], R[i, j])
            Min_RGB = min(B[i, j], G[i, j], R[i, j])

            if V[i, j] != 0.0:
                S[i, j] = ((V[i, j] - Min_RGB) / V[i, j])
            else:
                S[i, j] = 0.0

            if V[i, j] == Min_RGB:
                H[i, j] = 0
            elif V[i, j] == R[i, j]:
                H[i, j] = 60*(G[i, j] - B[i, j])/(V[i, j] - Min_RGB)
            elif V[i, j] == G[i, j]:
                H[i, j] = 120 + 60*(B[i, j] - R[i, j])/(V[i, j] - Min_RGB)
            elif V[i, j] == B[i, j]:
                H[i, j] = 240 + 60*(R[i, j] - G[i, j])/(V[i, j] - Min_RGB)

            if H[i, j] < 0:
                H[i, j] = H[i, j] + 360

    V = 255.0 * V
    S = 255.0 * S
    H = H/2
    hsv = np.round(cv2.merge((H, S, V)))
    return hsv.astype(np.uint8)

def imgExtraction(path):
    img = cv2.imread(path)
    hsvImg = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

    # hsvImg1 = RGBtoHSV(img)

    # cv2.imshow("cv", hsvImg)
    # cv2.imshow("saya", hsvImg1)
    # cv2.imshow("beda", hsvImg-hsvImg1)

    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    hue = hsvImg[:, :, 0].mean()
    saturation = hsvImg[:, :, 1].mean()
    value = hsvImg[:, :, 2].mean()
    [g0, g45, g90, g135] = glcm(img)
    return [hue, saturation, value, g0['asm'], g45['asm'], g90['asm'], g135['asm'], g0['kontras'],
            g45['kontras'], g90['kontras'], g135['kontras'], g0['idm'], g45['idm'], g90['idm'],
            g135['idm'], g0['entropi'], g45['entropi'], g90['entropi'], g135['entropi'], g0['korelasi'],
            g45['korelasi'], g90['korelasi'], g135['korelasi']]