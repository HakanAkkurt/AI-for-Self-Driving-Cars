import cv2
import numpy as np

def canny(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    canny = cv2.Canny(blur, 50, 150)
    return canny

def displayLines(image, lines):
    lineImage = np.zeros_like(image)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line.reshape(4)
            cv2.line(lineImage, (x1, y1), (x2, y2), (255, 0, 0), 10)
    return lineImage

def regionOfInterest(image):
    height = image.shape[0]
    polygons = np.array([
        [(200, height), (1100, height), (550, 250)]
    ])
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, polygons, 255)
    maskedImage = cv2.bitwise_and(image, mask)
    return maskedImage

image = cv2.imread('test_image.jpg')
laneImage = np.copy(image)
canny = canny(laneImage)
croppedImage = regionOfInterest(canny)
lines = cv2.HoughLinesP(croppedImage, 2, np.pi/180, 100, np.array([]), minLineLength = 40, maxLineGap = 5)
lineImage = displayLines(laneImage, lines)
comboImage = cv2.addWeighted(laneImage, 0.8, lineImage, 1, 1)
cv2.imshow('result', comboImage)
cv2.waitKey(0)