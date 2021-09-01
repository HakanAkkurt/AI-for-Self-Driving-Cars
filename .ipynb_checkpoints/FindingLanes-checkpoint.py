import cv2
import numpy as np

def makeCoordinates(image, lineParamaters):
    slope, intercept = lineParamaters
    y1 = image.shape[0]
    y2 = int(y1 * (3 / 5))
    x1 = int((y1 - intercept) / slope)
    x2 = int((y2 - intercept) / slope)

    return np.array([x1, y1, x2, y2])

def averageSlopeIntercept(image, lines):
    leftFit = []
    rightFit = []
    for line in lines:
        x1, y1, x2, y2 = line.reshape(4)
        parameters = np.polyfit((x1, x2), (y1, y2), 1)
        slope = parameters[0]
        intercept = parameters[1]
        if slope < 0:
            leftFit.append((slope, intercept))
        else:
            rightFit.append((slope, intercept))

    if len(leftFit) and len(rightFit):
        leftFitAverage  = np.average(leftFit, axis=0)
        rightFitAverage = np.average(rightFit, axis=0)
        leftLine  = makeCoordinates(image, leftFitAverage)
        rightLine = makeCoordinates(image, rightFitAverage)
        averagedLines = [leftLine, rightLine]
        return averagedLines

    elif len(leftFit):
        leftFitAverage  = np.average(leftFit, axis=0)
        leftLine  = makeCoordinates(image, leftFitAverage)
        averagedLines = [leftLine]
        return averagedLines
        
    elif len(rightFit):
        rightFitAverage = np.average(rightFit, axis=0)
        rightLine = makeCoordinates(image, rightFitAverage)
        averagedLines = [rightLine]
        return averagedLines


def canny(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    canny = cv2.Canny(blur, 50, 150)

    return canny

def displayLines(image, lines):
    lineImage = np.zeros_like(image)
    if lines is not None:
        for x1, y1, x2, y2 in lines:
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

cap = cv2.VideoCapture("test2.mp4")
while(cap.isOpened()):
    _, frame = cap.read()
    cannyImage = canny(frame)
    croppedImage = regionOfInterest(cannyImage)
    lines = cv2.HoughLinesP(croppedImage, 2, np.pi/180, 100, np.array([]), minLineLength = 40, maxLineGap = 5)
    averegedLines = averageSlopeIntercept(frame, lines)
    lineImage = displayLines(frame, averegedLines)
    comboImage = cv2.addWeighted(frame, 0.8, lineImage, 1, 1)
    cv2.imshow('result', comboImage)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()