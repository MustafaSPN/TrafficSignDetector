import cv2
import numpy as np
import statistics

templates = []
past = []
def colorSegmentation(image):
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    lower_red1 = np.array([0, 100, 100])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([160, 100, 100])
    upper_red2 = np.array([179, 255, 255])

    lower_blue = np.array([100, 100, 100])
    upper_blue = np.array([140, 255, 255])

    lower_white = np.array([0, 0, 180])
    upper_white = np.array([180, 50, 255])

    red_mask1 = cv2.inRange(hsv_image, lower_red1, upper_red1)
    red_mask2 = cv2.inRange(hsv_image, lower_red2, upper_red2)
    red_mask = cv2.bitwise_or(red_mask1, red_mask2)

    blue_mask = cv2.inRange(hsv_image, lower_blue, upper_blue)
    white_mask = cv2.inRange(hsv_image, lower_white, upper_white)

    combine_mask = cv2.bitwise_or(red_mask, blue_mask)
    combine_mask = cv2.bitwise_or(combine_mask,white_mask)
    segment_image = cv2.bitwise_and(image, image, mask=combine_mask)

    return segment_image
def loadTemplate(template_paths):
    templates = []
    for path in template_paths:
        template = cv2.imread(path)
        template = cv2.GaussianBlur(template,(7,7),2.6)
        template = cv2.cvtColor(template,cv2.COLOR_BGR2GRAY)
        template = cv2.Canny(template,210,220)
        templateTupple = (path[:-4],template)
        templates.append(templateTupple)
    return templates
def drawRectangle(image, text, x, y, width, height, color=(0, 255, 0), thickness=2):
    start_point = (x, y)
    end_point = (x + width, y + height)

    imageCopy = image.copy()
    cv2.rectangle(imageCopy, start_point, end_point, color, thickness)

    if text !="":
        font = cv2.FONT_HERSHEY_SIMPLEX
        fontScale = 0.5
        fontThickness = 1
        text_size, _ = cv2.getTextSize(text, font, fontScale, fontThickness)
        tw, th = text_size

        cv2.rectangle(imageCopy, (x + width + 10, y + 21), (x + width + 21 + tw, y + 19 - th), (0, 0, 0), -1)
        cv2.putText(imageCopy, text, (x + width + 20, y + 20), font, fontScale, (0, 255, 0), fontThickness)

    return imageCopy
def detectSign(orImage, image):
    contours, hierarchy = cv2.findContours(image,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
    originalImage = orImage.copy()
    global templates
    templatesCopy = templates.copy()

    for a,b in enumerate(contours):
        length = cv2.arcLength(b,True)
        approx = cv2.approxPolyDP(b,0.02*length,True)
        area = cv2.contourArea(b)
        x, y, h, w = cv2.boundingRect(approx)
        ratio = 0
        if w > h:
            ratio = w/h
        else:
            ratio = h/w
        if ratio < 3 and area > 1500 and hierarchy[0,a,3]==-1:
            originalImage = templateMatching(h, image, originalImage, templatesCopy, w, x, y)

            originalImage = drawRectangle(originalImage, "", x, y, w, h)

    return originalImage


def templateMatching(h, image, originalImage, templatesCopy, w, x, y):
    bestScore = 0
    bestScoreSign = ""
    global past

    for name, template in templatesCopy:
        resizedTemplate = cv2.resize(template, (w, h))
        croppedImage = image[y:y + h, x:x + w]
        result = cv2.matchTemplate(croppedImage, resizedTemplate, cv2.TM_CCOEFF_NORMED)
        _, max, _, _ = cv2.minMaxLoc(result)
        if (max > bestScore):
            bestScore = max
            bestScoreSign = name
    if (bestScore > 0.001):

        if len(past) > 50:
            past = []

        past.append(bestScoreSign)
    if len(past) > 0:
        originalImage = drawRectangle(originalImage,statistics.mode(past), x, y, w, h)
        print(statistics.mode(past))
    else:
        originalImage = drawRectangle(originalImage, "", x, y, w, h)
    return originalImage


import time
import time

import time


def main():
    global templates
    template_paths = ["no_enter.png", "bumps.png", "give_way.png", "pedestrian.png"]
    templates = loadTemplate(template_paths)
    frameWidth = 640
    frameHeight = 480
    cap = cv2.VideoCapture(0)
    cap.set(3, frameWidth)
    cap.set(4, frameHeight)

    prev_time = time.time()  # To store the previous time
    frame_count = 0  # To count the number of frames processed within a second
    fps = 0  # To store the FPS value

    while (True):
        success, image = cap.read()

        segmentedImage = colorSegmentation(image)
        bluredImage = cv2.GaussianBlur(segmentedImage, (7, 7), 2.0)
        grayImage = cv2.cvtColor(bluredImage, cv2.COLOR_BGR2GRAY)
        cannyImage = cv2.Canny(grayImage, 100, 200)

        detectedImage = detectSign(image, cannyImage)

        # Increment frame count
        frame_count += 1

        # Calculate FPS and update FPS value once per second
        current_time = time.time()
        elapsed_time = current_time - prev_time
        if elapsed_time >= 1.0:
            fps = frame_count / elapsed_time
            frame_count = 0
            prev_time = current_time

        # Display FPS
        cv2.putText(detectedImage, f"FPS: {int(fps)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow('Sign Detection', detectedImage)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
