import cv2 
import imutils
import numpy as np

def background_remove(image):
    if len(image.shape) == 3 and image.shape[2] == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

        

    new_width = int(image.shape[1] * 0.6)
    new_height = int(image.shape[0] * 0.6)

    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # cv2.imshow('Thresholded Image', cv2.resize(thresh, (new_width, new_height)))

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # cv2.drawContours(image, contours, -1, (0, 255, 0), 3)
    # cv2.imshow('Contours', cv2.resize(image, (new_width, new_height)))

    max_contour = None
    max_area = 0
    max_height = 0
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area = cv2.contourArea(cnt)
        if area > max_area or (area == max_area and h > max_height):
            max_area = area
            max_height = h
            max_contour = cnt

    roi = None
    if max_contour is not None:
    
        x, y, w, h = cv2.boundingRect(max_contour)
        roi = image[y:y+h, x:x+w]
        # cv2.imshow('Largest Contour ROI', cv2.resize(roi, (int(w*0.4), int(h*0.4))))

    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return roi