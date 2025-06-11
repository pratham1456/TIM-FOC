import cv2 
import imutils
import numpy as np
from background_remove import background_remove

def dent(image_path):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Image not found or unable to read.")
    roi = background_remove(image)
    if roi is None:
        raise ValueError("No suitable ROI found in the image.")
    new_width = int(roi.shape[1] * 0.5)
    new_height = int(roi.shape[0] * 0.5)

    blurred = cv2.GaussianBlur(roi, (3, 3), 0)
    edges = imutils.auto_canny(blurred)

    border_margin = 20
    height, width = edges.shape
    safe_mask = np.zeros_like(edges, dtype=np.uint8)
    cv2.rectangle(safe_mask, (border_margin, border_margin), (width - border_margin, height - border_margin), 255, -1)
    trimmed_edges = cv2.bitwise_and(edges, safe_mask)           

    contours, _ = cv2.findContours(trimmed_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for i,cnt in enumerate(contours):
        hull = cv2.convexHull(cnt)
        hull_area = cv2.contourArea(hull)
        cnt_area = cv2.contourArea(cnt)
        dent_area = hull_area - cnt_area

        if dent_area >= 8:  # Threshold for dent detection
            has_dent = True
            defect_name = "Dent Detected"
            cv2.drawContours(roi, [cnt], -1, (0, 0, 255), 2)
            cv2.putText(roi, defect_name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            if not has_dent:
                defect_name = "Ok"
                cv2.putText(roi, defect_name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
            # defect_name = "Ok"
            # cv2.putText(roi, defect_name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    return roi, defect_name

# roi, defect_name = dent(r"D:\rahul_jadhav\OneDrive - Percepta Innovations\Projects\TIM\Dataset\9_06_25\big_curve_light\dent\VCXU.2-201C.R\image0000361.bmp")
# print(f"Defect Name: {defect_name}")
# new_width = int(roi.shape[1] * 0.5)
# new_height = int(roi.shape[0] * 0.5)
# cv2.imshow('Dent Detection Result', cv2.resize(roi, (new_width, new_height)))
# cv2.waitKey(0)
# cv2.destroyAllWindows()