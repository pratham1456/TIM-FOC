import cv2 
import imutils
import numpy as np
from background_remove import background_remove
def peel_off(image_path):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Image not found or unable to read.")
    roi=background_remove(image)
    if roi is None:
        raise ValueError("No suitable ROI found in the image.")
    # color_roi=roi.copy()
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    new_width = int(roi.shape[1] * 0.5)
    new_height = int(roi.shape[0] * 0.5)

    averaging = cv2.blur(gray, (3, 3))
    edges=imutils.auto_canny(averaging)
    # cv2.imshow('Edges', cv2.resize(edges, (new_width, new_height)))

    border_margin = 30
    height,width= edges.shape
    safe_mask=np.zeros_like(edges, dtype=np.uint8)
    cv2.rectangle(safe_mask, (border_margin, border_margin), (width - border_margin, height - border_margin), 255, -1)


    trimmed_edges = cv2.bitwise_and(edges, safe_mask)
    defect_name='Ok'
    if np.count_nonzero(trimmed_edges) >0:
        roi[trimmed_edges>0] = [0, 0, 255]
        kernel=np.ones((3, 3), np.uint8)
        thick_edges=cv2.dilate(trimmed_edges, kernel, iterations=2)
        roi[thick_edges > 0] = [0, 0, 255]
        defect_name = "Peel Off Detected"
        cv2.putText(roi, defect_name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    else:
        cv2.putText(roi, defect_name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    
    return roi,defect_name