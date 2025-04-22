import cv2 as cv
import numpy as np

def detecting_area(frame):
    h, w, _ = frame.shape
    pts = np.array([[0,650], [600,500], [700,400], [1400,450], [1300,1050], [0,1050]])
    mask = np.zeros_like(frame)
    cv.fillPoly(mask, [pts], (1,1,1))
    return frame * mask


def draw_bounds(frame):
    color = (0,255,255)
    # draw ROI lines...
    # (copy your existing cv.line calls here)
    # then lightly shade inside
    pts = np.array([[0,650], [600,500], [700,400], [1400,450], [1300,1050], [0,1050]])
    mask = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)
    cv.fillPoly(mask, [pts], 1)
    overlay = np.zeros_like(frame)
    overlay[mask==1] = (221,218,250)
    frame[mask==1] = 0.7*frame[mask==1] + 0.3*overlay[mask==1]
    return frame
