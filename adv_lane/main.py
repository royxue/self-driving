import cv2
import numpy as np

from calibration import *
from curvature import *
from search import *
from threshold import cal_thresh_img
from vis import vis
from warp import warper


def process(image):
    objp, imgp = cal_calib_points('./camera_cal/calibration*.jpg')
    undist = cal_undistort(image, objp, imgp)

    thresh_s = (120, 255)
    thresh_sx = (20, 100)
    thresh_sy = (20, 100)
    thresh_mag = (0, 255)
    thresh_gdir = (0, np.pi/2)
    com = cal_thresh_img(undist, thresh_s, thresh_sx, thresh_sy, thresh_mag, thresh_gdir)
    
    src = np.float32([[200, 720], [1100, 720], [595, 450], [685, 450]])
    dst = np.float32([[300, 720], [980, 720], [300, 0], [980, 0]])
    warped = warper(com, src, dst)
    
    left_fit, right_fit = sliding_window_search(warped)
    
    res = vis(undist, warped, left_fit, right_fit, src, dst)
    
    left_c, right_c = cal_curvance(warped, left_fit, right_fit)
    
    return res

def main():
    return

if __name__ == "__main__":
    main()
