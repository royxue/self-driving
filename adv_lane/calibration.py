import cv2
import glob
import numpy as np


# Compute Camera Calibration Matrix
def cal_calib_points(filepath):
    objp = np.zeros((6*9, 3), np.float32)
    objp[:,:2]  = np.mgrid[0:9, 0:6].T.reshape(-1, 2)

    objpoints = [] # 3d points in real world
    imgpoints = [] # 2d points in image plane

    images = glob.glob(filepath)

    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        ret, corners = cv2.findChessboardCorners(gray, (9,6), None)

        if ret == True:
            objpoints.append(objp)
            imgpoints.append(corners)

    return objpoints, imgpoints

# Apply distortation on raw images
def cal_undistort(img, objp, imgp):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objp, imgp, gray.shape[::-1], None, None)

    if ret:
        dst = cv2.undistort(img, mtx, dist, None, mtx)
        return dst
    else:
        return None