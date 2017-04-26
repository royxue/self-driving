import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt

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

# Compute thresholded b-images
def hls_thresh(img, thresh=(0, 255)):
    # Get S Channel from the HLS image
    hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    s_channel = hls[:,:,2]

    sbin = np.zeros_like(s_channel)
    sbin[(s_channel>=thresh[0]) & (s_channel<=thresh[1])] = 1

    return sbin

def cal_sobel(img, sobel_kernel=3):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    sobelx = cv2.Sobel(gray, -1, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, -1, 0, 1, ksize=sobel_kernel)

    return sobelx, sobely

def sobel_thresh(sobel, thresh=(0, 255)):
    abs_sobel = np.absolute(sobel)
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))

    sbin = np.zeros_like(scaled_sobel)
    sbin[(scaled_sobel>=thresh[0])&(scaled_sobel<=thresh[1])] = 1

    return sbin

def mag_thresh(sobelx, sobely, thresh=(0, 255)):
    gradmag = np.sqrt(sobelx**2+sobely**2)
    gradmag = np.uint8(255*gradmag/np.max(gradmag))

    sbin = np.zeros_like(gradmag)
    sbin[(gradmag>=thresh[0])&(gradmag<=thresh[1])] = 1

    return sbin

def dir_thresh(sobelx, sobely, thresh=(0, np.pi/2)):
    absgraddir = np.arctan2(np.absolute(sobelx), np.absolute(sobely))

    sbin = np.zeros_like(absgraddir)
    sbin[(absgraddir>=thresh[0])&(absgraddir<=thresh[1])] = 1

    return sbin

def cal_thresh_img(img, thresh_s, thresh_sx, thresh_sy, thresh_mag, thresh_gdir):
    sobel_kernel = 3
    sobelx, sobely = cal_sobel(img, sobel_kernel)

    hls = hls_thresh(img, thresh_s)
    sobx = sobel_thresh(sobelx, thresh_sx)
    soby = sobel_thresh(sobely, thresh_sy)
    mag = mag_thresh(sobelx, sobely, thresh_mag)

    # Gradient direction use a bigger kernel for Sobel
    sobel_kernel = 15
    sobelx, sobely = cal_sobel(img, sobel_kernel)
    gdir = dir_thresh(sobelx, sobely, thresh_gdir)

    combined = np.zeros_like(hls)
    combined[(((sobx==1)&(soby==1)|((mag==1)&(gdir==1))))&(hls==1)] = 1

    return combined

# Apply a perspective transform to rectify binary image
def warper(img, src, dst):
    # Compute and apply perpective transform
    img_size = (img.shape[1], img.shape[0])
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_NEAREST)  # keep same size as input image
    return warped

# Detect lane pixel


def main():
    return

if __name__ == "__main__":
    main()
