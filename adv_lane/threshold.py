import cv2
import numpy as np


def hls_thresh(img, thresh=(0, 255)):
    """
    Generate HLS S Channel Threshold Binary
    """
    hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    s_channel = hls[:,:,2]

    sbin = np.zeros_like(s_channel)
    sbin[(s_channel>=thresh[0]) & (s_channel<=thresh[1])] = 1

    return sbin

def cal_sobel(img, sobel_kernel=3):
    """
    Generate Sobel X,Y Result
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)

    return sobelx, sobely

def sobel_thresh(sobel, thresh=(0, 255)):
    """
    """
    abs_sobel = np.absolute(sobel)
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))

    sbin = np.zeros_like(scaled_sobel)
    sbin[(scaled_sobel>=thresh[0])&(scaled_sobel<=thresh[1])] = 1

    return sbin

def mag_thresh(sobelx, sobely, thresh=(0, 255)):
    gradmag = np.sqrt(sobelx**2+sobely**2)

    #gradmag = np.uint8(255*gradmag/np.max(gradmag))

    scale_factor = np.max(gradmag)/255
    gradmag = (gradmag/scale_factor).astype(np.uint8)

    sbin = np.zeros_like(gradmag)
    sbin[(gradmag>=thresh[0])&(gradmag<=thresh[1])] = 1

    return sbin

def dir_thresh(sobelx, sobely, thresh=(0, np.pi/2)):
    absgraddir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))

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
    # combined[(((sobx==1)&(soby==1))|((mag==1)&(gdir==1)))|(hls==1)] = 1
    # Remove y direction for this situration
    combined[(sobx==1|((mag==1)&(gdir==1)))|(hls==1)] = 1

    return combined
