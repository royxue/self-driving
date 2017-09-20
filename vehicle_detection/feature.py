import numpy as np
import cv2
from skimage.feature import hog

def hog_feature(img, orient, pix_per_cell, cell_per_block, vis=False, feature_vec=True):
	if vis == True:
		features, hog_image = hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
			cells_per_block=(cell_per_block, cell_per_block), transform_sqrt=False,
			visualise=True, feature_vector=False)
		return features, hog_image
	else:
		features = hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
			cells_per_block=(cell_per_block, cell_per_block), transform_sqrt=False,
			visualise=False, feature_vector=feature_vec)
		return features

def bin_feature(img, size=(32, 32)):
	features = cv2.resize(img, size).ravel()
	return features

def color_feature(img, nbins=32, bins_range=(0, 256)):
	channel1_hist = np.histogram(img[:,:,0], bins=nbins, range=bins_range)
	channel2_hist = np.histogram(img[:,:,1], bins=nbins, range=bins_range)
	channel3_hist = np.histogram(img[:,:,2], bins=nbins, range=bins_range)

	hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))

	return hist_features

def extract_features(img, color_space='RGB', spatial_size=(32, 32),
						hist_bins=32, orient=9,
						pix_per_cell=8, cell_per_block=2, hog_channel=0,
						bin_f=True, color_f=True, hog_f=True):
	img_features = []

	if color_space != 'RGB':
		if color_space == 'HSV':
			feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
		elif color_space == 'LUV':
			feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2LUV)
		elif color_space == 'YUV':
			feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
		elif color_space == 'YCrCb':
			feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
		elif color_space == 'HLS':
			feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
		elif color_space == 'GRAY':
				feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
				feature_image = np.stack((feature_image, feature_image, feature_image), axis=2)  # keep shape
	else: feature_image = np.copy(img)
	if bin_f == True:
		bin_features = bin_feature(feature_image, size=spatial_size)
		img_features.append(bin_features)

	if color_f == True:
		color_features = color_feature(feature_image, nbins=hist_bins)
		img_features.append(color_features)

	if hog_f == True:
		if hog_channel == 'ALL':
			hog_features = []
			for channel in range(feature_image.shape[2]):
				hog_features.extend(hog_feature(feature_image[:,:,channel],
									orient, pix_per_cell, cell_per_block,
									vis=False, feature_vec=True))
		else:
			hog_features = hog_feature(feature_image[:,:,hog_channel], orient,
						pix_per_cell, cell_per_block, vis=False, feature_vec=True)
		img_features.append(hog_features)

	return np.concatenate(img_features)

