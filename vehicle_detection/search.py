import numpy as np
import cv2
from scipy.ndimage.measurements import label
from feature import *
from svm import *


def slide_window(img, x_start_stop=[None, None], y_start_stop=[None, None],
					xy_window=(64, 64), xy_overlap=(0.5, 0.5)):
	if x_start_stop[0] == None:
		x_start_stop[0] = 0
	if x_start_stop[1] == None:
		x_start_stop[1] = img.shape[1]
	if y_start_stop[0] == None:
		y_start_stop[0] = 0
	if y_start_stop[1] == None:
		y_start_stop[1] = img.shape[0]

	xspan = x_start_stop[1] - x_start_stop[0]
	yspan = y_start_stop[1] - y_start_stop[0]
	
	nx_pix_per_step = np.int(xy_window[0]*(1 - xy_overlap[0]))
	ny_pix_per_step = np.int(xy_window[1]*(1 - xy_overlap[1]))
	
	nx_windows = np.int(xspan/nx_pix_per_step) - 1
	ny_windows = np.int(yspan/ny_pix_per_step) - 1
	
	window_list = []
	
	for ys in range(ny_windows):
		for xs in range(nx_windows):
			
			startx = xs*nx_pix_per_step + x_start_stop[0]
			endx = startx + xy_window[0]
			starty = ys*ny_pix_per_step + y_start_stop[0]
			endy = starty + xy_window[1]

			
			window_list.append(((startx, starty), (endx, endy)))
	
	return window_list

def search_windows(img, windows, clf, scaler, color_space='RGB',
					spatial_size=(32, 32), hist_bins=32,
					hist_range=(0, 256), orient=9,
					pix_per_cell=8, cell_per_block=2,
					hog_channel=0, spatial_feat=True,
					hist_feat=True, hog_feat=True):

	on_windows = []
	for window in windows:
		test_img = cv2.resize(img[window[0][1]:window[1][1], window[0][0]:window[1][0]], (64, 64))
		features = extract_features(test_img, color_space=color_space,
							spatial_size=spatial_size, hist_bins=hist_bins,
							orient=orient, pix_per_cell=pix_per_cell,
							cell_per_block=cell_per_block,
							hog_channel=hog_channel, spatial_feat=spatial_feat,
							hist_feat=hist_feat, hog_feat=hog_feat)

		test_features = scaler.transform(np.array(features).reshape(1, -1))
		prediction = clf.predict(test_features)
		if prediction == 1:
			on_windows.append(window)

	return on_windows

def draw_boxes(img, bboxes, color=(255, 0, 0), thick=5):
	imcopy = np.copy(img)
	
	for bbox in bboxes:
		cv2.rectangle(imcopy, bbox[0], bbox[1], color, thick)
	
	return imcopy

def add_heat(heatmap, bbox_list):
	for box in bbox_list:
		heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1

	return heatmap

def apply_threshold(heatmap, threshold):
	heatmap[heatmap <= threshold] = 0
	
	return heatmap

def draw_labeled_bboxes(img, labels):
	
	for car_number in range(1, labels[1]+1):
		
		nonzero = (labels[0] == car_number).nonzero()
		nonzeroy = np.array(nonzero[0])
		nonzerox = np.array(nonzero[1])
		
		bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
		bbox_w = (bbox[1][0] - bbox[0][0])
		bbox_h = (bbox[1][1] - bbox[0][1])

		aspect_ratio = bbox_w / bbox_h  
		
		bbox_area = bbox_w * bbox_h

		if bbox_area < small_bbox_area and bbox[0][1] > close_y_thresh:
			small_box_close = True
		else:
			small_box_close = False

		if aspect_ratio > min_ar and aspect_ratio < max_ar and not small_box_close and bbox_area > min_bbox_area:
			cv2.rectangle(img, bbox[0], bbox[1], (255, 0, 0), 5)

	return img
