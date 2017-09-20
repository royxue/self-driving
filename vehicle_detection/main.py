import numpy as np
import cv2
import pickle

from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from moviepy.editor import VideoFileClip
from Windows import Windows
from svm import train
from search import *

color_space = 'RGB' 
orient = 30  
pix_per_cell = 16 
cell_per_block = 2 
hog_channel = 'ALL' 
spatial_size = (16, 16) 
hist_bins = 16	
bin_f= False 
color_f = False 
hog_f = True 
y_start_stop = [400, 720] 
x_start_stop = [0, 1280] 
pct_overlap = 0.7 
heatmap_thresh = 2
num_frames = 30 
min_ar, max_ar = 0.7, 3.0 
small_bbox_area, close_y_thresh = 80*80, 500
min_bbox_area = 40*40
use_pretrained = True 

hot_windows = Windows(num_frames)
svc = LinearSVC()
X_scaler = StandardScaler()


def annotate_image(image):
	"""
	Annotate the input image with detection boxes
	Returns annotated image
	"""
	global hot_windows, svc, X_scaler

	draw_image = np.copy(image)

	windows = slide_window(image, x_start_stop=(100, 1180), y_start_stop=(400, 500),
						xy_window=(96, 96), xy_overlap=(pct_overlap, pct_overlap))

	new_hot_windows = search_windows(image, windows, svc, X_scaler, color_space=color_space,
							spatial_size=spatial_size, hist_bins=hist_bins,
							orient=orient, pix_per_cell=pix_per_cell,
							cell_per_block=cell_per_block,
							hog_channel=hog_channel, bin_f=bin_f,
							color_f=color_f, hog_f=hog_f)
	
	hot_windows.add_windows(new_hot_windows)
	all_hot_windows = hot_windows.get_windows()

	heatmap = np.zeros((720, 1280))  
	heatmap = add_heat(heatmap, all_hot_windows)
	heatmap = apply_threshold(heatmap, heatmap_thresh)
	labels = label(heatmap)
	
	draw_img = draw_labeled_bboxes(np.copy(image), labels)

	return draw_img


def annotate_video(input_file, output_file):
	""" Given input_file video, save annotated video to output_file """
	global hot_windows, svc, X_scaler

	with open('model.p', 'rb') as f:
		save_dict = pickle.load(f)
	svc = save_dict['svc']
	X_scaler = save_dict['X_scaler']

	print('Loaded pre-trained model from model.p')

	video = VideoFileClip(input_file)
	annotated_video = video.fl_image(annotate_image)
	annotated_video.write_videofile(output_file, audio=False)


if __name__ == '__main__':
	annotate_video('project_video.mp4', 'out.mp4')

