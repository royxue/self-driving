import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from features import *

def train(cars, notcars, svc, X_scaler):
	car_features = []
	for car in cars:
		car_features.append(np.concatenate(extract_features(cars, color_space=color_space,
								spatial_size=spatial_size, hist_bins=hist_bins,
								orient=orient, pix_per_cell=pix_per_cell,
								cell_per_block=cell_per_block,
								hog_channel=hog_channel, bin_f=bin_f,
								color_f=color_f, hog_f=hog_f)))
	
	notcar_features = []
	for car in cars:
		car_features.append(np.concatenate(extract_features(notcars, color_space=color_space,
								spatial_size=spatial_size, hist_bins=hist_bins,
								orient=orient, pix_per_cell=pix_per_cell,
								cell_per_block=cell_per_block,
								hog_channel=hog_channel, bin_f=bin_f,
								color_f=color_f, hog_f=hog_f)))

	X = np.vstack((car_features, notcar_features, notcar_features)).astype(np.float64)
	X_scaler.fit(X)
	
	scaled_X = X_scaler.transform(X)
	
	y = np.hstack((np.ones(len(car_features)), np.zeros(3*len(notcar_features))))
	
	rand_state = np.random.randint(0, 100)
	X_train, X_test, y_train, y_test = train_test_split(
		scaled_X, y, test_size=0.05, random_state=rand_state)
	
	t=time.time()
	svc.fit(X_train, y_train)
