import os
import cv2
import imutils
import numpy as np
from tqdm import tqdm
import pandas as pd
def crop_img(img):
	"""
	Finds the extreme points on the image and crops the rectangular out of them
	"""
	gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
	gray = cv2.GaussianBlur(gray, (3, 3), 0)

	# threshold the image, then perform a series of erosions +
	# dilations to remove any small regions of noise
	thresh = cv2.threshold(gray, 45, 255, cv2.THRESH_BINARY)[1]
	thresh = cv2.erode(thresh, None, iterations=2)
	thresh = cv2.dilate(thresh, None, iterations=2)

	# find contours in thresholded image, then grab the largest one
	cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	cnts = imutils.grab_contours(cnts)
	c = max(cnts, key=cv2.contourArea)

	# find the extreme points
	extLeft = tuple(c[c[:, :, 0].argmin()][0])
	extRight = tuple(c[c[:, :, 0].argmax()][0])
	extTop = tuple(c[c[:, :, 1].argmin()][0])
	extBot = tuple(c[c[:, :, 1].argmax()][0])
	ADD_PIXELS = 0
	new_img = img[extTop[1]-ADD_PIXELS:extBot[1]+ADD_PIXELS, extLeft[0]-ADD_PIXELS:extRight[0]+ADD_PIXELS].copy()
	
	return new_img
	
if __name__ == "__main__":
	training = "full/Training"
	testing = "full/Testing"
	training_dir = os.listdir(training)
	testing_dir = os.listdir(testing)
	IMG_SIZE = 128

	for dir in training_dir:
		save_path = 'cleaned/trainingfull/'+ dir
		path = os.path.join(training,dir)
		image_dir = os.listdir(path)
		for img in image_dir:
			image = cv2.imread(os.path.join(path,img))
			new_img = crop_img(image)
			new_img = cv2.resize(new_img,(IMG_SIZE,IMG_SIZE))
			if not os.path.exists(save_path):
				os.makedirs(save_path)
			cv2.imwrite(save_path+'/'+img, new_img)
	
	for dir in testing_dir:
		save_path = 'cleaned/testingfull/'+ dir
		path = os.path.join(testing,dir)
		image_dir = os.listdir(path)
		for img in image_dir:
			image = cv2.imread(os.path.join(path,img))
			new_img = crop_img(image)
			new_img = cv2.resize(new_img,(IMG_SIZE,IMG_SIZE))
			if not os.path.exists(save_path):
				os.makedirs(save_path)
			cv2.imwrite(save_path+'/'+img, new_img)

	# training = "data/trainingfull"
	# training_dir = os.listdir(training)
	# IMG_SIZE = 128

	# for dir in training_dir:
	# 	save_path = 'cleaned/trainingfull/'+ dir
	# 	path = os.path.join(training,dir)
	# 	image_dir = os.listdir(path)
	# 	for img in image_dir:
	# 		image = cv2.imread(os.path.join(path,img))
	# 		new_img = crop_img(image)
	# 		new_img = cv2.resize(new_img,(IMG_SIZE,IMG_SIZE))
	# 		if not os.path.exists(save_path):
	# 			os.makedirs(save_path)
	# 		cv2.imwrite(save_path+'/'+img, new_img)





	root_dir = 'cleaned/trainingfull/'

	# Label to numerical mapping
	label_map = {
		'glioma': 0,
		'meningioma': 1,
		'notumor': 2,
		'pituitary': 3
	}

	labels = []
	data = []

	for label_name in os.listdir(root_dir):
		folder_path = os.path.join(root_dir, label_name)
		if os.path.isdir(folder_path) and label_name in label_map:
			for image_name in os.listdir(folder_path):
				if image_name.endswith('.jpg'):
					image_path = os.path.join(folder_path, image_name)
					image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
					# Flatten the image into a 1D array
					flattened = image.flatten()
			
					# Get the numerical label from our map
					label = label_map[label_name]

					# Add numerical label as the first element and append to data
					labels.append(label)
					data.append(list(flattened))
	# columns = ['label'] + [f'pixel_{i}' for i in range(len(data[0]) - 1)]

	# Convert data to a pandas DataFrame
	df = pd.DataFrame(data)
	df.to_csv('trainingfull.csv', index=False, header=False)
	df = pd.DataFrame(labels)
	df.to_csv('trainingfull_labels.csv', index=False, header=False)


























	# root_dir = 'cleaned/trainingfull/'

	# # Label to numerical mapping
	# label_map = {
	# 	'glioma_100': 0,
	# 	'meningioma_100': 1,
	# 	'notumor_100': 2,
	# 	'pituitary_100': 3
	# }

	# labels = []
	# data = []

	# for label_name in os.listdir(root_dir):
	# 	folder_path = os.path.join(root_dir, label_name)
	# 	if os.path.isdir(folder_path) and label_name in label_map:
	# 		for image_name in os.listdir(folder_path):
	# 			if image_name.endswith('.jpg'):
	# 				image_path = os.path.join(folder_path, image_name)
	# 				image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
	# 				# Flatten the image into a 1D array
	# 				flattened = image.flatten()
			
	# 				# Get the numerical label from our map
	# 				label = label_map[label_name]

	# 				# Add numerical label as the first element and append to data
	# 				labels.append(label)
	# 				data.append(list(flattened))
	# # columns = ['label'] + [f'pixel_{i}' for i in range(len(data[0]) - 1)]

	# # Convert data to a pandas DataFrame
	# df = pd.DataFrame(data)
	# df.to_csv('training100.csv', index=False, header=False)
	# df = pd.DataFrame(labels)
	# df.to_csv('training100_labels.csv', index=False, header=False)










	root_dir = 'cleaned/testingfull/'

	# Label to numerical mapping
	label_map = {
		'glioma': 0,
		'meningioma': 1,
		'notumor': 2,
		'pituitary': 3
	}

	labels = []
	data = []

	for label_name in os.listdir(root_dir):
		folder_path = os.path.join(root_dir, label_name)
		if os.path.isdir(folder_path) and label_name in label_map:
			for image_name in os.listdir(folder_path):
				if image_name.endswith('.jpg'):
					image_path = os.path.join(folder_path, image_name)
					image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
					# Flatten the image into a 1D array
					flattened = image.flatten()
			
					# Get the numerical label from our map
					label = label_map[label_name]

					# Add numerical label as the first element and append to data
					labels.append(label)
					data.append(list(flattened))
	# columns = ['label'] + [f'pixel_{i}' for i in range(len(data[0]) - 1)]

	# Convert data to a pandas DataFrame
	df = pd.DataFrame(data)
	df.to_csv('testingfull.csv', index=False, header=False)
	df = pd.DataFrame(labels)
	df.to_csv('testingfull_labels.csv', index=False, header=False)
