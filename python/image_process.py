
import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt

def conv2d(img : np.ndarray, kernel : np.ndarray):
	img_y, img_x = img.shape
	kernel_size = kernel.shape[0]
	kernel_vector = kernel.flatten()
	pad = (int)(kernel_size/2)

	feature = np.zeros((img_y + pad*2, img_x + pad*2))
	feature[pad:pad+img_y, pad:pad+img_x] = img

	output = np.zeros((img.shape))

	for i in range(img_y):
		for j in range(img_x):
			feature_vector = feature[i:i + kernel_size, j:j + kernel_size].flatten()
			conv = np.convolve(feature_vector, kernel_vector, mode = 'valid')
			output[i, j] = conv
	return output

def img_load(name, mode='gray'):
	img_dir = os.path.join(name)
	if not os.path.exists(img_dir):
		print("not exist file in path : ", img_dir)
	try:
		if mode == 'gray':
			return np.asarray((Image.open(img_dir)).convert('L'))
		elif mode == 'cmyk':
			return np.asarray((Image.open(img_dir)).convert('CMYK'))
		elif mode == 'rgb':
			return np.asarray((Image.open(img_dir)).convert('RGB'))
		elif mode == 'hsv':
			return np.asarray((Image.open(img_dir)).convert('HSV'))
		else:
			return np.array((Image.open(img_dir)).convert(mode))
	except:
		print(' mode or filename check..')

def clamping(value):
	if value<0:
		return 0
	elif value>255:
		return 255
	else:
		return (int)(value)

def clamping_from_array(array : np.ndarray):
	flat = array.flatten()
	for i in range(flat.shape[0]):
		flat[i] = clamping(flat[i])
	return np.reshape(flat, array.shape)

def gamma_process(array:np.ndarray, gamma_val:float, end_value = 255, bins = 256, mode='gray'):
	_, plot = plt.subplots(2,2, figsize=[10,8])
	plot[1,0].imshow(array, mode)
	plot[0,1].hist(array.flatten(), range = [0,255], color= 'blue')

	a = np.linspace(0, bins-1, bins)
	plot[0,0].plot(a, color = 'blue')
	b = a

	for i in range(bins-1):
		if(a[i] < end_value):
			b[i] = pow(a[i]/bins, gamma_val) * bins
	plot[0,0].plot(b, color = 'red', alpha=0.9)

	y = array.flatten()/bins
	for i in range(y.shape[0]):
		if(y[i] < end_value):
			y[i] = pow(y[i], gamma_val)
	y *= bins
	plot[0,1].hist(y, range = [0,255], color='red', alpha=0.8)
	y = np.reshape(y, array.shape)
	plot[1,1].imshow(y, mode)

	plt.show()
	return y

def hist_equalizer(array:np.ndarray, bins=256, mode='gray'):
	_, plot = plt.subplots(2,2,figsize=[10,8])
	plot[0,0].hist(array.flatten(), range = [0,255], color = 'blue')
	plot[1,0].imshow(array, mode)

	flat = array.flatten()
	hist = np.histogram(flat, bins = bins-1, range=(0,bins-1))

	hist_sum = np.zeros(bins-1)
	sum_v = 0
	temp = 0
	for i in range(0, bins-1):
		sum_v += hist[0][i]
		hist_sum[i] = sum_v
	for i in range(flat.shape[0]):
		temp = flat[i].astype(int) - 1
		flat[i] = (hist_sum[temp] * (bins-1)) / flat.shape[0]
	plot[0,1].hist(flat, range = [0,255], color = 'red')
	y = np.reshape(flat, array.shape)
	plot[1,1].imshow(y, mode)
	plt.show()
	return y

def imshow(array:np.ndarray, mode=None, max_pixel = 255):
	_, plot = plt.subplots(2, figsize=[10,8])
	img = plot[0].imshow(array, mode, vmin = 0, vmax=max_pixel)
	plt.colorbar(img)
	plot[1].hist(array.flatten())

def hist_stretching(array:np.ndarray, low=0, high=255, func='stretching', mode='gray'):
	_, plot = plt.subplots(2,2,figsize=[10,8])
	flat = array.flatten()
	plot[0,0].hist(flat, range=[0,255], color = 'blue')
	plot[1,0].imshow(array, mode)
	if func == 'end_in':
		for i in range(flat.shape[0]):
			if flat[i] < low:
				flat[i] = 0
			elif flat[i] > high:
				flat[i] = 255
			else:
				flat[i] = (flat[i] - low)*high /(high - low)

	else:
		low = np.min(flat)
		high = np.max(flat)
		for i in range(flat.shape[0]):
			flat[i] = (flat[i] - low) * 255 / (high - low)

	plot[0,1].hist(flat, range = [0,255], color = 'red')
	y = np.reshape(flat, array.shape)
	plot[1,1].imshow(y, mode)
	return y

def onscale(array : np.ndarray):
	array.astype(int)
	vector = array.flatten()
	min_v = np.min(vector)
	max_v = np.max(vector) - min_v
	for i in range(vector.shape[0]):
		vector[i] = (vector[i] - min_v) * (255/max_v)
	arr = np.reshape(vector, array.shape)
	return arr.astype(int)

def merge_to_3d_array(ch0, ch1, ch2, scale = True):
	array = np.zeros((ch1.shape[0], ch1.shape[1], 3))
	if scale:
		ch0 = onscale(ch0)
		ch1 = onscale(ch1)
		ch2 = onscale(ch2)
	array[:,:,0] = ch0
	array[:,:,1] = ch1
	array[:,:,2] = ch2
	return array.astype(int)

def merge_to_1d_array(ch0, ch1, ch2):
	array = np.zeros_like(ch0)
	array = (ch0 + ch1 + ch2)/3
	return array

def conv3d(image : np.ndarray, kernel : np.ndarray, show = True):
	ch0 = conv2d(image[:,:,0], kernel)
	ch1 = conv2d(image[:,:,1], kernel)
	ch2 = conv2d(image[:,:,2], kernel)
	array = merge2_3d(ch0, ch1, ch2)
	if show:
		imshow(array)
	return array

def rgb2gray(image : np.ndarray):
	# 0.2989 * R + 0.5870 * G + 0.1140 * B 
	img = (image[:,:,0]*0.2989 + image[:,:,1]*0.5870
	 + image[:,:,2]*0.1140)
	return img

def posterizing(image : np.ndarray, bins, mode = 'binary_r'):
	img_x, img_y = image.shape
	y = np.zeros_like(image, dtype=np.float32)
	bins_range = (int)(256/bins)
	for i in range(img_x):
		for j in range(img_y):
			for k in range(bins):
				if bins_range*k <= image[i][j] and image[i][j] < bins_range*(k+1):
					y[i][j] = bins_range*k
				elif image[i][j] >= bins_range*(k+1):
					y[i][j] = bins_range*(k+1)
	imshow(y, mode)
	return y

def sobel_edge_detection(image : np.ndarray, show = True, single_channel = False, mode = 'binary_r'):
	"""
	sobel edge detection : 
		https://en.wikipedia.org/wiki/Sobel_operator
	"""
	if not single_channel:
		img = rgb2gray(image)
	else:
		img = image
	# gaussian
	p1 = gaussian(img, show=False)
	
	sobel_x = np.array([[3, 0, -3], 
						[10, 0, -10],
						[3, 0, -3]])
	sobel_y = np.array([[3, 10, 3],
						[0, 0,  0],
						[-3, -10, -3]])
	gx = conv2d(p1, sobel_x)
	gy = conv2d(p1, sobel_y)

	p2 = np.sqrt(gx*gx + gy*gy)
	p3 = onscale(p2)
	if show:
		imshow(p3, mode)
	return p3

def split_ch(image : np.ndarray):
	CH = image.shape[2]
	if CH is 3:
		return image[:,:,0], image[:,:,1], image[:,:,2]
	elif CH is 4:
		return image[:,:,0], image[:,:,1], image[:,:,2], image[:,:,3]
	else:
		return image

def binarization(image : np.ndarray, threshold, show = True, max_pixel = 255, mode = 'binary_r'):
	flat = image.flatten()
	for i in range(flat.shape[0]):
		if flat[i] <= threshold:
			flat[i] = 0
		else:
			flat[i] = max_pixel
	y = np.reshape(flat, image.shape)
	if show:
		imshow(y, mode)
	return y

def edge_detection(image : np.ndarray, mode = 'binary_r'):
	x = rgb2gray(image)
	#edge_mask = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
	edge_mask = np.array([[1, 0, -1], [0, 0, 0], [-1, 0, 1]])
	#edge_mask = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
	y = onscale(conv2d(x, edge_mask))
	imshow(y, mode)
	return y

def gaussian(image : np.ndarray, iterate = 1, show = True, mode = 'binary_r'):
	#mask = np.array([[1/16, 1/8, 1/16], [1/8, 1/4, 1/8], [1/16, 1/8, 1/16]])
	mask = np.array([[1, 4, 6, 4, 1], [4, 16, 24, 16, 4], [6, 24, 36, 24, 6],
		[4, 16, 24, 16, 4], [1, 4, 6, 4, 1]])/256
	y = image
	for i in range(iterate):
		y = conv2d(y, mask)
	if show:
		imshow(y, mode)
	return y

def blurring(image : np.ndarray, iterate = 1, show = True, kernel_size = 3, blur_value = 9, mode = 'binary_r'):
	kernel = np.ones((kernel_size, kernel_size))/blur_value
	y = image
	for i in range(iterate):
		y = conv2d(y, kernel)
	if show:
		imshow(y, mode)
	return y

def binary_range(image : np.ndarray, min_value, max_value, show = True, mode = 'binary_r'):
	flat = image.flatten()
	min_value = clamping(min_value)
	max_value = clamping(max_value)
	for i in range(flat.shape[0]):
		if mode is 'binary_r':
			if min_value <= flat[i] and flat[i] < max_value:
				flat[i] = 0
			else:
				flat[i] = 255
		else:
			if min_value <= flat[i] and flat[i] < max_value:
				flat[i] = 255
			else:
				flat[i] = 0

	y = np.reshape(flat, image.shape)
	if show:
		imshow(y, mode)
	return y

def add_mask(a, b, mode = 'binary_r'):
	flat_a = a.flatten()
	flat_b = b.flatten()
	if mode == 'binary_r':
		y = np.ones_like(flat_a) * 255
		for i in range(flat_a.shape[0]):
			if flat_a[i] == 0 or flat_b[i] == 0:
				y[i] = 0
	elif mode == 'binary':
		y = np.zeros_like(flat_a)
		for i in range(flat_a.shape[0]):
			if flat_a[i] == 255 or flat_b[i] == 255:
				y[i] = 255

	return np.reshape(y, a.shape)

def box_mean_threshold(image:np.ndarray, box_size: int, mean_range = 10, show = True):
	if not image.ndim is 2:
		image = rgb2gray(image)
	image_x, image_y = image.shape

	y = image.copy()

	box_x = (int)(image_x/box_size)
	box_y = (int)(image_y/box_size)
	#print(box_x, box_y)

	for i in range(box_x+1):
		for j in range(box_y+1):
			if j < box_y and i < box_x:
				box = image[(i*box_size):(i+1)*box_size, (j*box_size):(j+1)*box_size]
				box_mean = np.mean(box)
				box = binarization(box, box_mean-mean_range, show = False)
				y[(i*box_size):(i+1)*box_size, (j*box_size):(j+1)*box_size] = box
			else:
				if i > box_x and j < box_y:
					box = image[(i-1)*box_size:, (j*box_size):(j+1)*box_size]
					box_mean = np.mean(box)
					box = binarization(box, box_mean-mean_range, show = False)
					y[(i-1)*box_size:, (j*box_size):(j+1)*box_size] = box

				elif i < box_x and j > box_y:
					box = image[(i*box_size):(i+1)*box_size, (j-1)*box_size:]
					box_mean = np.mean(box)
					box = binarization(box, box_mean-mean_range, show = False)
					y[(i*box_size):(i+1)*box_size, (j-1)*box_size:] = box 

				else:
					box = image[(i-1)*box_size:, (j-1)*box_size:]
					box_mean = np.mean(box)
					box = binarization(box, box_mean-mean_range, show = False)
					y[(i-1)*box_size:, (j-1)*box_size:] = box
	if show:
		imshow(y, 'gray')
	return y

def box_gaussian_threshold(image:np.ndarray, box_size:int, mean_range = 10, show=True):
	if not image.ndim is 2:
		image = rgb2gray(image)
	image_x, image_y = image.shape
	y = image.copy()

	box_x = (int)(image_x/box_size)
	box_y = (int)(image_y/box_size)

	for i in range(box_x+1):
		for j in range(box_y+1):
			if j < box_y and i < box_x:
				box = image[(i*box_size):(i+1)*box_size, (j*box_size):(j+1)*box_size]
				box_mean = np.mean(gaussian(box, show = False))
				box = binarization(box, box_mean-mean_range, show = False)
				y[(i*box_size):(i+1)*box_size, (j*box_size):(j+1)*box_size] = box
			else:
				if i > box_x and j < box_y:
					box = image[(i-1)*box_size:, (j*box_size):(j+1)*box_size]
					box_mean = np.mean(gaussian(box, show = False))
					box = binarization(box, box_mean-mean_range, show = False)
					y[(i-1)*box_size:, (j*box_size):(j+1)*box_size] = box

				elif i < box_x and j > box_y:
					box = image[(i*box_size):(i+1)*box_size, (j-1)*box_size:]
					box_mean = np.mean(gaussian(box, show = False))
					box = binarization(box, box_mean-mean_range, show = False)
					y[(i*box_size):(i+1)*box_size, (j-1)*box_size:] = box 
					
				else:
					box = image[(i-1)*box_size:, (j-1)*box_size:]
					box_mean = np.mean(gaussian(box, show = False))
					box = binarization(box, box_mean-mean_range, show = False)
					y[(i-1)*box_size:, (j-1)*box_size:] = box
	if show:
		imshow(y, 'gray')
	return y

def box_BHT_threshold(image:np.ndarray, box_size:int, mean_range = 10, show=True):
	if not image.ndim is 2:
		image = rgb2gray(image)
	image_x, image_y = image.shape
	y = image.copy()

	box_x = (int)(image_x/box_size)
	box_y = (int)(image_y/box_size)

	for i in range(box_x+1):
		for j in range(box_y+1):
			if j < box_y and i < box_x:
				box = image[(i*box_size):(i+1)*box_size, (j*box_size):(j+1)*box_size]
				threshold_value = BHT_threshold(get_hist(box))
				box = binarization(box, threshold_value-mean_range, show = False)
				y[(i*box_size):(i+1)*box_size, (j*box_size):(j+1)*box_size] = box
			else:
				if i > box_x and j < box_y:
					box = image[(i-1)*box_size:, (j*box_size):(j+1)*box_size]
					threshold_value = BHT_threshold(get_hist(box))
					box = binarization(box, threshold_value-mean_range, show = False)
					y[(i-1)*box_size:, (j*box_size):(j+1)*box_size] = box

				elif i < box_x and j > box_y:
					box = image[(i*box_size):(i+1)*box_size, (j-1)*box_size:]
					threshold_value = BHT_threshold(get_hist(box))
					box = binarization(box, threshold_value-mean_range, show = False)
					y[(i*box_size):(i+1)*box_size, (j-1)*box_size:] = box 
					
				else:
					box = image[(i-1)*box_size:, (j-1)*box_size:]
					threshold_value = BHT_threshold(get_hist(box))
					box = binarization(box, threshold_value-mean_range, show = False)
					y[(i-1)*box_size:, (j-1)*box_size:] = box
	if show:
		imshow(y, 'gray')
	return y

def get_hist(image:np.ndarray, bins = 256, show = False):
	flat = image.flatten()
	if show:
		imshow(img, 'gray', bins)
	return np.histogram(image, bins = bins, range = [0, bins])[0]

def _mean(x, y):
	return (int)((x+y)/2)

def BHT_threshold(hist, bins = 256, print_mean = False):
	"""
	 Balnced histogram thresholding :
	 	A. Anjos and H. Shahbazkia. Bi-Level Image Thresholding - A Fast Method.
	"""
	start, end = 0, bins-1
	mean = _mean(start, end)
	left_weight = np.sum(hist[start:mean])
	right_weight = np.sum(hist[mean+1:])

	while(start <= end):
		if(right_weight > left_weight):
			right_weight -= hist[end]
			end -= 1
			if _mean(start, end) < mean:
				left_weight -= hist[mean-1]
				right_weight += hist[mean-1]
				if mean > 1:
					mean -= 1
		else:
			left_weight -= hist[start]
			start += 1
			if _mean(start, end) >= mean:
				left_weight += hist[mean+1]
				right_weight -= hist[mean+1]
				if mean < 254:
					mean += 1
	if print_mean:
		print(" threshold value : ", mean)
	return mean

