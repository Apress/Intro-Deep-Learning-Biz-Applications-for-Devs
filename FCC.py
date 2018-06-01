import glob
import os
from PIL import Image
import numpy as np
from keras.layers import Input, Convolution2D, MaxPooling2D, UpSampling2D,
Dropout
from keras.models import Model
from keras import backend as K
from keras.callbacks import ModelCheckpoint
smooth = 1.

# define a weighted binary cross entropy function
def binary_crossentropy_2d_w(alpha):
	def loss(y_true, y_pred):
		bce = K.binary_crossentropy(y_pred, y_true)
		bce *= 1 + alpha * y_true
		bce /= alpha
		return K.mean(K.batch_flatten(bce), axis=-1)
	return loss

# define dice score to assess predictions
def dice_coef(y_true, y_pred):
	y_true_f = K.flatten(y_true)
	y_pred_f = K.flatten(y_pred)
	intersection = K.sum(y_true_f * y_pred_f)
	return (2. * intersection + smooth) / (K.sum(y_true_f) +
		K.sum(y_pred_f) + smooth)

def dice_coef_loss(y_true, y_pred):
	return 1 - dice_coef(y_true, y_pred)

def load_data(dir, boundary=False):
	X = []
	y = []
	# load images
	for f in sorted(glob.glob(dir + '/image??.png')):
		img = np.array(Image.open(f).convert('RGB'))
		X.append(img)
	# load masks
	for i, f in enumerate(sorted(glob.glob(dir + '/image??_mask.txt'))):
		if boundary:
			a = get_boundary_mask(f)
			y.append(np.expand_dims(a, axis=0))
		else:
			content = open(f).read().split('\n')[1:-1]
			a = np.array(content, 'i').reshape(X[i].shape[:2])
			a = np.clip(a, 0, 1).astype('uint8')
			y.append(np.expand_dims(a, axis=0))
	# stack data
	X = np.array(X) / 255.
	y = np.array(y)
	X = np.transpose(X, (0, 3, 1, 2))
	return X, y

# define the network model
def net_2_outputs(input_shape):
	input_img = Input(input_shape, name='input')
	x = Convolution2D(8, 3, 3, activation='relu',
		border_mode='same')(input_img)
	x = Convolution2D(8, 3, 3, activation='relu', border_mode='same')(x)
	x = Convolution2D(8, 3, 3, subsample=(1, 1), activation='relu',
		border_mode='same')(x)
	x = MaxPooling2D((2, 2), border_mode='same')(x)
	x = Convolution2D(16, 3, 3, activation='relu', border_mode='same')(x)
	x = Convolution2D(16, 3, 3, activation='relu', border_mode='same')(x)
	x = Convolution2D(16, 3, 3, subsample=(1, 1), activation='relu',
		border_mode='same')(x)
	x = MaxPooling2D((2, 2), border_mode='same')(x)
	x = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(x)
	x = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(x)
	x = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(x)
	# up
	x = UpSampling2D((2, 2))(x)
	x = Convolution2D(16, 3, 3, activation='relu', border_mode='same')(x)
	x = UpSampling2D((2, 2))(x)
	x = Convolution2D(8, 3, 3, activation='relu', border_mode='same')(x)
	output = Convolution2D(1, 3, 3, activation='sigmoid',
		border_mode='same', name='output')(x)
	model = Model(input_img, output=[output])
	model.compile(optimizer='adam', loss={'output':
		binary_crossentropy_2d_w(5)})
	return model

def train():
	X, y = load_data(DATA_DIR_TRAIN.replace('c_type', c_type),
	boundary=False) # load the data
	print(X.shape, y.shape) # make sure its the right shape
	h = X.shape[2]
	w = X.shape[3]
	training_data = ShuffleBatchGenerator(input_data={'input': X},
	output_data={'output': y, 'output_b': y_b}) # generate batches for
	training and testing
	training_data_aug = DataAugmentation(training_data,
	inplace_transfo=['mirror', 'transpose']) # apply some data
	augmentation
	net = net_2_outputs((X.shape[1], h, w))
	net.summary()
	model = net
	model.fit(training_data_aug, 300, 1, callbacks=[ProgressBarCallback()])
	net.save('model.hdf5' )

	# save predictions to disk
	res = model.predict(training_data, training_data.nb_elements)
	if not os.path.isdir('res'):
		os.makedirs('res')
	for i, img in enumerate(res[0]):
	Image.fromarray(np.squeeze(img) *
		255).convert('RGB').save('res/%s_res_%02d.jpg' % (c_type, i +1))
	for i, img in enumerate(res[1]):
		Image.fromarray(np.squeeze(img) *
			255).convert('RGB').save('res/%s_res_b%02d.jpg' % (c_type, i + 1))


if __name__ == '__main__':
	train()