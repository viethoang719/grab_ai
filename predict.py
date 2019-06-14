from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
from scipy.io import loadmat
import os
import numpy as np
import argparse
IMG_SIZE = 160
BATCH_SIZE = 64
IMAGE_SHAPE = (IMG_SIZE, IMG_SIZE, 3)

def preprocess_image(image):
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [IMG_SIZE, IMG_SIZE])
    image = (image/127.5) - 1
    return image
def load_and_preprocess_from_path(path):
    image = tf.io.read_file(path)
    return preprocess_image(image)

def prepare_data_set():
	cars_test_annos = loadmat(TEST_ANNOS_FILE)
	test_image_paths = cars_test_annos['annotations'][0]
	# Filter existing files only.
	print(test_image_paths[0][4][0])
	print(len(test_image_paths))
	test_image_paths = [TEST_FOLDER + image_path[4][0] for image_path in test_image_paths if  os.path.isfile(TEST_FOLDER + image_path[4][0])]
	test_image_label_ds = tf.data.Dataset.from_tensor_slices(test_image_paths[0:20])
	test_image_label_ds = test_image_label_ds.map(load_and_preprocess_from_path)
	test_image_label_ds = test_image_label_ds.batch(BATCH_SIZE)
	test_image_label_ds = test_image_label_ds.prefetch(buffer_size=BATCH_SIZE)
	return test_image_label_ds

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('--cars_test', help='path to cars test', default='.\\cars_test\\')
	parser.add_argument('--cars_test_annos', help='path to cars test annotations', default='.\\devkit\\cars_test_annos.mat')
	args = parser.parse_args()

	if args.cars_test:
		TEST_FOLDER = args.cars_test
	if args.cars_test_annos:
		TEST_ANNOS_FILE = args.cars_test_annos

	print(TEST_FOLDER)
	print(TEST_ANNOS_FILE)
	restore_model = tf.keras.models.load_model('grab_model.h5')
	predict_result = restore_model.predict(prepare_data_set())
	label_indexs = np.argmax(predict_result, axis=1)
	label_indexs = [label_index + 1 for label_index in label_indexs]
	with open('submission.txt', 'w') as f:
	    for item in label_indexs:
	        f.write("%s\n" % item)
