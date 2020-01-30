import os
import random

import matplotlib.pyplot as plt
import numpy as np
from skimage import io
from sklearn import linear_model
from sklearn import metrics
from ttictoc import TicToc

LARGE_DATASET_NAME = 'notMNIST_large/'
SMALL_DATASET_NAME = 'notMNIST_small/'
CLASSES = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']

def plot_images(u):
	fig, axises = plt.subplots(1, 10, figsize=(5, 5))
	axises = axises.ravel()

	for i in range(len(u)):
		axis = axises[i]
		axis.imshow(u[i], cmap='gray')
		axis.axis('off')

def train(total_sample_size, large_dataset_images, small_dataset_images):
	x_train = []
	y_train = []
	sample_size = total_sample_size / len(CLASSES)
	large_dataset_indices = {}
	for letter in CLASSES:
		large_dataset_indices[letter] = set()
		while len(large_dataset_indices[letter]) < sample_size:
			large_dataset_indices[letter].add(random.randint(0, len(large_dataset_images[letter]) - 1))
	for i in range(len(CLASSES)):
		letter = CLASSES[i]
		indices = large_dataset_indices[letter]
		for index in indices:
			image = large_dataset_images[letter][index]
			x_train.append(image)
			# y_train.append([1 if x == i else 0 for x in range(len(CLASSES))])
			y_train.append(i)
	x_train = np.array(x_train)
	y_train = np.array(y_train)
	print(x_train.shape)
	print(y_train.shape)
	x_test = []
	y_test = []
	for i in range(len(CLASSES)):
		letter = CLASSES[i]
		for image in small_dataset_images[letter]:
			x_test.append(image)
			# y_test.append([1 if x == i else 0 for x in range(len(CLASSES))])
			y_test.append(i)
	x_test = np.array(x_test)
	y_test = np.array(y_test)
	print(x_test.shape)
	print(y_test.shape)

	t = TicToc('learning')
	t.tic()
	# 50 iter elapsed: 11.2min finished # Accuracy ::  0.8896808860620682
	# 200 iter elapsed: 38.6min finished # Accuracy ::  0.8888035968856234
	mul_lr = linear_model.LogisticRegression(n_jobs=4, verbose=10, max_iter=200, solver='saga').fit(x_train, y_train)
	# mul_lr = linear_model.LogisticRegression(n_jobs=4, verbose=10, max_iter=100, solver='newton-cg').fit(x_train, y_train)
	accuracy = metrics.accuracy_score(y_test, mul_lr.predict(x_test))
	print("Multinomial Logistic regression Test Accuracy : ", accuracy)
	t.toc()
	print(t.elapsed)
	return accuracy

if __name__ == "__main__":

	large_dataset_filenames = {}
	for letter in CLASSES:
		for r, d, f in os.walk(LARGE_DATASET_NAME + letter):
			large_dataset_filenames[letter] = f
	print('Large dataset images filenames has been read')
	small_dataset_filenames = {}
	for letter in CLASSES:
		for r, d, f in os.walk(SMALL_DATASET_NAME + letter):
			small_dataset_filenames[letter] = f
	print('Small dataset images filenames has been read')

	samples = []
	for letter in CLASSES:
		index = random.randint(0, len(large_dataset_filenames[letter]) - 1)
		letter_image = io.imread(LARGE_DATASET_NAME + letter + "/" + large_dataset_filenames[letter][index], as_gray=True)
		samples.append(letter_image)
	plot_images(samples)
	plt.show()

	t = TicToc('reading datasets')
	t.tic()
	large_dataset_images = {}
	corrupted_filenames = {}
	corrupted_files_count = 0
	for letter in CLASSES:
		large_dataset_images[letter] = []
		corrupted_filenames = []
		for filename in large_dataset_filenames[letter]:
			try:
				large_dataset_images[letter].append(io.imread(LARGE_DATASET_NAME + letter + "/" + filename, as_gray=True).ravel())
			except:
				print(letter)
				print(filename)
				corrupted_filenames.append(filename)
				corrupted_files_count += 1
		for corrupted_filename in corrupted_filenames:
			large_dataset_filenames[letter].remove(corrupted_filename)
	print('Large dataset images has been read. Corrupted files count: %s' % corrupted_files_count)
	small_dataset_images = {}
	corrupted_files_count = 0
	for letter in CLASSES:
		small_dataset_images[letter] = []
		corrupted_filenames = []
		for filename in small_dataset_filenames[letter]:
			try:
				small_dataset_images[letter].append(io.imread(SMALL_DATASET_NAME + letter + "/" + filename, as_gray=True).ravel())
			except:
				print(letter)
				print(filename)
				corrupted_filenames.append(filename)
				corrupted_files_count += 1
		for corrupted_filename in corrupted_filenames:
			small_dataset_filenames[letter].remove(corrupted_filename)
	print('Small dataset images has been read. Corrupted files count: %s' % corrupted_files_count)
	t.toc()
	print(t.elapsed)

	# t = TicToc('detecting duplicates')
	# t.tic()
	# total_duplicates_count = 0
	# for letter in CLASSES:
	# 	print(letter)
	# 	duplicates_count = 0
	# 	duplicate_indices = set()
	# 	for i in range(len(small_dataset_images[letter])):
	# 		small_dataset_image = small_dataset_images[letter][i]
	# 		for j in range(len(large_dataset_images[letter])):
	# 			large_dataset_image = large_dataset_images[letter][j]
	# 			if np.array_equal(small_dataset_image, large_dataset_image):
	# 				duplicate_indices.add(j)
	# 				duplicates_count += 1
	# 	total_duplicates_count += duplicates_count
	# 	print('Duplicated count %s for letter %s' % (duplicates_count, letter))
	# 	for duplicate_index in duplicate_indices:
	# 		duplicate_filename = large_dataset_filenames[letter][duplicate_index]
	# 		os.remove(LARGE_DATASET_NAME + letter + "/" + duplicate_filename)
	# print('Total duplicates count %s' % total_duplicates_count)
	# t.toc()
	# print(t.elapsed)

	# t = TicToc('detecting duplicates in the same dataset')
	# t.tic()
	# for letter in CLASSES:
	# 	print(letter)
	# 	duplicate_indices = {}
	# 	for i in range(len(small_dataset_images[letter])):
	# 		small_dataset_image_i = small_dataset_images[letter][i]
	# 		for j in range(len(small_dataset_images[letter])):
	# 			if i == j:
	# 				continue
	#
	# 			small_dataset_image_j = small_dataset_images[letter][j]
	# 			if np.array_equal(small_dataset_image_i, small_dataset_image_j):
	# 				if i not in duplicate_indices and j not in duplicate_indices:
	# 					duplicate_indices[i] = set()
	# 				if i in duplicate_indices:
	# 					duplicate_indices[i].add(j)
	# 				elif j in duplicate_indices:
	# 					duplicate_indices[j].add(i)
	# 	for key, duplicate_indices_per_image in duplicate_indices.items():
	# 		for duplicate_index_per_image in duplicate_indices_per_image:
	# 			duplicate_filename = small_dataset_filenames[letter][duplicate_index_per_image]
	# 			try:
	# 				os.remove(SMALL_DATASET_NAME + letter + "/" + duplicate_filename)
	# 			except OSError:
	# 				pass
	# t.toc()
	# print(t.elapsed)

	# too slow
	# t = TicToc('detecting duplicates in the same dataset')
	# t.tic()
	# for letter in CLASSES:
	# 	print(letter)
	# 	duplicate_indices = {}
	# 	for i in range(len(large_dataset_images[letter])):
	# 		large_dataset_image_i = large_dataset_images[letter][i]
	# 		for j in range(len(large_dataset_images[letter])):
	# 			if i == j:
	# 				continue
	#
	# 			large_dataset_image_j = large_dataset_images[letter][j]
	# 			if np.array_equal(large_dataset_image_i, large_dataset_image_j):
	# 				if i not in duplicate_indices and j not in duplicate_indices:
	# 					duplicate_indices[i] = set()
	# 				if i in duplicate_indices:
	# 					duplicate_indices[i].add(j)
	# 				elif j in duplicate_indices:
	# 					duplicate_indices[j].add(i)
	# 	for key, duplicate_indices_per_image in duplicate_indices.items():
	# 		for duplicate_index_per_image in duplicate_indices_per_image:
	# 			duplicate_filename = large_dataset_filenames[letter][duplicate_index_per_image]
	# 			try:
	# 				os.remove(LARGE_DATASET_NAME + letter + "/" + duplicate_filename)
	# 			except OSError:
	# 				pass
	# t.toc()
	# print(t.elapsed)

	sample_sizes = [50, 100, 1000, 50000]
	accuracies = []
	for total_sample_size in sample_sizes:
		accuracies.append(train(total_sample_size, large_dataset_images, small_dataset_images))

	for i in range(len(sample_sizes)):
		plt.plot(sample_sizes[i], accuracies[i], 'bo')
		plt.text(sample_sizes[i] * (1 + 0.01), accuracies[i] * (1 + 0.01), sample_sizes[i], fontsize = 10)
	plt.show()
