import os
import numpy as np
import random

image_dataset_folder = "vso_images_with_cc/"
train_ratio = 0.7
validation_ratio = 0.15
test_ratio = 1 - (train_ratio + validation_ratio)

for subdir in os.listdir(image_dataset_folder):
	if (not subdir.endswith("_train")) and (not subdir.endswith("_validation")) and (not subdir.endswith("_test")): 

		subdir_path = image_dataset_folder + subdir + "/"
		print("Partioning " + subdir_path)
		number_of_images = int(len([name for name in os.listdir(subdir_path)]) / 2)

		if number_of_images > 0:

			train_dir_path = subdir_path[:-1] + "_train/"
			if not os.path.exists(train_dir_path):
				os.makedirs(train_dir_path)

			validation_dir_path = subdir_path[:-1] + "_validation/"
			if not os.path.exists(validation_dir_path):
				os.makedirs(validation_dir_path)

			test_dir_path = subdir_path[:-1] + "_test/"
			if not os.path.exists(test_dir_path):
				os.makedirs(test_dir_path)

			number_of_train_images = int(number_of_images * train_ratio)
			train_set_nums = random.sample(range(number_of_images), int(number_of_train_images))
			non_train_set_nums = np.arange(number_of_images).tolist()

			for item in train_set_nums:
				non_train_set_nums.remove(item)

			number_of_validation_images = int(len(non_train_set_nums) / 2)
			validation_set_nums = non_train_set_nums[:number_of_validation_images]
			test_set_nums = non_train_set_nums[number_of_validation_images:]

			count = 0
			for filename in os.listdir(subdir_path):
				if filename.endswith(".jpg"):
					filename = str(filename)
					if count in train_set_nums:
						os.rename(subdir_path + filename, subdir_path[:-1] + "_train/" + filename)
						print("Moved " + subdir_path + filename + " to " + subdir_path[:-1] + "_train/" + filename)
						os.rename(subdir_path + filename.replace(".jpg", ".json.txt"), subdir_path[:-1] + "_train/" + filename.replace(".jpg", ".json.txt"))
						print("Moved " + subdir_path + filename.replace(".jpg", ".json.txt") + " to " + subdir_path[:-1] + "_train/" + filename.replace(".jpg", ".json.txt"))
					elif count in validation_set_nums:
						os.rename(subdir_path + filename, subdir_path[:-1] + "_validation/" + filename)
						print("Moved " + subdir_path + filename + " to " + subdir_path[:-1] + "_validation/" + filename)
						os.rename(subdir_path + filename.replace(".jpg", ".json.txt"), subdir_path[:-1] + "_validation/" + filename.replace(".jpg", ".json.txt"))
						print("Moved " + subdir_path + filename.replace(".jpg", ".json.txt") + " to " + subdir_path[:-1] + "_validation/" + filename.replace(".jpg", ".json.txt"))
					elif count in test_set_nums:
						os.rename(subdir_path + filename, subdir_path[:-1] + "_test/" + filename)
						print("Moved " + subdir_path + filename + " to " + subdir_path[:-1] + "_test/" + filename)
						os.rename(subdir_path + filename.replace(".jpg", ".json.txt"), subdir_path[:-1] + "_test/" + filename.replace(".jpg", ".json.txt"))
						print("Moved " + subdir_path + filename.replace(".jpg", ".json.txt") + " to " + subdir_path[:-1] + "_test/" + filename.replace(".jpg", ".json.txt"))
					else:
						print("Invalid instance encountered at " + subdir_path + filename)
					count += 1

