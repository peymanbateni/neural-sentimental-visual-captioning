import os

main_folder_name = "vso_images_with_cc/"

for subdir in os.listdir(main_folder_name):

<<<<<<< HEAD
	for filename in os.listdir(main_folder_name + subdir):

		if filename.endswith("_train"):
			os.rmdir(main_folder_name + subdir + "/" + filename)
			print("Removed " + main_folder_name + subdir + "/" + filename)

		elif filename.endswith("_validation"):
                        os.rmdir(main_folder_name + subdir + "/" + filename)
                        print("Removed " + main_folder_name + subdir + "/" + filename)

		elif filename.endswith("_test"):
			os.rmdir(main_folder_name + subdir + "/" + filename)
			print("Removed " + main_folder_name + subdir + "/" + filename)
=======
        for filename in os.listdir(main_folder_name + subdir):

                if filename.endswith("_train"):
                        os.rmdir(main_folder_name + subdir + "/" + filename)
                        print("Removed " + main_folder_name + subdir + "/" + filename)

                elif filename.endswith("_validation"):
                        os.rmdir(main_folder_name + subdir + "/" + filename)
                        print("Removed " + main_folder_name + subdir + "/" + filename)

                elif filename.endswith("_test"):
                        os.rmdir(main_folder_name + subdir + "/" + filename)
                        print("Removed " + main_folder_name + subdir + "/" + filename)
>>>>>>> df1b56ba04016d335d3a6f3fe7e536d9c29e55a9
