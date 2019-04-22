import os
from PIL import Image

vso_images_folder = "data/vso/vso_images_with_cc/"
min_image_dim = 224

total_image_count = 0
image_issue_count = 0
size_issue_count = 0
RGB_conversion_issue_count = 0
log_file_name = "image_removing_logs.txt"
log_file = open(log_file_name, "a+")

for subdir in os.listdir(vso_images_folder):
    for filename in os.listdir(vso_images_folder + subdir):
        if filename.endswith(".jpg"):
           total_image_count += 1
           try:
               img = Image.open(vso_images_folder + subdir + "/" + filename)
           except:
               image_issue_count += 1
               print("Image is corrupted at", subdir + "/" + filename)
               log_file.write("Image is corrupted at " + subdir + "/" + filename + "\n")
               os.remove(vso_images_folder + subdir + "/" + filename)
               print("Removed file", vso_images_folder + subdir + "/" + filename)
               log_file.write("Removed file " + vso_images_folder + subdir + "/" + filename + "\n")
               continue
           height, width = img.size
           if height < min_image_dim or width < min_image_dim:
               size_issue_count += 1
               print("Image is smaller than required dimensions at", subdir + "/" + filename)
               log_file.write("Image is smaller than required dimensions at " + subdir + "/" + filename + "\n")
               os.remove(vso_images_folder + subdir + "/" + filename)
               print("Removed file", vso_images_folder + subdir + "/" + filename)
               log_file.write("Removed file " + vso_images_folder + subdir + "/" + filename + "\n")
               continue
           try:
               img_rgb = img.convert('RGB')
           except:
               RGB_conversion_issue_count += 1
               print("RGB conversion issue at", subdir + "/" + filename)
               log_file.write("RGB conversion issue at " + subdir + "/" + filename + "\n")
               os.remove(vso_images_folder + subdir + "/" + filename)
               print("Removed file", vso_images_folder + subdir + "/" + filename)
               log_file.write("Removed file " + vso_images_folder + subdir + "/" + filename + "\n")
               continue

print("Total number of images:", total_image_count)
log_file.write("Total number of images: " + str(total_image_count))
print("Number of photos removed due to image problems:", image_issue_count)
log_file.write("Number of photos removed due to image problems: " + str(image_issue_count))
print("Number of photos removed due to dimension issues:", size_issue_count)
log_file.write("Number of photos removed due to dimension issues: " + str(size_issue_count))
print("Number of photos removed due to RGB conversion issues:", RGB_conversion_issue_count)
log_file.write("Number of photos removed due to RGB conversion issues: " + str(RGB_conversion_issue_count))
