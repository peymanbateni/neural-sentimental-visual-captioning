from collections import Counter, defaultdict
from IPython import display
from nltk import word_tokenize
from nltk.translate.bleu_score import sentence_bleu
from PIL import Image
from torch import nn
from torch.autograd import Variable
from torchvision import models, transforms

import json
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import random
import torch
import torch.nn.functional as F
import os

img_size = 256
loader = transforms.Compose([
  transforms.Resize(img_size),
  transforms.CenterCrop(img_size),
  transforms.ToTensor(),
])

def load_image(filename, volatile=False):
    """
    Simple function to load and preprocess the images.
    """
    image = Image.open(filename).convert('RGB')
    image_tensor = loader(image).float()
    image_var = Variable(image_tensor, volatile=volatile).unsqueeze(0)
    return image_var.cuda()

#print(load_image('data/vso/vso_images_with_cc/amazing_flowers/1066918516_e27cbf795e.jpg'))

vso_images_folder = "data/vso/vso_images_with_cc/"

train_anp_tags = []
train_image_names = []
train_image_to_anp_tag = {}
for subdir in os.listdir(vso_images_folder):
        if subdir.endswith("_train"):
                train_anp_tags.append(subdir.replace("_train", "").replace("_", " "))
                for filename in os.listdir(vso_images_folder + subdir):
                        if filename.endswith(".jpg"):
                                train_image_names.append(vso_images_folder + subdir + "/"  + filename)
                                train_image_to_anp_tag[vso_images_folder + subdir + "/"  + filename] = subdir.replace("_train", "").replace("_", " ")

validation_anp_tags = []
validation_image_names = []
validation_image_to_anp_tag = {}
for subdir in os.listdir(vso_images_folder):
        if subdir.endswith("_validation"):
                validation_anp_tags.append(subdir.replace("_validation", "").replace("_", " "))
                for filename in os.listdir(vso_images_folder + subdir):
                        if filename.endswith(".jpg"):
                                validation_image_names.append(vso_images_folder + subdir + "/"  + filename)
                                validation_image_to_anp_tag[vso_images_folder + subdir + "/"  + filename] = subdir.replace("_validation", "").replace("_", " ")

test_anp_tags = []
test_image_names = []
test_image_to_anp_tag = {}
for subdir in os.listdir(vso_images_folder):
        if subdir.endswith("_test"):
                test_anp_tags.append(subdir.replace("_test", "").replace("_", " "))
                for filename in os.listdir(vso_images_folder + subdir):
                        if filename.endswith(".jpg"):
                                test_image_names.append(vso_images_folder + subdir + "/"  + filename)
                                test_image_to_anp_tag[vso_images_folder + subdir + "/"  + filename] = subdir.replace("_test", "").replace("_", " ")

print(train_image_to_anp_tag)
print(validation_image_to_anp_tag)
print(test_image_to_anp_tag)
