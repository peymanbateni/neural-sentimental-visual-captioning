
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

from models.ANP_classifier import *

USE_CUDA = False # switch to true when training on GPU(s)

def train_pass(image_input, target_output, model, optimizer, criterion):
    """
    Given batch of images, completes one pass of training on the model,
    using the given optimizer and criterion.
    """

    if USE_CUDA:
        image_input = image_input.cuda()
        target_output = target_output.cuda()
        model = model.cuda()
        optimizer = optimizer.cuda()
        criterion = criterion.cuda()

    optimizer.zero_grad()
    model_output = model(image_input)
    loss = criterion(model_output, target_output)
    loss.backward()
    optimizer.step()

    return loss.data.cpu().numpy

# TODO: Add code for iterating through data and training the model
