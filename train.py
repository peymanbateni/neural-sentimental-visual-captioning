
from collections import Counter, defaultdict
from gensim.models import Word2Vec
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

# TODO: Add code for training the model
