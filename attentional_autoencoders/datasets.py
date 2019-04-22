from torch.utils import data
from PIL import Image
from torch.autograd import Variable
from torchvision import transforms
from torch import device as torchDevice, cuda, LongTensor
import re
img_size = 224
device = torchDevice("cuda:0" if cuda.is_available() else "cpu")  # sets device for model and PyTorch tensors

class ANPDataset(data.Dataset):
    def __init__(self, partition, split, captions):
#        'Initialization'
        self.split = split
        assert self.split in {'train', 'validation', 'test'}
        imgs_addrs = partition[split]
        self.imgs_addrs = []
        self.captions = {}
        if('coco' in imgs_addrs[0]):
            for addr in imgs_addrs:
                for i, cap in enumerate(captions[addr]):
                    i_addr = addr + str(i) 
                    self.imgs_addrs.append(i_addr)
                    self.captions.update({i_addr:cap})
        else:
            self.imgs_addrs = imgs_addrs
            self.captions = captions
        self.caplens = {key: len(value) for key, value in self.captions.items()}
        self.loader = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor()
        ])
    def __len__(self):
 #       'Denotes the total number of samples'
        return len(self.imgs_addrs)

    def __getitem__(self, index):
#       'Generates one sample of data'
    # Select sample
        ID = self.imgs_addrs[index]
    # Load data and get label
        image = Image.open(re.sub(r'jpg[0-9]?', 'jpg', ID)).convert('RGB')
        image = self.loader(image).float()
        image = Variable(image, volatile=False)
        #image.to(device)
        caption = LongTensor(self.captions[ID])
        caplen = LongTensor([self.caplens[ID]])
        if (self.split == 'train'):
              return image, caption, caplen
        else:
              allcaps = LongTensor([self.captions[ID]])
              return image, caption, caplen, allcaps
