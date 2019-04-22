from os import listdir
from PIL import Image, ImageFile
from torchvision import transforms
from IPython import display
from random import randint
import pickle
import json
from nltk import word_tokenize
from collections import Counter
ImageFile.LOAD_TRUNCATED_IMAGES = True

def pad_caption_array(cap_array, max_length):
    cap_array += ['<pad>']*(max_length-len(cap_array))
    return cap_array

def replace_right(source, target, replacement, replacements=None):
    if (source.count("_") <= 1):
        return source
    return replacement.join(source.rsplit(target, replacements))


def splitTrainValAndTestData(data_folder, img_size=224):
    if ('coco' in data_folder):
        file_name = 'coco_dictionaries.p'
    else:
        file_name = 'img_addr_dictionaries.p'
    
    print("Starting split of data.")
    try:
        with open(file_name, 'rb') as fp:
           print('Trying to retrieve from file.')
           word2index, img2encoded_anps, train_image_addresses, validation_image_addresses, test_image_addresses = pickle.load(fp)
        print('Successfully retrieved from file.')
    except Exception as e:
        print("Could not find file", file_name)
        print('Extracting information from data folder.')
        count = 0

        train_image_addresses = []
        train_image_to_anp_tag = {}

        validation_image_addresses = []
        validation_image_to_anp_tag = {}

        test_image_addresses = []
        test_image_to_anp_tag = {}

        max_train_caplen = 0
        max_val_caplen = 0
        max_test_caplen = 0
        if ('coco' in file_name):
            folder = '../data/coco/'
            train_id2addr = {}
            val_id2addr = {}
            with open(folder + 'annotations_trainval2017/' + 'captions_train2017.json', 'rb') as f:
                train_info = json.load(f)
            with open(folder + 'annotations_trainval2017/' + 'captions_val2017.json', 'rb') as f:
                val_info = json.load(f)
            subfolder = 'train2017/'
            for img_info in train_info['images']:
                train_id2addr.update({img_info['id']:folder + subfolder + img_info['file_name']})
            subfolder = 'val2017/'
            for img_info in val_info['images']:
                val_id2addr.update({img_info['id']:folder + subfolder + img_info['file_name']})
            folder = '../processed/'
            for filename in listdir(folder):
                with open(folder + filename, 'rb') as f:
                    if 'anp_caption' in filename:
                        if 'test' in filename:
                            test_id2anp_tag = pickle.load(f)
                        elif 'train' in filename:
                            train_id2anp_tag = pickle.load(f)
                        elif 'val' in filename:
                            val_id2anp_tag = pickle.load(f)
            train_id2anp_tag.update(test_id2anp_tag)
            for img_id, addr in train_id2addr.items():
                if randint(0,100) == 50 or randint(0,500) == 220:
                    test_image_addresses.append(addr)
                    test_image_to_anp_tag.update({addr:train_id2anp_tag[img_id]})
                else:
                    train_image_addresses.append(addr)
                    train_image_to_anp_tag.update({addr:train_id2anp_tag[img_id]})
                for cap in train_id2anp_tag[img_id]:
                    length = len(word_tokenize(cap)) + 2
                    if max_train_caplen < length:
                        max_train_caplen = length
    #        for img_id, addr in val_id2addr.items():
     #           if randint(0,1000) == 10:
      #              test_image_addresses.append(addr)
       #             test_image_to_anp_tag.update({addr:val_id2anp_tag[img_id]})
        #        else:
         #           validation_image_addresses.append(addr)
          #          validation_image_to_anp_tag.update({addr:val_id2anp_tag[img_id]})
           #     for cap in val_id2anp_tag[img_id]:
            #        length = cap.count(" ")
             #       if max_val_caplen < length:
              #          max_val_caplen = length
            max_test_caplen = max(max_val_caplen, max_train_caplen)
            validation_image_addresses = test_image_addresses
            img2anp={img: [pad_caption_array(["<SOS>"] + word_tokenize(anp) + ["<EOS>"], max_train_caplen) for anp in captions] for img, captions in train_image_to_anp_tag.items()}
           # img2anp.update({img: [["<SOS>"] + pad_caption_array(word_tokenize(anp), max_val_caplen) + ["<EOS>"] for anp in captions] for img, captions in validation_image_to_anp_tag.items()})
            img2anp.update({img: [pad_caption_array(["<SOS>"] + word_tokenize(anp) + ["<EOS>"], max_test_caplen) for anp in captions] for img, captions in test_image_to_anp_tag.items()})
            word_counts = Counter([word for img, captions in img2anp.items() for anp in captions for word in anp])
            index2word = [e for e in word_counts]
            if('<pad>' not in index2word):
                index2word += ['<pad>']
            elif('<unk>' not in index2word):
                index2word += ['<unk>']   
            word2index = {word: index for index, word in enumerate(index2word)}
            img2encoded_anps = {img: [[word2index[word]
                for word in anp] for anp in captions] for img, captions in img2anp.items()}
        else:
            for subdir in listdir(data_folder):
                if subdir.endswith("_train"):
                   for filename in listdir(data_folder + subdir):
                        if filename.endswith(".jpg"):
                            img_addr = data_folder + subdir + "/" + filename
                            try:
                                height, width = Image.open(img_addr).size
                                if height < img_size or width < img_size:
                                   print("Bad image " + str(img_addr) + "\n   Width is: "
                                      + str(width) + "\n   Height is: " + str(height))
                                else:
                                    train_image_addresses.append(img_addr)
                                    train_caption = replace_right(
                                        subdir, "_train", "", 1).replace("_", " ")
                                    length = train_caption.count(" ")
                                    if (max_train_caplen < length):
                                        max_train_caplen = length
                                    train_image_to_anp_tag[img_addr] = train_caption
                            except:
                                print("Error opening ", img_addr)
                elif subdir.endswith("_validation"):
                    for filename in listdir(data_folder + subdir):
                            if filename.endswith(".jpg"):
                                img_addr = data_folder + subdir + "/" + filename
                            try:
                                height, width = Image.open(img_addr).size
                                if (height < img_size or width < img_size):
                                   print("Bad image " + str(img_addr) + "\n   Width is: "
                                      + str(width) + "\n   Height is: " + str(height))
                                else:
                                    validation_image_addresses.append(img_addr)
                                    val_caption = subdir.replace(
                                        "_validation", "").replace("_", " ")
                                    length = val_caption.count(" ")
                                    if (max_val_caplen < length):
                                        max_val_caplen = length
                                    validation_image_to_anp_tag[img_addr] = val_caption
                            except Exception as e:
                                print("Error", e)
                                print("Error opening ", img_addr)
                elif subdir.endswith("_test"):
                    for filename in listdir(data_folder + subdir):
                        if filename.endswith(".jpg"):
                            img_addr = data_folder + subdir + "/" + filename
                            try:
                                height, width = Image.open(img_addr).size
                                if (height < img_size or width < img_size):
                                   print("Bad image " + str(img_addr) + "\n   Width is: "
                                      + str(width) + "\n   Height is: " + str(height))
                                else:
                                    test_image_addresses.append(img_addr)
                                    test_caption = subdir.replace(
                                        "_test", "").replace("_", " ")
                                    length = test_caption.count(" ")
                                    if (max_test_caplen < length):
                                        max_test_caplen=length
                                    test_image_to_anp_tag[img_addr]=test_caption
                            except:
                                print("Error opening ", img_addr)
                img2anp={img: ["<SOS>"] + pad_caption_array(word_tokenize(anp),
                                                            max_train_caplen) + ["<EOS>"]
                                                        for img, anp in train_image_to_anp_tag.items()}
                img2anp.update({img: ["<SOS>"] + pad_caption_array(word_tokenize(anp),
                                                                   max_val_caplen) + ["<EOS>"]
                               for img, anp in validation_image_to_anp_tag.items()})
                img2anp.update({img: ["<SOS>"] + pad_caption_array(word_tokenize(anp),
                                                                   max_test_caplen) + ["<EOS>"] for img,
                                anp in test_image_to_anp_tag.items()})
                word_counts = Counter([word for img, anp in img2anp.items()
                                    for word in anp])
                index2word = [e for e in word_counts]
                if('<pad>' not in index2word):
                    index2word += ['<pad>']
                elif('<unk>' not in index2word):
                    index2word += ['<unk>']   
                word2index = {word: index for index, word in enumerate(index2word)}
                img2encoded_anps = {img: [word2index[word]
                    for word in anp] for img, anp in img2anp.items()}

#    display.display(display.Image('data/vso/vso_images_with_cc/adorable_smile_validation/272536646_a621ce4cb1.jpg'))
    print("Finished splitting data.")
    with open(file_name, 'wb') as fp:
        pickle.dump((word2index, img2encoded_anps, train_image_addresses,
                    validation_image_addresses, test_image_addresses), fp, protocol=pickle.HIGHEST_PROTOCOL)
    print("Number of train images: ", len(train_image_addresses))
    print("Number of validation images: ", len(validation_image_addresses))
    print("Number of test images: ", len(test_image_addresses))

    return word2index, img2encoded_anps, train_image_addresses, validation_image_addresses, test_image_addresses
