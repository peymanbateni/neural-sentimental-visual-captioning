from time import time
import torch.backends.cudnn as cudnn
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision.transforms import Normalize
from torch.nn import CrossEntropyLoss
from torch.nn.utils.rnn import pack_padded_sequence
from datasets import *
from utils import *
from nltk.translate.bleu_score import corpus_bleu
from models import *
from preprocess import *
from torch import device as torchDevice, load, multiprocessing, cuda
from nltk import word_tokenize
from torch import max as tmax
from collections import Counter

# Data parameters
# folder with data files saved by create_input_files.py
data_folder = '../data/coco/'
data_name = 'coco_imgs'  # base name shared by data files

# Model parameters
emb_dim = 512  # dimension of word embeddings
attention_dim = 512  # dimension of attention linear layers
decoder_dim = 512  # dimension of decoder RNN
dropout = 0.6
# sets device for model and PyTorch tensors
device = torchDevice("cuda:0" if cuda.is_available() else "cpu")
# set to true only if inputs to model are fixed size; otherwise lot of computational overhead
cudnn.benchmark = True

# Training parameters
start_epoch = 0
# number of epochs to train for (if early stopping is not triggered)
epochs = 10
# keeps track of number of epochs since there's been an improvement in validation BLEU
epochs_since_improvement = 0
batch_size = 20
workers = 1  # for data-loading; right now, only 1 works with h5py
encoder_lr = 1e-4  # learning rate for encoder if fine-tuning
decoder_lr = 4e-4  # learning rate for decoder
grad_clip = 5.  # clip gradients at an absolute value of
alpha_c = 1.  # regularization parameter for 'doubly stochastic attention', as in the paper
bleu_weights = (0.5, 0.5)  # weights for the bleu score
best_bleu2 = 0.  # BLEU-2 score right now
print_freq = 100  # print training/validation stats every __ batches
fine_tune_encoder = True  # fine-tune encoder?
#checkpoint = "BEST_checkpoint_vso_224_imgs.pth.tar"  # path to checkpoint, None if none
#checkpoint = "BEST_checkpoint_coco_imgs.pth.tar"
checkpoint = None
# Dictionaries and prep for dataloader


def main():
    """
    Training and validation.
    """
    global start_time, word2index, partition, anps, best_bleu2, epochs_since_improvement, checkpoint, start_epoch, data_name
    start_time = time()

    word2index, img2encoded_anps, train_image_addresses, validation_image_addresses, test_image_addresses = splitTrainValAndTestData(data_folder)
    partition = {}
    partition['train'] = train_image_addresses
    partition['validation'] = validation_image_addresses
    partition['test'] = test_image_addresses
    print("Size of partitions (train, val) is: " + str(len(partition['train'])) + ", "
          + str(len(partition['validation'])))
    print("Size of anps dictionary: " + str(len(img2encoded_anps)))
    params = {'batch_size': batch_size,
              'shuffle': True,
              'num_workers': workers,
              'pin_memory': True}
    encoder_dim = (len(word2index))

    # Read word map

    # Initialize / load checkpoint
    if checkpoint is None:
        decoder = DecoderWithAttention(attention_dim=attention_dim,
                                       embed_dim=emb_dim,
                                       decoder_dim=decoder_dim,
                                       vocab_size=len(word2index),
                                       dropout=dropout)
        decoder_optimizer = Adam(params=filter(lambda p: p.requires_grad, decoder.parameters()),
                                 lr=decoder_lr)
        encoder = ANPClassifier()
        encoder.fine_tune(fine_tune_encoder)
        encoder_optimizer = Adam(params=filter(lambda p: p.requires_grad, encoder.parameters()),
                                 lr=encoder_lr) if fine_tune_encoder else None

    else:
        checkpoint = load(checkpoint)
        start_epoch = checkpoint['epoch'] + 1
        epochs_since_improvement = checkpoint['epochs_since_improvement']
        best_bleu2 = checkpoint['bleu-2']
        decoder = checkpoint['decoder']
        decoder_optimizer = checkpoint['decoder_optimizer']
        encoder = checkpoint['encoder']
        encoder_optimizer = checkpoint['encoder_optimizer']
        if fine_tune_encoder is True and encoder_optimizer is None:
            encoder.fine_tune(fine_tune_encoder)
            encoder_optimizer = Adam(params=filter(lambda p: p.requires_grad, encoder.parameters()),
                                     lr=encoder_lr)

    # Move to GPU, if available
    decoder = decoder.to(device)
    encoder = encoder.to(device)

    # Loss function
    criterion = CrossEntropyLoss().to(device)

    # Custom dataloaders
    normalize = Normalize(mean=[0.485, 0.456, 0.406],
                          std=[0.229, 0.224, 0.225])
    params = {'batch_size': batch_size,
              'shuffle': True,
              'num_workers': workers,
              'pin_memory': True}
    train_loader = DataLoader(
        ANPDataset(partition, 'train', img2encoded_anps),
        **params)
    val_loader = DataLoader(
        ANPDataset(partition, 'validation', img2encoded_anps),
        **params)

    # Epochs
    for epoch in range(start_epoch, epochs):
        # Decay learning rate if there is no improvement for 8 consecutive epochs, and terminate training after 20
        if epochs_since_improvement == 2:
            break
        if epochs_since_improvement > 0 and epochs_since_improvement % 8 == 0:
            adjust_learning_rate(decoder_optimizer, 0.8)
            if fine_tune_encoder:
                adjust_learning_rate(encoder_optimizer, 0.8)

        # One epoch's training
        train(train_loader=train_loader,
              encoder=encoder,
              decoder=decoder,
              criterion=criterion,
              encoder_optimizer=encoder_optimizer,
              decoder_optimizer=decoder_optimizer,
              epoch=epoch)

        # One epoch's validation
        recent_bleu2 = validate(val_loader=val_loader,
                                encoder=encoder,
                                decoder=decoder,
                                criterion=criterion)
    
        # Check if there was an improvement
        is_best = recent_bleu2 > best_bleu2
        best_bleu2 = max(recent_bleu2, best_bleu2)
        if not is_best:
            epochs_since_improvement += 1
            print("\nEpochs since last improvement: %d\n" %
                  (epochs_since_improvement,))
        else:
            epochs_since_improvement = 0
    
        # Save checkpoint
        save_checkpoint(data_name, epoch, epochs_since_improvement, encoder, decoder, encoder_optimizer,
                        decoder_optimizer, recent_bleu2, is_best)


def train(train_loader, encoder, decoder, criterion, encoder_optimizer, decoder_optimizer, epoch):


    """
    Performs one epoch's training.
    :param train_loader: DataLoader for training data
    :param encoder: encoder model
    :param decoder: decoder model
    :param criterion: loss layer
    :param encoder_optimizer: optimizer to update encoder's weights (if fine-tuning)
    :param decoder_optimizer: optimizer to update decoder's weights
    :param epoch: epoch number
    """

    decoder.train()  # train mode (dropout and batchnorm is used)
    encoder.train()

    batch_time = AverageMeter()  # forward prop. + back prop. time
    data_time = AverageMeter()  # data loading time
    losses = AverageMeter()  # loss (per word decoded)
    top1accs = AverageMeter()  # top5 accuracy 
    top5accs = AverageMeter()  # top5 accuracy
    top10accs = AverageMeter()  # top5 accurac
    top20accs = AverageMeter()  # top5 accuracy
    start = time()

    # Batches
    for i, (imgs, caps, caplens) in enumerate(train_loader):
        data_time.update(time() - start)

        # Move to GPU, if available
        imgs = imgs.to(device)
        caps = caps.to(device)
        caplens = caplens.to(device)

        # Forward prop.
        imgs = encoder(imgs)
        scores, caps_sorted, decode_lengths, alphas, sort_ind = decoder(
            imgs, caps, caplens)

        # Since we decoded starting with <start>, the targets are all words after <start>, up to <end>
        targets = caps_sorted[:, 1:]

        # Remove timesteps that we didn't decode at, or are pads
        # pack_padded_sequence is an easy trick to do this
        scores, _ = pack_padded_sequence(scores, decode_lengths, batch_first=True)
        targets, _ = pack_padded_sequence(
            targets, decode_lengths, batch_first=True)

        # Calculate loss
        loss = criterion(scores, targets)

        # Add doubly stochastic attention regularization
        loss += alpha_c * ((1. - alphas.sum(dim=1)) ** 2).mean()

        # Back prop.
        decoder_optimizer.zero_grad()
        if encoder_optimizer is not None:
            encoder_optimizer.zero_grad()
        loss.backward()

        # Clip gradients
        if grad_clip is not None:
            clip_gradient(decoder_optimizer, grad_clip)
            if encoder_optimizer is not None:
                clip_gradient(encoder_optimizer, grad_clip)

        # Update weights
        decoder_optimizer.step()
        if encoder_optimizer is not None:
            encoder_optimizer.step()

        # Keep track of metrics
        top1 = accuracy(scores, targets, 1)
        top5 = accuracy(scores, targets, 5)
        top10 = accuracy(scores, targets, 10)
        top20 = accuracy(scores, targets, 20)

        losses.update(loss.item(), sum(decode_lengths))
        top1accs.update(top1, sum(decode_lengths))
        top5accs.update(top5, sum(decode_lengths))
        top10accs.update(top10, sum(decode_lengths))
        top20accs.update(top20, sum(decode_lengths))
        batch_time.update(time() - start)

        start = time()

        # Print status
        if i % print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  '\n     Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  '\n     Data Load Time {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  '\n     Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  '\n     Top-1 Accuracy {top1.val:.3f} ({top1.avg:.3f})'
                  '\n     Top-5 Accuracy {top5.val:.3f} ({top5.avg:.3f})'
                  '\n     Top-10 Accuracy {top10.val:.3f} ({top10.avg:.3f})'
                  '\n     Top-20 Accuracy {top20.val:.3f} ({top20.avg:.3f})'.format(epoch, i, len(train_loader),
                                                                                 batch_time=batch_time,
                                                                                 data_time=data_time, loss=losses,
                                                                                 top1=top1accs,
                                                                                 top5=top5accs,
                                                                                 top10=top10accs,
                                                                                 top20=top20accs))


def validate(val_loader, encoder, decoder, criterion):


    """
    Performs one epoch's validation.
    :param val_loader: DataLoader for validation data.
    :param encoder: encoder model
    :param decoder: decoder model
    :param criterion: loss layer
    :return: BLEU-2 score
    """
    decoder.eval()  # eval mode (no dropout or batchnorm)
    if encoder is not None:
        encoder.eval()

    batch_time = AverageMeter()
    losses = AverageMeter()
    top1accs = AverageMeter()
    top5accs = AverageMeter()
    top10accs = AverageMeter()
    top20accs = AverageMeter()

    start = time()

    references = list()  # references (true captions) for calculating BLEU-2 score
    hypotheses = list()  # hypotheses (predictions)

    # Batches
    for i, (imgs, caps, caplens, allcaps) in enumerate(val_loader):

        # Move to device, if available
        imgs = imgs.to(device)
        caps = caps.to(device)
        caplens = caplens.to(device)

        # Forward prop.
        if encoder is not None:
            imgs = encoder(imgs)
        scores, caps_sorted, decode_lengths, alphas, sort_ind = decoder(
            imgs, caps, caplens)

        # Since we decoded starting with <start>, the targets are all words after <start>, up to <end>
        targets = caps_sorted[:, 1:]

        # Remove timesteps that we didn't decode at, or are pads
        # pack_padded_sequence is an easy trick to do this
        scores_copy = scores.clone()
        scores, _ = pack_padded_sequence(scores, decode_lengths, batch_first=True)
        targets, _ = pack_padded_sequence(
            targets, decode_lengths, batch_first=True)

        # Calculate loss
        loss = criterion(scores, targets)

        # Add doubly stochastic attention regularization
        loss += alpha_c * ((1. - alphas.sum(dim=1)) ** 2).mean()

        # Keep track of metrics
        losses.update(loss.item(), sum(decode_lengths))
        top1 = accuracy(scores, targets, 1)
        top1accs.update(top1, sum(decode_lengths))
        top5 = accuracy(scores, targets, 5)
        top5accs.update(top5, sum(decode_lengths))
        top10 = accuracy(scores, targets, 10)
        top10accs.update(top10, sum(decode_lengths))
        top20 = accuracy(scores, targets, 20)
        top20accs.update(top20, sum(decode_lengths))


        batch_time.update(time() - start)

        start = time()

        if i % print_freq == 0:
            print('Validation: [{0}/{1}]\t'
                      'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Top-1 Accuracy {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Top-5 Accuracy {top5.val:.3f} ({top5.avg:.3f})\t'
                      'Top-10 Accuracy {top10.val:.3f} ({top10.avg:.3f})\t'
                      'Top-20 Accuracy {top20.val:.3f} ({top20.avg:.3f})\t'.format(i, len(val_loader), batch_time=batch_time,
                                                                                loss=losses, top1=top1accs, top5=top5accs, top10=top10accs, top20=top20accs)) 

        # Store references (true captions), and hypothesis (prediction) for each image
        # If for n images, we have n hypotheses, and references a, b, c... for each image, we need -
        # references = [[ref1a, ref1b, ref1c], [ref2a, ref2b], ...], hypotheses = [hyp1, hyp2, ...]

        # References
        allcaps = allcaps[sort_ind]
        for j in range(allcaps.shape[0]):
            img_caps = allcaps[j].tolist()
            img_captions = list(
                map(lambda c: [w for w in c if w not in {word2index['<SOS>'],
                                                         word2index['<pad>']}],
                    img_caps))  # remove <start> and pads
            references.append(img_captions)
           # references = caps
        # Hypotheses
        _, preds = tmax(scores_copy, dim=2)
        preds = preds.tolist()
        temp_preds = list()
        for j, p in enumerate(preds):
            temp_preds.append(preds[j][:decode_lengths[j]])  # remove pads
        preds = temp_preds
        hypotheses.extend(preds)
        assert len(references) == len(hypotheses)

        # Calculate BLEU-2 scores
        bleu2 = corpus_bleu(references, hypotheses, weights=bleu_weights)

        print(
            '\n * LOSS - {loss.avg:.3f}, TOP-1 ACCURACY - {top1.avg:.3f}, TOP-5 ACCURACY - {top5.avg:.3f}, TOP-10 ACCURACY - {top10.avg:.3f}, TOP-20 ACCURACY - {top20.avg:.3f}, BLEU-2 - {bleu}\n'.format(
                loss=losses,
                top1=top1accs,
                top5=top5accs,
                top10=top10accs,
                top20=top20accs,
                bleu=bleu2))
    print('    Total training time:', time()-start_time)
    return bleu2


if __name__ == '__main__':
    multiprocessing.set_start_method('forkserver', force=True)
    main()
