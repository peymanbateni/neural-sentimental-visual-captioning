import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
from datasets import *
from utils import *
from nltk.translate.bleu_score import corpus_bleu
import torch.nn.functional as F
from tqdm import tqdm

from preprocess import *
# Parameters
data_folder = './'  # folder with data files saved by create_input_files.py
data_name = 'vso_224_imgs'  # base name shared by data files
checkpoint = './BEST_checkpoint_' + data_name + '.pth.tar'  # model checkpoint
# sets device for model and PyTorch tensors
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# set to true only if inputs to model are fixed size; otherwise lot of computational overhead
cudnn.benchmark = True

# Load model
checkpoint = torch.load(checkpoint, map_location=device)
decoder = checkpoint['decoder']
decoder = decoder.to(device)
decoder.eval()
encoder = checkpoint['encoder']
encoder = encoder.to(device)
encoder.eval()

# Normalization transform
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])


def evaluate(beam_size):
    """
    Evaluation
    :param beam_size: beam size at which to generate captions for evaluation
    :return: BLEU-2 score
    """

    word2index, img2encoded_anps, train_image_addresses, validation_image_addresses, test_image_addresses = splitTrainValAndTestData(
        "../data/vso/vso_images_with_cc/")

    partition = {}
    partition['test'] = test_image_addresses
    # DataLoader
    loader = torch.utils.data.DataLoader(
        ANPDataset(partition, 'test', img2encoded_anps),
        batch_size=1, shuffle=True, num_workers=1, pin_memory=True)

    # TODO: Batched Beam Search
    # Therefore, do not use a batch_size greater than 1 - IMPORTANT!

    # Lists to store references (true captions), and hypothesis (prediction) for each image
    # If for n images, we have n hypotheses, and references a, b, c... for each image, we need -
    # references = [[ref1a, ref1b, ref1c], [ref2a, ref2b], ...], hypotheses = [hyp1, hyp2, ...]
    references = list()
    hypotheses = list()
    vocab_size = len(word2index)
    # For each image
    for i, (image, caps, caplens, allcaps) in enumerate(
            tqdm(loader, desc="EVALUATING AT BEAM SIZE " + str(beam_size))):
        k = beam_size
        # Move to GPU device, if available
        image = image.to(device)  # (1, 3, 256, 256)

        # Encode
        encoder_out = encoder(image)  # (1, enc_image_size, enc_image_size, encoder_dim)
        enc_image_size = encoder_out.size(1)
        encoder_dim = encoder_out.size(3)

        # Flatten encoding
        encoder_out = encoder_out.view(1, -1, encoder_dim)  # (1, num_pixels, encoder_dim)
        num_pixels = encoder_out.size(1)

        # We'll treat the problem as having a batch size of k
        encoder_out = encoder_out.expand(k, num_pixels, encoder_dim)  # (k, num_pixels, encoder_dim)

        # Tensor to store top k previous words at each step; now they're just <start>
        k_prev_words = torch.LongTensor([[word2index['<SOS>']]] * k).to(device)  # (k, 1)
        # Tensor to store top k sequences; now they're just <start>
        seqs = k_prev_words  # (k, 1)

        # Tensor to store top k sequences' scores; now they're just 0
        top_k_scores = torch.zeros(k, 1).to(device)  # (k, 1)

        # Lists to store completed sequences and scores
        complete_seqs = list()
        complete_seqs_scores = list()

        # Start decoding
        step = 1
        h, c = decoder.init_hidden_state(encoder_out)

        # s is a number less than or equal to k, because sequences are removed from this process once they hit <EOS>
        while True:
            embeddings = decoder.embedding(
                k_prev_words).squeeze(1)  # (s, embed_dim)
            # (s, encoder_dim), (s, num_pixels)
            awe, _ = decoder.attention(encoder_out, h)
            # gating scalar, (s, encoder_dim)
            gate = decoder.sigmoid(decoder.f_beta(h))
            awe = gate * awe
            h, c = decoder.decode_step(
                torch.cat([embeddings, awe], dim=1), (h, c))  # (s, decoder_dim)

            scores = decoder.fc(h)  # (s, vocab_size)
            scores = F.log_softmax(scores, dim=1)

            # Add
            scores = top_k_scores.expand_as(scores) + scores  # (s, vocab_size)

            # For the first step, all k points will have the same scores (since same k previous words, h, c)
            if step == 1:
                top_k_scores, top_k_words = scores[0].topk(
                    k, 0, True, True)  # (s)
            else:
                # Unroll and find top scores, and their unrolled indices
                # (s)
                top_k_scores, top_k_words = scores.view(
                    -1).topk(k, 0, True, True)

            # Convert unrolled indices to actual indices of scores
            prev_word_inds = top_k_words / vocab_size  # (s)
            next_word_inds = top_k_words % vocab_size  # (s)
            # Add new words to sequences
            seqs = torch.cat([seqs[prev_word_inds],
                              next_word_inds.unsqueeze(0)], dim=1)  # (s, step+1)

            # Which sequences are incomplete (didn't reach <EOS>)?
            incomplete_inds = [ind for ind, next_word in enumerate(next_word_inds) if
                               next_word != word2index['<EOS>']]
            complete_inds = list(
                set(range(len(next_word_inds))) - set(incomplete_inds))

            # Set aside complete sequences
            if len(complete_inds) > 0:
                complete_seqs.extend(seqs[complete_inds].tolist())
                complete_seqs_scores.extend(top_k_scores[complete_inds])
            k -= len(complete_inds)  # reduce beam length accordingly

            # Proceed with incomplete sequences
            if k == 0:
                break
            seqs = seqs[incomplete_inds]
            h = h[prev_word_inds[incomplete_inds]]
            c = c[prev_word_inds[incomplete_inds]]
            encoder_out = encoder_out[prev_word_inds[incomplete_inds]]
            top_k_scores = top_k_scores[incomplete_inds].unsqueeze(1)
            k_prev_words = next_word_inds[incomplete_inds].unsqueeze(1)

            # Break if things have been going on too long
            if step > 50:
                break
            step += 1

        i = complete_seqs_scores.index(max(complete_seqs_scores))
        seq = complete_seqs[i]

        # References
        img_caps = allcaps[0].tolist()
        img_captions = list(
            map(lambda c: [w for w in c if w not in {word2index['<SOS>'],
                                                     word2index['<EOS>'],
                                                     word2index['<pad>']}],
                img_caps))  # remove <SOS> and pads
        references.append(img_captions)

        # Hypotheses
        hypotheses.append([w for w in seq if w not in {
                          word2index['<SOS>'], word2index['<EOS>'], word2index['<pad>']}])

        assert len(references) == len(hypotheses)

    # Calculate BLEU-2 scores
    bleu2 = corpus_bleu(references, hypotheses, weights=(0.5,0.5))

    return bleu2


if __name__ == '__main__':
    beam_size = 1
    print("\nBLEU-2 score @ beam size of %d is %.4f." %
          (beam_size, evaluate(beam_size)))
