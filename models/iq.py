"""Contains code for the IQ model.
"""

from math import log
# from utils import vocab
from models import transformer_layers
from models.decoder_transformer import GVTransformerDecoder
from models.transformer_layers import Latent, generate_pad_mask
from models.encoder_transformer import GVTransformerEncoder
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoder_cnn import EncoderCNN
from .encoder_rnn import EncoderRNN
from .decoder_rnn import DecoderRNN
from .mlp import MLP


class IQ(nn.Module):
    """Information Maximization question generation.
    """
    def __init__(self, latent_transformer, vocab, args, num_att_layers=2):
        super(IQ, self).__init__()
        self.vocab = vocab
        self.vocab_size = len(self.vocab.word2idx)
        self.latent_transformer = latent_transformer
        self.args = args

        # Nihir: Set up embedding
        self.embedding = self.embedder()


        # Setup image encoder.
        self.encoder_cnn = EncoderCNN(args.hidden_dim)

        self.latent_layer = Latent(args)
        self.answer_encoder = GVTransformerEncoder(self.embedding, self.latent_layer, self.latent_transformer, args)

        self.decoder = GVTransformerDecoder(self.embedding, self.latent_transformer, self.vocab_size, vocab, args)

        # Setup image reconstruction.
        self.image_reconstructor = MLP(
                args.hidden_dim, args.pwffn_dim, args.hidden_dim,
                num_layers=num_att_layers)


    def switch_GVT_train_mode(self, new_mode):
        self.latent_transformer = new_mode
        self.answer_encoder.latent_transformer = new_mode
        self.decoder.latent_transformer = new_mode


    def embedder(self):
        init_embeddings = np.random.randn(self.vocab_size, self.args.emb_dim) * 0.01 
        print('Embeddings: %d x %d' % (self.vocab_size, self.args.emb_dim))
        if self.args.emb_file is not None:
            print('Loading embedding file: %s' % self.args.emb_file)
            pre_trained = 0
            for line in open(self.args.emb_file).readlines():
                sp = line.split()
                if(len(sp) == self.args.emb_dim + 1):
                    if sp[0] in self.vocab.word2idx:
                        pre_trained += 1
                        init_embeddings[self.vocab.word2idx[sp[0]]] = [float(x) for x in sp[1:]]
                else:
                    print(sp[0])
            print('Pre-trained: %d (%.2f%%)' % (pre_trained, pre_trained * 100.0 / self.vocab_size))
        embedding = nn.Embedding(self.vocab_size, self.args.emb_dim, padding_idx=self.vocab.word2idx[self.vocab.SYM_PAD])
        embedding.weight.data.copy_(torch.FloatTensor(init_embeddings))
        embedding.weight.data.requires_grad = True
        return embedding


    def forward(self, images, answers, response, target):
        """Passes the image and the question through a model and generates answers.

        Args:
            images: Batch of image Variables.
            answers: Batch of answer Variables.
            categories: Batch of answer Variables.
            alengths: List of answer lengths.
            questions: Batch of question Variables.
            teacher_forcing_ratio: Whether to predict with teacher forcing.
            decode_function: What to use when choosing a word from the
                distribution over the vocabulary.

        Returns:
            - outputs: The output scores for all steps in the RNN.
            - hidden: The hidden states of all the RNNs.
            - ret_dict: A dictionary of attributes. See DecoderRNN.py for details.
        """
        # features is (N * args.hidden_dim)
        image_features = self.encoder_cnn(images)

        # z-path. transformer_posteriors is a tuple: (mean_posterior, logvar_posterior)
        encoder_outputs, transformer_kld_loss, z, transformer_posteriors, src_mask = self.answer_encoder(answers, response, image_features)
        output, target_embedding, z_logit = self.decoder(encoder_outputs, target, image_features, z, src_mask)

        if self.latent_transformer: # experiement without requiring the latent mode enabled?
            reconstructed_image_features = self.image_reconstructor(z)
        else:
            reconstructed_image_features = self.image_reconstructor(encoder_outputs[:, 0])

        return output, z_logit, transformer_kld_loss, (image_features, reconstructed_image_features)


    def decode_greedy(self, images, answers, max_decode_length = 50):
        image_features = self.encoder_cnn(images)
        src_mask = generate_pad_mask(answers)
        embedded_context = self.embedding(answers)
        encoder_outputs = self.answer_encoder.encoder(embedded_context, src_mask)
        encoder_outputs[:, 0] = encoder_outputs[:, 0] + image_features # TEST THISSS

        z = 0
        if self.latent_transformer:
            _, z, _ = self.latent_layer(encoder_outputs[:, 0], None)

        ys = torch.ones(answers.shape[0], 1).fill_(self.vocab.word2idx[self.vocab.SYM_PAD]).long().to(self.args.device)


        decoded_words = []
        for i in range(max_decode_length + 1):
            pred_targets_logit = self.decoder.inference_forward(encoder_outputs, ys, image_features, z, src_mask)
            _, pred_next_word = torch.max(pred_targets_logit[:, -1], dim=1)

            decoded_words.append(['<end>' if token.item() == self.vocab.word2idx[self.vocab.SYM_EOS] else self.vocab.idx2word[token.item()] for token in pred_next_word.view(-1)])

            ys = torch.cat([ys, pred_next_word.unsqueeze(1)], dim=1)

        sentence = []
        for _, row in enumerate(np.transpose(decoded_words)):
            st = ''
            for e in row:
                if e == '<end>': break
                else: st+= e + ' '
            sentence.append(st)
        return sentence
