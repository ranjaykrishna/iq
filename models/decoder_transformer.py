from models.transformer_layers import Decoder, generate_pad_mask
from torch import nn
import torch

class GVTransformerDecoder(nn.Module):
    def __init__(self, embedding, latent_transformer, vocab_size, vocab, args):
        super().__init__()

        self.embedding = embedding
        self.latent_transformer = latent_transformer
        self.vocab = vocab
        self.args = args

        self.decoder = Decoder(args.emb_dim, hidden_size = args.hidden_dim, num_layers=args.num_layers, num_heads=args.num_heads, 
                            total_key_depth=args.hidden_dim, total_value_depth=args.hidden_dim,
                            filter_size=args.pwffn_dim, device=args.device)

        self.output = nn.Linear(args.hidden_dim, vocab_size)
        self.z_classifier = nn.Linear(args.hidden_dim, vocab_size)

    def forward(self, encoder_outputs, target, image_features, z, src_mask):

        sos_token = torch.LongTensor([self.vocab.word2idx[self.vocab.SYM_SOQ]] * encoder_outputs.size(0)).unsqueeze(1)
        sos_token = sos_token.to(self.args.device)

        target_shifted = torch.cat((sos_token, target[:, :-1]), 1)
        trg_key_padding_mask = generate_pad_mask(target_shifted)
        target_embedding = self.embedding(target_shifted)

        target_embedding[:, 0] = target_embedding[:, 0] + image_features # z = 0 if we're pretraining
        z_logit = None
        if self.latent_transformer:
            target_embedding[:, 0] = target_embedding[:, 0] + z
            z_logit = self.z_classifier(z + image_features)

        # decoder_outputs = self.transformer_decoder(target_embedding, encoder_outputs, trg_mask, src_mask.unsqueeze(1))
        decoder_outputs, _ = self.decoder(target_embedding, encoder_outputs, (src_mask, trg_key_padding_mask))

        output = self.output(decoder_outputs)
        return output, target_embedding, z_logit

    def inference_forward(self, encoder_outputs, inference_input, image_features, z, src_mask):
        trg_key_padding_mask = generate_pad_mask(inference_input)
        pred_targets_embedding = self.embedding(inference_input)
        pred_targets_embedding[:, 0] = pred_targets_embedding[:, 0] + z + image_features
        decoder_outputs, _ = self.decoder(pred_targets_embedding, encoder_outputs, (src_mask, trg_key_padding_mask))
        return self.output(decoder_outputs)