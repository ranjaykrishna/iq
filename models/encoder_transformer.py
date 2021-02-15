from models.transformer_layers import Encoder, generate_pad_mask
from torch import nn

class GVTransformerEncoder(nn.Module):
    def __init__(self, embedding, latent_layer, latent_transformer, args):
        super().__init__()

        self.embedding = embedding
        self.latent_transformer = latent_transformer
        self.latent_layer = latent_layer
        
        self.encoder = Encoder(args.emb_dim, args.hidden_dim, num_layers=args.num_layers, num_heads=args.num_heads, 
                                total_key_depth=args.hidden_dim, total_value_depth=args.hidden_dim,
                                filter_size=args.pwffn_dim)
        
        self.r_encoder = Encoder(args.emb_dim, args.hidden_dim, num_layers=args.num_layers, num_heads=args.num_heads, 
                                total_key_depth=args.hidden_dim, total_value_depth=args.hidden_dim,
                                filter_size=args.pwffn_dim)

        

    def forward(self, context, response, image_features):
        res_mask = generate_pad_mask(response)
        embedded_response = self.embedding(response)
        response_encoder_outputs = self.r_encoder(embedded_response, res_mask)

        src_mask = generate_pad_mask(context)
        embedded_context = self.embedding(context)
        encoder_outputs = self.encoder(embedded_context, src_mask)

        # TEST WHETHER THIS IS BENEFICIAL/DETRIMENTAL
        encoder_outputs[:, 0] = encoder_outputs[:, 0] + image_features
        kld_loss, z, posteriors = None, None, None
        if self.latent_transformer:
            kld_loss, z, posteriors = self.latent_layer(encoder_outputs[:, 0], response_encoder_outputs[:, 0])

        return encoder_outputs, kld_loss, z, posteriors, src_mask