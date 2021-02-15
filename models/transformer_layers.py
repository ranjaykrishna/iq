from typing import Optional
from torch.nn import ModuleList, Module, Linear, Dropout, LayerNorm, ReLU
from torch import Tensor
import torch.functional as F
import torch
import copy
import math
from torch import nn
import numpy as np


def generate_pad_mask(data, pad_idx=0):
    return data.data.eq(pad_idx).unsqueeze(1)

class Latent(nn.Module):
    def __init__(self, args, dropout=0):
        super(Latent, self).__init__()

        self.args = args

        latent_layer_network_architecture = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(args.latent_dim*2, args.latent_dim*2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(args.latent_dim*2, args.latent_dim*2)
        )

        self.mean_logvar_prior = nn.Sequential(
            nn.Linear(args.hidden_dim, args.latent_dim*2),
            *copy.deepcopy(latent_layer_network_architecture)

        )

        self.mean_logvar_posterior = nn.Sequential(
            nn.Linear(args.hidden_dim*2, args.latent_dim*2),
            *copy.deepcopy(latent_layer_network_architecture)
        )

    def forward(self, x, x_p):
        mean_logvar_prior = self.mean_logvar_prior(x)
        mean_prior, logvar_prior = mean_logvar_prior[:, :self.args.latent_dim], mean_logvar_prior[:, self.args.latent_dim:]

        eps = torch.randn(x.size()).to(self.args.device)
        std = torch.exp(0.5 * logvar_prior)
        z = eps * std + mean_prior
        kld_loss = 0

        mean_posterior, logvar_posterior = None, None
        if x_p is not None: # if x_p IS None, then we're in eval mode.
            mean_logvar_posterior = self.mean_logvar_posterior(torch.cat((x_p, x), dim=-1))
            mean_posterior, logvar_posterior = mean_logvar_posterior[:, :self.args.latent_dim], mean_logvar_posterior[:, self.args.latent_dim:]
            kld_loss = gaussian_kld(mean_posterior, logvar_posterior, mean_prior, logvar_prior)
            kld_loss = torch.mean(kld_loss)

            std = torch.exp(0.5 * logvar_posterior)
            z = eps * std + mean_posterior
        return kld_loss, z, (mean_posterior, logvar_posterior)



def _gen_bias_mask(max_length):
    """
    Generates bias values (-Inf) to mask future timesteps during attention
    """
    np_mask = np.triu(np.full([max_length, max_length], -np.inf), 1)
    torch_mask = torch.from_numpy(np_mask).type(torch.FloatTensor)
    
    return torch_mask.unsqueeze(0).unsqueeze(1)


def _get_attn_subsequent_mask(size, device="cpu"):
    """
    Get an attention mask to avoid using the subsequent info.
    Args:
        size: int
    Returns:
        (`LongTensor`):
        * subsequent_mask `[1 x size x size]`
    """
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    subsequent_mask = torch.from_numpy(subsequent_mask).to(device)
    return subsequent_mask


class Encoder(nn.Module):
    """
    A Transformer Encoder module. 
    Inputs should be in the shape [batch_size, length, hidden_size]
    Outputs will have the shape [batch_size, length, hidden_size]
    Refer Fig.1 in https://arxiv.org/pdf/1706.03762.pdf
    """
    def __init__(self, embedding_size, hidden_size, num_layers, num_heads, total_key_depth, total_value_depth,
                 filter_size, max_length=1000, input_dropout=0, layer_dropout=0, 
                 attention_dropout=0.1, relu_dropout=0.1, use_mask=False):
        """
        Parameters:
            embedding_size: Size of embeddings
            hidden_size: Hidden size
            num_layers: Total layers in the Encoder
            num_heads: Number of attention heads
            total_key_depth: Size of last dimension of keys. Must be divisible by num_head
            total_value_depth: Size of last dimension of values. Must be divisible by num_head
            output_depth: Size last dimension of the final output
            filter_size: Hidden size of the middle layer in FFN
            max_length: Max sequence length (required for timing signal)
            input_dropout: Dropout just after embedding
            layer_dropout: Dropout for each layer
            attention_dropout: Dropout probability after attention (Should be non-zero only during training)
            relu_dropout: Dropout probability after relu in FFN (Should be non-zero only during training)
            use_mask: Set to True to turn on future value masking
        """
        
        super(Encoder, self).__init__()
        self.num_layers = num_layers
        self.timing_signal = _gen_timing_signal(max_length, hidden_size)
        
        
        params =(hidden_size, 
                 total_key_depth or hidden_size,
                 total_value_depth or hidden_size,
                 filter_size, 
                 num_heads, 
                 _gen_bias_mask(max_length) if use_mask else None,
                 layer_dropout, 
                 attention_dropout, 
                 relu_dropout)
        
        self.embedding_proj = nn.Linear(embedding_size, hidden_size, bias=False)
        self.enc = nn.ModuleList([EncoderLayer(*params) for _ in range(num_layers)])
        
        self.layer_norm = LayerNorm(hidden_size)
        self.input_dropout = nn.Dropout(input_dropout)
        

    def forward(self, inputs, mask):
        #Add input dropout
        x = self.input_dropout(inputs)
        
        # Project to hidden size
        x = self.embedding_proj(x)

        # Add timing signal
        x += self.timing_signal[:, :inputs.shape[1], :].type_as(inputs.data)
        
        for i in range(self.num_layers):
            x = self.enc[i](x, mask)
    
        y = self.layer_norm(x)
        return y


class Decoder(nn.Module):
    """
    A Transformer Decoder module. 
    Inputs should be in the shape [batch_size, length, hidden_size]
    Outputs will have the shape [batch_size, length, hidden_size]
    Refer Fig.1 in https://arxiv.org/pdf/1706.03762.pdf
    """
    def __init__(self, embedding_size, hidden_size, num_layers, num_heads, total_key_depth, total_value_depth,
                 filter_size, max_length=200, input_dropout=0, layer_dropout=0, 
                 attention_dropout=0.1, relu_dropout=0.1, device="cpu"):
        """
        Parameters:
            embedding_size: Size of embeddings
            hidden_size: Hidden size
            num_layers: Total layers in the Encoder
            num_heads: Number of attention heads
            total_key_depth: Size of last dimension of keys. Must be divisible by num_head
            total_value_depth: Size of last dimension of values. Must be divisible by num_head
            output_depth: Size last dimension of the final output
            filter_size: Hidden size of the middle layer in FFN
            max_length: Max sequence length (required for timing signal)
            input_dropout: Dropout just after embedding
            layer_dropout: Dropout for each layer
            attention_dropout: Dropout probability after attention (Should be non-zero only during training)
            relu_dropout: Dropout probability after relu in FFN (Should be non-zero only during training)
        """
        
        super(Decoder, self).__init__()
        self.num_layers = num_layers
        self.timing_signal = _gen_timing_signal(max_length, hidden_size)

        self.mask = _get_attn_subsequent_mask(max_length, device)

        params =(hidden_size, 
                 total_key_depth or hidden_size,
                 total_value_depth or hidden_size,
                 filter_size, 
                 num_heads, 
                 _gen_bias_mask(max_length), # mandatory
                 None,
                 layer_dropout, 
                 attention_dropout, 
                 relu_dropout)
        
        self.dec = nn.Sequential(*[DecoderLayer(*params) for l in range(num_layers)])
        
        self.embedding_proj = nn.Linear(embedding_size, hidden_size, bias=False)
        self.layer_norm = LayerNorm(hidden_size)
        self.input_dropout = nn.Dropout(input_dropout)

    def forward(self, inputs, encoder_output, mask):
        mask_src, mask_trg = mask
        dec_mask = torch.gt(mask_trg + self.mask[:, :mask_trg.size(-1), :mask_trg.size(-1)], 0)

        #Add input dropout
        x = self.input_dropout(inputs)
        x = self.embedding_proj(x)
            
        # Add timing signal
        x += self.timing_signal[:, :inputs.shape[1], :].type_as(inputs.data)
        
        # Run decoder
        y, _, attn_dist, _ = self.dec((x, encoder_output, [], (mask_src,dec_mask)))

        # Final layer normalization
        y = self.layer_norm(y)
        return y, attn_dist



class EncoderLayer(nn.Module):
    """
    Represents one Encoder layer of the Transformer Encoder
    Refer Fig. 1 in https://arxiv.org/pdf/1706.03762.pdf
    NOTE: The layer normalization step has been moved to the input as per latest version of T2T
    """
    def __init__(self, hidden_size, total_key_depth, total_value_depth, filter_size, num_heads,
                 bias_mask=None, layer_dropout=0.1, attention_dropout=0.1, relu_dropout=0.1):
        """
        Parameters:
            hidden_size: Hidden size
            total_key_depth: Size of last dimension of keys. Must be divisible by num_head
            total_value_depth: Size of last dimension of values. Must be divisible by num_head
            output_depth: Size last dimension of the final output
            filter_size: Hidden size of the middle layer in FFN
            num_heads: Number of attention heads
            bias_mask: Masking tensor to prevent connections to future elements
            layer_dropout: Dropout for this layer
            attention_dropout: Dropout probability after attention (Should be non-zero only during training)
            relu_dropout: Dropout probability after relu in FFN (Should be non-zero only during training)
        """
        
        super(EncoderLayer, self).__init__()
        
        self.multi_head_attention = MultiHeadAttention(hidden_size, total_key_depth, total_value_depth, 
                                                       hidden_size, num_heads, bias_mask, attention_dropout)
        
        self.positionwise_feed_forward = PositionwiseFeedForward(hidden_size, filter_size, hidden_size,
                                                                 layer_config='ll', padding = 'both', 
                                                                 dropout=relu_dropout)
        self.dropout = nn.Dropout(layer_dropout)
        self.layer_norm_mha = LayerNorm(hidden_size)
        self.layer_norm_ffn = LayerNorm(hidden_size)
        # self.layer_norm_end = LayerNorm(hidden_size)
        
    def forward(self, inputs, mask=None):
        x = inputs
        
        # Layer Normalization
        x_norm = self.layer_norm_mha(x)
        
        # Multi-head attention
        y, _ = self.multi_head_attention(x_norm, x_norm, x_norm, mask)
        
        # Dropout and residual
        x = self.dropout(x + y)
        
        # Layer Normalization
        x_norm = self.layer_norm_ffn(x)
        
        # Positionwise Feedforward
        y = self.positionwise_feed_forward(x_norm)
        
        # Dropout and residual
        y = self.dropout(x + y)
        
        # y = self.layer_norm_end(y)
        return y



class DecoderLayer(nn.Module):
    """
    Represents one Decoder layer of the Transformer Decoder
    Refer Fig. 1 in https://arxiv.org/pdf/1706.03762.pdf
    NOTE: The layer normalization step has been moved to the input as per latest version of T2T
    """
    def __init__(self, hidden_size, total_key_depth, total_value_depth, filter_size, num_heads,
                 bias_mask,vocab_size=None , layer_dropout=0, attention_dropout=0.1, relu_dropout=0.1):
        """
        Parameters:
            hidden_size: Hidden size
            total_key_depth: Size of last dimension of keys. Must be divisible by num_head
            total_value_depth: Size of last dimension of values. Must be divisible by num_head
            output_depth: Size last dimension of the final output
            filter_size: Hidden size of the middle layer in FFN
            num_heads: Number of attention heads
            bias_mask: Masking tensor to prevent connections to future elements
            layer_dropout: Dropout for this layer
            attention_dropout: Dropout probability after attention (Should be non-zero only during training)
            relu_dropout: Dropout probability after relu in FFN (Should be non-zero only during training)
        """
        
        super(DecoderLayer, self).__init__()
        
        self.multi_head_attention_dec = MultiHeadAttention(hidden_size, total_key_depth, total_value_depth, 
                                                       hidden_size, num_heads, bias_mask, attention_dropout)

        self.multi_head_attention_enc_dec = MultiHeadAttention(hidden_size, total_key_depth, total_value_depth, 
                                                       hidden_size, num_heads, None, attention_dropout)
        
        self.positionwise_feed_forward = PositionwiseFeedForward(hidden_size, filter_size, hidden_size,
                                                                 layer_config='ll', padding = 'left', 
                                                                 dropout=relu_dropout)
        self.dropout = nn.Dropout(layer_dropout)
        self.layer_norm_mha_dec = LayerNorm(hidden_size)
        self.layer_norm_mha_enc = LayerNorm(hidden_size)
        self.layer_norm_ffn = LayerNorm(hidden_size)
        # self.layer_norm_end = LayerNorm(hidden_size)

        
    def forward(self, inputs):
        """
        NOTE: Inputs is a tuple consisting of decoder inputs and encoder output
        """

        x, encoder_outputs, attention_weight, mask = inputs
        mask_src, dec_mask = mask
        
        # Layer Normalization before decoder self attention
        x_norm = self.layer_norm_mha_dec(x)
        
        # Masked Multi-head attention
        y, _ = self.multi_head_attention_dec(x_norm, x_norm, x_norm, dec_mask)
        
        # Dropout and residual after self-attention
        x = self.dropout(x + y)

        # Layer Normalization before encoder-decoder attention
        x_norm = self.layer_norm_mha_enc(x)

        # Multi-head encoder-decoder attention
        y, attention_weight = self.multi_head_attention_enc_dec(x_norm, encoder_outputs, encoder_outputs, mask_src)

        # Dropout and residual after encoder-decoder attention
        x = self.dropout(x + y)
        
        # Layer Normalization
        x_norm = self.layer_norm_ffn(x)
        
        # Positionwise Feedforward
        y = self.positionwise_feed_forward(x_norm)
        
        # Dropout and residual after positionwise feed forward layer
        y = self.dropout(x + y)
        
        # y = self.layer_norm_end(y)
        
        # Return encoder outputs as well to work with nn.Sequential
        return y, encoder_outputs, attention_weight, mask


class PositionwiseFeedForward(nn.Module):
    """
    Does a Linear + RELU + Linear on each of the timesteps
    """
    def __init__(self, input_depth, filter_size, output_depth, layer_config='ll', padding='left', dropout=0.0):
        """
        Parameters:
            input_depth: Size of last dimension of input
            filter_size: Hidden size of the middle layer
            output_depth: Size last dimension of the final output
            layer_config: ll -> linear + ReLU + linear
                          cc -> conv + ReLU + conv etc.
            padding: left -> pad on the left side (to mask future data), 
                     both -> pad on both sides
            dropout: Dropout probability (Should be non-zero only during training)
        """
        super(PositionwiseFeedForward, self).__init__()
        
        layers = []
        sizes = ([(input_depth, filter_size)] + 
                 [(filter_size, filter_size)]*(len(layer_config)-2) + 
                 [(filter_size, output_depth)])

        for lc, s in zip(list(layer_config), sizes):
            if lc == 'l':
                layers.append(nn.Linear(*s))
            else:
                raise ValueError("Unknown layer type {}".format(lc))

        self.layers = nn.ModuleList(layers)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, inputs):
        x = inputs
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers):
                x = self.relu(x)
                x = self.dropout(x)

        return x



class MultiHeadAttention(nn.Module):
    """
    Multi-head attention as per https://arxiv.org/pdf/1706.03762.pdf
    Refer Figure 2
    """
    def __init__(self, input_depth, total_key_depth, total_value_depth, output_depth, 
                 num_heads, bias_mask=None, dropout=0.0):
        """
        Parameters:
            input_depth: Size of last dimension of input
            total_key_depth: Size of last dimension of keys. Must be divisible by num_head
            total_value_depth: Size of last dimension of values. Must be divisible by num_head
            output_depth: Size last dimension of the final output
            num_heads: Number of attention heads
            bias_mask: Masking tensor to prevent connections to future elements
            dropout: Dropout probability (Should be non-zero only during training)
        """
        super(MultiHeadAttention, self).__init__()
        # Checks borrowed from 
        # https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/layers/common_attention.py
        # if total_key_depth % num_heads != 0:
        #     raise ValueError("Key depth (%d) must be divisible by the number of "
        #                      "attention heads (%d)." % (total_key_depth, num_heads))
        # if total_value_depth % num_heads != 0:
        #     raise ValueError("Value depth (%d) must be divisible by the number of "
        #                      "attention heads (%d)." % (total_value_depth, num_heads))
            
        if total_key_depth % num_heads != 0:
            print("Key depth (%d) must be divisible by the number of "
                             "attention heads (%d)." % (total_key_depth, num_heads))
            total_key_depth = total_key_depth - (total_key_depth % num_heads)
        if total_value_depth % num_heads != 0:
            print("Value depth (%d) must be divisible by the number of "
                             "attention heads (%d)." % (total_value_depth, num_heads))
            total_value_depth = total_value_depth - (total_value_depth % num_heads)

        self.num_heads = num_heads
        self.query_scale = (total_key_depth//num_heads)**-0.5 ## sqrt
        self.bias_mask = bias_mask
        
        # Key and query depth will be same
        self.query_linear = nn.Linear(input_depth, total_key_depth, bias=False)
        self.key_linear = nn.Linear(input_depth, total_key_depth, bias=False)
        self.value_linear = nn.Linear(input_depth, total_value_depth, bias=False)
        self.output_linear = nn.Linear(total_value_depth, output_depth, bias=False)
        
        self.dropout = nn.Dropout(dropout)
    
    def _split_heads(self, x):
        """
        Split x such to add an extra num_heads dimension
        Input:
            x: a Tensor with shape [batch_size, seq_length, depth]
        Returns:
            A Tensor with shape [batch_size, num_heads, seq_length, depth/num_heads]
        """
        if len(x.shape) != 3:
            raise ValueError("x must have rank 3")
        shape = x.shape
        return x.view(shape[0], shape[1], self.num_heads, shape[2]//self.num_heads).permute(0, 2, 1, 3)
    
    def _merge_heads(self, x):
        """
        Merge the extra num_heads into the last dimension
        Input:
            x: a Tensor with shape [batch_size, num_heads, seq_length, depth/num_heads]
        Returns:
            A Tensor with shape [batch_size, seq_length, depth]
        """
        if len(x.shape) != 4:
            raise ValueError("x must have rank 4")
        shape = x.shape
        return x.permute(0, 2, 1, 3).contiguous().view(shape[0], shape[2], shape[3]*self.num_heads)
        
    def forward(self, queries, keys, values, mask):
        
        # Do a linear for each component
        queries = self.query_linear(queries)
        keys = self.key_linear(keys)
        values = self.value_linear(values)
        
        # Split into multiple heads
        queries = self._split_heads(queries)
        keys = self._split_heads(keys)
        values = self._split_heads(values)
        
        # Scale queries
        queries *= self.query_scale
        
        # Combine queries and keys
        logits = torch.matmul(queries, keys.permute(0, 1, 3, 2))
        
        if mask is not None:
            mask = mask.unsqueeze(1)  # [B, 1, 1, T_values]
            logits = logits.masked_fill(mask, -1e18)


        # # Add bias to mask future values
        # if self.bias_mask is not None:
        #     logits += self.bias_mask[:, :, :logits.shape[-2], :logits.shape[-1]].type_as(logits.data)
        
        ## attention weights 
        attetion_weights = logits.sum(dim=1)/self.num_heads

        # Convert to probabilites
        weights = nn.functional.softmax(logits, dim=-1)

        # Dropout
        weights = self.dropout(weights)
        
        # Combine with values to get context
        contexts = torch.matmul(weights, values)
        
        # Merge heads
        contexts = self._merge_heads(contexts)
        #contexts = torch.tanh(contexts)
        
        # Linear to get output
        outputs = self.output_linear(contexts)
        
        return outputs, attetion_weights



def gaussian_kld(recog_mu, recog_logvar, prior_mu, prior_logvar):
    kld = -0.5 * torch.sum(1 + (recog_logvar - prior_logvar)
                               - torch.div(torch.pow(prior_mu - recog_mu, 2), torch.exp(prior_logvar))
                               - torch.div(torch.exp(recog_logvar), torch.exp(prior_logvar)), dim=-1)
    return kld

def _gen_timing_signal(length, channels, min_timescale=1.0, max_timescale=1.0e4):
    """
    Generates a [1, length, channels] timing signal consisting of sinusoids
    Adapted from:
    https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/layers/common_attention.py
    """
    position = np.arange(length)
    num_timescales = channels // 2
    log_timescale_increment = ( math.log(float(max_timescale) / float(min_timescale)) / (float(num_timescales) - 1))
    inv_timescales = min_timescale * np.exp(np.arange(num_timescales).astype(np.float) * -log_timescale_increment)
    scaled_time = np.expand_dims(position, 1) * np.expand_dims(inv_timescales, 0)

    signal = np.concatenate([np.sin(scaled_time), np.cos(scaled_time)], axis=1)
    signal = np.pad(signal, [[0, 0], [0, channels % 2]], 'constant', constant_values=[0.0, 0.0])
    signal =  signal.reshape([1, length, channels])

    return torch.from_numpy(signal).type(torch.FloatTensor)